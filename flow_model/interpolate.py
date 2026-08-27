import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import wandb

from .config import Config
from .data import load_all_volumes, split_case_ids
from .ode import invert, sample
from .velocity_model import FlowMatchingUNet3D
from .viz_utils import log_interpolation_step, render_single


def interp(x_a: torch.Tensor, x_b: torch.Tensor, alpha: float, method: str = "slerp") -> torch.Tensor:
    """
    method='linear': (1-alpha)*x_a + alpha*x_b
    method='slerp': spherical interpolation along the great circle between x_a and x_b,
        preserving vector norm (linear interpolation of two ~N(0,I) points drifts toward
        the origin, off the Gaussian's typical shell, handing the model an out-of-distribution
        low-norm input at mid-alphas; slerp avoids this).
    """
    if method == "linear":
        return (1 - alpha) * x_a + alpha * x_b

    if method == "slerp":
        a_flat = x_a.flatten()
        b_flat = x_b.flatten()
        cos_theta = torch.dot(a_flat, b_flat) / (a_flat.norm() * b_flat.norm() + 1e-8)
        cos_theta = cos_theta.clamp(-1.0, 1.0)
        theta = torch.acos(cos_theta)
        sin_theta = torch.sin(theta)
        if sin_theta.abs() < 1e-6:
            return (1 - alpha) * x_a + alpha * x_b
        w_a = torch.sin((1 - alpha) * theta) / sin_theta
        w_b = torch.sin(alpha * theta) / sin_theta
        return w_a * x_a + w_b * x_b

    raise ValueError(f"Unknown method: {method}")


def interpolate_real(model: FlowMatchingUNet3D, phantom_a: torch.Tensor, phantom_b: torch.Tensor,
                      n_steps: int, alphas: List[float], num_classes: int = 4,
                      method: str = "slerp", device: str = "cpu") -> List[Tuple[float, torch.Tensor]]:
    """
    phantom_a/b: int64 label volumes [K, H, W] (values 0..num_classes-1).
    Returns a list of (alpha, label_volume) tuples, in order.
    """
    x1_a = F.one_hot(phantom_a.long(), num_classes).permute(3, 0, 1, 2).float().unsqueeze(0).to(device)
    x1_b = F.one_hot(phantom_b.long(), num_classes).permute(3, 0, 1, 2).float().unsqueeze(0).to(device)

    x0_a = invert(model, x1_a, n_steps)
    x0_b = invert(model, x1_b, n_steps)

    results = []
    for alpha in alphas:
        z = interp(x0_a, x0_b, alpha, method=method)
        x1_hat = sample(model, n_steps, x0=z, device=device)
        results.append((alpha, x1_hat.argmax(dim=1)[0].cpu()))
    return results


def interpolate_noise(model: FlowMatchingUNet3D, n_steps: int, alphas: List[float],
                       num_classes: int = 4, shape=(26, 128, 128), method: str = "slerp",
                       seed: Optional[int] = None,
                       device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[float, torch.Tensor]]]:
    """
    Samples z_A, z_B ~ N(0,I) directly (no inversion) and interpolates between them.
    Returns (z_a_label, z_b_label, results) where z_a_label/z_b_label are the decoded
    endpoint samples (argmax label volumes) — used as the reference renders for these
    endpoints since the raw noise vectors themselves aren't meaningful to visualize.
    """
    if seed is not None:
        torch.manual_seed(seed)

    full_shape = (num_classes, *shape)
    z_a = torch.randn(1, *full_shape, device=device)
    z_b = torch.randn(1, *full_shape, device=device)

    z_a_label = sample(model, n_steps, x0=z_a, device=device).argmax(dim=1)[0].cpu()
    z_b_label = sample(model, n_steps, x0=z_b, device=device).argmax(dim=1)[0].cpu()

    results = []
    for alpha in alphas:
        z = interp(z_a, z_b, alpha, method=method)
        x1_hat = sample(model, n_steps, x0=z, device=device)
        results.append((alpha, x1_hat.argmax(dim=1)[0].cpu()))
    return z_a_label, z_b_label, results


def get_device(preferred: str = "auto") -> torch.device:
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Real-phantom and noise interpolation via flow matching")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=Path("mri_images_3D"))
    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--method", type=str, default="slerp", choices=["slerp", "linear"])
    parser.add_argument("--base_ch", type=int, default=8)
    parser.add_argument("--embed_channels", type=int, default=16)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--wandb_project", type=str, default="diffusion_project")
    parser.add_argument("--wandb_mode", type=str, default="online")
    args = parser.parse_args()

    device = get_device(args.device)

    model = FlowMatchingUNet3D(num_classes=args.num_classes, base_ch=args.base_ch,
                                embed_channels=args.embed_channels).to(device)
    # weights_only=False: trusted, locally-generated checkpoint
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    case_volumes = load_all_volumes(args.data_dir)
    _, val_ids = split_case_ids(list(case_volumes.keys()), val_frac=args.val_frac, seed=args.seed)
    if len(val_ids) < 2:
        raise RuntimeError("Need at least 2 validation case IDs to pick a real interpolation pair")

    case_a, case_b = val_ids[0], val_ids[1]
    phantom_a = torch.from_numpy(case_volumes[case_a][0])
    phantom_b = torch.from_numpy(case_volumes[case_b][0])

    wandb.init(project=args.wandb_project, mode=args.wandb_mode,
               name=f"interpolate_{case_a}_{case_b}", config=vars(args))

    # --- real-phantom interpolation: reference renders are the actual ground-truth phantoms ---
    render_a = render_single(phantom_a, case_a)
    render_b = render_single(phantom_b, case_b)

    real_results = interpolate_real(model, phantom_a, phantom_b, args.n_steps, args.alphas,
                                     num_classes=args.num_classes, method=args.method, device=device)
    for alpha, label in real_results:
        render_interp = render_single(label, f"real_alpha_{alpha:.2f}")
        log_interpolation_step("interpolation/real", render_a, render_b, render_interp,
                                alpha, phantom_a, phantom_b, case_a, case_b)

    # --- noise interpolation: reference renders are the decoded endpoint samples ---
    z_a_label, z_b_label, noise_results = interpolate_noise(
        model, args.n_steps, args.alphas, num_classes=args.num_classes,
        method=args.method, seed=args.seed, device=device)
    render_noise_a = render_single(z_a_label, "noise A")
    render_noise_b = render_single(z_b_label, "noise B")
    for alpha, label in noise_results:
        render_interp = render_single(label, f"noise_alpha_{alpha:.2f}")
        log_interpolation_step("interpolation/noise", render_noise_a, render_noise_b, render_interp,
                                alpha, z_a_label, z_b_label, "noise A", "noise B")

    print(f"Logged real interpolation ({case_a} -> {case_b}) and noise interpolation to wandb "
          f"({len(args.alphas)} steps each).")


if __name__ == "__main__":
    main()
