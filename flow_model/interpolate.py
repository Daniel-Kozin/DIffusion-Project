import argparse
import random as pyrandom
from itertools import combinations
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import wandb

from .config import Config
from .data import load_all_volumes, rotate_label_volume, split_case_ids
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

    seed: if given, reseeds the global RNG before drawing z_a/z_b (useful for a single
    reproducible pair). Pass None (and seed the RNG once yourself beforehand) to draw a
    fresh, distinct pair each call — e.g. when looping over multiple noise pairs.
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
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--n_pairs", type=int, default=10,
                         help="Number of real-phantom pairs and number of noise pairs to interpolate "
                              "(each pair distinct; real pairs are drawn as distinct combinations of "
                              "validation case IDs, so individual phantoms may recur across pairs).")
    parser.add_argument("--method", type=str, default="slerp", choices=["slerp", "linear"])
    parser.add_argument("--base_ch", type=int, default=8)
    parser.add_argument("--embed_channels", type=int, default=16)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--wandb_project", type=str, default="diffusion_project")
    parser.add_argument("--wandb_mode", type=str, default="online")
    parser.add_argument("--run_tag", type=str, default=None,
                         help="Appended to the wandb run name, e.g. to identify which checkpoint this "
                              "interpolation run is from.")
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
        raise RuntimeError("Need at least 2 validation case IDs to form a real interpolation pair")

    all_pairs = list(combinations(val_ids, 2))
    n_real_pairs = min(args.n_pairs, len(all_pairs))
    if n_real_pairs < args.n_pairs:
        print(f"[interpolate] only {len(all_pairs)} distinct validation pairs available "
              f"({len(val_ids)} case IDs) — using {n_real_pairs} instead of {args.n_pairs}")
    real_pairs = pyrandom.Random(args.seed).sample(all_pairs, n_real_pairs)
    rot_rng = pyrandom.Random(args.seed + 1)

    run_name = f"interpolate_{args.n_pairs}pairs_{len(args.alphas)}alphas"
    if args.run_tag:
        run_name = f"{run_name}_{args.run_tag}"
    wandb.init(project=args.wandb_project, mode=args.wandb_mode, name=run_name, config=vars(args))

    # --- real-phantom interpolation: reference renders are the actual ground-truth phantoms ---
    for pair_idx, (case_a, case_b) in enumerate(real_pairs):
        # Randomly rotate each endpoint (independently) so pairs aren't stuck at their
        # original in-dataset orientation, which tended to leave lumps clustered close
        # together across pairs; same 8-way augmentation used during training.
        orient_a, orient_b = rot_rng.randrange(8), rot_rng.randrange(8)
        phantom_a = torch.from_numpy(rotate_label_volume(case_volumes[case_a][0], orient_a).copy())
        phantom_b = torch.from_numpy(rotate_label_volume(case_volumes[case_b][0], orient_b).copy())

        render_a = render_single(phantom_a, case_a)
        render_b = render_single(phantom_b, case_b)
        render_a_top = render_single(phantom_a, case_a, look_up=True)
        render_b_top = render_single(phantom_b, case_b, look_up=True)

        real_results = interpolate_real(model, phantom_a, phantom_b, args.n_steps, args.alphas,
                                         num_classes=args.num_classes, method=args.method, device=device)
        for alpha, label in real_results:
            render_interp = render_single(label, f"real_alpha_{alpha:.2f}")
            render_interp_top = render_single(label, f"real_alpha_{alpha:.2f}_top", look_up=True)
            log_interpolation_step(f"real/pair{pair_idx}_{case_a}_{case_b}",
                                    render_a, render_b, render_interp,
                                    alpha, phantom_a, phantom_b, case_a, case_b,
                                    render_a_top=render_a_top, render_b_top=render_b_top,
                                    render_interp_top=render_interp_top, vol_interp=label)
        print(f"Logged real interpolation pair {pair_idx} ({case_a} -> {case_b}).")

    # --- noise interpolation: reference renders are the decoded endpoint samples ---
    torch.manual_seed(args.seed)  # seed once; each loop iteration then draws a fresh, distinct pair
    for pair_idx in range(args.n_pairs):
        z_a_label, z_b_label, noise_results = interpolate_noise(
            model, args.n_steps, args.alphas, num_classes=args.num_classes,
            method=args.method, seed=None, device=device)
        render_noise_a = render_single(z_a_label, f"noise A pair{pair_idx}")
        render_noise_b = render_single(z_b_label, f"noise B pair{pair_idx}")
        render_noise_a_top = render_single(z_a_label, f"noise A pair{pair_idx} top", look_up=True)
        render_noise_b_top = render_single(z_b_label, f"noise B pair{pair_idx} top", look_up=True)
        for alpha, label in noise_results:
            render_interp = render_single(label, f"noise_pair{pair_idx}_alpha_{alpha:.2f}")
            render_interp_top = render_single(label, f"noise_pair{pair_idx}_alpha_{alpha:.2f}_top", look_up=True)
            log_interpolation_step(f"noise/pair{pair_idx}",
                                    render_noise_a, render_noise_b, render_interp,
                                    alpha, z_a_label, z_b_label, "noise A", "noise B",
                                    render_a_top=render_noise_a_top, render_b_top=render_noise_b_top,
                                    render_interp_top=render_interp_top, vol_interp=label)
        print(f"Logged noise interpolation pair {pair_idx}.")

    print(f"Done: {args.n_pairs} real pairs and {args.n_pairs} noise pairs, "
          f"{len(args.alphas)} alpha steps each, front + top-down views.")


if __name__ == "__main__":
    main()
