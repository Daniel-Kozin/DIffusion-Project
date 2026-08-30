"""
Sweeps the "invert only partway" idea from future_idea.md: instead of inverting both real
endpoints all the way to t=0 before interpolating, invert to some intermediate t_mid > 0,
slerp there, and integrate forward only the remaining distance. For each t_mid this reports
two quantitative metrics (no rendering needed to compute them):

  centroid_mae_voxels    mean absolute distance (in voxels) between where a naive linear
                         blend of the two real endpoints' lump positions would put the lump,
                         and where the model's generated interpolation actually puts it.
  avg_anatomy_components number of disconnected chunks the generated non-background anatomy
                         splits into (real single phantoms are always 1 connected piece).
  avg_lump_components    same, but just for the lump class (real single-lump phantoms are
                         always exactly 1 lump).
Lower is better on all three.

    ./run_sweep_tmid.sh --checkpoint checkpoints/latest.pt
"""
import argparse
import random as pyrandom
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from scipy.ndimage import label as cc_label

from .data import load_all_volumes, rotate_label_volume, split_case_ids
from .interpolate import interp
from .ode import invert, sample
from .velocity_model import FlowMatchingUNet3D
from .viz_utils import lump_centroid_hw, log_interpolation_step, render_single


def get_device(preferred: str = "auto") -> torch.device:
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_components(label_volume: torch.Tensor, classes) -> int:
    """Number of connected components (26-connectivity in 3D) of voxels whose class is in
    `classes`. 0 if there are no such voxels at all."""
    mask = torch.zeros_like(label_volume, dtype=torch.bool)
    for c in classes:
        mask |= (label_volume == c)
    mask = mask.numpy()
    if not mask.any():
        return 0
    structure = np.ones((3, 3, 3), dtype=int)  # full 26-connectivity
    _, n = cc_label(mask, structure=structure)
    return int(n)


def main():
    parser = argparse.ArgumentParser(description="Sweep t_mid (partial inversion) and report quality metrics")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=Path("mri_images_3D"))
    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--t_mids", type=float, nargs="+",
                         default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--n_pairs", type=int, default=20)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--method", type=str, default="slerp", choices=["slerp", "linear"])
    parser.add_argument("--base_ch", type=int, default=8)
    parser.add_argument("--embed_channels", type=int, default=16)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--wandb_project", type=str, default="diffusion_project")
    parser.add_argument("--wandb_mode", type=str, default="online")
    parser.add_argument("--run_tag", type=str, default=None)
    parser.add_argument("--render_n_examples", type=int, default=5,
                         help="For every t_mid, also render this many example pairs (front + "
                              "top view each), grouped into their own wandb folder per t_mid. "
                              "0 disables rendering entirely.")
    args = parser.parse_args()

    device = get_device(args.device)

    model = FlowMatchingUNet3D(num_classes=args.num_classes, base_ch=args.base_ch,
                                embed_channels=args.embed_channels).to(device)
    # weights_only=False: trusted, locally-generated checkpoint
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    case_volumes = load_all_volumes(args.data_dir)
    val_ids = split_case_ids(list(case_volumes.keys()), val_frac=args.val_frac, seed=args.seed)[1]
    all_pairs = list(combinations(val_ids, 2))
    n_real_pairs = min(args.n_pairs, len(all_pairs))
    real_pairs = pyrandom.Random(args.seed).sample(all_pairs, n_real_pairs)
    rot_rng = pyrandom.Random(args.seed + 1)

    # Pre-rotate the phantoms once so every t_mid in the sweep is tested on the exact same
    # pairs/orientations -- otherwise differences across t_mid could just be different pairs.
    pairs_data = []
    for case_a, case_b in real_pairs:
        orient_a, orient_b = rot_rng.randrange(8), rot_rng.randrange(8)
        phantom_a = torch.from_numpy(rotate_label_volume(case_volumes[case_a][0], orient_a).copy())
        phantom_b = torch.from_numpy(rotate_label_volume(case_volumes[case_b][0], orient_b).copy())
        x1_a = F.one_hot(phantom_a.long(), args.num_classes).permute(3, 0, 1, 2).float().unsqueeze(0).to(device)
        x1_b = F.one_hot(phantom_b.long(), args.num_classes).permute(3, 0, 1, 2).float().unsqueeze(0).to(device)
        pairs_data.append({
            "case_a": case_a, "case_b": case_b,
            "phantom_a": phantom_a, "phantom_b": phantom_b,
            "pos_a": lump_centroid_hw(phantom_a), "pos_b": lump_centroid_hw(phantom_b),
            "x1_a": x1_a, "x1_b": x1_b,
        })

    run_name = f"sweep_tmid_{len(args.t_mids)}vals_{n_real_pairs}pairs"
    if args.run_tag:
        run_name = f"{run_name}_{args.run_tag}"
    wandb.init(project=args.wandb_project, mode=args.wandb_mode, name=run_name, config=vars(args))

    n_render_pairs = min(args.render_n_examples, len(pairs_data))

    for t_mid in args.t_mids:
        centroid_errors = []
        anatomy_components = []
        lump_components = []

        for pi, pd in enumerate(pairs_data):
            x_mid_a = invert(model, pd["x1_a"], args.n_steps, t_start=1.0, t_end=t_mid)
            x_mid_b = invert(model, pd["x1_b"], args.n_steps, t_start=1.0, t_end=t_mid)

            do_render = pi < n_render_pairs
            if do_render:
                render_a = render_single(pd["phantom_a"], pd["case_a"])
                render_b = render_single(pd["phantom_b"], pd["case_b"])
                render_a_top = render_single(pd["phantom_a"], pd["case_a"], look_up=True)
                render_b_top = render_single(pd["phantom_b"], pd["case_b"], look_up=True)

            for alpha in args.alphas:
                z = interp(x_mid_a, x_mid_b, alpha, method=args.method)
                out = sample(model, args.n_steps, x0=z, device=device, t_start=t_mid, t_end=1.0)
                label = out.argmax(dim=1)[0].cpu()

                if pd["pos_a"] is not None and pd["pos_b"] is not None:
                    pos_expected = ((1 - alpha) * pd["pos_a"][0] + alpha * pd["pos_b"][0],
                                     (1 - alpha) * pd["pos_a"][1] + alpha * pd["pos_b"][1])
                    pos_actual = lump_centroid_hw(label)
                    if pos_actual is not None:
                        err = ((pos_expected[0] - pos_actual[0]) ** 2
                               + (pos_expected[1] - pos_actual[1]) ** 2) ** 0.5
                        centroid_errors.append(err)

                anatomy_components.append(count_components(label, classes=(1, 2, 3)))
                lump_components.append(count_components(label, classes=(3,)))

                if do_render:
                    render_interp = render_single(label, f"tmid{t_mid:.2f}_alpha{alpha:.2f}")
                    render_interp_top = render_single(label, f"tmid{t_mid:.2f}_alpha{alpha:.2f}_top",
                                                       look_up=True)
                    log_interpolation_step(
                        f"tmid_{t_mid:.2f}/pair{pi}_{pd['case_a']}_{pd['case_b']}_alpha{alpha:.2f}",
                        render_a, render_b, render_interp, alpha,
                        pd["phantom_a"], pd["phantom_b"], pd["case_a"], pd["case_b"],
                        render_a_top=render_a_top, render_b_top=render_b_top,
                        render_interp_top=render_interp_top, vol_interp=label)

        centroid_mae = float(np.mean(centroid_errors)) if centroid_errors else float("nan")
        avg_anatomy = float(np.mean(anatomy_components))
        avg_lump = float(np.mean(lump_components))

        print(f"t_mid={t_mid:.2f}  centroid_mae_voxels={centroid_mae:.2f}  "
              f"avg_anatomy_components={avg_anatomy:.2f}  avg_lump_components={avg_lump:.2f}  "
              f"(n={len(centroid_errors)} pair-alpha combos)")
        wandb.log({
            "t_mid": t_mid,
            "centroid_mae_voxels": centroid_mae,
            "avg_anatomy_components": avg_anatomy,
            "avg_lump_components": avg_lump,
        })

    print("Done sweeping t_mid.")


if __name__ == "__main__":
    main()
