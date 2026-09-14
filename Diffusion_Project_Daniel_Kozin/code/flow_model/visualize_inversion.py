"""
Visualizes the inversion round-trip on real (augmented) data: a real phantom -> inverted
noise pre-image z -> re-generated reconstruction, logged to wandb as a before / z / after
figure per example. This is the same round-trip idea as sanity_checks.py's round_trip
check, but rendered visually instead of only reported as a voxel-agreement number.

    ./run_interpolate.sh --checkpoint checkpoints/latest.pt   # (unrelated, for reference)
    KMP_DUPLICATE_LIB_OK=TRUE conda run -n uni python -m flow_model.visualize_inversion \
        --checkpoint checkpoints/latest.pt --n_examples 2
"""
import argparse
import random as pyrandom
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import wandb

from .data import load_all_volumes, rotate_label_volume, split_case_ids
from .ode import invert, sample
from .velocity_model import FlowMatchingUNet3D
from .viz_utils import grid_figure, render_label_volumes


def get_device(preferred: str = "auto") -> torch.device:
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Visualize the invert -> z -> reconstruct round trip")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=Path("mri_images_3D"))
    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--n_examples", type=int, default=2,
                         help="How many real (randomly rotated) phantoms to invert and log.")
    parser.add_argument("--base_ch", type=int, default=8)
    parser.add_argument("--embed_channels", type=int, default=16)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--wandb_project", type=str, default="diffusion_project")
    parser.add_argument("--wandb_mode", type=str, default="online")
    parser.add_argument("--run_tag", type=str, default=None)
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
    rng = pyrandom.Random(args.seed)
    chosen_cases = rng.sample(val_ids, min(args.n_examples, len(val_ids)))

    run_name = "visualize_inversion"
    if args.run_tag:
        run_name = f"{run_name}_{args.run_tag}"
    wandb.init(project=args.wandb_project, mode=args.wandb_mode, name=run_name, config=vars(args))

    for i, case_id in enumerate(chosen_cases):
        # Randomly rotate the real phantom before inverting it — the model was trained on
        # 8x rotation-augmented data, so this checks inversion works on the augmented
        # distribution it actually saw, not just the raw on-disk orientation.
        orientation = rng.randrange(8)
        before_label = torch.from_numpy(
            rotate_label_volume(case_volumes[case_id][0], orientation).copy())

        x1 = F.one_hot(before_label.long(), args.num_classes).permute(3, 0, 1, 2).float().unsqueeze(0).to(device)
        x0_hat = invert(model, x1, args.n_steps)
        z_label = x0_hat.argmax(dim=1)[0].cpu()

        x1_hat = sample(model, args.n_steps, x0=x0_hat, device=device)
        after_label = x1_hat.argmax(dim=1)[0].cpu()

        agreement = (after_label == before_label).float().mean().item()

        names = [f"before ({case_id}, rot={orientation})", "z (noise pre-image)",
                 f"after (reconstruction, {agreement:.1%} voxel match)"]
        front_imgs = render_label_volumes([before_label, z_label, after_label], names=names)
        top_imgs = render_label_volumes([before_label, z_label, after_label], names=names, look_up=True)

        fig = grid_figure(front_imgs + top_imgs,
                           titles=names + [f"{n} (top)" for n in names], max_per_row=3)
        wandb.log({f"inversion/example_{i}_{case_id}": wandb.Image(fig)})
        plt.close(fig)

        print(f"Logged inversion example {i} ({case_id}, rotation={orientation}): "
              f"voxel agreement={agreement:.4f}")

    print(f"Done: {len(chosen_cases)} inversion example(s) logged to wandb.")


if __name__ == "__main__":
    main()
