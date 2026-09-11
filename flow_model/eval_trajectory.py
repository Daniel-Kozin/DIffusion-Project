"""
Full-dataset evaluation plus a t=0->1 trajectory visualization for a single rotate180
checkpoint. Two things eval_rotate180.py doesn't do: (1) reports the full per-pair
distribution (std/min/max), not just the mean, since this task has repeatedly shown real
pair-to-pair variance that a mean alone hides; (2) renders evenly-spaced snapshots of the ODE
integration itself for a few example pairs, showing how the prediction actually evolves from
the original phantom (t=0) to the final output (t=1), not just the before/after.

    ./run_eval_trajectory.sh --checkpoint checkpoints/latest.pt --n_eval_pairs 50 --n_trajectory_examples 3
"""
import argparse
import random
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import wandb

from .data import build_rotation_pair_datasets
from .eval_rotate180 import evaluate_checkpoint, load_model_from_checkpoint, summarize
from .metrics import lump_centroid_hw, mask_dice
from .ode import sample
from .train import get_device
from .viz_utils import render_label_volumes


def spread(values: dict) -> dict:
    """std/min/max per metric list from evaluate_checkpoint(), nan if <2 points."""
    out = {}
    for k, v in values.items():
        if len(v) >= 2:
            out[f"{k}_std"] = statistics.stdev(v)
            out[f"{k}_min"] = min(v)
            out[f"{k}_max"] = max(v)
        else:
            out[f"{k}_std"] = out[f"{k}_min"] = out[f"{k}_max"] = float("nan")
    return out


def render_trajectory_figure(model, x0: torch.Tensor, x1: torch.Tensor, n_steps: int, n_frames: int,
                              device: torch.device, title: str):
    """
    Integrates x0 -> x1_hat over n_steps (capturing every intermediate state), decodes
    n_frames evenly-spaced snapshots (including t=0 and t=1) to label volumes, renders each,
    and plots them in a row alongside the true target for reference. Also tracks each frame's
    lump centroid and its Dice-vs-target (on the lump+pillar combined mask) so the plot shows
    not just what each frame looks like, but how close it already is to the true answer.
    """
    x0_b, x1_b = x0.unsqueeze(0).to(device), x1.unsqueeze(0).to(device)
    _, trajectory = sample(model, n_steps, x0=x0_b, device=device, t_start=0.0, t_end=1.0,
                            return_trajectory=True)
    total = len(trajectory)
    frame_steps = [round(i * (total - 1) / (n_frames - 1)) for i in range(n_frames)]
    t_values = [s / (total - 1) for s in frame_steps]

    expected_label = x1.argmax(dim=0)
    frame_labels = [trajectory[s].argmax(dim=1)[0].cpu() for s in frame_steps]
    frame_imgs = render_label_volumes(frame_labels, names=[f"{title}_t{t:.2f}" for t in t_values])
    dice_vs_target = [mask_dice(fl, expected_label, (2, 3)) or 0.0 for fl in frame_labels]

    fig, axes = plt.subplots(2, n_frames, figsize=(2.2 * n_frames, 5.2),
                              gridspec_kw={"height_ratios": [3, 1]})
    for col in range(n_frames):
        axes[0, col].imshow(frame_imgs[col])
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])
        axes[0, col].set_title(f"t={t_values[col]:.2f}", fontsize=9)
        for spine in axes[0, col].spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("tab:blue" if col == 0 else ("tab:orange" if col == n_frames - 1 else "gray"))
            spine.set_linewidth(2)

    for ax in axes[1, :]:
        ax.remove()
    ax_trend = fig.add_subplot(2, 1, 2)
    ax_trend.plot(t_values, dice_vs_target, marker="o", color="tab:green")
    ax_trend.set_xlabel("t")
    ax_trend.set_ylabel("combined Dice\nvs true target")
    ax_trend.set_ylim(-0.05, 1.05)
    ax_trend.grid(alpha=0.3)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    return fig, dice_vs_target


def main():
    parser = argparse.ArgumentParser(description="Full-dataset eval + t=0->1 trajectory visualization")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=Path("mri_images_3D"))
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--val_frac", type=float, default=0.2,
                         help="MUST match the training run's --val_frac.")
    parser.add_argument("--seed", type=int, default=0, help="MUST match the training run's --seed.")
    parser.add_argument("--n_eval_pairs", type=int, default=-1, help="-1 = full val set.")
    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--method", type=str, default="euler", choices=["euler", "heun"])
    parser.add_argument("--n_trajectory_examples", type=int, default=3,
                         help="How many validation pairs get a full t=0->1 trajectory figure.")
    parser.add_argument("--n_trajectory_frames", type=int, default=10,
                         help="Evenly-spaced snapshots per trajectory (including t=0 and t=1).")
    parser.add_argument("--trajectory_selection", type=str, default="spread",
                         choices=["spread", "best", "worst", "random"],
                         help="Which pairs get a trajectory figure: 'spread' = evenly across "
                              "the combined-Dice distribution (a good + a bad + a middling "
                              "example), 'best'/'worst' = top/bottom by final combined Dice, "
                              "'random' = uniform random pick.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--wandb_project", type=str, default="diffusion_project")
    parser.add_argument("--wandb_mode", type=str, default="online")
    parser.add_argument("--run_tag", type=str, default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    model, epoch_trained = load_model_from_checkpoint(args.checkpoint, device)
    print(f"[eval_trajectory] checkpoint epoch {epoch_trained}")

    _, val_ds = build_rotation_pair_datasets(args.data_dir, val_frac=args.val_frac,
                                              seed=args.seed, num_classes=args.num_classes)
    val_case_ids = sorted({val_ds.index[i][0] for i in range(len(val_ds))})
    print(f"[eval_trajectory] val_frac={args.val_frac} seed={args.seed} -> "
          f"{len(val_case_ids)} val case_ids: {val_case_ids}")
    print(f"[eval_trajectory] {len(val_ds)} val pairs total")
    if len(val_ds) == 0:
        raise RuntimeError("No validation pairs found -- check --data_dir/--val_frac/--seed")

    rng = random.Random(args.seed)
    if args.n_eval_pairs <= 0 or args.n_eval_pairs >= len(val_ds):
        eval_indices = list(range(len(val_ds)))
    else:
        eval_indices = rng.sample(range(len(val_ds)), args.n_eval_pairs)
    print(f"[eval_trajectory] evaluating {len(eval_indices)} pairs (metrics-only pass, no rendering)")

    run_name = f"eval_trajectory_{Path(args.checkpoint).stem}_{len(eval_indices)}pairs"
    if args.run_tag:
        run_name = f"{run_name}_{args.run_tag}"
    wandb.init(project=args.wandb_project, mode=args.wandb_mode, name=run_name, config=vars(args))

    # --- 1. full-dataset metrics, including spread (not just mean) ---
    values = evaluate_checkpoint(model, val_ds, eval_indices, device, args.n_steps, method=args.method)
    s = summarize(values)
    sp = spread(values)
    print(f"[eval_trajectory] combined_dice: mean={s['combined_dice_mean']:.4f}  "
          f"std={sp['combined_dice_std']:.4f}  min={sp['combined_dice_min']:.4f}  max={sp['combined_dice_max']:.4f}")
    print(f"[eval_trajectory] lump_dice:     mean={s['lump_dice_mean']:.4f}  "
          f"std={sp['lump_dice_std']:.4f}  min={sp['lump_dice_min']:.4f}  max={sp['lump_dice_max']:.4f}")
    print(f"[eval_trajectory] pillar_dice:   mean={s['pillar_dice_mean']:.4f}  "
          f"std={sp['pillar_dice_std']:.4f}  min={sp['pillar_dice_min']:.4f}  max={sp['pillar_dice_max']:.4f}")
    wandb.log({"checkpoint_epoch": epoch_trained, **{f"full_eval/{k}": v for k, v in {**s, **sp}.items()}})

    # histogram of the per-pair distribution -- the mean alone has repeatedly hidden
    # bimodal/high-variance behavior on this task
    fig_hist, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values["combined_dice"], bins=20, color="tab:blue", edgecolor="white")
    ax.set_xlabel("combined Dice (this pair)")
    ax.set_ylabel("# validation pairs")
    ax.set_title(f"Per-pair combined Dice distribution (n={len(values['combined_dice'])})")
    plt.tight_layout()
    wandb.log({"full_eval/combined_dice_histogram": wandb.Image(fig_hist)})
    plt.close(fig_hist)

    # --- 2. t=0->1 trajectory visualization for a handful of example pairs ---
    order = sorted(range(len(eval_indices)), key=lambda i: values["combined_dice"][i]
                   if i < len(values["combined_dice"]) else 0.0)
    n_traj = min(args.n_trajectory_examples, len(eval_indices))
    if args.trajectory_selection == "best":
        chosen = order[-n_traj:][::-1]
    elif args.trajectory_selection == "worst":
        chosen = order[:n_traj]
    elif args.trajectory_selection == "random":
        chosen = rng.sample(range(len(eval_indices)), n_traj)
    else:  # spread
        chosen = [order[round(i * (len(order) - 1) / max(n_traj - 1, 1))] for i in range(n_traj)]

    print(f"[eval_trajectory] rendering {n_traj} trajectory figures "
          f"({args.n_trajectory_frames} frames each, selection={args.trajectory_selection})...")
    for rank, i in enumerate(chosen):
        idx = eval_indices[i]
        x0, x1 = val_ds[idx]
        case_id, variant_idx, orientation = val_ds.index[idx]
        title = f"{case_id} o{orientation} (final combined Dice={values['combined_dice'][i]:.3f})"
        fig_traj, dice_curve = render_trajectory_figure(model, x0, x1, args.n_steps, args.n_trajectory_frames,
                                                          device, title)
        wandb.log({f"trajectory/{args.trajectory_selection}_{rank}_{case_id}_o{orientation}": wandb.Image(fig_traj)})
        plt.close(fig_traj)

    print("[eval_trajectory] done.")


if __name__ == "__main__":
    main()
