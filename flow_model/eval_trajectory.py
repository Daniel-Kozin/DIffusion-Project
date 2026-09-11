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
from matplotlib.gridspec import GridSpecFromSubplotSpec
import torch
import wandb

from .data import build_rotation_pair_datasets
from .eval_rotate180 import evaluate_checkpoint, load_model_from_checkpoint, summarize
from .metrics import lump_centroid_hw
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
    n_frames evenly-spaced snapshots (including t=0 and t=1) to label volumes, and renders:
      - left: the lump centroid's (W, H) position at every one of those frames, connected in
        order and color-graded t=0 (blue) -> t=1 (red), plus the true target position as a
        green star -- the same position-schematic language used in rotation_result_figure,
        generalized from 3 fixed points to the whole path.
      - top right: original (t=0) and final prediction (t=1), shown large.
      - bottom right: every frame as a filmstrip, in order.
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

    frame_positions = [lump_centroid_hw(fl) for fl in frame_labels]
    expected_position = lump_centroid_hw(expected_label)
    H, W = x0.shape[-2], x0.shape[-1]
    fallback = (W / 2, H / 2)
    ws = [p[0] if p is not None else fallback[0] for p in frame_positions]
    hs = [p[1] if p is not None else fallback[1] for p in frame_positions]

    # constrained_layout (not tight_layout) is required here: the top (orig/pred) and bottom
    # (filmstrip) rows are each their OWN nested gridspec spanning the full available width,
    # rather than sharing one grid with unused columns -- tight_layout doesn't compact nested
    # gridspecs correctly and leaves a large blank gap where the unused columns used to be.
    fig = plt.figure(figsize=(1.7 * n_frames, 5.5), layout="constrained")
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 2.6])

    # --- left: lump centroid position at every t, connected and color-graded by t ---
    ax_left = fig.add_subplot(outer[0, 0])
    ax_left.plot(ws, hs, "-", color="gray", alpha=0.5, zorder=1, linewidth=1)
    sc = ax_left.scatter(ws, hs, c=t_values, cmap="coolwarm", s=90, zorder=3,
                          edgecolor="black", linewidth=0.5)
    if expected_position is not None:
        ax_left.scatter([expected_position[0]], [expected_position[1]], s=260, color="tab:green",
                         marker="*", zorder=4, label="expected (true 180°)")
        ax_left.annotate("expected", expected_position, textcoords="offset points", xytext=(10, -10),
                          fontsize=9, color="tab:green")
        ax_left.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax_left.annotate("t=0 (original)", (ws[0], hs[0]), textcoords="offset points", xytext=(-10, 10),
                      ha="right", fontsize=9, color="tab:blue")
    ax_left.annotate("t=1 (predicted)", (ws[-1], hs[-1]), textcoords="offset points", xytext=(10, 10),
                      ha="left", fontsize=9, color="tab:red")
    fig.colorbar(sc, ax=ax_left, fraction=0.046, pad=0.04).set_label("t")
    ax_left.set_xlim(0, W)
    ax_left.set_ylim(H, 0)  # inverted so row 0 is at the top, matching image orientation
    ax_left.set_aspect("equal")
    ax_left.set_xlabel("W (voxels)")
    ax_left.set_ylabel("H (voxels)")
    ax_left.set_title("Lump centroid position, t=0 -> 1")

    # --- top right: original (t=0) and predicted (t=1), large -- own nested 1x2 grid so it
    # fills the full row width with no unused columns ---
    right = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 1], height_ratios=[1, 3.2])
    top_gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=right[0, 0])
    bottom_gs = GridSpecFromSubplotSpec(1, n_frames, subplot_spec=right[1, 0])

    ax_orig = fig.add_subplot(top_gs[0, 0])
    ax_orig.imshow(frame_imgs[0], aspect="equal")
    ax_orig.set_title("original (t=0)", fontsize=10, color="tab:blue")
    ax_orig.set_xticks([])
    ax_orig.set_yticks([])
    for spine in ax_orig.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("tab:blue")
        spine.set_linewidth(3)

    ax_pred = fig.add_subplot(top_gs[0, 1])
    ax_pred.imshow(frame_imgs[-1], aspect="equal")
    ax_pred.set_title("predicted (t=1)", fontsize=10, color="tab:red")
    ax_pred.set_xticks([])
    ax_pred.set_yticks([])
    for spine in ax_pred.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("tab:red")
        spine.set_linewidth(3)

    # --- bottom right: every frame, in order -- own nested 1xn grid, own full row width ---
    for col in range(n_frames):
        ax = fig.add_subplot(bottom_gs[0, col])
        ax.imshow(frame_imgs[col], aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"t={t_values[col]:.2f}", fontsize=10)
        color = "tab:blue" if col == 0 else ("tab:red" if col == n_frames - 1 else "gray")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(color)
            spine.set_linewidth(1.5)

    fig.suptitle(title, fontsize=12)
    return fig


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
    parser.add_argument("--eval_train_too", type=lambda v: str(v).lower() in ("1", "true", "yes"), default=False,
                         help="Also evaluate a matching-size random sample of TRAINING pairs, logged "
                              "alongside the validation numbers -- directly checks overfitting "
                              "(train much better than val => memorizing, not generalizing).")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--wandb_project", type=str, default="diffusion_project")
    parser.add_argument("--wandb_mode", type=str, default="online")
    parser.add_argument("--run_tag", type=str, default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    model, epoch_trained = load_model_from_checkpoint(args.checkpoint, device)
    print(f"[eval_trajectory] checkpoint epoch {epoch_trained}")

    train_ds, val_ds = build_rotation_pair_datasets(args.data_dir, val_frac=args.val_frac,
                                                      seed=args.seed, num_classes=args.num_classes)
    val_case_ids = sorted({val_ds.index[i][0] for i in range(len(val_ds))})
    print(f"[eval_trajectory] val_frac={args.val_frac} seed={args.seed} -> "
          f"{len(val_case_ids)} val case_ids: {val_case_ids}")
    print(f"[eval_trajectory] {len(val_ds)} val pairs total, {len(train_ds)} train pairs total")
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

    def report(tag: str, values: dict) -> None:
        s = summarize(values)
        sp = spread(values)
        print(f"[eval_trajectory][{tag}] combined_dice: mean={s['combined_dice_mean']:.4f}  "
              f"std={sp['combined_dice_std']:.4f}  min={sp['combined_dice_min']:.4f}  max={sp['combined_dice_max']:.4f}")
        print(f"[eval_trajectory][{tag}] lump_dice:     mean={s['lump_dice_mean']:.4f}  "
              f"std={sp['lump_dice_std']:.4f}  min={sp['lump_dice_min']:.4f}  max={sp['lump_dice_max']:.4f}")
        print(f"[eval_trajectory][{tag}] pillar_dice:   mean={s['pillar_dice_mean']:.4f}  "
              f"std={sp['pillar_dice_std']:.4f}  min={sp['pillar_dice_min']:.4f}  max={sp['pillar_dice_max']:.4f}")
        wandb.log({"checkpoint_epoch": epoch_trained, **{f"full_eval_{tag}/{k}": v for k, v in {**s, **sp}.items()}})

    # --- 1. full validation-set metrics, including spread (not just mean) ---
    values = evaluate_checkpoint(model, val_ds, eval_indices, device, args.n_steps, method=args.method)
    report("val", values)

    # --- 1b. optional: matching-size train-set sample, to check overfitting ---
    if args.eval_train_too:
        n_train_sample = min(len(eval_indices), len(train_ds))
        train_indices = rng.sample(range(len(train_ds)), n_train_sample)
        print(f"[eval_trajectory] evaluating {len(train_indices)} TRAIN pairs for comparison...")
        train_values = evaluate_checkpoint(model, train_ds, train_indices, device, args.n_steps, method=args.method)
        report("train", train_values)
        val_dice = summarize(values)["combined_dice_mean"]
        train_dice = summarize(train_values)["combined_dice_mean"]
        gap = train_dice - val_dice
        print(f"[eval_trajectory] train-val combined_dice gap: {gap:+.4f} "
              f"({'overfitting signal' if gap > 0.1 else 'no strong overfitting signal'})")
        wandb.log({"full_eval_train_val_gap/combined_dice": gap})

    # histogram of the per-pair distribution -- the mean alone has repeatedly hidden
    # bimodal/high-variance behavior on this task
    fig_hist, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values["combined_dice"], bins=20, color="tab:blue", edgecolor="white")
    ax.set_xlabel("combined Dice (this pair)")
    ax.set_ylabel("# validation pairs")
    ax.set_title(f"Per-pair combined Dice distribution (n={len(values['combined_dice'])})")
    plt.tight_layout()
    wandb.log({"full_eval_val/combined_dice_histogram": wandb.Image(fig_hist)})
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
        fig_traj = render_trajectory_figure(model, x0, x1, args.n_steps, args.n_trajectory_frames, device, title)
        wandb.log({f"trajectory/{args.trajectory_selection}_{rank}_{case_id}_o{orientation}": wandb.Image(fig_traj)})
        plt.close(fig_traj)

    print("[eval_trajectory] done.")


if __name__ == "__main__":
    main()
