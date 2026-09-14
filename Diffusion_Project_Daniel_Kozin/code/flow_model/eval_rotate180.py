"""
Compares rotate180 checkpoints (e.g. the 100/250/500/1000-epoch milestones) on held-out
validation pairs. Unlike the old interpolation task, this task has EXACT ground truth for
every prediction (the true 180-degree-rotated phantom). The only metrics tracked are
lump/pillar/combined mask IoU and Dice (predicted vs true binary mask, on the hard decoded
labels) -- centroid distance and voxel agreement were dropped after confirming they can look
good for a diffuse or oversized blob whose center of mass happens to land near the true
position without the shape actually matching. No ODE inversion is used: x0 is already real
data, so evaluation is a single forward integration per pair.

    ./run_eval_rotate180.sh --checkpoints checkpoints/epoch_0100.pt checkpoints/epoch_0250.pt \
        checkpoints/epoch_0500.pt checkpoints/epoch_1000.pt
"""
import argparse
import random
from pathlib import Path

import torch
import wandb

from .data import build_rotation_pair_datasets
from .metrics import rotation_metrics
from .ode import sample
from .train import get_device
from .velocity_model import FlowMatchingUNet3D
from .viz_utils import log_rotation_result


def load_model_from_checkpoint(path: Path, device: torch.device):
    """Loads a checkpoint, auto-detecting model architecture from its own saved config dict
    (mirrors sanity_checks.py's round_trip_reconstruction_check) instead of requiring the
    caller to pass matching --base_ch/--embed_channels flags by hand."""
    # weights_only=False: trusted, locally-generated checkpoint
    state = torch.load(path, map_location=device, weights_only=False)
    cfg = state.get("config", {})
    model = FlowMatchingUNet3D(num_classes=cfg.get("num_classes", 4), base_ch=cfg.get("base_ch", 8),
                                embed_channels=cfg.get("embed_channels", 16),
                                dropout=cfg.get("dropout", 0.1)).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state.get("epoch", "?")


def evaluate_checkpoint(model, val_ds, eval_indices, device: torch.device, n_steps: int,
                         method: str = "euler", render_indices=None, tag_prefix: str = "eval") -> dict:
    """
    Runs a forward integration (no inversion -- x0 is already real data) for every index in
    eval_indices and aggregates the trusted mask metrics (lump/pillar/combined IoU and Dice).
    Shared by eval_rotate180.py (compares several checkpoints) and any other script that wants
    a full-dataset metrics pass against one checkpoint.

    render_indices: optional subset of eval_indices to also log the qualitative 6-image
    figure for (tagged f"{tag_prefix}/pair_{case_id}_o{orientation}"). None/empty disables
    rendering entirely.

    Returns per-metric lists (not just means) so the caller can compute std/min/max or a
    histogram without re-running -- averages alone hide the per-pair variance this task has
    shown throughout (see project_summary.tex Part II).
    """
    render_indices = render_indices or set()
    lump_ious, lump_dices, pillar_ious, pillar_dices, combined_ious, combined_dices = [], [], [], [], [], []

    for idx in eval_indices:
        x0, x1 = val_ds[idx]
        case_id, variant_idx, orientation = val_ds.index[idx]
        x1_hat = sample(model, n_steps, x0=x0.unsqueeze(0).to(device), device=device,
                         method=method, t_start=0.0, t_end=1.0)
        pred_label = x1_hat.argmax(dim=1)[0].cpu()
        expected_label = x1.argmax(dim=0)
        orig_label = x0.argmax(dim=0)

        m = rotation_metrics(pred_label, expected_label)
        if m["lump_iou"] is not None:
            lump_ious.append(m["lump_iou"])
            lump_dices.append(m["lump_dice"])
        if m["pillar_iou"] is not None:
            pillar_ious.append(m["pillar_iou"])
            pillar_dices.append(m["pillar_dice"])
        if m["combined_iou"] is not None:
            combined_ious.append(m["combined_iou"])
            combined_dices.append(m["combined_dice"])

        if idx in render_indices:
            log_rotation_result(f"{tag_prefix}/pair_{case_id}_o{orientation}", orig_label, pred_label, expected_label)

    return {
        "lump_iou": lump_ious, "lump_dice": lump_dices,
        "pillar_iou": pillar_ious, "pillar_dice": pillar_dices,
        "combined_iou": combined_ious, "combined_dice": combined_dices,
    }


def summarize(values: dict) -> dict:
    """mean of each metric list from evaluate_checkpoint(), nan if empty."""
    return {f"{k}_mean": (sum(v) / len(v) if v else float("nan")) for k, v in values.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Compare rotate180 checkpoints on held-out validation pairs with exact ground truth.")
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True,
                         help="Checkpoint paths to compare, e.g. epoch_0100.pt epoch_0250.pt epoch_0500.pt epoch_1000.pt")
    parser.add_argument("--data_dir", type=Path, default=Path("mri_images_3D"))
    parser.add_argument("--num_classes", type=int, default=4,
                         help="Used to build the dataset's one-hot encoding -- must match what --checkpoints were trained with.")
    parser.add_argument("--val_frac", type=float, default=0.2,
                         help="MUST match the training run's --val_frac, or the reconstructed val split "
                              "differs from what the model was actually held out on.")
    parser.add_argument("--seed", type=int, default=0,
                         help="MUST match the training run's --seed (see --val_frac note).")
    parser.add_argument("--n_eval_pairs", type=int, default=-1,
                         help="-1 = full val set (quantitative-only pass, no rendering, cheap).")
    parser.add_argument("--n_render_examples", type=int, default=5,
                         help="Subset of the eval pairs that also get the qualitative 6-image figure.")
    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--method", type=str, default="euler", choices=["euler", "heun"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--wandb_project", type=str, default="diffusion_project")
    parser.add_argument("--wandb_mode", type=str, default="online")
    parser.add_argument("--run_tag", type=str, default=None)
    args = parser.parse_args()

    device = get_device(args.device)

    train_ds, val_ds = build_rotation_pair_datasets(args.data_dir, val_frac=args.val_frac,
                                                      seed=args.seed, num_classes=args.num_classes)
    val_case_ids = sorted({val_ds.index[i][0] for i in range(len(val_ds))})
    print(f"[eval_rotate180] val_frac={args.val_frac} seed={args.seed} -> "
          f"{len(val_case_ids)} val case_ids: {val_case_ids}")
    print(f"[eval_rotate180] {len(val_ds)} val pairs total")
    if len(val_ds) == 0:
        raise RuntimeError("No validation pairs found -- check --data_dir/--val_frac/--seed")

    rng = random.Random(args.seed)
    if args.n_eval_pairs <= 0 or args.n_eval_pairs >= len(val_ds):
        eval_indices = list(range(len(val_ds)))
    else:
        eval_indices = rng.sample(range(len(val_ds)), args.n_eval_pairs)
    n_render = min(args.n_render_examples, len(eval_indices))
    render_indices = set(rng.sample(eval_indices, n_render)) if n_render > 0 else set()
    print(f"[eval_rotate180] evaluating {len(eval_indices)} pairs, rendering {len(render_indices)} of them")

    run_name = f"eval_rotate180_{len(args.checkpoints)}ckpts_{len(eval_indices)}pairs"
    if args.run_tag:
        run_name = f"{run_name}_{args.run_tag}"
    wandb.init(project=args.wandb_project, mode=args.wandb_mode, name=run_name, config=vars(args))

    table = wandb.Table(columns=["checkpoint", "epoch", "lump_iou_mean", "lump_dice_mean",
                                  "pillar_iou_mean", "pillar_dice_mean", "combined_iou_mean", "combined_dice_mean"])
    summary_rows = []

    for ckpt_path in args.checkpoints:
        model, epoch_trained = load_model_from_checkpoint(ckpt_path, device)
        ckpt_name = ckpt_path.stem
        print(f"[eval_rotate180] {ckpt_name} (epoch {epoch_trained})...")

        values = evaluate_checkpoint(model, val_ds, eval_indices, device, args.n_steps, method=args.method,
                                      render_indices=render_indices, tag_prefix=f"eval/{ckpt_name}")
        s = summarize(values)

        print(f"  lump_IoU={s['lump_iou_mean']:.4f}  lump_Dice={s['lump_dice_mean']:.4f}  "
              f"pillar_IoU={s['pillar_iou_mean']:.4f}  pillar_Dice={s['pillar_dice_mean']:.4f}  "
              f"combined_IoU={s['combined_iou_mean']:.4f}  combined_Dice={s['combined_dice_mean']:.4f}")

        wandb.log({
            "checkpoint_epoch": epoch_trained,
            "eval/lump_iou_mean": s["lump_iou_mean"],
            "eval/lump_dice_mean": s["lump_dice_mean"],
            "eval/pillar_iou_mean": s["pillar_iou_mean"],
            "eval/pillar_dice_mean": s["pillar_dice_mean"],
            "eval/combined_iou_mean": s["combined_iou_mean"],
            "eval/combined_dice_mean": s["combined_dice_mean"],
        })
        table.add_data(ckpt_name, epoch_trained, s["lump_iou_mean"], s["lump_dice_mean"],
                        s["pillar_iou_mean"], s["pillar_dice_mean"], s["combined_iou_mean"], s["combined_dice_mean"])
        summary_rows.append((ckpt_name, epoch_trained, s["combined_dice_mean"]))

    wandb.log({"eval/summary_table": table})

    print("-" * 60)
    print("Ranked by combined (lump+pillar) Dice -- the trusted metric (higher is better):")
    for ckpt_name, epoch_trained, combined_dice_mean in sorted(summary_rows, key=lambda r: -r[2]):
        print(f"  {ckpt_name} (epoch {epoch_trained}): combined_dice={combined_dice_mean:.4f}")
    print("-" * 60)


if __name__ == "__main__":
    main()
