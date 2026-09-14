"""
Splits phantoms by lump left/right position (set A = lump right of center, set B = left) and
showcases a handful of examples from each, plus a whole-dataset centroid scatter as a sanity
check that the split boundary is doing what's expected.

    ./run_showcase_split.sh --n_per_side 10
"""
import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import wandb

from .data import load_all_volumes, rotate_label_volume, split_case_ids
from .metrics import lump_centroid_hw
from .viz_utils import grid_figure, render_label_volumes

# Volumes are 128x128 in-plane; the vertical bisector through the center splits the circle
# into a right half (set A) and a left half (set B), matching "circle top to bottom, right
# side is A, left side is B."
CENTER_W = 63.5


def enumerate_instances(case_volumes, case_ids):
    """Every (case_id, variant_idx, orientation) triple over all 8 orientations, its rotated
    label volume, and its lump centroid (w, h) -- or None if it has no lump voxels."""
    instances = []
    for case_id in case_ids:
        for variant_idx in range(len(case_volumes[case_id])):
            raw = case_volumes[case_id][variant_idx]
            for orientation in range(8):
                label_t = torch.from_numpy(rotate_label_volume(raw, orientation).copy())
                pos = lump_centroid_hw(label_t)
                instances.append((case_id, variant_idx, orientation, label_t, pos))
    return instances


def side_of(pos):
    """'A' (right), 'B' (left), or None (no lump voxels, or exactly on the center bisector --
    measure-zero on real data, guarded defensively rather than silently misclassified)."""
    if pos is None:
        return None
    w, _ = pos
    if w > CENTER_W:
        return "A"
    if w < CENTER_W:
        return "B"
    return None


def pick_examples(instances, side, n, max_per_case, rng):
    """Seeded selection of up to n instances of the given side, spread across distinct
    case_ids first (up to max_per_case each), then filling with repeats if the side has
    fewer distinct cases than n."""
    pool = [inst for inst in instances if side_of(inst[4]) == side]
    rng.shuffle(pool)

    picked, picked_keys, per_case_count = [], set(), {}
    for inst in pool:
        case_id = inst[0]
        if per_case_count.get(case_id, 0) >= max_per_case:
            continue
        picked.append(inst)
        picked_keys.add(inst[:3])
        per_case_count[case_id] = per_case_count.get(case_id, 0) + 1
        if len(picked) >= n:
            return picked

    for inst in pool:
        if inst[:3] in picked_keys:
            continue
        picked.append(inst)
        picked_keys.add(inst[:3])
        if len(picked) >= n:
            break
    return picked


def _title(inst):
    case_id, variant_idx, orientation, _, pos = inst
    w = pos[0] if pos is not None else float("nan")
    return f"{case_id}_v{variant_idx}_o{orientation} (w={w:.0f})"


def main():
    parser = argparse.ArgumentParser(description="Split phantoms by lump left/right position and showcase examples")
    parser.add_argument("--data_dir", type=Path, default=Path("mri_images_3D"))
    parser.add_argument("--split", type=str, default="all", choices=["all", "train", "val"])
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_per_side", type=int, default=10)
    parser.add_argument("--max_per_case", type=int, default=1)
    parser.add_argument("--wandb_project", type=str, default="diffusion_project")
    parser.add_argument("--wandb_mode", type=str, default="online")
    parser.add_argument("--run_tag", type=str, default=None)
    args = parser.parse_args()

    case_volumes = load_all_volumes(args.data_dir)
    if args.split == "all":
        case_ids = list(case_volumes.keys())
    else:
        train_ids, val_ids = split_case_ids(list(case_volumes.keys()), val_frac=args.val_frac, seed=args.seed)
        case_ids = train_ids if args.split == "train" else val_ids

    print(f"[showcase_split] enumerating all 8 orientations for {len(case_ids)} case_ids ({args.split})...")
    instances = enumerate_instances(case_volumes, case_ids)

    n_no_lump = sum(1 for inst in instances if inst[4] is None)
    n_on_bisector = sum(1 for inst in instances if inst[4] is not None and side_of(inst[4]) is None)
    n_A = sum(1 for inst in instances if side_of(inst[4]) == "A")
    n_B = sum(1 for inst in instances if side_of(inst[4]) == "B")
    print(f"[showcase_split] {len(instances)} total instances: {n_A} on right (A), {n_B} on left (B), "
          f"{n_no_lump} with no lump voxels, {n_on_bisector} exactly on the center bisector")

    rng = random.Random(args.seed)
    examples_A = pick_examples(instances, "A", args.n_per_side, args.max_per_case, rng)
    examples_B = pick_examples(instances, "B", args.n_per_side, args.max_per_case, rng)
    if len(examples_A) < args.n_per_side or len(examples_B) < args.n_per_side:
        print(f"[showcase_split] warning: only found {len(examples_A)} set-A / {len(examples_B)} "
              f"set-B examples (requested {args.n_per_side} each)")

    imgs_A = render_label_volumes([inst[3] for inst in examples_A], names=[_title(i) for i in examples_A])
    imgs_B = render_label_volumes([inst[3] for inst in examples_B], names=[_title(i) for i in examples_B])
    fig_A = grid_figure(imgs_A, titles=[_title(i) for i in examples_A])
    fig_B = grid_figure(imgs_B, titles=[_title(i) for i in examples_B])

    # Cheap whole-dataset sanity check: every instance's centroid, color-coded by side, with
    # the W=63.5 boundary drawn in -- confirms the split holds beyond the handful shown above.
    scatter_fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter([inst[4][0] for inst in instances if side_of(inst[4]) == "A"],
               [inst[4][1] for inst in instances if side_of(inst[4]) == "A"],
               s=8, color="tab:blue", alpha=0.5, label=f"A (right, n={n_A})")
    ax.scatter([inst[4][0] for inst in instances if side_of(inst[4]) == "B"],
               [inst[4][1] for inst in instances if side_of(inst[4]) == "B"],
               s=8, color="tab:orange", alpha=0.5, label=f"B (left, n={n_B})")
    ax.axvline(CENTER_W, color="gray", linestyle="dotted")
    ax.set_xlim(0, 128)
    ax.set_ylim(128, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("W (voxels)")
    ax.set_ylabel("H (voxels)")
    ax.set_title("Lump centroid, all instances")
    ax.legend(fontsize=8)
    plt.tight_layout()

    run_name = f"split_showcase_{args.n_per_side}per_side_{args.split}"
    if args.run_tag:
        run_name = f"{run_name}_{args.run_tag}"
    wandb.init(project=args.wandb_project, mode=args.wandb_mode, name=run_name, config=vars(args))
    wandb.log({
        "split_showcase/set_A": wandb.Image(fig_A),
        "split_showcase/set_B": wandb.Image(fig_B),
        "split_showcase/centroid_scatter_all": wandb.Image(scatter_fig),
        "split_showcase/n_A_total": n_A,
        "split_showcase/n_B_total": n_B,
        "split_showcase/n_no_lump": n_no_lump,
        "split_showcase/n_on_bisector": n_on_bisector,
    })
    plt.close(fig_A)
    plt.close(fig_B)
    plt.close(scatter_fig)

    print(f"Logged {len(examples_A)} set-A and {len(examples_B)} set-B examples, plus the full "
          f"{len(instances)}-instance centroid scatter.")


if __name__ == "__main__":
    main()
