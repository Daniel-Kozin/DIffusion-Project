import math
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from visualize_inserts_3d import visualize_inserts_3d


def render_label_volumes(label_volumes: List[torch.Tensor], names: Optional[List[str]] = None) -> List[np.ndarray]:
    """
    label_volumes: list of int class-index tensors [K, H, W] (values 0-3).
    Returns a list of RGB screenshot arrays, one per volume.
    """
    if names is None:
        names = [f"vol_{i}" for i in range(len(label_volumes))]

    vis = visualize_inserts_3d(off_screen=True, show_text=False)
    for vol, name in zip(label_volumes, names):
        vis.load_array(vol, name=name)

    imgs = vis.show()
    return imgs if imgs is not None else []


def render_single(label_volume: torch.Tensor, name: str = "vol") -> Optional[np.ndarray]:
    """Render a single label volume [K, H, W] to an RGB screenshot array."""
    imgs = render_label_volumes([label_volume], [name])
    return imgs[0] if imgs else None


def lump_centroid_hw(label_volume: torch.Tensor, lump_class: int = 3) -> Optional[Tuple[float, float]]:
    """
    Centroid of the lump voxels in a label volume [K, H, W], projected onto the (H, W)
    plane (averaged over depth). Returns (w, h) so it plots naturally on an (x, y) axis,
    or None if the volume has no voxels of `lump_class`.
    """
    mask = (label_volume == lump_class)
    if mask.sum() == 0:
        return None
    coords = mask.nonzero(as_tuple=False).float()  # [N, 3] -> (k, h, w) indices
    h = coords[:, 1].mean().item()
    w = coords[:, 2].mean().item()
    return (w, h)


def grid_figure(imgs: List[np.ndarray], titles: Optional[List[str]] = None,
                 max_per_row: int = 5) -> plt.Figure:
    n = len(imgs)
    cols = min(max_per_row, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_2d(axes)
    if axes.shape != (rows, cols):
        axes = axes.reshape(rows, cols)

    for i in range(n):
        r, c = divmod(i, cols)
        axes[r, c].imshow(imgs[i])
        if titles is not None:
            axes[r, c].set_title(titles[i], fontsize=9)
        axes[r, c].axis("off")

    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")

    plt.tight_layout()
    return fig


def log_grid_to_wandb(tag: str, label_volumes: List[torch.Tensor], titles: List[str]) -> None:
    if wandb.run is None:
        print(f"[viz_utils] wandb.run is None, skipping log for '{tag}'")
        return

    imgs = render_label_volumes(label_volumes, names=titles)
    if not imgs:
        print(f"[viz_utils] no images rendered, skipping log for '{tag}'")
        return

    fig = grid_figure(imgs, titles=titles)
    wandb.log({tag: wandb.Image(fig)})
    plt.close(fig)


def interpolation_step_figure(render_a: np.ndarray, render_b: np.ndarray, render_interp: np.ndarray,
                               alpha: float, pos_a: Optional[Tuple[float, float]],
                               pos_b: Optional[Tuple[float, float]],
                               volume_hw: Tuple[int, int] = (128, 128),
                               label_a: str = "A", label_b: str = "B") -> plt.Figure:
    """
    Left: the two endpoints plotted at their actual lump centroid location in the (H, W)
    slice plane, connected by a dotted line, with a marker at the alpha-interpolated
    position between them.
    Right: A and B renders on top, the interpolated render centered below them.

    pos_a/pos_b: (w, h) lump centroid from lump_centroid_hw(), or None if that phantom
    has no lump voxels (falls back to the volume center, annotated as such).
    """
    H, W = volume_hw
    fallback = (W / 2, H / 2)
    pos_a_used = pos_a if pos_a is not None else fallback
    pos_b_used = pos_b if pos_b is not None else fallback
    pos_interp = ((1 - alpha) * pos_a_used[0] + alpha * pos_b_used[0],
                   (1 - alpha) * pos_a_used[1] + alpha * pos_b_used[1])

    fig = plt.figure(figsize=(11, 5.5))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1])

    # --- left: lump position schematic ---
    ax_left = fig.add_subplot(gs[:, 0:2])
    ax_left.plot([pos_a_used[0], pos_b_used[0]], [pos_a_used[1], pos_b_used[1]],
                 linestyle="dotted", color="gray", zorder=1)
    ax_left.scatter([pos_a_used[0]], [pos_a_used[1]], s=140, color="tab:blue", zorder=3)
    ax_left.scatter([pos_b_used[0]], [pos_b_used[1]], s=140, color="tab:orange", zorder=3)
    ax_left.scatter([pos_interp[0]], [pos_interp[1]], s=220, color="tab:red", marker="*", zorder=4)
    label_a_text = label_a if pos_a is not None else f"{label_a} (no lump)"
    label_b_text = label_b if pos_b is not None else f"{label_b} (no lump)"
    # Point B's label away from A (and vice versa) so labels don't collide when the two
    # centroids are close together; the alpha marker's label sits off to the side.
    dx, dy = pos_b_used[0] - pos_a_used[0], pos_b_used[1] - pos_a_used[1]
    ax_left.annotate(label_a_text, pos_a_used, textcoords="offset points",
                      xytext=(-10 - 6 * (dx > 0), -10 - 6 * (dy > 0)),
                      ha="right" if dx > 0 else "left", fontsize=10, color="tab:blue")
    ax_left.annotate(label_b_text, pos_b_used, textcoords="offset points",
                      xytext=(10 + 6 * (dx > 0), 10 + 6 * (dy > 0)),
                      ha="left" if dx > 0 else "right", fontsize=10, color="tab:orange")
    ax_left.annotate(f"alpha={alpha:.2f}", pos_interp, textcoords="offset points",
                      xytext=(16, 0), ha="left", va="center", fontsize=10, color="tab:red")
    ax_left.set_xlim(0, W)
    ax_left.set_ylim(H, 0)  # inverted so row 0 is at the top, matching image orientation
    ax_left.set_aspect("equal")
    ax_left.set_xlabel("W (voxels)")
    ax_left.set_ylabel("H (voxels)")
    ax_left.set_title("Lump centroid position")

    # --- right: A / B on top, interpolated result centered below ---
    ax_a = fig.add_subplot(gs[0, 2])
    ax_a.imshow(render_a)
    ax_a.set_title(label_a, fontsize=10)
    ax_a.axis("off")

    ax_b = fig.add_subplot(gs[0, 3])
    ax_b.imshow(render_b)
    ax_b.set_title(label_b, fontsize=10)
    ax_b.axis("off")

    ax_interp = fig.add_subplot(gs[1, 2:4])
    ax_interp.imshow(render_interp)
    ax_interp.set_title(f"interpolated (alpha={alpha:.2f})", fontsize=10)
    ax_interp.axis("off")

    plt.tight_layout()
    return fig


def log_interpolation_step(tag: str, render_a: np.ndarray, render_b: np.ndarray, render_interp: np.ndarray,
                            alpha: float, vol_a: torch.Tensor, vol_b: torch.Tensor,
                            label_a: str = "A", label_b: str = "B", lump_class: int = 3) -> None:
    """
    vol_a/vol_b: label volumes [K, H, W] for the two endpoints, used to locate each
    phantom's lump centroid for the left-panel schematic.
    """
    if wandb.run is None:
        print(f"[viz_utils] wandb.run is None, skipping log for '{tag}'")
        return
    if render_a is None or render_b is None or render_interp is None:
        print(f"[viz_utils] missing render(s), skipping log for '{tag}' at alpha={alpha}")
        return

    pos_a = lump_centroid_hw(vol_a, lump_class=lump_class)
    pos_b = lump_centroid_hw(vol_b, lump_class=lump_class)
    volume_hw = (vol_a.shape[-2], vol_a.shape[-1])

    fig = interpolation_step_figure(render_a, render_b, render_interp, alpha, pos_a, pos_b,
                                     volume_hw=volume_hw, label_a=label_a, label_b=label_b)
    wandb.log({tag: wandb.Image(fig)})
    plt.close(fig)
