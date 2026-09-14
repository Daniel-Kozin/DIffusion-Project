import fcntl
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from visualize_inserts_3d import visualize_inserts_3d

from .metrics import lump_centroid_hw, rotation_metrics

# visualize_inserts_3d's off-screen VTK rendering has no real X server to draw against on
# this machine (see the "bad X server connection" warning) and isn't safe for concurrent
# use: two processes rendering at once corrupt each other's screenshot buffer into static/
# garbled output. This machine regularly runs multiple training/interpolation jobs at once
# (one per GPU), so a cross-process file lock serializes actual render calls to avoid it.
_RENDER_LOCK_PATH = Path("/tmp/diffusion_project_vtk_render.lock")


def render_label_volumes(label_volumes: List[torch.Tensor], names: Optional[List[str]] = None,
                          look_up: bool = False) -> List[np.ndarray]:
    """
    label_volumes: list of int class-index tensors [K, H, W] (values 0-3).
    look_up: use visualize_inserts_3d's top-down camera preset instead of the default
        angled view.
    Returns a list of RGB screenshot arrays, one per volume.
    """
    if names is None:
        names = [f"vol_{i}" for i in range(len(label_volumes))]

    with open(_RENDER_LOCK_PATH, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        vis = None
        try:
            vis = visualize_inserts_3d(off_screen=True, show_text=False, look_up=look_up)
            for vol, name in zip(label_volumes, names):
                vis.load_array(vol, name=name)
            imgs = vis.show()
        finally:
            # render_all()'s off-screen path never closes the VTK render window/GL context
            # it opens (only the interactive quit_program() path does) -- every render call
            # through here leaked one. Over hours of periodic previews across multiple
            # concurrent training jobs this accumulates fast and degrades/corrupts
            # subsequent renders. Close explicitly, inside the lock, so a leaked context
            # can never be attributed to "two processes rendering at once" confusion again.
            if vis is not None:
                vis.plotter.close()
            fcntl.flock(lock_file, fcntl.LOCK_UN)

    return imgs if imgs is not None else []


def render_single(label_volume: torch.Tensor, name: str = "vol", look_up: bool = False) -> Optional[np.ndarray]:
    """Render a single label volume [K, H, W] to an RGB screenshot array."""
    imgs = render_label_volumes([label_volume], [name], look_up=look_up)
    return imgs[0] if imgs else None


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
                               label_a: str = "A", label_b: str = "B",
                               render_a_top: Optional[np.ndarray] = None,
                               render_b_top: Optional[np.ndarray] = None,
                               render_interp_top: Optional[np.ndarray] = None,
                               pos_actual: Optional[Tuple[float, float]] = None) -> plt.Figure:
    """
    Left: the two endpoints plotted at their actual lump centroid location in the (H, W)
    slice plane, connected by a dotted line, with a star at the alpha-interpolated
    *expected* position between them (naive linear interpolation of the two endpoint
    centroids) and, when pos_actual is given, a second marker at the lump centroid the
    model *actually* generated for this alpha — a dotted line between the two makes the
    gap between "expected" and "actual" directly visible.
    Right: A, B, and the interpolated render, angled view on top and top-down view
    below (top-down renders are optional; when omitted only the angled row is drawn).

    pos_a/pos_b: (w, h) lump centroid from lump_centroid_hw(), or None if that phantom
    has no lump voxels (falls back to the volume center, annotated as such).
    pos_actual: (w, h) lump centroid of the actual generated interpolated volume, or None
    if it has no lump voxels (or the caller doesn't have it) — nothing extra is drawn.
    """
    H, W = volume_hw
    fallback = (W / 2, H / 2)
    pos_a_used = pos_a if pos_a is not None else fallback
    pos_b_used = pos_b if pos_b is not None else fallback
    pos_interp = ((1 - alpha) * pos_a_used[0] + alpha * pos_b_used[0],
                   (1 - alpha) * pos_a_used[1] + alpha * pos_b_used[1])

    has_top = render_a_top is not None and render_b_top is not None and render_interp_top is not None
    n_right_rows = 2 if has_top else 1
    fig = plt.figure(figsize=(13.5, 3 + 2.75 * n_right_rows))
    gs = fig.add_gridspec(n_right_rows, 5, width_ratios=[1, 1, 1, 1, 1])

    # --- left: lump position schematic ---
    ax_left = fig.add_subplot(gs[:, 0:2])
    ax_left.plot([pos_a_used[0], pos_b_used[0]], [pos_a_used[1], pos_b_used[1]],
                 linestyle="dotted", color="gray", zorder=1)
    ax_left.scatter([pos_a_used[0]], [pos_a_used[1]], s=140, color="tab:blue", zorder=3)
    ax_left.scatter([pos_b_used[0]], [pos_b_used[1]], s=140, color="tab:orange", zorder=3)
    ax_left.scatter([pos_interp[0]], [pos_interp[1]], s=220, color="tab:red", marker="*", zorder=4,
                     label="expected (linear)")
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

    if pos_actual is not None:
        ax_left.plot([pos_interp[0], pos_actual[0]], [pos_interp[1], pos_actual[1]],
                     linestyle="dotted", color="tab:green", zorder=2, linewidth=1.5)
        ax_left.scatter([pos_actual[0]], [pos_actual[1]], s=200, color="tab:green", marker="X", zorder=5,
                        label="actual (generated)")
        ax_left.annotate("actual", pos_actual, textcoords="offset points", xytext=(10, -10),
                         ha="left", fontsize=10, color="tab:green")
        ax_left.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax_left.annotate(f"alpha={alpha:.2f}", pos_interp, textcoords="offset points",
                      xytext=(16, 0), ha="left", va="center", fontsize=10, color="tab:red")
    ax_left.set_xlim(0, W)
    ax_left.set_ylim(H, 0)  # inverted so row 0 is at the top, matching image orientation
    ax_left.set_aspect("equal")
    ax_left.set_xlabel("W (voxels)")
    ax_left.set_ylabel("H (voxels)")
    ax_left.set_title("Lump centroid position")

    # --- right: A / B / interpolated, angled view on row 0, top-down view on row 1 ---
    # A/B panels get a colored title + border matching their dot color on the left schematic,
    # so it's unambiguous which render corresponds to which endpoint.
    interp_title = f"interpolated (alpha={alpha:.2f})"
    colors = ["tab:blue", "tab:orange", None]

    def _render_panel(ax, img, title, color):
        ax.imshow(img)
        ax.set_title(title, fontsize=10, color=color if color else "black")
        ax.set_xticks([])
        ax.set_yticks([])
        if color:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
        else:
            for spine in ax.spines.values():
                spine.set_visible(False)

    row0 = [(label_a, render_a), (label_b, render_b), (interp_title, render_interp)]
    for col, (title, img) in enumerate(row0):
        ax = fig.add_subplot(gs[0, 2 + col])
        _render_panel(ax, img, title, colors[col])

    if has_top:
        row1 = [(f"{label_a} (top)", render_a_top), (f"{label_b} (top)", render_b_top),
                (f"{interp_title} (top)", render_interp_top)]
        for col, (title, img) in enumerate(row1):
            ax = fig.add_subplot(gs[1, 2 + col])
            _render_panel(ax, img, title, colors[col])

    plt.tight_layout()
    # wandb.Image() calls fig.savefig() without bbox_inches="tight", and long titles on the
    # rightmost column (e.g. "interpolated (alpha=0.50) (top)") can extend past what
    # tight_layout() reserved within the raw figsize, clipping the saved PNG. An explicit
    # right margin guarantees the title fits without depending on the caller's savefig args.
    fig.subplots_adjust(right=0.94, left=0.06)
    return fig


def log_interpolation_step(tag: str, render_a: np.ndarray, render_b: np.ndarray, render_interp: np.ndarray,
                            alpha: float, vol_a: torch.Tensor, vol_b: torch.Tensor,
                            label_a: str = "A", label_b: str = "B", lump_class: int = 3,
                            render_a_top: Optional[np.ndarray] = None,
                            render_b_top: Optional[np.ndarray] = None,
                            render_interp_top: Optional[np.ndarray] = None,
                            vol_interp: Optional[torch.Tensor] = None) -> None:
    """
    vol_a/vol_b: label volumes [K, H, W] for the two endpoints, used to locate each
    phantom's lump centroid for the left-panel schematic.
    render_*_top: optional top-down renders of the same three volumes; when given, the
    figure gains a second row showing them below the default angled-view row.
    vol_interp: the actual generated interpolated label volume [K, H, W] — when given, its
    lump centroid is plotted alongside the naive expected (linearly-interpolated) position
    so the two can be compared directly.
    """
    if wandb.run is None:
        print(f"[viz_utils] wandb.run is None, skipping log for '{tag}'")
        return
    if render_a is None or render_b is None or render_interp is None:
        print(f"[viz_utils] missing render(s), skipping log for '{tag}' at alpha={alpha}")
        return

    pos_a = lump_centroid_hw(vol_a, lump_class=lump_class)
    pos_b = lump_centroid_hw(vol_b, lump_class=lump_class)
    pos_actual = lump_centroid_hw(vol_interp, lump_class=lump_class) if vol_interp is not None else None
    volume_hw = (vol_a.shape[-2], vol_a.shape[-1])

    fig = interpolation_step_figure(render_a, render_b, render_interp, alpha, pos_a, pos_b,
                                     volume_hw=volume_hw, label_a=label_a, label_b=label_b,
                                     render_a_top=render_a_top, render_b_top=render_b_top,
                                     render_interp_top=render_interp_top, pos_actual=pos_actual)
    wandb.log({tag: wandb.Image(fig)})
    plt.close(fig)


def rotation_result_figure(render_orig: np.ndarray, render_pred: np.ndarray, render_expected: np.ndarray,
                            pos_orig: Optional[Tuple[float, float]], pos_pred: Optional[Tuple[float, float]],
                            pos_expected: Optional[Tuple[float, float]],
                            volume_hw: Tuple[int, int] = (128, 128),
                            render_orig_top: Optional[np.ndarray] = None,
                            render_pred_top: Optional[np.ndarray] = None,
                            render_expected_top: Optional[np.ndarray] = None,
                            metrics: Optional[Dict] = None) -> plt.Figure:
    """
    Left: three lump-centroid markers -- original (blue), expected/ground-truth 180-degree
    position (orange), and the model's predicted position (green X, same "actual/generated"
    visual language as interpolation_step_figure's green-X marker). A dotted gray line
    original->expected shows the true 180-degree path; a dotted green line expected->predicted
    makes the model's error directly visible as a gap.
    Right: original / predicted / expected renders, angled view on top and (if given) a
    top-down row below -- 6 images total when both angles are supplied.

    pos_*: (w, h) lump centroid from lump_centroid_hw(), or None if that volume has no lump
    voxels (falls back to the volume center, annotated as such).
    metrics: optional dict from metrics.rotation_metrics() -- when given, lump IoU/Dice (the
    trusted metrics for this task) are annotated as a text box on the left panel.
    """
    H, W = volume_hw
    fallback = (W / 2, H / 2)
    pos_orig_used = pos_orig if pos_orig is not None else fallback
    pos_pred_used = pos_pred if pos_pred is not None else fallback
    pos_expected_used = pos_expected if pos_expected is not None else fallback

    has_top = render_orig_top is not None and render_pred_top is not None and render_expected_top is not None
    n_right_rows = 2 if has_top else 1
    fig = plt.figure(figsize=(13.5, 3 + 2.75 * n_right_rows))
    gs = fig.add_gridspec(n_right_rows, 5, width_ratios=[1, 1, 1, 1, 1])

    # --- left: original / predicted / expected lump position schematic ---
    ax_left = fig.add_subplot(gs[:, 0:2])
    ax_left.plot([pos_orig_used[0], pos_expected_used[0]], [pos_orig_used[1], pos_expected_used[1]],
                 linestyle="dotted", color="gray", zorder=1)
    ax_left.plot([pos_expected_used[0], pos_pred_used[0]], [pos_expected_used[1], pos_pred_used[1]],
                 linestyle="dotted", color="tab:green", zorder=2, linewidth=1.5)
    ax_left.scatter([pos_orig_used[0]], [pos_orig_used[1]], s=140, color="tab:blue", zorder=3,
                     label="original")
    ax_left.scatter([pos_expected_used[0]], [pos_expected_used[1]], s=200, color="tab:orange",
                     marker="*", zorder=4, label="expected (true 180°)")
    ax_left.scatter([pos_pred_used[0]], [pos_pred_used[1]], s=160, color="tab:green", marker="X",
                     zorder=5, label="predicted")

    def _label(pos, given, text, color, dx_ref, dy_ref):
        text = text if given is not None else f"{text} (no lump)"
        ax_left.annotate(text, pos, textcoords="offset points",
                          xytext=(-10 - 6 * (dx_ref > 0), -10 - 6 * (dy_ref > 0)),
                          ha="right" if dx_ref > 0 else "left", fontsize=10, color=color)

    dx, dy = pos_expected_used[0] - pos_orig_used[0], pos_expected_used[1] - pos_orig_used[1]
    _label(pos_orig_used, pos_orig, "original", "tab:blue", -dx, -dy)
    _label(pos_expected_used, pos_expected, "expected", "tab:orange", dx, dy)
    _label(pos_pred_used, pos_pred, "predicted", "tab:green",
           pos_pred_used[0] - pos_expected_used[0], pos_pred_used[1] - pos_expected_used[1])

    ax_left.legend(loc="upper right", fontsize=8, framealpha=0.9)
    if metrics is not None:
        lump_iou = metrics.get("lump_iou")
        lump_dice = metrics.get("lump_dice")
        combined_dice = metrics.get("combined_dice")
        text = f"lump IoU: {lump_iou:.3f}" if lump_iou is not None else "lump IoU: n/a"
        text += "\nlump Dice: " + (f"{lump_dice:.3f}" if lump_dice is not None else "n/a")
        text += "\ncombined Dice: " + (f"{combined_dice:.3f}" if combined_dice is not None else "n/a")
        ax_left.text(0.02, 0.02, text, transform=ax_left.transAxes, fontsize=9, va="bottom", ha="left",
                     bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"))
    ax_left.set_xlim(0, W)
    ax_left.set_ylim(H, 0)  # inverted so row 0 is at the top, matching image orientation
    ax_left.set_aspect("equal")
    ax_left.set_xlabel("W (voxels)")
    ax_left.set_ylabel("H (voxels)")
    ax_left.set_title("Lump centroid position")

    # --- right: original / predicted / expected, angled view on row 0, top-down on row 1 ---
    colors = ["tab:blue", "tab:green", "tab:orange"]

    def _render_panel(ax, img, title, color):
        ax.imshow(img)
        ax.set_title(title, fontsize=10, color=color)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(color)
            spine.set_linewidth(3)

    row0 = [("original", render_orig), ("predicted", render_pred), ("expected", render_expected)]
    for col, (title, img) in enumerate(row0):
        ax = fig.add_subplot(gs[0, 2 + col])
        _render_panel(ax, img, title, colors[col])

    if has_top:
        row1 = [("original (top)", render_orig_top), ("predicted (top)", render_pred_top),
                ("expected (top)", render_expected_top)]
        for col, (title, img) in enumerate(row1):
            ax = fig.add_subplot(gs[1, 2 + col])
            _render_panel(ax, img, title, colors[col])

    plt.tight_layout()
    # See interpolation_step_figure's identical note: wandb.Image() savefig()'s without
    # bbox_inches="tight", so an explicit margin guarantees titles aren't clipped.
    fig.subplots_adjust(right=0.94, left=0.06)
    return fig


def log_rotation_result(tag: str, orig_label: torch.Tensor, pred_label: torch.Tensor,
                         expected_label: torch.Tensor, lump_class: int = 3,
                         top_down: bool = True) -> Optional[Dict]:
    """
    orig_label/pred_label/expected_label: label volumes [K, H, W]. Renders all three in one
    batched render_label_volumes() call (angled) plus, if top_down, one more (look_up=True) --
    2 render calls / VTK-lock acquisitions total for up to 6 images, rather than 6 individual
    render_single() calls. Computes metrics.rotation_metrics(pred_label, expected_label),
    builds rotation_result_figure(), logs it to wandb, and returns the metrics dict (unlike
    log_interpolation_step's `-> None`) since both train_rotate180.py's periodic preview and
    eval_rotate180.py's aggregation need this same dict, computed once rather than twice.
    """
    m = rotation_metrics(pred_label, expected_label, lump_class=lump_class)

    if wandb.run is None:
        print(f"[viz_utils] wandb.run is None, skipping log for '{tag}'")
        return m

    vols = [orig_label, pred_label, expected_label]
    names = [f"{tag}_original", f"{tag}_predicted", f"{tag}_expected"]
    renders = render_label_volumes(vols, names=names, look_up=False)
    renders_top = render_label_volumes(vols, names=names, look_up=True) if top_down else [None, None, None]
    if len(renders) != 3 or (top_down and len(renders_top) != 3):
        print(f"[viz_utils] missing render(s), skipping log for '{tag}'")
        return m

    pos_orig = lump_centroid_hw(orig_label, lump_class=lump_class)
    pos_pred = lump_centroid_hw(pred_label, lump_class=lump_class)
    pos_expected = lump_centroid_hw(expected_label, lump_class=lump_class)
    volume_hw = (orig_label.shape[-2], orig_label.shape[-1])

    fig = rotation_result_figure(renders[0], renders[1], renders[2], pos_orig, pos_pred, pos_expected,
                                  volume_hw=volume_hw,
                                  render_orig_top=renders_top[0], render_pred_top=renders_top[1],
                                  render_expected_top=renders_top[2], metrics=m)
    wandb.log({tag: wandb.Image(fig)})
    plt.close(fig)
    return m
