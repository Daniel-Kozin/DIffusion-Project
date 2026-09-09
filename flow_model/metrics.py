"""Pure numeric helpers (no VTK/wandb imports) shared by the rotation-180 training preview,
eval_rotate180.py, showcase_split.py, and sweep_tmid.py."""
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from scipy.ndimage import label as cc_label


def count_components(label_volume: torch.Tensor, classes: Iterable[int]) -> int:
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


def lump_centroid_distance(pos_a: Optional[Tuple[float, float]],
                            pos_b: Optional[Tuple[float, float]]) -> Optional[float]:
    """Euclidean distance in voxels between two (w, h) centroids, or None if either is None."""
    if pos_a is None or pos_b is None:
        return None
    return ((pos_a[0] - pos_b[0]) ** 2 + (pos_a[1] - pos_b[1]) ** 2) ** 0.5


def voxel_class_agreement(pred_label: torch.Tensor, target_label: torch.Tensor) -> float:
    """Fraction of voxels where pred_label and target_label have the same class index."""
    return (pred_label == target_label).float().mean().item()


def rotation_metrics(pred_label: torch.Tensor, expected_label: torch.Tensor,
                      lump_class: int = 3) -> Dict[str, Optional[float]]:
    """Single source of truth for the quantitative comparison used by both
    train_rotate180.py's periodic preview and eval_rotate180.py's checkpoint comparison, so
    the two can't drift apart."""
    pos_pred = lump_centroid_hw(pred_label, lump_class)
    pos_expected = lump_centroid_hw(expected_label, lump_class)
    return {
        "voxel_agreement": voxel_class_agreement(pred_label, expected_label),
        "centroid_error_voxels": lump_centroid_distance(pos_pred, pos_expected),
        "pred_has_lump": pos_pred is not None,
        "pred_lump_components": count_components(pred_label, classes=(lump_class,)),
        "expected_lump_components": count_components(expected_label, classes=(lump_class,)),
    }


@torch.no_grad()
def directional_signal_metrics(model, x0_batch: torch.Tensor, x1_batch: torch.Tensor,
                                t_values: Tuple[float, ...] = (0.1, 0.3, 0.7, 0.9)) -> Dict[str, float]:
    """
    Diagnostic on the RAW velocity prediction (before ODE integration/decoding): whether
    v_pred = model(xt, t) points in the right direction and has the right magnitude relative
    to the true target v_star = x1 - x0, restricted to voxels that actually need to change
    (v_star != 0). This can reveal real learning progress well before it's strong enough to
    flip the discrete decode after full ODE integration -- e.g. a model whose decoded output
    still looks identical to x0 can already have a strongly correct (but too weak) directional
    signal here.

    t=0.5 is deliberately excluded from the default t_values: it's the exact 50/50 blend of
    x0's and x1's one-hot vectors, a maximally ambiguous input where cosine similarity is
    structurally near zero regardless of training quality -- not a meaningful signal to track.

    x0_batch/x1_batch: [B, C, K, H, W] one-hot batches, already on model's device, model in
    eval mode. Pools all "needs to change" voxels across the whole batch (not per-item) before
    computing cosine similarity/magnitude ratio, for simplicity and to stay cheap on a large B.
    Returns nan for both fields if no voxel in the batch needs to change (shouldn't happen on
    real data).
    """
    v_star = x1_batch - x0_batch
    changed_mask = (v_star.abs().sum(dim=1, keepdim=True) > 0).expand_as(v_star)
    if not changed_mask.any():
        return {"cos_sim_mean": float("nan"), "magnitude_ratio_mean": float("nan")}

    cos_sims, mag_ratios = [], []
    for t_val in t_values:
        t = torch.full((x0_batch.shape[0],), t_val, device=x0_batch.device)
        xt = (1 - t_val) * x0_batch + t_val * x1_batch
        v_pred = model(xt, t)

        vp = v_pred[changed_mask]
        vs = v_star[changed_mask]
        cos_sims.append(torch.nn.functional.cosine_similarity(vp.unsqueeze(0), vs.unsqueeze(0), dim=1).item())
        mag_ratios.append((vp.abs().mean() / (vs.abs().mean() + 1e-8)).item())

    return {
        "cos_sim_mean": float(np.mean(cos_sims)),
        "magnitude_ratio_mean": float(np.mean(mag_ratios)),
    }
