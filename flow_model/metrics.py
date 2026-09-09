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
