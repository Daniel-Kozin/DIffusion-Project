from pathlib import Path

import numpy as np

from flow_model.data import build_datasets, rotate_label_volume

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "mri_images_3D"


def test_datasets_non_empty_and_one_hot():
    train_ds, val_ds = build_datasets(DATA_DIR, val_frac=0.2, num_orientations=8, seed=0)
    assert len(train_ds) > 0
    assert len(val_ds) > 0

    sample = train_ds[0]
    assert sample.shape == (4, 26, 128, 128)
    voxel_sums = sample.sum(dim=0)
    assert bool((voxel_sums == 1.0).all())


def test_rotation_identity_at_orientation_zero():
    label = np.random.randint(0, 4, size=(3, 16, 16)).astype(np.int64)
    rotated = rotate_label_volume(label, orientation=0)
    assert np.array_equal(rotated, label)


def test_rotation_180_matches_double_flip():
    label = np.zeros((3, 16, 16), dtype=np.int64)
    label[:, 2, 5] = 3  # asymmetric marker
    rotated = rotate_label_volume(label, orientation=4)  # -4*45 = -180 degrees
    expected = np.flip(label, axis=(1, 2))
    assert np.array_equal(rotated, expected)


def test_rotation_preserves_discrete_labels_no_blending():
    label = np.zeros((3, 16, 16), dtype=np.int64)
    label[:, 4:8, 4:8] = 1
    label[:, 8:10, 8:10] = 3
    rotated = rotate_label_volume(label, orientation=1)  # 45 degrees, non-axis-aligned
    unique_vals = set(np.unique(rotated).tolist())
    assert unique_vals.issubset({0, 1, 2, 3})
