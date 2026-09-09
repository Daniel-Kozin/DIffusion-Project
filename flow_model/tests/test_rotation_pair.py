import numpy as np
import torch

from flow_model.data import RotationPairDataset, rotate_label_volume
from flow_model.metrics import rotation_metrics
from flow_model.train import flow_matching_loss
from flow_model.velocity_model import FlowMatchingUNet3D


def _make_label():
    label = np.zeros((3, 16, 16), dtype=np.int64)
    label[:, 2, 5] = 3  # asymmetric lump marker, off-center
    return label


def test_rotation_pair_dataset_covers_all_orientations_bidirectionally():
    label = _make_label()
    ds = RotationPairDataset({"c": [label]}, ["c"], num_classes=4)
    assert len(ds) == 8
    orientations = sorted(ds.index[i][2] for i in range(8))
    assert orientations == list(range(8))


def test_rotation_pair_dataset_x1_is_exact_180_of_x0():
    label = _make_label()
    ds = RotationPairDataset({"c": [label]}, ["c"], num_classes=4)
    for idx in range(len(ds)):
        case_id, variant_idx, orientation = ds.index[idx]
        x0, x1 = ds[idx]
        expected_x0 = rotate_label_volume(label, orientation)
        expected_x1 = rotate_label_volume(label, (orientation + 4) % 8)
        assert np.array_equal(x0.argmax(dim=0).numpy(), expected_x0)
        assert np.array_equal(x1.argmax(dim=0).numpy(), expected_x1)


def test_rotation_metrics_on_synthetic_shift():
    pred = torch.zeros(3, 16, 16, dtype=torch.long)
    expected = torch.zeros(3, 16, 16, dtype=torch.long)
    pred[:, 2, 5] = 3
    expected[:, 2, 8] = 3  # shifted 3 voxels along W

    m = rotation_metrics(pred, expected)
    assert m["pred_has_lump"] is True
    assert m["centroid_error_voxels"] is not None
    assert abs(m["centroid_error_voxels"] - 3.0) < 1e-6
    assert m["pred_lump_components"] == 1
    assert m["expected_lump_components"] == 1
    assert 0.0 <= m["voxel_agreement"] <= 1.0


def test_flow_matching_loss_with_real_x0_backward():
    # UNet3D's two downsampling levels assume the full (26, 128, 128) volume shape (see
    # test_model.py) -- a smaller depth/H/W breaks its skip-connection concat.
    model = FlowMatchingUNet3D(num_classes=4, base_ch=4, embed_channels=8)
    x0 = torch.rand(1, 4, 26, 128, 128)
    x1 = torch.rand(1, 4, 26, 128, 128)
    loss = flow_matching_loss(model, x1, torch.device("cpu"), x0=x0)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
