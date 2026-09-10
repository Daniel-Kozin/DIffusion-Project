import numpy as np
import torch

from flow_model.data import RotationPairDataset, rotate_label_volume
from flow_model.metrics import directional_signal_metrics, mask_dice, mask_iou, rotation_metrics
from flow_model.train import flow_matching_loss, soft_dice_loss
from flow_model.train_rotate180 import EMA, compute_pixel_loss_weight
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
    # disjoint single-voxel masks -> IoU/Dice both 0 (no overlap)
    assert m["lump_iou"] == 0.0
    assert m["lump_dice"] == 0.0


def test_mask_iou_and_dice_perfect_and_zero_overlap():
    a = torch.zeros(2, 4, 4, dtype=torch.long)
    a[0, 1, 1] = 3
    a[0, 2, 2] = 3
    b_perfect = a.clone()
    b_disjoint = torch.zeros(2, 4, 4, dtype=torch.long)
    b_disjoint[1, 3, 3] = 3

    assert mask_iou(a, b_perfect, 3) == 1.0
    assert mask_dice(a, b_perfect, 3) == 1.0
    assert mask_iou(a, b_disjoint, 3) == 0.0
    assert mask_dice(a, b_disjoint, 3) == 0.0
    assert mask_iou(torch.zeros_like(a), torch.zeros_like(a), 3) is None


def test_mask_iou_penalizes_oversized_blob_more_than_centroid_would():
    # true: a single lump voxel. predicted: covers it plus a much larger surrounding blob --
    # the "huge blob near the right spot" failure this metric is meant to catch. Centroid
    # distance could look fine here (the blob's center can be near the true voxel) while IoU
    # stays low, since the union grows much faster than the intersection.
    true_label = torch.zeros(1, 8, 8, dtype=torch.long)
    true_label[0, 4, 4] = 3
    pred_label = torch.zeros(1, 8, 8, dtype=torch.long)
    pred_label[0, 2:7, 2:7] = 3  # 25-voxel blob containing the true voxel

    iou = mask_iou(pred_label, true_label, 3)
    assert iou is not None and iou < 0.1  # 1 / 25 = 0.04


def test_mask_iou_combined_classes():
    pred = torch.zeros(1, 4, 4, dtype=torch.long)
    true = torch.zeros(1, 4, 4, dtype=torch.long)
    pred[0, 0, 0] = 2  # pillar, correct
    pred[0, 1, 1] = 3  # lump, correct
    true[0, 0, 0] = 2
    true[0, 1, 1] = 3

    assert mask_iou(pred, true, (2, 3)) == 1.0
    assert mask_dice(pred, true, (2, 3)) == 1.0
    # single-class views still see only their own class
    assert mask_iou(pred, true, 2) == 1.0
    assert mask_iou(pred, true, 3) == 1.0


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


def test_directional_signal_metrics_shape_and_range():
    model = FlowMatchingUNet3D(num_classes=4, base_ch=4, embed_channels=8)
    model.eval()
    x0 = torch.rand(2, 4, 26, 128, 128)
    x1 = torch.rand(2, 4, 26, 128, 128)
    m = directional_signal_metrics(model, x0, x1)
    assert -1.0 - 1e-6 <= m["cos_sim_mean"] <= 1.0 + 1e-6
    assert m["magnitude_ratio_mean"] >= 0.0


def test_directional_signal_metrics_nan_when_nothing_changes():
    model = FlowMatchingUNet3D(num_classes=4, base_ch=4, embed_channels=8)
    model.eval()
    x0 = torch.rand(1, 4, 26, 128, 128)
    m = directional_signal_metrics(model, x0, x0.clone())  # x1 == x0 everywhere, no changed voxels
    assert m["cos_sim_mean"] != m["cos_sim_mean"]  # NaN
    assert m["magnitude_ratio_mean"] != m["magnitude_ratio_mean"]


def test_flow_matching_loss_pixel_weight_zero_matches_omitted():
    # pixel_loss_weight=0.0 must be bit-identical to not passing it at all -- every existing
    # caller (train.py's unconditional loop, sanity_checks.py) omits it and must see zero
    # behavior change.
    torch.manual_seed(0)
    model = FlowMatchingUNet3D(num_classes=4, base_ch=4, embed_channels=8)
    x0 = torch.rand(1, 4, 26, 128, 128)
    x1 = torch.rand(1, 4, 26, 128, 128)

    torch.manual_seed(1)
    loss_omitted = flow_matching_loss(model, x1, torch.device("cpu"), x0=x0)
    torch.manual_seed(1)
    loss_explicit_zero = flow_matching_loss(model, x1, torch.device("cpu"), x0=x0, pixel_loss_weight=0.0)
    assert torch.allclose(loss_omitted, loss_explicit_zero)


def test_pixel_loss_is_not_redundant_with_velocity_loss():
    # Regression guard: an earlier (rejected) version of this pixel loss used MSE on the
    # one-step endpoint estimate x1_hat = xt + (1-t)*v_pred, which is algebraically EXACTLY
    # (1-t)^2 times the velocity MSE term -- same minimizer, same gradient direction, just a
    # per-example reweighting, not a new training signal. The actual fix uses cross-entropy
    # instead, which has a genuinely different gradient landscape. This test checks that the
    # gradient with pixel_loss_weight>0 is NOT just a positive rescaling of the velocity-only
    # gradient (which is what the redundant MSE version would have produced).
    torch.manual_seed(0)
    model = FlowMatchingUNet3D(num_classes=4, base_ch=4, embed_channels=8)
    x0 = torch.rand(1, 4, 26, 128, 128)
    x1 = torch.rand(1, 4, 26, 128, 128)

    def flat_grad(pixel_loss_weight):
        model.zero_grad()
        torch.manual_seed(42)  # same sampled t and same forward pass for a fair comparison
        loss = flow_matching_loss(model, x1, torch.device("cpu"), x0=x0, pixel_loss_weight=pixel_loss_weight)
        loss.backward()
        return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])

    g_velocity_only = flat_grad(0.0)
    g_with_pixel = flat_grad(1.0)

    # A TRUE algebraic redundancy (like the rejected MSE version) makes g_with_pixel an exact
    # positive scalar multiple of g_velocity_only, so cos_sim is 1.0 up to float32 rounding
    # (~1e-6). 0.99999 leaves ample margin above that noise floor while still failing on an
    # actual identity -- a small, untrained, random-input toy model can legitimately show
    # high (but not identity-level) correlation between two genuinely different losses.
    cos_sim = torch.nn.functional.cosine_similarity(g_velocity_only.unsqueeze(0), g_with_pixel.unsqueeze(0)).item()
    assert cos_sim < 0.99999, (
        f"pixel loss gradient is suspiciously close to an exact scalar multiple of the "
        f"velocity-only gradient (cos_sim={cos_sim:.6f}) -- looks like the redundant-MSE bug"
    )


def test_soft_dice_loss_zero_for_perfect_match():
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    target[0, 1, 1] = 3  # a single lump voxel
    probs = torch.nn.functional.one_hot(target, num_classes=4).permute(0, 3, 1, 2).float()
    loss = soft_dice_loss(probs, target, classes=(3,))
    assert loss.item() < 1e-3


def test_soft_dice_loss_penalizes_over_prediction_that_cross_entropy_barely_sees():
    # True: a single lump voxel. Predicted: covers that voxel PLUS 10 extra background
    # voxels confidently classified as lump -- the "huge blob" failure mode this loss
    # targets. Dice should be clearly penalized (it directly measures volume overlap);
    # class-weighted CE would barely react since the 10 false positives are weighted at
    # their TRUE class (background, weight=1.0), not at the wrongly-predicted lump weight.
    target = torch.zeros(1, 6, 6, dtype=torch.long)
    target[0, 2, 2] = 3
    probs = torch.zeros(1, 4, 6, 6)
    probs[0, 0] = 1.0  # everything defaults to confident background
    # confidently (mis)predict lump at the true voxel plus 10 background voxels
    lump_voxels = [(2, 2)] + [(r, c) for r in range(6) for c in range(6) if (r, c) != (2, 2)][:10]
    for r, c in lump_voxels:
        probs[0, :, r, c] = 0.0
        probs[0, 3, r, c] = 1.0

    loss = soft_dice_loss(probs, target, classes=(3,))
    # true=1 voxel, predicted=11 voxels, intersection=1 -> dice = 2*1/(1+11) = 1/6, loss = 5/6
    assert loss.item() > 0.5


def test_compute_pixel_loss_weight_interpolates_and_handles_resume():
    class Cfg:
        pixel_loss_weight = 1.0
        pixel_loss_weight_final = 0.3
        epochs = 100

    cfg = Cfg()
    assert compute_pixel_loss_weight(cfg, 0) == 1.0
    assert abs(compute_pixel_loss_weight(cfg, 100) - 0.3) < 1e-9
    mid = compute_pixel_loss_weight(cfg, 50)
    assert 0.3 < mid < 1.0
    # continues correctly for a run resumed partway through -- schedule depends only on the
    # absolute epoch_num/epochs, not on how many epochs *this session* has run
    assert compute_pixel_loss_weight(cfg, 75) == compute_pixel_loss_weight(cfg, 75)


def test_ema_tracks_and_smooths_model_weights():
    torch.manual_seed(0)
    model = FlowMatchingUNet3D(num_classes=4, base_ch=4, embed_channels=8)
    ema = EMA(model, decay=0.9)
    param_name = next(k for k, v in model.state_dict().items() if torch.is_floating_point(v))
    old_ema_value = ema.shadow[param_name].clone()

    # simulate a large, sudden parameter jump (like one noisy SGD step)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)

    raw_value = model.state_dict()[param_name]
    new_ema_value = ema.shadow[param_name]
    # EMA should have moved toward the new value but by less than the full jump
    assert not torch.allclose(new_ema_value, raw_value)
    assert (new_ema_value - old_ema_value).abs().sum() < (raw_value - old_ema_value).abs().sum()


def test_ema_load_state_dict_replaces_shadow():
    model = FlowMatchingUNet3D(num_classes=4, base_ch=4, embed_channels=8)
    ema = EMA(model, decay=0.999)
    new_state = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
    ema.load_state_dict(new_state)
    assert all(torch.equal(v, torch.zeros_like(v)) for v in ema.shadow.values())
