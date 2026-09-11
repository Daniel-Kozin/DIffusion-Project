import torch

from flow_model.ode import invert, sample
from flow_model.velocity_model import FlowMatchingUNet3D


def _tiny_model():
    return FlowMatchingUNet3D(num_classes=4, base_ch=4, embed_channels=8)


def test_sample_shape():
    model = _tiny_model()
    x1_hat = sample(model, n_steps=5, batch_size=2, shape=(4, 26, 128, 128), device="cpu")
    assert x1_hat.shape == (2, 4, 26, 128, 128)


def test_sample_with_given_x0_shape():
    model = _tiny_model()
    x0 = torch.randn(1, 4, 26, 128, 128)
    x1_hat = sample(model, n_steps=5, x0=x0, device="cpu")
    assert x1_hat.shape == x0.shape


def test_invert_shape():
    model = _tiny_model()
    x1 = torch.randn(1, 4, 26, 128, 128)
    x0_hat = invert(model, x1, n_steps=5)
    assert x0_hat.shape == x1.shape


def test_sample_return_trajectory_matches_final_and_has_n_steps_plus_one_frames():
    model = _tiny_model()
    x0 = torch.randn(1, 4, 26, 128, 128)
    x1_hat, trajectory = sample(model, n_steps=5, x0=x0, device="cpu", return_trajectory=True)

    assert len(trajectory) == 6  # t_start, after each of 5 steps
    assert torch.equal(trajectory[0], x0)  # first frame is the exact starting state
    assert torch.equal(trajectory[-1], x1_hat)  # last frame matches the returned final state
    assert all(t.shape == x0.shape for t in trajectory)


def test_sample_without_return_trajectory_is_unaffected():
    # default (omitted) return_trajectory must stay a bare tensor for every existing caller
    torch.manual_seed(0)
    model = _tiny_model()
    x0 = torch.randn(1, 4, 26, 128, 128)
    out = sample(model, n_steps=5, x0=x0, device="cpu")
    assert isinstance(out, torch.Tensor)


def test_invert_then_sample_runs_and_stays_finite():
    """
    On an untrained model we can't expect an accurate reconstruction — this only
    checks the invert -> sample round trip runs end-to-end and produces finite,
    correctly-shaped output. Accuracy against a real checkpoint is checked
    separately in sanity_checks.round_trip_reconstruction_check.
    """
    model = _tiny_model()
    x1 = torch.randn(1, 4, 26, 128, 128)

    x0_hat = invert(model, x1, n_steps=10)
    x1_hat = sample(model, n_steps=10, x0=x0_hat, device="cpu")

    assert x1_hat.shape == x1.shape
    assert torch.isfinite(x1_hat).all()
