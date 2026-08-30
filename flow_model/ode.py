from typing import Optional, Tuple

import torch

from .velocity_model import FlowMatchingUNet3D


@torch.no_grad()
def sample(model: FlowMatchingUNet3D, n_steps: int, x0: Optional[torch.Tensor] = None,
           batch_size: int = 1, shape: Tuple[int, int, int] = (4, 26, 128, 128),
           device: str = "cpu", method: str = "euler",
           t_start: float = 0.0, t_end: float = 1.0) -> torch.Tensor:
    """
    Integrates dx/dt = v_theta(x, t) forward from t=t_start to t=t_end (default: 0 to 1,
    i.e. full noise to full data). Pass t_start>0 to resume a partial trajectory (e.g. the
    output of invert() with a matching t_end) rather than always starting from noise.

    x0: state at t=t_start; if None, sampled ~ N(0, I) with the given batch_size/shape
    (only meaningful when t_start=0, i.e. actually pure noise).
    Returns x_t_end: [B, C, D, H, W] continuous (caller does .argmax(dim=1) for the label volume).
    """
    model.eval()
    if x0 is None:
        x = torch.randn(batch_size, *shape, device=device)
    else:
        x = x0.to(device).clone()

    B = x.shape[0]
    dt = (t_end - t_start) / n_steps

    for i in range(n_steps):
        t = torch.full((B,), t_start + i * dt, device=device)
        v = model(x, t)
        if method == "euler":
            x = x + v * dt
        elif method == "heun":
            t_next = torch.full((B,), t_start + (i + 1) * dt, device=device)
            x_pred = x + v * dt
            v_next = model(x_pred, t_next)
            x = x + 0.5 * (v + v_next) * dt
        else:
            raise ValueError(f"Unknown method: {method}")

    return x


@torch.no_grad()
def invert(model: FlowMatchingUNet3D, x1: torch.Tensor, n_steps: int,
           method: str = "euler", t_start: float = 1.0, t_end: float = 0.0) -> torch.Tensor:
    """
    Integrates the same learned ODE backward from t=t_start (default 1, a real one-hot
    phantom) to t=t_end (default 0, full noise/latent x0_hat). Pass t_end>0 for partial
    inversion --- stopping partway back toward noise instead of going all the way, so the
    result is still partway pulled toward the real phantom's own anatomy.

    x1: [B, C, D, H, W] one-hot real phantom(s) (or any state at t=t_start).
    """
    model.eval()
    device = x1.device
    x = x1.clone()
    B = x.shape[0]
    dt = (t_start - t_end) / n_steps

    for i in range(n_steps):
        t = torch.full((B,), t_start - i * dt, device=device)
        v = model(x, t)
        if method == "euler":
            x = x - v * dt
        elif method == "heun":
            t_next = torch.full((B,), t_start - (i + 1) * dt, device=device)
            x_pred = x - v * dt
            v_next = model(x_pred, t_next)
            x = x - 0.5 * (v + v_next) * dt
        else:
            raise ValueError(f"Unknown method: {method}")

    return x
