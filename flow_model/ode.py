from typing import Optional, Tuple

import torch

from .velocity_model import FlowMatchingUNet3D


@torch.no_grad()
def sample(model: FlowMatchingUNet3D, n_steps: int, x0: Optional[torch.Tensor] = None,
           batch_size: int = 1, shape: Tuple[int, int, int] = (4, 26, 128, 128),
           device: str = "cpu", method: str = "euler") -> torch.Tensor:
    """
    Integrates dx/dt = v_theta(x, t) forward from t=0 to t=1.

    x0: if None, sampled ~ N(0, I) with the given batch_size/shape.
    Returns x1_hat: [B, C, D, H, W] continuous (caller does .argmax(dim=1) for the label volume).
    """
    model.eval()
    if x0 is None:
        x = torch.randn(batch_size, *shape, device=device)
    else:
        x = x0.to(device).clone()

    B = x.shape[0]
    dt = 1.0 / n_steps

    for i in range(n_steps):
        t = torch.full((B,), i * dt, device=device)
        v = model(x, t)
        if method == "euler":
            x = x + v * dt
        elif method == "heun":
            t_next = torch.full((B,), (i + 1) * dt, device=device)
            x_pred = x + v * dt
            v_next = model(x_pred, t_next)
            x = x + 0.5 * (v + v_next) * dt
        else:
            raise ValueError(f"Unknown method: {method}")

    return x


@torch.no_grad()
def invert(model: FlowMatchingUNet3D, x1: torch.Tensor, n_steps: int,
           method: str = "euler") -> torch.Tensor:
    """
    Integrates the same learned ODE backward from t=1 (a real one-hot phantom)
    to t=0, recovering the approximate noise/latent x0_hat.

    x1: [B, C, D, H, W] one-hot real phantom(s).
    """
    model.eval()
    device = x1.device
    x = x1.clone()
    B = x.shape[0]
    dt = 1.0 / n_steps

    for i in range(n_steps):
        t = torch.full((B,), 1.0 - i * dt, device=device)
        v = model(x, t)
        if method == "euler":
            x = x - v * dt
        elif method == "heun":
            t_next = torch.full((B,), 1.0 - (i + 1) * dt, device=device)
            x_pred = x - v * dt
            v_next = model(x_pred, t_next)
            x = x - 0.5 * (v + v_next) * dt
        else:
            raise ValueError(f"Unknown method: {method}")

    return x
