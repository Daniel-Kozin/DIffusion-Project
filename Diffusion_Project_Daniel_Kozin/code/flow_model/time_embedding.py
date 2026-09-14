import math

import torch


def sinusoidal_time_embedding(t: torch.Tensor, dim: int, max_period: float = 1000.0) -> torch.Tensor:
    """
    Standard transformer/DDPM sinusoidal embedding.

    Parameters
    ----------
    t : torch.Tensor
        Shape [B], float values in [0, 1].
    dim : int
        Output embedding dimension.
    max_period : float
        t is scaled by this before computing frequencies, so t in [0,1] still
        spans a useful frequency range instead of collapsing to near-zero angles.

    Returns
    -------
    torch.Tensor
        Shape [B, dim].
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = (t.float() * max_period)[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb
