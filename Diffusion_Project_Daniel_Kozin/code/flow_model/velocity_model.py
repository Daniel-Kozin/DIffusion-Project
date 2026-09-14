import torch
import torch.utils.checkpoint
from torch import nn

from .time_embedding import sinusoidal_time_embedding
from .unet3d.unet_3d import UNet3D


class TimeMLP(nn.Module):
    def __init__(self, embed_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_channels, embed_channels),
            nn.SiLU(),
            nn.Linear(embed_channels, embed_channels),
        )

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        return self.net(t_emb)


class FlowMatchingUNet3D(nn.Module):
    """Unconditional flow-matching velocity model: v_theta(x, t)."""

    def __init__(self, num_classes: int = 4, base_ch: int = 8, embed_channels: int = 16,
                 dropout: float = 0.1, use_grad_checkpoint: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.embed_channels = embed_channels
        self.use_grad_checkpoint = use_grad_checkpoint

        self.time_mlp = TimeMLP(embed_channels)
        self.unet = UNet3D(in_channels=num_classes + embed_channels,
                            out_channels=num_classes, base_ch=base_ch, dropout=dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: [B, num_classes, D, H, W] float
        t: [B] float in [0, 1]
        returns v_pred: [B, num_classes, D, H, W]
        """
        B, C, D, H, W = x.shape

        t_emb = sinusoidal_time_embedding(t, self.embed_channels)
        t_emb = self.time_mlp(t_emb)
        t_emb = t_emb.view(B, self.embed_channels, 1, 1, 1).expand(B, self.embed_channels, D, H, W)

        h = torch.cat([x, t_emb], dim=1)  # [B, num_classes + embed_channels, D, H, W]

        if self.use_grad_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(self.unet, h, use_reentrant=False)
        return self.unet(h)
