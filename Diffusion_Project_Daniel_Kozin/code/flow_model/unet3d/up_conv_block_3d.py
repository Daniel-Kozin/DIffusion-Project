import torch
from torch import nn

from .conv_block_3d import conv_block_3d


class up_conv_block_3d(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0, use_residual=True, padding=0):
        super().__init__()
        # when padding = 1, output padding is to make 6 -> 13 instead of 12
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2,
                                     output_padding=(padding, 0, 0))
        self.conv_block = conv_block_3d(out_ch * 2, out_ch, dropout, use_residual)  # after skip concat

    def forward(self, x, skip):
        x = self.up(x)  # [B, out_ch, 2*D, 2*H, 2*W]
        x = torch.cat([x, skip], dim=1)  # [B, out_ch + skip_channels, 2*D, 2*H, 2*W]
        x = self.conv_block(x)  # [B, out_ch, 2*D, 2*H, 2*W]
        return x
