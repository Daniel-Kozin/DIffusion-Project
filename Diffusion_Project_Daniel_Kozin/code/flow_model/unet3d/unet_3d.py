from torch import nn

from .conv_block_3d import conv_block_3d
from .self_attention_3d import self_attention_3d
from .up_conv_block_3d import up_conv_block_3d


class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels=4, base_ch=8, dropout=0.0):
        """
        in_channels: input channels
        out_channels: output channels
        base_ch: features at first layer
        attention_resolutions: which levels to apply self-attention
        """
        super().__init__()
        # Encoder
        self.enc1 = conv_block_3d(in_channels, base_ch, dropout)
        self.enc2 = conv_block_3d(base_ch, base_ch * 2, dropout)
        self.pool = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = conv_block_3d(base_ch * 2, base_ch * 4, dropout)
        self.attn_bottleneck = self_attention_3d(base_ch * 4)

        # Decoder
        self.up2 = up_conv_block_3d(base_ch * 4, base_ch * 2, dropout, padding=1)
        self.up1 = up_conv_block_3d(base_ch * 2, base_ch, dropout)

        # Output
        self.out_conv = nn.Conv3d(base_ch, out_channels, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)  # [B, base_ch, 26, 128, 128]
        e2 = self.pool(e1)  # [B, base_ch, 13, 64, 64]
        e3 = self.enc2(e2)  # [B, 2*base_ch, 13, 64, 64]

        # Bottleneck
        b1 = self.pool(e3)  # [B, 2*base_ch, 6, 32, 32]
        b2 = self.bottleneck(b1)  # [B, 4*base_ch, 6, 32, 32]
        b3 = self.attn_bottleneck(b2)  # [B, 4*base_ch, 6, 32, 32]

        # Decoder
        d2 = self.up2(b3, e3)  # [B, 2*base_ch, 13, 64, 64]
        d1 = self.up1(d2, e1)  # [B, base_ch,26, 128, 128]

        # Output
        out = self.out_conv(d1)  # [B, out_channels, 26, 128, 128]
        return out
