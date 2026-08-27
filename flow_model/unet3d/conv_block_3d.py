from torch import nn


class conv_block_3d(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        # 3x3x3 kernel doesn't change the resolution
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout)
        )
        # if ub_ch != out_ch dim doesnt match
        if use_residual and in_ch != out_ch:
            self.res_conv = nn.Conv3d(in_ch, out_ch, 1)
        else:
            self.res_conv = None

    def forward(self, x):
        out = self.conv(x)
        if self.use_residual:
            res = self.res_conv(x) if self.res_conv else x
            out = out + res
        return out
