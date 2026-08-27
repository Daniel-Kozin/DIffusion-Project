import torch
from torch import nn


class self_attention_3d(nn.Module):
	def __init__(self, channels):
		super().__init__()
		self.channels = channels
		# kernel = 1 doesn't change the output size
		self.q = nn.Conv3d(channels, channels, 1)
		self.k = nn.Conv3d(channels, channels, 1)
		self.v = nn.Conv3d(channels, channels, 1)
		self.proj_out = nn.Conv3d(channels, channels, 1)

	def forward(self, x):
		B, C, D, H, W = x.shape
		q = self.q(x).reshape(B, C, -1)  # [B, C, DHW]
		k = self.k(x).reshape(B, C, -1)  # [B, C, DHW]
		v = self.v(x).reshape(B, C, -1)  # [B, C, DHW]

		attn = torch.softmax(torch.bmm(q.permute(0, 2, 1), k) / C ** 0.5, dim=-1)  # [B, DHW, DHW]
		out = torch.bmm(v, attn.permute(0, 2, 1))  # [B, C, DHW]
		out = out.reshape(B, C, D, H, W)

		# residual
		res_out = self.proj_out(out) + x
		return res_out
