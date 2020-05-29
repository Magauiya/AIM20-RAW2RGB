import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

class UNET(nn.Module):
	def __init__(self, cfg):
		super(UNET, self).__init__()
		self.upscale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
		self.inc = DoubleConv(cfg.in_channel, 64)
		self.down1 = Down(64, 128)
		self.down2 = Down(128, 256)
		self.down3 = Down(256, 512)
		self.down4 = Down(512, 512)
		self.up1 = Up(1024, 256)
		self.up2 = Up(512, 128)
		self.up3 = Up(256, 64)
		self.up4 = Up(128, 64)
		self.outc = OutConv(64, cfg.out_channel)

	def forward(self, x):
		x = self.upscale(x)
		x1 = self.inc(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		out = self.up1(x5, x4)
		out = self.up2(out, x3)
		out = self.up3(out, x2)
		out = self.up4(out, x1)
		out = self.outc(out)
		
		return out