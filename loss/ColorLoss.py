import torch
from torch.nn import CosineSimilarity
from torch.nn.functional import interpolate


class ColorLoss(torch.nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()
        self.color_criterion = CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, x, y):
        hue_sim = self.color_criterion(interpolate(x, scale_factor=0.5), interpolate(y, scale_factor=0.5))

        color_loss = 1. - torch.mean(hue_sim)
        return color_loss
