import torch
from torch import nn
from ... import loader


class AdaptiveModel(nn.Module):
    def __init__(self, in_channels, out_channels, adaptive_layer, main_model):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.adaptive_layer = loader(adaptive_layer, in_channels=in_channels)
        inner_channels = self.adaptive_layer.out_channels
        self.main_model = loader(main_model, in_channels=inner_channels, out_channels=out_channels)

    def forward(self, x):
        xx = self.adaptive_layer(x)
        return self.main_model(xx)
