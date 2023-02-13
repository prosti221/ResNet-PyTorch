'''
Implementing a 34-layer ResNet model as described in the paper.
This has a fixed number of layers and is not adapted to change.

Paper: https://arxiv.org/pdf/1512.03385.pdf
'''
import torch
import torch.nn as nn
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride1=1, stride2=1):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
                Conv2dAuto(in_channels, out_channels, kernel_size=(k_size, k_size), stride=stride1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                Conv2dAuto(out_channels, out_channels, kernel_size=(k_size, k_size), stride=stride2),
                nn.BatchNorm2d(out_channels)
                )

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.stride1 = stride1
        self.stride2 = stride2

    def forward(self, x):
        I = x
        out = self.block(x)

        if self.stride1 != self.stride2:
            I = nn.functional.pad(I[:, :, ::2, ::2], (0, 0, 0, 0, self.out_channels//4, self.out_channels//4), "constant", 0)
        out += I
        return nn.ReLU()(out)

class ConvStack(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride1=1, stride2=1, depth=2):
        super(ConvStack, self).__init__()
        self.layers = [ResidualBlock(in_channels, out_channels, k_size, stride1, stride2)]

        for i in range(1, depth):
            self.layers.append(ResidualBlock(out_channels, out_channels, k_size, stride2, stride2))

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)

'''
Credits to: https://github.com/FrancescoSaverioZuppichini/ResNet/blob/master/ResNet.ipynb
for Conv2dAuto
'''
class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2)

# Reduced is set true for an 18-layer network instead of 37
class ResNet(nn.Module):
    def __init__(self, num_classes=1000, reduced=False):
        super(ResNet, self).__init__()
        stack_depth = [3, 8, 12, 6] if not reduced else [2, 2, 2, 2]
        self.model = nn.Sequential(
                Conv2dAuto(3, 64, kernel_size=(7, 7)),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=1),
                ConvStack(64, 64, k_size=3, stride1=1, stride2=1, depth=stack_depth[0]),
                ConvStack(64, 128, k_size=3, stride1=2, stride2=1, depth=stack_depth[1]),
                ConvStack(128, 256, k_size=3, stride1=2, stride2=1, depth=stack_depth[2]),
                ConvStack(256, 512, k_size=3, stride1=2, stride2=1, depth=stack_depth[3]),
                nn.AdaptiveAvgPool2d((1, 1)),
                )
        self.fc = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        return self.fc(self.model(x).view(-1, 512))
