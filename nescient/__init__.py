__version__ = "0.1.0"

import numpy as np
import torch
from torch import nn


class ConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 5, (5, 5))
        self.linear = nn.Linear(5, 50)
        self.out = nn.Linear(50, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.linear(x)
        return self.out(x)


net = ConvNet()
print(net)
