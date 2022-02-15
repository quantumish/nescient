__version__ = "0.1.0"

import numpy as np
import torch
from concrete.torch.compile import compile_torch_model
from torch import nn


class ConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, (8, 8), stride=8)
        self.linear = nn.Linear(32 * 32, 512)
        self.out = nn.Linear(512, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.linear(x)
        return self.out(x)


class ConvNetWrapper:
    """I am bad at naming things."""

    def __init__(self):
        self.module = compile_torch_model(
            ConvNet(),
            torch.rand(1, 1, 256, 256),
            n_bits=7,
        )

    def infer(self):
        enc_x = np.array([np.random.randn(1, 1, 256, 256)]).astype(np.uint8)
        out = self.module.forward_fhe.run(enc_x)
        return self.module.dequantize_output(np.array(out, dtype=np.float32))
