"""
Library for encrypted classification of images.
"""
__version__ = "0.1.0"

import numpy as np
import torch
from torch import nn
import crypten


class ConvNet(torch.nn.Module):
    """Simple conv net for binary classification of images."""
    def __init__(self):
        """Initializes the model"""        
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, (8, 8), stride=8)
        self.linear = nn.Linear(32, 16)
        self.out = nn.Linear(16, 2)

    def forward(self, x):
        """
        Forward propagates the model.
        
        Arguments:
        - x: a (1,1,256,256) black and white image.
        """
        x = self.conv1(x)
        x = self.linear(x)
        return self.out(x)


class ConvNetWrapper:
    """Wraps a ConvNet() into an encrypted model"""
    def __init__(self, model=ConvNet()):
        """
        Arguments: 
        - model: a ConvNet
        """
        self.graph = crypten.nn.from_pytorch(
            model,
            torch.rand(1, 1, 256, 256),
        ).encrypt()

    def encrypted_infer(self, x):
        """
        Peforms encrypted inference on an input.
        Argument: 
        - x: a (1,1,256,256) black and white image. 
        """
        self.graph.forward(x)

    # def train(self, inputs, labels):
    #     pass
    
