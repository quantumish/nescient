"""
Library for encrypted classification of images.
"""
__version__ = "0.1.0"

import numpy as np
import torch
from torch import nn
import crypten


class ConvNet(torch.nn.Module):
    """Simple conv net for classification of chest X rays."""
    def __init__(self):
        """Initializes the model"""        
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, (8, 8), stride=4)
        self.conv2 = nn.Conv2d(1, 1, (4, 4), stride=2)
        self.linear = nn.Linear(3844, 120)
        self.out = nn.Linear(120, 11)

    def forward(self, x):
        """
        Forward propagates the model.
        
        Arguments:
        - x: a (1, 1, 512, 512) grayscale image.
        """
        x = self.conv1(x)
        x = self.conv2(x)    
        x = self.linear(torch.flatten(x, 1))
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
            torch.rand(1, 1, 512, 512),
        ).encrypt()

    def encrypted_infer(self, x):
        """
        Peforms encrypted inference on an input.
        Argument: 
        - x: a (1, 1, 512, 512) black and white image. 
        """
        self.graph.forward(x)

    # def train(self, inputs, labels):
    #     pass
    
