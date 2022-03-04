"""Tests standard functionality of nescient."""

import nescient
import torch
import crypten
import doctest

def test_convnet():
    """Test if a ConvNet can be instantiated and forward propagated."""
    net = nescient.ConvNet()
    x = torch.rand(1,1,256,256)
    net.forward(x)

    
def test_wrapper():
    """Test if a ConvNetWrapper can be instantiated and forward propagated."""
    x = torch.rand(1,1,256,256)
    x = crypten.cryptensor(x)
    net = nescient.ConvNetWrapper()
    net.encrypted_infer(x)
