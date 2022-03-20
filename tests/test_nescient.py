"""Tests standard functionality of nescient."""

import doctest
from itertools import islice

import crypten
import nescient
import torch


def test_convnet():
    """Test if a ConvNet can be instantiated and forward propagated."""
    net = nescient.ConvNet()
    x = torch.rand(1, 1, 320, 390)
    net.forward(x)


def test_train_sanity():
    """Model sanity check: make sure the model is able to train at all."""
    net = nescient.ConvNet()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    img = torch.rand(1, 1, 320, 390)
    label = torch.tensor([[1.0]])
    outputs = net(img)
    orig_loss = criterion(outputs, label)
    for epoch in range(100):
        outputs = net(img)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    assert loss.item() < orig_loss
    assert loss.item() < 0.0001
    
    
def test_wrapper():
    """Test if a ConvNetWrapper can be instantiated and forward propagated."""
    x = torch.rand(1, 1, 320, 390)
    x = crypten.cryptensor(x)
    net = nescient.ConvNetWrapper(nescient.ConvNet())
    net.infer(x)
