"""Tests standard functionality of nescient."""

import nescient
import torch
import crypten
import doctest

def test_convnet():
    """Test if a ConvNet can be instantiated and forward propagated."""
    net = nescient.ConvNet()
    x = torch.rand(1,1,512,512)
    net.forward(x)

def test_train_sanity():
    """Model sanity check: make sure the model is able to train at all."""
    net = nescient.ConvNet()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
    i = torch.rand(1,1,512,512)
    label = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    outputs = net(i)
    orig_loss = criterion(outputs, label)
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = net(i)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
    assert(loss.item() < orig_loss)
    return loss

def test_train_well_sanity():
    """Model sanity check: make sure the model can train to near zero loss on a single datapoint."""
    loss = test_train_sanity()
    assert(loss.item() < 0.01)
    
def test_wrapper():
    """Test if a ConvNetWrapper can be instantiated and forward propagated."""
    x = torch.rand(1,1,512,512)
    x = crypten.cryptensor(x)
    net = nescient.ConvNetWrapper()
    net.encrypted_infer(x)
