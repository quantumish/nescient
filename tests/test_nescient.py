"""Tests standard functionality of nescient."""

import doctest
from itertools import islice

import crypten
import nescient
import torch


def test_convnet():
    """Test if a ConvNet can be instantiated and forward propagated."""
    net = nescient.ConvNet()
    x = torch.rand(1, 1, 512, 512)
    net.forward(x)


def test_train_sanity():
    """Model sanity check: make sure the model is able to train at all."""
    net = nescient.ConvNet()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    i = torch.rand(1, 1, 512, 512)
    label = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    outputs = net(i)
    orig_loss = criterion(outputs, label)
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = net(i)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
    assert loss.item() < orig_loss
    return loss


def test_train_well_sanity():
    """Test if model can train to near zero loss on a single datapoint."""
    loss = test_train_sanity()
    assert loss.item() < 0.01


def test_train_epoch():
    net = nescient.ConvNet()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    print()
    for n in range(100):
        it = nescient.DataIterator(
            "/home/quantumish/aux/CheXpert-v1.0-small/train.csv", "/home/quantumish/aux"
        )
        iters = 0
        max_iters = 0
        losses = []
        for i in it:
            if i == None:
                continue
            # print(i[0])
            if iters > max_iters:
                break
            optimizer.zero_grad()
            outputs = net(i[0])
            loss = criterion(outputs, i[1])
            losses.append(loss.item())
            # print("Iteration {}/{} (running avg. loss {})".format(iters, max_iters, sum(losses)/(iters+1)), end='\n' if iters == max_iters else '\r')
            print("{}\n{}\n{}\n\n".format(outputs.tolist(), i[1].tolist(), loss.item()))
            loss.backward()
            optimizer.step()
            iters += 1
        print("Epoch {}/100: avg. loss of {}".format(n, sum(losses) / (max_iters + 1)))
    print(loss.item())
    assert 1 == 0


def test_wrapper():
    """Test if a ConvNetWrapper can be instantiated and forward propagated."""
    x = torch.rand(1, 1, 512, 512)
    x = crypten.cryptensor(x)
    net = nescient.ConvNetWrapper()
    net.encrypted_infer(x)
