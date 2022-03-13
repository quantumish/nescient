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
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    it = nescient.DataIterator(
        "/home/quantumish/aux/CheXpert-v1.0-small/train.csv", "/home/quantumish/aux"
    )
    img, label = None, None
    for i in it:
        if i != None:
            img, label = i[0], i[1]
            break
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
    return loss


def test_train_epoch():
    # net = nescient.ConvNet()
    # criterion = torch.nn.MSELoss()
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    # print()
    # for n in range(100):
    #     it = nescient.DataIterator(
    #         "/home/quantumish/aux/CheXpert-v1.0-small/train.csv", "/home/quantumish/aux"
    #     )
    #     iters = 0
    #     max_iters = 0
    #     losses = []
    #     for i in it:
    #         if i == None:
    #             continue
    #         # print(i[0])
    #         if iters > max_iters:
    #             break
    #         outputs = net(i[0])
    #         loss = criterion(outputs, i[1])
    #         losses.append(loss.item())
    #         # print("Iteration {}/{} (running avg. loss {})".format(iters, max_iters, sum(losses)/(iters+1)), end='\n' if iters == max_iters else '\r')
    #         # print("{}\n{}\n{}\n\n".format(outputs.tolist(), i[1].tolist(), loss.item()))
    #         loss.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()
    #         iters += 1
    #     print("Epoch {}/100: avg. loss of {}".format(n, sum(losses) / (max_iters + 1)))
    # print(loss.item())
    # assert 1 == 0
    pass


def test_wrapper():
    """Test if a ConvNetWrapper can be instantiated and forward propagated."""
    x = torch.rand(1, 1, 320, 390)
    x = crypten.cryptensor(x)
    net = nescient.ConvNetWrapper()
    net.encrypted_infer(x)


def test_docs():
    """Test all documentation examples of nescient and ensure they run without errors."""
    assert(doctest.testmod(nescient)[0] == 0)
