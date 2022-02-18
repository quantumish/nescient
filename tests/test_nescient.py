import nescient
import torch


def test_version():
    assert nescient.__version__ == "0.1.0"


def test_convnet():
    net = nescient.ConvNet()
    x = torch.rand(16)
    net.forward(x)


def test_wrapper():
    net = nescient.ConvNetWrapper()
    net.infer()
