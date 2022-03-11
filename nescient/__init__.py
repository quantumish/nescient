"""Library for encrypted classification of images."""
__version__ = "0.1.0"

import crypten
import numpy
import torch
from PIL import Image
from torch import nn
from torchvision.transforms import ToTensor


class DataIterator:
    def __init__(self, csv_path: str, data_folder: str):
        self.data_folder = data_folder
        self.file = open(csv_path, "r")

    def __iter__(self):
        next(self.file)  # skip header line
        return self

    def __next__(self):
        line = next(self.file).split(",")
        raw_image = Image.open(self.data_folder + "/" + line[0])
        # print(raw_image.size, line[0])
        if raw_image.size != (320, 390):
            # print("Skipping!")
            return None        
        image = torch.tensor(
            [numpy.array(
                (Image.open(self.data_folder + "/" + line[0])).getdata(),
                dtype=numpy.float32
            ).reshape(1, 320, 390)]
        )

        def sketchy_float(i):
            try:
                return float(i)
            except:
                return 0.0

            
        label = torch.tensor([[0 if i == '' else sketchy_float(i) for i in line[5:]]])
        return (image, label)


class ConvNet(torch.nn.Module):
    """Simple conv net for classification of chest X rays."""

    def __init__(self):
        """Initialize the model."""
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, (32, 32), stride=4)
        self.conv2 = nn.Conv2d(1, 1, (4, 4), stride=2)
        self.linear = nn.Linear(1786, 120)
        self.out = nn.Linear(120, 14)

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
    """Wrap a ConvNet() into an encrypted model."""

    def __init__(self, model=ConvNet()):
        """Initialize the ConvNetWrapper.

        Uses either the default constructor of ConvNet or an existing ConvNet.

        Arguments:
        - model: a ConvNet
        """
        self.graph = crypten.nn.from_pytorch(
            model,
            torch.rand(1, 1, 512, 512),
        ).encrypt()

    def encrypted_infer(self, x):
        """Peforms encrypted inference on an input.

        Argument:
        - x: a (1, 1, 512, 512) black and white image.
        """
        self.graph.forward(x)

    # def train(self, inputs, labels):
    #     pass
