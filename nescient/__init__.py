"""Library for encrypted classification of images."""
__version__ = "0.1.0"

import crypten
import numpy
import torch
from PIL import Image
from torch import nn
from torchvision.transforms import ToTensor


class DataIterator:
    """Iterator over the CheXpert dataset."""

    def __init__(self, csv_path: str, data_folder: str):
        """Initialize iterator.

        Arguments:
        - `csv_path`: a path to the train.csv file.
        - `data_folder`: path to the overarching dataset folder.
        """
        self.data_folder = data_folder
        self.file = open(csv_path, "r")
        next(self.file)

    def __iter__(self):
        """Get an iterator from self.

        Mostly for maintaining compatibility with Python interfaces.
        """
        return self

    def __next__(self):
        """Get next (image, label) pair from dataset.

        Processes a line of the train.csv, reads the associated image,
        and returns it as well as its labels.

        Returns: a tuple of a 1x1x320x390 PyTorch tensor and a 1x14
        PyTorch tensor.
        """
        line = next(self.file).split(",")
        raw_image = Image.open(self.data_folder + "/" + line[0])
        if raw_image.size != (320, 390):            
            return None
        data = raw_image.getdata()
        image = torch.FloatTensor(data)
        image = image.reshape(1, 1, 320, 390)
        
        def sketchy_float(i):
            try:
                return float(i)
            except:
                return 0.0

        label = torch.tensor([[0.0 if i == '' else sketchy_float(i) for i in line[5:]]])
        return (image, label)


class ConvNet(torch.nn.Module):
    """Simple conv net for classification of chest X rays."""

    def __init__(self):
        """Initialize the model."""
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 5, (64, 64)),
            nn.Conv2d(5, 5, (32, 32)),
            nn.Conv2d(5, 5, (16, 16), stride=4),
            nn.Conv2d(5, 5, (4, 4), stride=2),
            nn.Flatten(),
            nn.Linear(850*5, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Sigmoid(),
            nn.Linear(64, 64),
            nn.Sigmoid(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 14),
        )

    def forward(self, x):
        """Forward propagates the model.

        Arguments:
        - x: a (1, 1, 320, 390) grayscale image.
        """
        return self.model(x)


class ConvNetWrapper:
    """Wrap a ConvNet() into an encrypted model."""

    def __init__(self, model=ConvNet()):
        """Initialize the ConvNetWrapper.

        Uses either the default constructor of ConvNet or an existing ConvNet.

        Arguments:
        - model: a ConvNet

        Example:
        >>> import nescient
        >>> net = nescient.ConvNet()
        >>> # do things...
        >>> encrypted_net = nescient.ConvNetWrapper(net)
        """
        self.graph = crypten.nn.from_pytorch(
            model,
            torch.rand(1, 1, 320, 390),
        ).encrypt()

    def encrypted_infer(self, x):
        """Peforms encrypted inference on an input.

        Argument:
        - x: a (1, 1, 320, 390) black and white image.
        """
        self.graph.forward(x)
