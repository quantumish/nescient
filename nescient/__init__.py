"""Library for encrypted classification of images."""
__version__ = "0.1.0"

import math

import crypten
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset


class CheXpertDataset(Dataset):
    """CheXpert dataset."""

    def __init__(self, csv_path: str, data_folder: str):
        """Initialize iterator.

        Arguments:
        - `csv_path`: a path to the train.csv file.
        - `data_folder`: path to the overarching dataset folder.
        """
        self.data_folder = data_folder
        self.data = pd.read_csv(csv_path)

    def __len__(self):
        """Get the size of the dataset - aka the number of images."""
        return self.data.shape[0]

    def __getitem__(self, n):
        """Get nth (image, label) pair from dataset.

        Returns: a tuple of a 1x1x320x390 PyTorch tensor and a 1x14 PyTorch tensor.
        """
        row = self.data.iloc[n, :]
        raw_image = Image.open("{}/{}".format(self.data_folder, row[1]))
        data = raw_image.getdata()
        image = torch.FloatTensor([data])
        image = image.reshape(1, raw_image.size[0], raw_image.size[1])
        # print(row)
        if raw_image.size[0] > raw_image.size[1]:
            image = torch.transpose(image, 1, 2)
        image = nn.functional.pad(image, (0, 390-image.shape[2]), mode="replicate")
        # print(row[0], row[9], row[5+7])
        label = float(row["Lung Lesion"])
        # print(float(row["Lung Lesion"]))
        label = torch.tensor([0.0 if label == -1.0 or math.isnan(label) else label])
        return (image/255, label)


class ConvNet(torch.nn.Module):
    """Simple conv net for classification of chest X rays."""

    def __init__(self, batch_size):
        """Initialize the model."""
        super().__init__()
        self.model = nn.Sequential(
        #    nn.BatchNorm2d(1),
            # nn.Dropout(0.1),
            nn.Conv2d(1, 16, (12, 12), stride=3),
            nn.ReLU(),
            nn.Conv2d(16, 8, (6, 6), stride=2),
            nn.ReLU(),            
            nn.MaxPool2d(4),
            nn.Flatten(1),
            # nn.Dropout(0.05),
            nn.Linear(1440, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
            # nn.Linear(128, 64),
            # nn.ReLU(),
            # nn.Linear(64, 32),
            # nn.GELU(),
            # nn.Linear(32, 16),
            # nn.GELU(),

    def forward(self, x):
        """Forward propagates the model.

        Arguments:
        - x: a (1, 1, 320, 390) grayscale image.
        """
        return self.model(x)


class ConvNetWrapper:
    """Wrap a ConvNet() into an encrypted model."""

    def __init__(self, model=ConvNet(5)):
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
