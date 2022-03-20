"""Library for encrypted classification of images."""
__version__ = "0.1.0"

import math
import os

from typing import Tuple
import crypten
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch import Tensor, FloatTensor
from torch.utils.data import Dataset

class CheXpertDataset(Dataset):
    """Implementation of PyTorch Dataset for the CheXpert-v1.0-small dataset."""

    def __init__(self, csv_path: str, data_folder: str):
        """Initialize dataset.

        Arguments:
        - `csv_path`: a path to the train.csv file.
        - `data_folder`: path to the overarching dataset folder.
        """
        self.data_folder = data_folder
        self.data = pd.read_csv(csv_path)

    def __len__(self) -> int:
        """Get the size of the dataset - aka the number of images."""
        return self.data.shape[0]

    def __getitem__(self, n: int) -> Tuple[Tensor, Tensor]:
        """Get nth (image, label) pair from dataset.

        Returns: a tuple of a 1x1x320x390 image scaled from 0 to 1 and a label scaled from 0 to 1.
        """        
        row = self.data.iloc[n, :]

        # get raw data of image and reformat it as a 2d tensor
        raw_image = Image.open(os.path.join(self.data_folder, row[1]))
        data = raw_image.getdata()        
        image = torch.tensor([data], dtype=torch.float32)
        image = image.reshape(1, raw_image.size[0], raw_image.size[1])

        # transpose image if x > y, and pad if dimensions are wrong
        if raw_image.size[0] > raw_image.size[1]:
            image = torch.transpose(image, 1, 2)            
        image = nn.functional.pad(image, (0, 390-image.shape[2]), mode="replicate")

        # get label, normalize it and image
        label = float(row["Lung Lesion"])
        label = 0.0 if label == -1.0 or math.isnan(label) else label
        return (image/255, torch.tensor([label]))


class ConvNet(torch.nn.Module):
    """Simple conv net for classification of chest X rays."""

    def __init__(self):
        """Initialize the model."""
        super().__init__()
        self.model = nn.Sequential(
            # nn.Dropout(0.1),            
            nn.Conv2d(1, 3, (16, 16)),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.Conv2d(16, 8, (6, 6), stride=2),
            # nn.ReLU(),            
            nn.MaxPool2d(4),
            nn.Flatten(1),
            nn.Dropout(0.3),
            nn.Linear(21204, 1),
            # nn.BatchNorm1d(256),
            # nn.ReLU(),
            # nn.Linear(512, 64),
            # nn.ReLU(),
            # nn.Linear(64, 1),
            # nn.ReLU(),
            # nn.Linear(32, 16),
            # nn.ReLU(),
            # nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        

    def forward(self, x: Tensor) -> Tensor:
        """Forward propagates the model.

        Arguments:
        - x: a (1, 1, 320, 390) grayscale image.
        """
        return self.model(x)


class ConvNetWrapper:
    """Wrap a ConvNet() into an encrypted model."""

    def __init__(self, model: ConvNet):
        """Initialize the ConvNetWrapper.

        Uses either the default constructor of ConvNet or an existing ConvNet.

        Arguments:
        - model: a ConvNet

        Example:
        >>> import nescient
        >>> import crypten
        >>> crypten.init()
        >>> net = nescient.ConvNet()
        >>> # do things...
        >>> encrypted_net = nescient.ConvNetWrapper(net)
        """
        self.graph = crypten.nn.from_pytorch(
            model,
            torch.rand(1, 1, 320, 390),
        ).encrypt()

    def infer(self, x: Tensor) -> Tensor:
        """Peforms encrypted inference on an input.

        Argument:
        - x: a (1, 1, 320, 390) black and white image.
        """
        self.graph.forward(x)
