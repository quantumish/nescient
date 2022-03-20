"""Script responsible for training `nescient` on the CheXpert dataset."""

import math
from itertools import islice
import sys

import pandas as pd
import torch
import tqdm
from torch.utils.data import DataLoader
import crypten
import nescient

if len(sys.argv) < 3:
    print("usage: python main.py <path-to-train-csv> <path-to-data-folder>")
    exit(0)

torch.backends.cudnn.benchmark = True  # Performance tweak for GPUs
crypten.init()  # CrypTen requires initialization before use
cuda = torch.device('cuda')

# Hyperparameters and network setup
batch_sz = 8
epochs = 30
net = nescient.ConvNet().to(cuda)
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, weight_decay=1e-4)
criterion = torch.nn.BCELoss()

# Initialize data
dataset = nescient.CheXpertDataset(sys.argv[1], sys.argv[2])
train, test = torch.utils.data.random_split(
    dataset,
    [int(0.9 * len(dataset)), len(dataset)-int(0.9 * len(dataset))]
)
trainloader = DataLoader(train, batch_size=batch_sz, num_workers=16, pin_memory=True, shuffle=True)
testloader = DataLoader(test, batch_size=batch_sz, num_workers=16, pin_memory=True)

for n in range(epochs):
    total_loss = 0
    total_right = 0
    prog_loader = tqdm.tqdm(trainloader)
    prog_loader.set_description("Training")
    for batch in prog_loader:
        inputs = batch[0].cuda()
        labels = batch[1].cuda()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        total_loss += float(loss)*batch_sz        
        for x,i in enumerate(list(outputs)):
            if float(i) < 0.5 and float(labels[x]) == 0.0:
                total_right += 1
            elif float(i) > 0.5 and float(labels[x]) == 1.0:
                total_right += 1
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    total_val_loss = 0
    total_val_right = 0
    prog_tloader = tqdm.tqdm(testloader)
    prog_tloader.set_description("Validating")
    for batch in prog_tloader:
        inputs = batch[0].cuda()
        labels = batch[1].cuda()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        total_val_loss += float(loss)*batch_sz
        for x,i in enumerate(list(outputs)):
            if float(i) < 0.5 and float(labels[x]) == 0.0:
                total_val_right += 1
            elif float(i) > 0.5 and float(labels[x]) == 1.0:
                total_val_right += 1
            
    print("Epoch {}/{}: train loss {}, train acc {}%, val loss {}, val acc {}".format(
        n,
        epochs,
        round(total_loss/(len(train)), 3),
        round((total_right/len(train))*100, 3),
        round(total_val_loss/(len(test)), 3),
        round((total_val_right/len(test))*100, 3)
    ))

torch.save(net.state_dict(), "./checkpoint.weights")
encrypted_net = nescient.ConvNetWrapper(net.cpu())
