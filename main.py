import math
from itertools import islice

import pandas as pd
import torch
import tqdm
from torch.utils.data import DataLoader

import nescient

torch.backends.cudnn.benchmark = True

cuda = torch.device('cuda')
batch_sz = 1
net = nescient.ConvNet(batch_sz).to(cuda)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)
dataset = nescient.CheXpertDataset(
   "./p-shuf.csv", "/home/quantumish/aux"
)
dataloader = DataLoader(dataset, batch_size=batch_sz, num_workers=16, pin_memory=True)

epochs = 100
for n in range(epochs):
    total_loss = torch.zeros([1], device=cuda)
    total_right = torch.zeros([1], device=cuda)
    total_out = torch.zeros([1], device=cuda)
    prog_loader = tqdm.tqdm(dataloader)
    for batch in prog_loader:
        inputs = batch[0].cuda()
        labels = batch[1].cuda()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        total_loss[0] += loss.item()
        # print(inputs, outputs.item(), labels.item(), loss.item())
        if outputs < 0.5 and labels == 0.0:
            total_right += 1
        elif outputs > 0.5 and labels == 1.0:
            total_right += 1
        total_out += outputs.item()
        loss.backward()
        prog_loader.set_description("Loss: {:05}".format(round(loss.item(), 3)))
        optimizer.step()
        optimizer.zero_grad()

    print("Epoch {}/{}: avg. loss of {}, avg. out of {}, accuracy of {}%".format(
        n,
        epochs,
        total_loss.item() / (len(dataset) + 1),
        total_out.item() / (len(dataset)),
        (total_right.item()/len(dataset))*100
    ))
