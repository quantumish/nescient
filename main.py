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
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=3e-3)
dataset = nescient.CheXpertDataset(
   "./p-shuf.csv", "/home/quantumish/aux"
)
dataloader = DataLoader(dataset, batch_size=batch_sz, num_workers=16, pin_memory=True)

epochs = 100
for n in range(epochs):
    total_loss = torch.zeros([1], device=cuda)
    total_right = torch.zeros([1], device=cuda)
    # losses = []
    prog_loader = tqdm.tqdm(dataloader)
    for batch in prog_loader:
        inputs = batch[0].cuda()
        labels = batch[1].cuda()
        outputs = net(inputs)
        # print(outputs, labels)
        loss = criterion(outputs, labels)
        total_loss[0] += loss.item()
        if outputs[0][0] < 0.5 and labels == -1.0:
            total_right += 1
        elif outputs[0][0] > 0.5 and labels == 1.0:
            total_right += 1
        # losses.append(loss.item())
        # print("Iteration {}/{} (running avg. loss {})".format(iters, max_iters, sum(losses)/(iters+1)), end='\n' if iters == max_iters else '\r')
        #print(outputs.shape, batch[1].shape)
        # print("{}\n{}\n{}\n\n".format(outputs.tolist(), batch[1].tolist(), loss.item()))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    print("Epoch {}/{}: avg. loss of {}, accuracy of {}%".format(
        n,
        epochs,
        total_loss[0] / (len(dataset) + 1),
        (total_right.item()/1000)*100
    ))
