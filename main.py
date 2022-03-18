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
optimizer = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
dataset = nescient.CheXpertDataset(
   "./small-p.csv", "/home/quantumish/aux"
)
train, test = torch.utils.data.random_split(dataset, [int(0.9 * len(dataset)), len(dataset)-int(0.9 * len(dataset))])
dataloader = DataLoader(train, batch_size=batch_sz, num_workers=16, pin_memory=True)
testloader = DataLoader(test, batch_size=batch_sz, num_workers=16, pin_memory=True)
print(len(dataset), len(test), len(train))

epochs = 300
for n in range(epochs):
    total_loss = torch.zeros([1], device=cuda)
    total_right = torch.zeros([1], device=cuda)
    prog_loader = tqdm.tqdm(dataloader)
    prog_loader.set_description("Training")
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
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    total_val_loss = torch.zeros([1], device=cuda)
    total_val_right = torch.zeros([1], device=cuda)
    prog_tloader = tqdm.tqdm(testloader)
    prog_tloader.set_description("Validating")
    for batch in prog_tloader:
        inputs = batch[0].cuda()
        labels = batch[1].cuda()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        total_val_loss[0] += loss.item()        
        if outputs < 0.5 and labels == 0.0:
            total_val_right += 1
        elif outputs > 0.5 and labels == 1.0:
            total_val_right += 1            
        
        
    
    print("Epoch {}/{}: train loss {}, train acc {}%, val loss {}, val acc {}".format(
        n,
        epochs,
        round(total_loss.item()/(len(train)), 3),
        round((total_right.item()/len(train))*100, 3),
        round(total_val_loss.item()/(len(test)), 3),
        round((total_val_right.item()/len(test))*100, 3)
    ))

torch.save(net.state_dict(), "./checkpoint.weights")
