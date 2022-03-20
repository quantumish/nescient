import math
from itertools import islice

import pandas as pd
import torch
import tqdm
from torch.utils.data import DataLoader

import nescient

torch.backends.cudnn.benchmark = True

cuda = torch.device('cuda')
batch_sz = 8
net = nescient.ConvNet(batch_sz).to(cuda)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)
dataset = nescient.CheXpertDataset(
   "./all.csv", "/home/quantumish/aux"
)
train, test = torch.utils.data.random_split(dataset, [int(0.9 * len(dataset)), len(dataset)-int(0.9 * len(dataset))])
dataloader = DataLoader(train, batch_size=batch_sz, num_workers=16, pin_memory=True, shuffle=True)
testloader = DataLoader(test, batch_size=batch_sz, num_workers=16, pin_memory=True)
print(len(dataset), len(test), len(train))

epochs = 100
for n in range(epochs):
    total_loss = 0
    total_right = 0
    prog_loader = tqdm.tqdm(dataloader)
    prog_loader.set_description("Training")
    for batch in prog_loader:
        inputs = batch[0].cuda()
        labels = batch[1].cuda()
#        print(labels.item())
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        total_loss += float(loss)*batch_sz
        # prog_loader.set_description("{:05} {} {}".format(
        #     round(float(loss), 2),
        #     float(outputs),
        #     float(labels)
        # ))
        # print(inputs, outputs.item(), labels.item(), loss.item())
#        print(outputs, labels)
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
        # print(outputs, labels)
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
