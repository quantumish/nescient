import math
from itertools import islice

import pandas as pd
import torch
import tqdm
from torch.utils.data import DataLoader

import nescient

# torch.backends.cudnn.benchmark = True

# cuda = torch.device('cuda')
batch_sz = 1
# net = nescient.ConvNet(batch_sz).to(cuda)
# criterion = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
#dataset = nescient.CheXpertDataset(
#    "/home/quantumish/aux/CheXpert-v1.0-small/train.csv", "/home/quantumish/aux"
# )
#dataloader = DataLoader(dataset, batch_size=batch_sz, num_workers=16, pin_memory=True)

def databench(sz):
    print("Stats for first {} datapoints:\n".format(sz))
    pos = [0] * 14
    neu = [0] * 14
    neg = [0] * 14
    data = pd.read_csv("/home/quantumish/aux/CheXpert-v1.0-small/train.csv")
    for i in range(sz):
        row = data.iloc[i, :]
        label = [0.0 if math.isnan(float(i)) else float(i) for i in row[5:]]
        # label = torch.tensor([[0.0 if math.isnan(i) else float(i) for i in label]])
        for j, x in enumerate(label):
            if x == 0.0:
                neu[j] += 1
            elif x == 1.0:
                pos[j] += 1
            elif x == -1.0:
                neg[j] += 1

    names = ["No Finding","Enlarged Cardiomediastinum","Cardiomegaly","Lung Opacity","Lung Lesion","Edema","Consolidation","Pneumonia","Atelectasis","Pneumothorax","Pleural Effusion","Pleural Other","Fracture","Support Devices"]
    
    for i in range(14):
        print("{}: {}% pos, {}% neg, {}% neu".format(
            names[i],
            round((pos[i]/sz)*100, 2),
            round((neg[i]/sz)*100, 2),
            round((neu[i]/sz)*100, 2)
        ))
    print("\n\n")
        
# databench(100)
# databench(1000)
# # databench(223414)
        
# for n in range(10):
#     iters = 0
#     total_loss = torch.zeros([1], device=cuda)
#     max_iters = 100
#     # losses = []
#     prog_loader = tqdm.tqdm(dataloader)
#     for batch in prog_loader:
#         if iters > max_iters:
#             break
#         inputs = batch[0].cuda()
#         labels = batch[1].cuda().reshape(batch_sz, 14)
#         outputs = net(inputs)
#         # print(outputs, labels)
#         loss = criterion(outputs, labels)
#         total_loss[0] += loss.item()
#         # losses.append(loss.item())
#         # print("Iteration {}/{} (running avg. loss {})".format(iters, max_iters, sum(losses)/(iters+1)), end='\n' if iters == max_iters else '\r')
#         #print(outputs.shape, batch[1].shape)
#         #print("{}\n{}\n{}\n\n".format(outputs.tolist(), batch[1].tolist(), loss.item()))
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad(set_to_none=True)
#         iters += 1
#     print("Epoch {}/100: avg. loss of {}".format(n, total_loss[0] / (max_iters + 1)))
 
