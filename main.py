import torch
import tqdm
from torch.utils.data import DataLoader

import nescient

torch.backends.cudnn.benchmark = True

cuda = torch.device('cuda')
batch_sz = 1
net = nescient.ConvNet(batch_sz).to(cuda)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
dataset = nescient.CheXpertDataset(
    "/home/quantumish/aux/CheXpert-v1.0-small/train.csv", "/home/quantumish/aux"
)
dataloader = DataLoader(dataset, batch_size=batch_sz, num_workers=16, pin_memory=True)
total_loss = torch.zeros([1], device=cuda)
for n in range(1):
    iters = 0
    max_iters = 5000
    # losses = []
    prog_loader = tqdm.tqdm(dataloader)
    for batch in prog_loader:
        if iters > max_iters:
            break
        inputs = batch[0].cuda()
        labels = batch[1].cuda().reshape(batch_sz, 14)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        total_loss[0] += loss.item()
        # losses.append(loss.item())
        # print("Iteration {}/{} (running avg. loss {})".format(iters, max_iters, sum(losses)/(iters+1)), end='\n' if iters == max_iters else '\r')
        #print(outputs.shape, batch[1].shape)
        #print("{}\n{}\n{}\n\n".format(outputs.tolist(), batch[1].tolist(), loss.item()))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        iters += 1
    print("Epoch {}/100: avg. loss of {}".format(n, total_loss[0] / (max_iters + 1)))
