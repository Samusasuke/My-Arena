#%%
import torch as t
from torchvision import transforms,datasets
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import DataLoader, Subset
from tqdm.notebook import tqdm
import wandb
import time


device = t.device("cuda" if t.cuda.is_available() else "cpu")
print(f'using device = {device}')
#%%
@dataclass
class CnnArgs():
    in_size : t.Size = t.Size((28,28))
    out_size : int = 10
    width : int = 10
    device: str = device

@dataclass
class TrainArgs():
    lr: float = 0.001
    epochs : int = 1
    batch_size : int = 256
    denominator : int = 1
    shift: int = 0

class SmallCnn(t.nn.Module):
    def __init__(self, args: CnnArgs = CnnArgs()):
        super().__init__()
        self.in_size = args.in_size
        self.width = args.width
        self.out_size = args.out_size 
        self.conv1 = t.nn.Conv2d(in_channels = 1, out_channels = self.width, kernel_size= 5, padding = 'same')
        self.pool1 = t.nn.MaxPool2d(2, 2)
        self.conv2 = t.nn.Conv2d(in_channels = self.width, out_channels = 2*self.width ,kernel_size = 3, padding = 'same')
        self.avg =   t.nn.AvgPool2d(3)
        self.out = t.nn.Linear(in_features = 2*self.width*16, out_features = args.out_size)


    def forward(self, x: t.Tensor):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.avg(F.relu(self.conv2(x)))
        return self.out(x.flatten(start_dim = 1))
# %%
mnist_transform = transforms.ToTensor()

# %%

def get_dataloaders (batch_size, denominator, shift):
    train_set = datasets.MNIST(root = './data', train = True,download = True, transform = mnist_transform)
    test_set =  datasets.MNIST(root = './data', train = False,download = True, transform = mnist_transform)

    if denominator>1.0:
        train_set = Subset(train_set, range(shift,len(train_set), denominator))
    return DataLoader(train_set, batch_size, True ), DataLoader(test_set, batch_size, False)
# %%

def train_loop(cnn_args:CnnArgs = CnnArgs(), train_args : TrainArgs = TrainArgs()):


    @t.inference_mode
    def test_accuracy():
        correct = 0
        tested = 0
        for x, y in test_loader:
            x , y  = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim = -1)
            correct += (preds == y).sum().item()
            tested += y.shape[0]
        return correct/tested, loss(logits, y)
    
    def training_step():
        for x, y in tqdm(train_loader):
            x , y  = x.to(device), y.to(device)
            logits = model(x)
            l = loss(logits, y)
            l.backward()
            opt.step()
            opt.zero_grad()
            
    epochs = train_args.epochs
    batch_size = train_args.batch_size
    lr = train_args.lr


    train_loader, test_loader = get_dataloaders(batch_size, denominator = train_args.denominator, shift = train_args.shift)
    model = SmallCnn(cnn_args).to(device)
    opt = t.optim.AdamW(model.parameters(), lr)
    loss = F.cross_entropy
    

    for epoch in range(epochs):
        t_start = time.time()
        training_step()
        compute = time.time() - t_start
        acc, loss = test_accuracy()
    return acc, loss, compute
# %%

# %%



results = []
for width in (3,5,7,10):
    for denominator in (1,2,4,8):
        for lr_mul in (0.5,1.0,2.0):
            for batch_size in (32,64,128,256):
                for shift in range(3):
                    lr = lr_mul * (28*28*width)**(-0.5)
                    cnn_args = CnnArgs(width = width)
                    train_args = TrainArgs(lr = lr, batch_size = batch_size, denominator= denominator ,shift = shift)
                    acc, loss, compute = train_loop(cnn_args, train_args)
                    log = (width, denominator, lr, batch_size, compute, acc, loss, compute)
                    print(log)
                    results.append(log)

t.save(results, './results')
