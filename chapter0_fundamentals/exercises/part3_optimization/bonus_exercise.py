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

        training_step()

        acc, loss = test_accuracy()
    return acc, loss
# %%

wandb.finish()
# %%


sweep_hyper_config = {
    "name": "Powerlaw-Sweep",
    "method": "random",
    "metric": {"goal": "maximize", "name": "accuracy"},
    "parameters": {
        "learning_rate" : {'max' : 0.1, 'min' : 0.0001, 'distribution' : 'log_uniform_values'},
        "batch_size": {"values": [32,64,128,256,512]}},
    }


def sweep_hyper_fn():
    wandb.init(project = 'power_law_cnn')
    cnn_args = CnnArgs()
    train_args = TrainArgs(lr = wandb.config['learning_rate'], batch_size = wandb.config['batch_size'] )
    accuracy = train_loop(cnn_args, train_args)
    wandb.log({'accuracy': accuracy})
# sweep_id = wandb.sweep(sweep=sweep_hyper_config, project='power_law_cnn')
# wandb.agent(sweep_id=sweep_id, function=sweep_hyper_fn, count=40)
wandb.finish()
# %%
results = t.load('./results')
results = [t[:-1] for t in results]
print(len(results))
#%%

clean_results = []
best = {}
default = (float('inf'),)
for width, denominator, lr, batch_size, compute, accuracy, loss in results:
    weird_lr = int(lr**(-2))
    try:
        loss = loss.item()
    except AttributeError:
        pass
    key = (width, denominator, weird_lr, batch_size)
    if best.get(key, default)[0] < loss:
        continue
    if loss > 0.05:
        continue
    best[key] = (loss, accuracy, compute)

for (width, den, weird_lr, batch_size) , (loss, accuracy, compute) in best.items():
    lr = weird_lr**(-0.5)
    size = 60000/den
    clean_results.append((width, size, lr, batch_size, compute, accuracy, loss))


print(f'len (results ) = {len(results)}')
for i in range(5):
    print(results[i])

results = clean_results

#%%

import matplotlib.pyplot as plt

# Example list of training parameters
training_parameters = results

# Extracting data for plotting
widths = [params[0] for params in training_parameters]
dataset_sizes = [params[1] for params in training_parameters]
val_losses = [params[6] for params in training_parameters]


# Mapping widths to colors (adjust colormap and normalization as needed)
colors = widths  # Using width as a proxy for color
cmap = plt.get_cmap('viridis')  # Colormap for colors
normalize = plt.Normalize(min(widths), max(widths))  # Normalize widths for colormap

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(dataset_sizes, val_losses, c=colors, cmap=cmap, norm=normalize, alpha=0.7, edgecolors='k')

# Adding colorbar
cbar = plt.colorbar()
cbar.set_label('Width')

# Labeling the plot
plt.xlabel('Dataset Size')
plt.ylabel('Validation Loss')
plt.title('Scatter Plot: Dataset Size vs Validation Loss')

plt.grid(True)
plt.tight_layout()
plt.show()
