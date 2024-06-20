#%%
import torch as t
import torch.nn as nn
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
from my_cnn import BlockGroup, AveragePool
#%%
@dataclass
class ResNetArgs():
    in_size : t.Size = t.Size((32,32))
    n_classes : int = 10
    n_blocks_per_group=[2 ,3, 3, 2]
    out_features_per_group=[64, 128, 256, 512]
    first_strides_per_group=[1, 2, 2, 2]

    device: str = device

@dataclass
class TrainArgs():
    learning_rate: float = 0.0003
    epochs : int = 1
    batch_size : int = 256
    denominator : int = 1
    shift: int = 0
    log: bool = False
    print: bool = False
# %%

class SmallResNet(nn.Module):
    def __init__(
        self,
        args
    ):

        super().__init__()
        self.in_feats0 = args.out_features_per_group[0]
        self.n_blocks_per_group = args.n_blocks_per_group
        self.out_features_per_group = args.out_features_per_group
        self.first_strides_per_group = args.first_strides_per_group
        self.n_classes = args.n_classes

        self.in_layers = nn.Sequential(
            nn.Conv2d(3, self.in_feats0, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_feats0),
            nn.ReLU(),
        )

        self.all_in_feats = [self.in_feats0] + self.out_features_per_group[:-1]
        self.residual_layers = nn.Sequential(
            *(
                BlockGroup(*args)
                for args in zip(
                    self.n_blocks_per_group,
                    self.all_in_feats,
                    self.out_features_per_group,
                    self.first_strides_per_group,
                )
            )
        )

        self.out_layers = nn.Sequential(
            AveragePool(),
            nn.Linear(self.out_features_per_group[-1], self.n_classes),
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, channels, height, width)
        Return: shape (batch, n_classes)
        '''
        # SOLUTION
        x = self.in_layers(x)
        x = self.residual_layers(x)
        x = self.out_layers(x)
        return x

#%%
mean = [0.5,0.5,0.5]
sd = [0.25,0.25,0.25]
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding = 4, padding_mode= 'reflect'),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean, sd)])

val_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, sd)])

# %%

def get_dataloaders (batch_size, denominator, shift):
    train_set = datasets.CIFAR10(root = './data', train = True,download = True, transform = train_transform)
    test_set =  datasets.CIFAR10(root = './data', train = False,download = True, transform = val_transform)

    if denominator>1.0:
        train_set = Subset(train_set, range(shift,len(train_set), denominator))
    return DataLoader(train_set, batch_size, True ), DataLoader(test_set, batch_size, False)
# %%

class ResNetTrainer:
    def __init__(self, model_args : ResNetArgs, train_args: TrainArgs):
        self.trainArgs = train_args
        self.model = SmallResNet(model_args).to(device)
        self.batch_size = train_args.batch_size
        self.train_loader, self.val_loader =  get_dataloaders(self.trainArgs.batch_size,self.trainArgs.denominator, self.trainArgs.shift)
        self.optimizer = t.optim.Adam(self.model.parameters(), lr=train_args.learning_rate, weight_decay = 0.001)
        self.lr_sched = t.optim.lr_scheduler.OneCycleLR(optimizer = self.optimizer, max_lr = train_args.learning_rate, epochs = train_args.epochs, steps_per_epoch = len(self.train_loader))
        if train_args.log:
            wandb.init()
        
    def to_device(self, *args):
        return [x.to(device) for x in args]

    def training_step(self, imgs: t.Tensor, labels: t.Tensor) -> t.Tensor:
        imgs, labels = self.to_device(imgs, labels)
        logits = self.model(imgs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.lr_sched.step()
        if self.trainArgs.log:
            wandb.log({'train_loss':loss.item()})
        if self.trainArgs.print:
            print(f'loss {loss.item()}')
        return loss.item()

    @t.inference_mode()
    def validation_step(self, imgs: t.Tensor, labels: t.Tensor) -> t.Tensor:
        imgs, labels = self.to_device(imgs, labels)
        logits = self.model(imgs)
        return (logits.argmax(dim=1) == labels).sum()

    def train(self):

        for epoch in range(self.trainArgs.epochs):

            # Load data
            

            # Training loop (includes updating progress bar, and logging loss)
            self.model.train()
            for imgs, labels in tqdm(self.train_loader):
                loss = self.training_step(imgs, labels)

            # Compute accuracy by summing n_correct over all batches, and dividing by number of items
            self.model.eval()
            accuracy = sum(self.validation_step(imgs, labels) for imgs, labels in self.val_loader) / 10000

            # Update progress bar description to include accuracy, and log accuracy
            if self.trainArgs.log:
                wandb.log({'test_accuracy': accuracy})
            if self.trainArgs.print:
                print(f'acc: {accuracy}')

# %%
sweep_config = {
    "name": "sweepdemo",
    "method": "random",
    "metric": {"goal": "minimize", "name": "train_loss"},
    "parameters": {
        "learning_rate" : {'max' : 0.005, 'min' : 0.0005, 'distribution' : 'log_uniform_values'},
        "batch_size": {"values": [256,512]},},
    }

def sweep_fn():
    wandb.init(project='my-resnet-sweep')
    modelArgs = ResNetArgs()
    trainArgs = TrainArgs(learning_rate = wandb.config["learning_rate"], batch_size= wandb.config['batch_size'], log = True)
    trainer = ResNetTrainer(modelArgs, trainArgs)
    trainer.train()
    

# %%

# modelArgs = ResNetArgs()
# wandb.init(project = 'big_resnet_run')
# trainArgs = TrainArgs(epochs = 10, log = True)
# lr = trainArgs.learning_rate
# trainer = ResNetTrainer(modelArgs, trainArgs)
# trainer.train()

# old_model = trainer.model
# trainArgs = TrainArgs(epochs = 10, log = True, learning_rate= lr/5)
# trainer = ResNetTrainer(modelArgs, trainArgs)
# trainer.model = old_model
# trainer.train()

# old_model = trainer.model
# trainArgs = TrainArgs(epochs = 10, log = True, learning_rate= lr/15)
# trainer = ResNetTrainer(modelArgs, trainArgs)
# trainer.model = old_model

# trainer.train()
# wandb.finish()

# t.save('./model')

#%%


modelArgs = ResNetArgs()
trainArgs = TrainArgs(epochs = 40, log = True)
trainer = ResNetTrainer(modelArgs, trainArgs)
trainer.train()
# %%
model = SmallResNet(ResNetArgs())
print(sum([p.numel() for p in model.parameters()]))
# %%
