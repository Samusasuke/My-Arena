#%%
import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
import torch as t
from torch import Tensor, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from typing import Callable, Iterable, Tuple, Optional
from jaxtyping import Float
from dataclasses import dataclass, replace
from tqdm.notebook import tqdm
from pathlib import Path
import numpy as np
from IPython.display import display, HTML

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part3_optimization"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import bar, imshow, plot_train_loss_and_test_accuracy_from_trainer
from part2_cnns.solutions import  ResNet34
from part2_cnns.solutions_bonus import get_resnet_for_feature_extraction
from part3_optimization.utils import plot_fn, plot_fn_with_points
import part3_optimization.tests as tests

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"

# %%
def pathological_curve_loss(x: t.Tensor, y: t.Tensor):
    # Example of a pathological curvature. There are many more possible, feel free to experiment here!
    x_loss = t.tanh(x) ** 2 + 0.01 * t.abs(x)
    y_loss = t.sigmoid(y)
    return x_loss + y_loss

# %%

def bivariate_gaussian(x, y, x_mean=0.0, y_mean=0.0, x_sig=1.0, y_sig=1.0):
    norm = 1 / (2 * np.pi * x_sig * y_sig)
    x_exp = (-1 * (x - x_mean) ** 2) / (2 * x_sig**2)
    y_exp = (-1 * (y - y_mean) ** 2) / (2 * y_sig**2)
    return norm * t.exp(x_exp + y_exp)

def neg_trimodal_func(x, y):
    z = -bivariate_gaussian(x, y, x_mean=1.0, y_mean=-0.5, x_sig=0.2, y_sig=0.2)
    z -= bivariate_gaussian(x, y, x_mean=-1.0, y_mean=0.5, x_sig=0.2, y_sig=0.2)
    z -= bivariate_gaussian(x, y, x_mean=-0.5, y_mean=-0.8, x_sig=0.2, y_sig=0.2)
    return z


# plot_fn(pathological_curve_loss)
# %%

def opt_fn_with_sgd(fn: Callable, xy: t.Tensor, lr=0.001, momentum=0.98, n_iters: int = 100):
    '''
    Optimize the a given function starting from the specified point.

    xy: shape (2,). The (x, y) starting point.
    n_iters: number of steps.
    lr, momentum: parameters passed to the torch.optim.SGD optimizer.

    Return: (n_iters, 2). The (x,y) BEFORE each step. So out[0] is the starting point.
    '''
    optimizer = t.optim.SGD([xy],lr, momentum)
    ans = t.zeros((n_iters, 2))
    for it in range(n_iters):
        ans[it,:] = xy.data.detach()
        x, y = xy
        f = fn(x,y)
        f.backward()
        optimizer.step()
        optimizer.zero_grad()
    return ans
# %%

points = []

optimizer_list = [
    (optim.SGD, {"lr": 0.1, "momentum": 0.0}),
    (optim.SGD, {"lr": 0.02, "momentum": 0.99}),
]

for optimizer_class, params in optimizer_list:
    xy = t.tensor([2.5, 2.5], requires_grad=True)
    xys = opt_fn_with_sgd(pathological_curve_loss, xy=xy, lr=params['lr'], momentum=params['momentum'])

    points.append((xys, optimizer_class, params))

# plot_fn_with_points(pathological_curve_loss, points=points)
# %%

class SGD:
    def __init__(
        self, 
        params: Iterable[t.nn.parameter.Parameter], 
        lr: float, 
        momentum: float = 0.0, 
        weight_decay: float = 0.0
    ):
        '''Implements SGD with momentum.

        Like the PyTorch version, but assume nesterov=False, maximize=False, and dampening=0
            https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD

        '''
        params = list(params) # turn params into a list (because it might be a generator)
        self.params = params
        self.lr = lr
        self.mu = momentum
        self.lmda = weight_decay
        self.b = [t.zeros_like(param) for param in params]

    def zero_grad(self) -> None:
        '''Zeros all gradients of the parameters in `self.params`.
        '''
        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:
        '''Performs a single optimization step of the SGD algorithm.
        '''
        for par_ind, param in enumerate(self.params):
            g = param.grad
            
            g+= self.lmda*param.data.detach()

            if self.mu !=0:
                self.b[par_ind] *= self.mu
                self.b[par_ind] += g
                
                g = self.b[par_ind]

            param.data -= self.lr*g


    def __repr__(self) -> str:
        return f"SGD(lr={self.lr}, momentum={self.mu}, weight_decay={self.lmda})"


tests.test_sgd(SGD)
# %%

class RMSprop:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-08,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
    ):
        '''Implements RMSprop.

        Like the PyTorch version, but assumes centered=False
            https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html

        '''
        params = list(params) # turn params into a list (because it might be a generator)
        self.params = params
        self.lr = lr
        self.eps = eps
        self.mu = momentum
        self.lmda = weight_decay
        self.alpha = alpha
        self.v = [t.zeros_like(param) for param in params]
        self.b = [t.zeros_like(param) for param in params]


    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:
        for par_ind, param in enumerate(self.params):

            g = param.grad
            
            g+= self.lmda*param.data.detach()

            self.v[par_ind] *= self.alpha
            self.v[par_ind] +=(1-self.alpha)*(g.square())

            v = self.v[par_ind]


            self.b[par_ind] *= self.mu
            self.b[par_ind] += g/(v.sqrt() + self.eps)
            
            g = self.b[par_ind]

            param.data -= self.lr*g

    def __repr__(self) -> str:
        return f"RMSprop(lr={self.lr}, eps={self.eps}, momentum={self.mu}, weight_decay={self.lmda}, alpha={self.alpha})"


tests.test_rmsprop(RMSprop)
# %%

class Adam:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.0,
    ):
        '''Implements Adam.

        Like the PyTorch version, but assumes amsgrad=False and maximize=False
            https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
        '''
        params = list(params) # turn params into a list (because it might be a generator)
        self.params = params
        self.lr = lr
        self.eps = eps
        self.mu = betas[0]
        self.lmda = weight_decay
        self.alpha = betas[1]
        self.t = 1
        self.m = [t.zeros_like(param) for param in params]
        self.v = [t.zeros_like(param) for param in params]


    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:
        for par_ind, param in enumerate(self.params):

            g = param.grad
            
            g+= self.lmda*param

            self.m[par_ind] *= self.mu
            self.m[par_ind] += (1-self.mu)*g

            self.v[par_ind] *= self.alpha
            self.v[par_ind] +=(1-self.alpha)*(g.square())

            m_norm = self.m[par_ind]/(1-self.mu**self.t)
            v_norm = self.v[par_ind]/(1-self.alpha**self.t)


            param.data -= self.lr*m_norm/(v_norm.sqrt() + self.eps)
        self.t+=1

    def __repr__(self) -> str:
        return f"Adam(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps}, weight_decay={self.lmda})"


tests.test_adam(Adam)
# %%

class AdamW:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.0,
    ):
        '''Implements Adam.

        Like the PyTorch version, but assumes amsgrad=False and maximize=False
            https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
        '''
        params = list(params) # turn params into a list (because it might be a generator)
        self.params = params
        self.lr = lr
        self.eps = eps
        self.mu = betas[0]
        self.lmda = weight_decay
        self.alpha = betas[1]
        self.t = 1
        self.m = [t.zeros_like(param) for param in params]
        self.v = [t.zeros_like(param) for param in params]

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:
        for par_ind, param in enumerate(self.params):

            g = param.grad
            
            param-= self.lr*self.lmda*param

            self.m[par_ind] *= self.mu
            self.m[par_ind] += (1-self.mu)*g

            self.v[par_ind] *= self.alpha
            self.v[par_ind] +=(1-self.alpha)*(g.square())

            m_norm = self.m[par_ind]/(1-self.mu**self.t)
            v_norm = self.v[par_ind]/(1-self.alpha**self.t)


            param.data -= self.lr*m_norm/(v_norm.sqrt() + self.eps)
        self.t+=1

    def __repr__(self) -> str:
        return f"AdamW(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps}, weight_decay={self.lmda})"


tests.test_adamw(AdamW)
# %%

def opt_fn(fn: Callable, xy: t.Tensor, optimizer_class, optimizer_hyperparams: dict, n_iters: int = 100):
    '''Optimize the a given function starting from the specified point.

    optimizer_class: one of the optimizers you've defined, either SGD, RMSprop, or Adam
    optimzer_kwargs: keyword arguments passed to your optimiser (e.g. lr and weight_decay)
    '''
    optimizer = optimizer_class([xy],**optimizer_hyperparams)
    ans = t.zeros((n_iters, 2))
    for it in range(n_iters):
        ans[it,:] = xy.data.detach()
        x, y = xy
        f = fn(x,y)
        f.backward()
        optimizer.step()
        optimizer.zero_grad()
    return ans
# %%

points = []

optimizer_list = [
    (SGD, {"lr": 0.03, "momentum": 0.99}),
    (RMSprop, {"lr": 0.02, "alpha": 0.99, "momentum": 0.9}),
    # (Adam, {"lr": 0.2, "betas": (0.99, 0.9), "weight_decay": 0.005}),
    # (AdamW, {"lr": 0.2, "betas": (0.99, 0.9), "weight_decay": 0.005})
]
func = bivariate_gaussian

for optimizer_class, params in optimizer_list:
    xy = t.tensor([2.5, 2.5], requires_grad=True)
    xys = opt_fn(func, xy=xy, optimizer_class=optimizer_class, optimizer_hyperparams=params)
    points.append((xys, optimizer_class, params))

# plot_fn_with_points(func, points=points)


# plot_fn(neg_trimodal_func, x_range=(-2, 2), y_range=(-2, 2))
# %%

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGENET_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def get_cifar(subset: int = 1):
    cifar_trainset = datasets.CIFAR10(root ='./data', train = True, download = True, transform = IMAGENET_TRANSFORM)
    cifar_testset  = datasets.CIFAR10(root = './data', train = False, download = True, transform= IMAGENET_TRANSFORM)

    if subset>1:
        cifar_trainset = Subset(cifar_trainset, indices=range(0, len(cifar_trainset), subset))
        cifar_testset = Subset(cifar_testset, indices=range(0, len(cifar_testset), subset))

    return cifar_trainset, cifar_testset

cifar_trainset, cifar_testset = get_cifar()

imshow(
    cifar_trainset.data[:15],
    facet_col=0,
    facet_col_wrap=5,
    facet_labels=[cifar_trainset.classes[i] for i in cifar_trainset.targets[:15]],
    title="CIFAR-10 images",
    height=600
)
# %%

@dataclass
class ResNetTrainingArgs():
    batch_size: int = 64
    epochs: int = 3
    learning_rate: float = 1e-3
    n_classes: int = 10
    subset: int = 10

#%%

class ResNetTrainer:
    def __init__(self, args: ResNetTrainingArgs):
        self.args = args
        self.model = get_resnet_for_feature_extraction(args.n_classes).to(device)
        self.optimizer = t.optim.Adam(self.model.out_layers[-1].parameters(), lr=args.learning_rate)
        self.trainset, self.testset = get_cifar(subset=args.subset)
        self.logged_variables = {"loss": [], "accuracy": []}

    def to_device(self, *args):
        return [x.to(device) for x in args]

    def training_step(self, imgs: Tensor, labels: Tensor) -> t.Tensor:
        imgs, labels = self.to_device(imgs, labels)
        logits = self.model(imgs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss

    @t.inference_mode()
    def validation_step(self, imgs: Tensor, labels: Tensor) -> t.Tensor:
        imgs, labels = self.to_device(imgs, labels)
        logits = self.model(imgs)
        return (logits.argmax(dim=1) == labels).sum()

    def train(self):

        for epoch in range(self.args.epochs):

            # Load data
            train_dataloader = DataLoader(self.trainset, batch_size=self.args.batch_size, shuffle=True)
            val_dataloader = DataLoader(self.testset, batch_size=self.args.batch_size, shuffle=True)
            progress_bar = tqdm(total=len(train_dataloader))

            # Training loop (includes updating progress bar, and logging loss)
            self.model.train()
            for imgs, labels in train_dataloader:
                loss = self.training_step(imgs, labels)
                self.logged_variables["loss"].append(loss.item())
                progress_bar.update()
                progress_bar.set_description(f"Epoch {epoch+1}/{self.args.epochs}, Loss = {loss:.2f}")

            # Compute accuracy by summing n_correct over all batches, and dividing by number of items
            self.model.eval()
            accuracy = sum(self.validation_step(imgs, labels) for imgs, labels in val_dataloader) / len(self.testset)

            # Update progress bar description to include accuracy, and log accuracy
            progress_bar.set_description(f"Epoch {epoch+1}/{self.args.epochs}, Loss = {loss:.2f}, Accuracy = {accuracy:.2f}")
            self.logged_variables["accuracy"].append(accuracy.item())
# %%

args = ResNetTrainingArgs()
trainer = ResNetTrainer(args)
# trainer.train()

# plot_train_loss_and_test_accuracy_from_trainer(trainer, title="Feature extraction with ResNet34")
# %%

def test_resnet_on_random_input(model: ResNet34, n_inputs: int = 3):
    indices = np.random.choice(len(cifar_trainset), n_inputs).tolist()
    classes = [cifar_trainset.classes[cifar_trainset.targets[i]] for i in indices]
    imgs = cifar_trainset.data[indices]
    device = next(model.parameters()).device
    with t.inference_mode():
        x = t.stack(list(map(IMAGENET_TRANSFORM, imgs)))
        logits: t.Tensor = model(x.to(device))
    probs = logits.softmax(-1)
    if probs.ndim == 1: probs = probs.unsqueeze(0)
    for img, label, prob in zip(imgs, classes, probs):
        display(HTML(f"<h2>Classification probabilities (true class = {label})</h2>"))
        imshow(
            img, 
            width=200, height=200, margin=0,
            xaxis_visible=False, yaxis_visible=False
        )
        bar(
            prob,
            x=cifar_trainset.classes,
            template="ggplot2", width=600, height=400,
            labels={"x": "Classification", "y": "Probability"}, 
            text_auto='.2f', showlegend=False,
        )


# test_resnet_on_random_input(trainer.model)

import wandb

# %%

@dataclass
class ResNetTrainingArgsWandb(ResNetTrainingArgs):
    wandb_project: Optional[str] = 'day3-resnet'
    wandb_name: Optional[str] = None


# %%

class ResNetTrainerWandb(ResNetTrainer):
    def __init__(self, args: ResNetTrainingArgsWandb):
        self.args = args
        self.model = get_resnet_for_feature_extraction(args.n_classes).to(device)
        self.optimizer = t.optim.Adam(self.model.out_layers[-1].parameters(), lr=args.learning_rate)
        self.trainset, self.testset = get_cifar(subset=args.subset)
        wandb.init(project= args.wandb_project, name = args.wandb_name, config = args)
        wandb.watch(self.model.out_layers[-1], log = 'all', log_freq = 20)

    def train(self):
        step = 0
        for epoch in range(self.args.epochs):

            # Load data
            train_dataloader = DataLoader(self.trainset, batch_size=self.args.batch_size, shuffle=True)
            val_dataloader = DataLoader(self.testset, batch_size=self.args.batch_size, shuffle=True)
            progress_bar = tqdm(total=len(train_dataloader))

            # Training loop (includes updating progress bar, and logging loss)
            self.model.train()
            for imgs, labels in train_dataloader:
                loss = self.training_step(imgs, labels)
                wandb.log({"loss":loss.item()}, step = step)
                step+=1
                progress_bar.update()
                progress_bar.set_description(f"Epoch {epoch+1}/{self.args.epochs}, Loss = {loss:.2f}")

            # Compute accuracy by summing n_correct over all batches, and dividing by number of items
            self.model.eval()
            accuracy = sum(self.validation_step(imgs, labels) for imgs, labels in val_dataloader) / len(self.testset)

            # Update progress bar description to include accuracy, and log accuracy
            progress_bar.set_description(f"Epoch {epoch+1}/{self.args.epochs}, Loss = {loss:.2f}, Accuracy = {accuracy:.2f}")
            wandb.log({"accuracy":accuracy.item()}, step = step)

        wandb.finish()

args = ResNetTrainingArgsWandb()
trainer = ResNetTrainerWandb(args)
# trainer.train()
wandb.finish()
# %%

sweep_config = {
    "name": "sweepdemo",
    "method": "random",
    "metric": {"goal": "maximize", "name": "accuracy"},
    "parameters": {
        "learning_rate" : {'max' : 0.1, 'min' : 0.0001, 'distribution' : 'log_uniform_values'},
        "batch_size": {"values": [128,256,512]},
        "epochs": {"values": [4, 8, 12,16]}},
    }

# tests.test_sweep_config(sweep_config)


# %%
class ResNetTrainerWandbSweeps(ResNetTrainerWandb):
    '''
    New training class made specifically for hyperparameter sweeps, which overrides the values in
    `args` with those in `wandb.config` before defining model/optimizer/datasets.
    '''
    def __init__(self, args: ResNetTrainingArgsWandb):

        # Initialize
        wandb.init(name=args.wandb_name)

        # Update args with the values in wandb.config
        self.args = args
        self.args.batch_size = wandb.config["batch_size"]
        self.args.epochs = wandb.config["epochs"]
        self.args.learning_rate = wandb.config["learning_rate"]

        # Perform the previous steps (initialize model & other important objects)
        self.model = get_resnet_for_feature_extraction(self.args.n_classes).to(device)
        self.optimizer = t.optim.Adam(self.model.out_layers[-1].parameters(), lr=self.args.learning_rate)
        self.trainset, self.testset = get_cifar(subset=self.args.subset)
        self.step = 0

        wandb.watch(self.model.out_layers[-1], log="all", log_freq=20)
# %%

def train():
    args = ResNetTrainingArgsWandb()
    trainer = ResNetTrainerWandbSweeps(args)
    trainer.train()

sweep_id = wandb.sweep(sweep=sweep_config, project='day3-resnet-sweep')
wandb.agent(sweep_id=sweep_id, function=train, count=40)
wandb.finish()
# %%
