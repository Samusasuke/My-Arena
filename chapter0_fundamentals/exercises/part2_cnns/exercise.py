#%%
import os
import sys
import numpy as np
import einops
from typing import Union, Optional, Tuple, List, Dict
import torch as t
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
import functools
from pathlib import Path
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from tqdm.notebook import tqdm
from dataclasses import dataclass
from PIL import Image
import json

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part2_cnns"
print(exercises_dir)
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow, line, bar
import part2_cnns.tests as tests
from part2_cnns.utils import print_param_count

MAIN = __name__ == "__main__"

device = t.device("cuda" if t.cuda.is_available() else "cpu")
print(device)
# %%

class ReLU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return t.maximum(x,t.zeros_like(x))


tests.test_relu(ReLU)
# %%

#HELPERS
def initialize(shape, lim):
    return nn.Parameter(t.rand(shape)*(2*lim)-lim)

def extra_repr(module, atts):
    return ', '.join([f'{att} = {getattr(module, att)}' for att in atts])
# %%

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        '''
        A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lim = (in_features**(-0.5))
        self.weight = initialize((out_features, in_features), self.lim)
        if bias:
            self.bias = initialize((out_features,), self.lim)
        else:
            self.bias = None

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (*, in_features)
        Return: shape (*, out_features)
        '''
        y = x@(self.weight.T)
        if not (self.bias is None):
            y += self.bias
        return y

    def extra_repr(self) -> str:
        return extra_repr(self, ['in_features', 'out_features', ])


tests.test_linear_forward(Linear)
tests.test_linear_parameters(Linear)
tests.test_linear_no_bias(Linear)
# %%

class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: t.Tensor) -> t.Tensor:
        '''
        Flatten out dimensions from start_dim to end_dim, inclusive of both.
        '''
        return t.flatten(input, start_dim=self.start_dim, end_dim=self.end_dim)

    def extra_repr(self) -> str:
        return extra_repr(self, ['start_dim', 'end_dim'])


tests.test_flatten(Flatten)
# %%

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.F = Flatten(1,-1)
        self.L1 = Linear(28**2, 100, True)
        self.R1 = ReLU()
        self.L2 = Linear(100,10,True)
        self.layers = [getattr(self, att) for att in ['F', 'L1', 'R1', 'L2']]
    def forward(self, x: t.Tensor) -> t.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


tests.test_mlp(SimpleMLP)
# %%

MNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def get_mnist(subset: int = 1):
    '''Returns MNIST training data, sampled by the frequency given in `subset`.'''
    mnist_trainset = datasets.MNIST(root="./data", train=True, download=True, transform=MNIST_TRANSFORM)
    mnist_testset = datasets.MNIST(root="./data", train=False, download=True, transform=MNIST_TRANSFORM)

    if subset > 1:
        mnist_trainset = Subset(mnist_trainset, indices=range(0, len(mnist_trainset), subset))
        mnist_testset = Subset(mnist_testset, indices=range(0, len(mnist_testset), subset))

    return mnist_trainset, mnist_testset


mnist_trainset, mnist_testset = get_mnist()
mnist_trainloader = DataLoader(mnist_trainset, batch_size=64, shuffle=True)
mnist_testloader = DataLoader(mnist_testset, batch_size=64, shuffle=False)
# %%

@dataclass
class SimpleMLPTrainingArgs():
    '''
    Defining this class implicitly creates an __init__ method, which sets arguments as 
    given below, e.g. self.batch_size = 64. Any of these arguments can also be overridden
    when you create an instance, e.g. args = SimpleMLPTrainingArgs(batch_size=128).
    '''
    batch_size: int = 64
    epochs: int = 3
    learning_rate: float = 1e-3
    subset: int = 10


def train(args: SimpleMLPTrainingArgs):
    '''
    Trains the model, using training parameters from the `args` object.
    '''
    model = SimpleMLP().to(device)

    mnist_trainset, mnist_testset = get_mnist(subset=args.subset)
    mnist_trainloader = DataLoader(mnist_trainset, batch_size=args.batch_size, shuffle=True)
    mnist_testloader = DataLoader(mnist_testset, batch_size=args.batch_size, shuffle=False)

    optimizer = t.optim.Adam(model.parameters(), lr=args.learning_rate)

    loss_list = []
    accuracy_list = []

    for epoch in tqdm(range(args.epochs)):
        # Validation loop
        num_correct_classifications = 0
        for imgs, labels in mnist_testloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            with t.inference_mode():
                logits = model(imgs)
            predictions = t.argmax(logits, dim=1)
            num_correct_classifications += (predictions == labels).sum().item()
        accuracy = num_correct_classifications / len(mnist_testset)
        accuracy_list.append(accuracy)   

        # Training loop
        for imgs, labels in mnist_trainloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_list.append(loss.item())   

        

    line(
        loss_list, 
        yaxis_range=[0, max(loss_list) + 0.1],
        labels={"x": "Num batches seen", "y": "Cross entropy loss"}, 
        title="SimpleMLP training on MNIST",
        width=700,
    )
    line(
        accuracy_list, 
        yaxis_range=[0, 1],
        labels={"x": "Num epochs", "y": "Accuracy"}, 
        title="SimpleMLP test accuracy on MNIST",
        width=700,
    )


args = SimpleMLPTrainingArgs()
train(args)
# %%

class Conv2d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0
    ):
        '''
        Same as torch.nn.Conv2d with bias=False.

        Name your weight field `self.weight` for compatibility with the PyTorch version.
        '''
        super().__init__()
        k = (in_channels * kernel_size**2 )**(-0.5)
        self.kernel_shape = t.Size((kernel_size, kernel_size))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.weight = initialize((out_channels, in_channels, kernel_size, kernel_size),k)


    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Apply the functional conv2d, which you can import.'''
        return F.conv2d(x, self.weight, None, self.stride, self.padding)

    def extra_repr(self) -> str:
        return extra_repr(self, ['in_channels', 'out_channels', 'kernel_shape', 'stride', 'padding'])


tests.test_conv2d_module(Conv2d)
m = Conv2d(in_channels=24, out_channels=12, kernel_size=3, stride=2, padding=1)
print(f"Manually verify that this is an informative repr: {m}")
# %%

class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: int, stride: Optional[int] = None, padding: int = 1):
        super().__init__()
        self.kernel_shape = t.Size((kernel_size, kernel_size))
        self.stride = stride
        self.padding = padding


    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Call the functional version of max_pool2d.'''
        return F.max_pool2d(x, self.kernel_shape, self.stride, self.padding)

    def extra_repr(self) -> str:
        '''Add additional information to the string representation of this class.'''
        return extra_repr(self, ['kernel_shape', 'stride', 'padding'])



tests.test_maxpool2d_module(MaxPool2d)
m = MaxPool2d(kernel_size=3, stride=2, padding=1)
print(f"Manually verify that this is an informative repr: {m}")
# %%
class Sequential(nn.Module):
    _modules: Dict[str, nn.Module]

    def __init__(self, *modules: nn.Module):
        super().__init__()
        if len(modules) == 1 and isinstance(modules[0], OrderedDict):
            d = modules[0]
            for name, module in d.items():
                self._modules[name] = module
        else:
            for index, mod in enumerate(modules):
                self._modules[str(index)] = mod

    def __getitem__(self, index: int) -> nn.Module:
        index %= len(self._modules) # deal with negative indices
        return self._modules[str(index)]

    def __setitem__(self, index: int, module: nn.Module) -> None:
        index %= len(self._modules) # deal with negative indices
        self._modules[str(index)] = module

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Chain each module together, with the output from one feeding into the next one.'''
        for mod in self._modules.values():
            x = mod(x)
        return x
    
#%%
class BatchNorm2d(nn.Module):
    # The type hints below aren't functional, they're just for documentation
    running_mean: Float[Tensor, "num_features"]
    running_var: Float[Tensor, "num_features"]
    num_batches_tracked: Int[Tensor, ""] # This is how we denote a scalar tensor

    def __init__(self, num_features: int, eps=1e-05, momentum=0.1):
        '''
        Like nn.BatchNorm2d with track_running_stats=True and affine=True.

        Name the learnable affine parameters `weight` and `bias` in that order.
        '''
        super().__init__()
        self.eps = eps
        self.num_features = num_features
        self.register_buffer('running_mean', t.zeros((num_features,)))
        self.register_buffer('running_var', t.ones((num_features,)))
        self.register_buffer('num_batches_tracked', t.tensor(0))
        self.weight = nn.Parameter(t.ones((num_features,)))
        self.bias = nn.Parameter(t.zeros((num_features,)))
        self.momentum = momentum
        self.updater = lambda x, old : old*(1.0-momentum) + x*momentum

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        Normalize each channel.

        Compute the variance using `torch.var(x, unbiased=False)`
        Hint: you may also find it helpful to use the argument `keepdim`.

        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        '''
        if self.training:
            self.num_batches_tracked+=1
            assert x.shape[1] == self.num_features
            mean = x.mean(dim = (0,2,3))
            var = x.var(dim = (0,2,3), unbiased=False)
            self.running_mean = self.updater(mean, self.running_mean)
            self.running_var = self.updater(var, self.running_var)
        else:
            mean = self.running_mean
            var = self.running_var
            
        z_score = (x - mean[:,None,None])/(t.sqrt(var)[:,None,None] + self.eps)
        return z_score * self.weight[:,None,None] + self.bias[:,None,None] 

    def extra_repr(self) -> str:
        return extra_repr(self, ['num_features','momentum' ])



tests.test_batchnorm2d_module(BatchNorm2d)
tests.test_batchnorm2d_forward(BatchNorm2d)
tests.test_batchnorm2d_running_mean(BatchNorm2d)
# %%

class AveragePool(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, channels, height, width)
        Return: shape (batch, channels)
        '''
        return x.mean(dim = (2,3))
# %%
from collections import OrderedDict
class ResidualBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, first_stride=1):
        '''
        A single residual block with optional downsampling.

        For compatibility with the pretrained model, declare the left side branch first using a `Sequential`.

        If first_stride is > 1, this means the optional (conv + bn) should be present on the right branch. Declare it second using another `Sequential`.
        '''
        super().__init__()
        left = OrderedDict()
        left['conv1'] = Conv2d(in_feats, out_feats, 3, first_stride, 1)
        left['bn1'] = BatchNorm2d(out_feats)
        left['ReLU'] = ReLU()
        left['conv2'] = Conv2d(out_feats, out_feats, 3, 1, 1)
        left['bn2'] =BatchNorm2d(out_feats)
        # print(f'left_blocks = {left_blocks}')
        self.left = Sequential(left)
                                 
        if first_stride>1:
            self.right = Sequential(Conv2d(in_feats, out_feats, 1, first_stride), BatchNorm2d(out_feats))
        else:
            self.right = nn.Identity()
        self.relu = ReLU()
    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / stride, width / stride)

        If no downsampling block is present, the addition should just add the left branch's output to the input.
        '''
        return self.relu(self.left(x)+self.right(x))
# %%

class BlockGroup(nn.Module):
    def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride=1):
        super().__init__()
        
        '''An n_blocks-long sequence of ResidualBlock where only the first block uses the provided stride.'''
        blocks = OrderedDict()
        blocks['SBlock']= ResidualBlock(in_feats, out_feats, first_stride)
        for i in range(n_blocks - 1):
            blocks[f'Block_{i}'] = ResidualBlock(out_feats, out_feats, 1)
        self.blocks = Sequential(blocks)
    def forward(self, x: t.Tensor):
        return self.blocks(x)
# %%

class ResNet34(nn.Module):
    def __init__(
        self,
        n_blocks_per_group=[3, 4, 6, 3],
        out_features_per_group=[64, 128, 256, 512],
        first_strides_per_group=[1, 2, 2, 2],
        n_classes=1000,
    ):
        # SOLUTION
        super().__init__()
        in_feats0 = 64
        self.n_blocks_per_group = n_blocks_per_group
        self.out_features_per_group = out_features_per_group
        self.first_strides_per_group = first_strides_per_group
        self.n_classes = n_classes

        self.in_layers = Sequential(
            Conv2d(3, in_feats0, kernel_size=7, stride=2, padding=3),
            BatchNorm2d(in_feats0),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        all_in_feats = [in_feats0] + out_features_per_group[:-1]
        self.residual_layers = Sequential(
            *(
                BlockGroup(*args)
                for args in zip(
                    n_blocks_per_group,
                    all_in_feats,
                    out_features_per_group,
                    first_strides_per_group,
                )
            )
        )

        self.out_layers = Sequential(
            AveragePool(),
            Linear(out_features_per_group[-1], n_classes),
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

my_resnet = ResNet34()
# %%

def copy_weights(my_resnet: ResNet34, pretrained_resnet: models.resnet.ResNet) -> ResNet34:
    '''Copy over the weights of `pretrained_resnet` to your resnet.'''

    # Get the state dictionaries for each model, check they have the same number of parameters & buffers
    mydict = my_resnet.state_dict()
    pretraineddict = pretrained_resnet.state_dict()
    print
    assert len(mydict) == len(pretraineddict), "Mismatching state dictionaries."

    # Define a dictionary mapping the names of your parameters / buffers to their values in the pretrained model
    state_dict_to_load = {
        mykey: pretrainedvalue
        for (mykey, myvalue), (pretrainedkey, pretrainedvalue) in zip(mydict.items(), pretraineddict.items())
    }

    # Load in this dictionary to your model
    my_resnet.load_state_dict(state_dict_to_load)

    return my_resnet


pretrained_resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
my_resnet = copy_weights(my_resnet, pretrained_resnet)

# %%
import utils
utils.print_param_count(my_resnet, pretrained_resnet)
# %%

IMAGE_FILENAMES = [
    "chimpanzee.jpg",
    "golden_retriever.jpg",
    "platypus.jpg",
    "frogs.jpg",
    "fireworks.jpg",
    "astronaut.jpg",
    "iguana.jpg",
    "volcano.jpg",
    "goofy.jpg",
    "dragonfly.jpg",
]

IMAGE_FOLDER = section_dir / "resnet_inputs"

images = [Image.open(IMAGE_FOLDER / filename) for filename in IMAGE_FILENAMES]
# %%

images[0]
# %%
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGENET_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

prepared_images = t.stack([IMAGENET_TRANSFORM(img) for img in images], dim=0)

assert prepared_images.shape == (len(images), 3, IMAGE_SIZE, IMAGE_SIZE)
# %%

def predict(model, images: t.Tensor) -> t.Tensor:
    '''
    Returns the predicted class for each image (as a 1D array of ints).
    '''
    model.eval()
    y : t.Tensor = model(images)
    return y.argmax(dim = 1)


with open(section_dir / "imagenet_labels.json") as f:
    imagenet_labels = list(json.load(f).values())

# Check your predictions match those of the pretrained model
my_predictions = predict(my_resnet, prepared_images)
pretrained_predictions = predict(pretrained_resnet, prepared_images)
assert all(my_predictions == pretrained_predictions)
print("All predictions match!")

# Print out your predictions, next to the corresponding images
for img, label in zip(images, my_predictions):
    print(f"Class {label}: {imagenet_labels[label]}")
    display(img)
    print()
# %%

class NanModule(nn.Module):
    '''
    Define a module that always returns NaNs (we will use hooks to identify this error).
    '''
    def forward(self, x):
        return t.full_like(x, float('nan'))


model = nn.Sequential(
    nn.Identity(),
    NanModule(),
    nn.Identity()
)


def hook_check_for_nan_output(module: nn.Module, input: Tuple[t.Tensor], output: t.Tensor) -> None:
    '''
    Hook function which detects when the output of a layer is NaN.
    '''
    if t.isnan(output).any():
        raise ValueError(f"NaN output from {module}")


def add_hook(module: nn.Module) -> None:
    '''
    Register our hook function in a module.

    Use model.apply(add_hook) to recursively apply the hook to model and all submodules.
    '''
    module.register_forward_hook(hook_check_for_nan_output)


def remove_hooks(module: nn.Module) -> None:
    '''
    Remove all hooks from module.

    Use module.apply(remove_hooks) to do this recursively.
    '''
    module._backward_hooks.clear()
    module._forward_hooks.clear()
    module._forward_pre_hooks.clear()


model = model.apply(add_hook)
input = t.randn(3)

try:
    output = model(input)
except ValueError as e:
    print(e)

model = model.apply(remove_hooks)

# %%

test_input = t.tensor(
    [[0, 1, 2, 3, 4], 
    [5, 6, 7, 8, 9], 
    [10, 11, 12, 13, 14], 
    [15, 16, 17, 18, 19]], dtype=t.float
)

import torch as t
from collections import namedtuple

TestCase = namedtuple("TestCase", ["output", "size", "stride"])

test_cases = [
    TestCase(
        output=t.tensor([0, 1, 2, 3]), 
        size=(4,),
        stride=(1,),
    ),
    TestCase(
        output=t.tensor([[0, 2], [5, 7]]), 
        size=(2, 2),
        stride=(5, 2),
    ),

    TestCase(
        output=t.tensor([0, 1, 2, 3, 4]),
        size=(5,),
        stride=(1,),
    ),

    TestCase(
        output=t.tensor([0, 5, 10, 15]),
        size=(4,),
        stride=(5,),
    ),

    TestCase(
        output=t.tensor([
            [0, 1, 2], 
            [5, 6, 7]
        ]), 
        size=(2,3),
        stride=(5,1),
    ),

    TestCase(
        output=t.tensor([
            [0, 1, 2], 
            [10, 11, 12]
        ]), 
        size=(2,3),
        stride=(10,1)
    ),

    TestCase(
        output=t.tensor([
            [0, 0, 0], 
            [11, 11, 11]
        ]), 
        size=(2,3),
        stride=(11,0),
    ),
    TestCase(
        output=t.tensor([0, 6, 12, 18]), 
        size=(4,),
        stride=(6,),
    ),
]

for (i, test_case) in enumerate(test_cases):
    if (test_case.size is None) or (test_case.stride is None):
        print(f"Test {i} failed: attempt missing.")
    else:
        actual = test_input.as_strided(size=test_case.size, stride=test_case.stride)
        if (test_case.output != actual).any():
            print(f"Test {i} failed:")
            print(f"Expected: {test_case.output}")
            print(f"Actual: {actual}\n")
        else:
            print(f"Test {i} passed!\n")
# %%

def as_strided_trace(mat: Float[Tensor, "i j"]) -> Float[Tensor, ""]:
    '''
    Returns the same as `torch.trace`, using only `as_strided` and `sum` methods.
    '''
    shape = mat.size()
    stride = mat.stride()
    return mat.as_strided((shape[0],), (stride[0]+stride[1],)).sum()

tests.test_trace(as_strided_trace)
# %%

def as_strided_mv(mat: Float[Tensor, "i j"], vec: Float[Tensor, "j"]) -> Float[Tensor, "i"]:
    '''
    Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
    '''
    l = len(vec)
    s = vec.as_strided(mat.shape, (0,vec.stride(0)))
    # print(mat, s)
    return (mat * s).sum(dim = 1)


tests.test_mv(as_strided_mv)
tests.test_mv2(as_strided_mv)
# %%

def as_strided_mm(matA: Float[Tensor, "i j"], matB: Float[Tensor, "j k"]) -> Float[Tensor, "i k"]:
    '''
    Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
    '''
    assert matA.shape[1] == matB.shape[0]
    m, n = matA.shape
    n, p = matB.shape
    a = matA.as_strided((m,n,p), (matA.stride(0), matA.stride(1), 0))
    b = matB.as_strided((m,n,p), (0, matB.stride(0), matB.stride(1)))
    return (a*b).sum(dim = 1)

tests.test_mm(as_strided_mm)
tests.test_mm2(as_strided_mm)
# %%

def conv1d_minimal_simple(x: Float[Tensor, "w"], weights: Float[Tensor, "kw"]) -> Float[Tensor, "ow"]:
    '''
    Like torch's conv1d using bias=False and all other keyword arguments left at their default values.

    Simplifications: batch = input channels = output channels = 1.

    x: shape (width,)
    weights: shape (kernel_width,)

    Returns: shape (output_width,)
    '''
    # print(x,weights)
    w = x.shape[0]
    kw = weights.shape[0]
    ow = w - kw + 1
    a = x.as_strided((ow,kw), (x.stride(0),x.stride(0)))
    b = weights.as_strided((ow,kw), (0, weights.stride(0)))
    return (a*b).sum(dim = 1)

tests.test_conv1d_minimal_simple(conv1d_minimal_simple)
# %%

def conv2d_minimal(x: Float[Tensor, "b ic h w"], weights: Float[Tensor, "oc ic kh kw"]) -> Float[Tensor, "b oc oh ow"]:
    '''
    Like torch's conv2d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    '''
    b,ic, h, w = x.shape
    oc, _, kh, kw = weights.shape
    oh = h - kh + 1
    ow = w - kw + 1
    m_shape =  (b          , ic               , oc               , oh         , kh               , ow         , kw               )
    a_stride = (x.stride(0), x.stride(1)      , 0                , x.stride(2), x.stride(2)      , x.stride(3), x.stride(3)      )
    b_stride = (0          , weights.stride(1), weights.stride(0), 0          , weights.stride(2), 0          , weights.stride(3))
    a = x.as_strided(m_shape, a_stride)
    b = weights.as_strided(m_shape, b_stride)
    return (a*b).sum(dim = (1,4,6))


tests.test_conv2d_minimal(conv2d_minimal)
# %%

def pad1d(x: t.Tensor, left: int, right: int, pad_value: float) -> t.Tensor:
    '''Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, width), dtype float32

    Return: shape (batch, in_channels, left + right + width)
    '''
    b, i, w = x.shape
    out = t.full((b, i, w+left+right), pad_value).to(t.float)
    out[...,left:left+w] = x
    return out


tests.test_pad1d(pad1d)
tests.test_pad1d_multi_channel(pad1d)
# %%

def pad2d(x: t.Tensor, left: int, right: int, top: int, bottom: int, pad_value: float) -> t.Tensor:
    '''Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, height, width), dtype float32

    Return: shape (batch, in_channels, top + height + bottom, left + width + right)
    '''
    b, i, h, w = x.shape
    out = t.full((b, i, h+top+bottom, w+left+right), pad_value).to(t.float)
    out[...,top:top+h,left:left+w] = x
    return out


tests.test_pad2d(pad2d)
tests.test_pad2d_multi_channel(pad2d)
# %%

def conv1d(
    x: Float[Tensor, "b ic w"], 
    weights: Float[Tensor, "oc ic kw"], 
    stride: int = 1, 
    padding: int = 0
) -> Float[Tensor, "b oc ow"]:
    '''
    Like torch's conv1d using bias=False.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    '''
    x = pad1d(x, right = padding, left = padding, pad_value = 0)
    if False:
        x = x.to(t.long)
    b, ic, w = x.shape
    oc, ic, kw = weights.shape
    ow = (w-kw)//stride + 1
    m_shape = (b, oc, ow, ic, kw)

    x_s, w_s = x.stride(), weights.stride()
    a_s = (x_s[0],0,stride*x_s[2], x_s[1], x_s[2])
    b_s = (0, w_s[0], 0, w_s[1], w_s[2])
    a = x.as_strided(m_shape, a_s)
    b = weights.as_strided(m_shape, b_s)
    return (a*b).sum(dim = (-1,-2))



tests.test_conv1d(conv1d)
# %%

IntOrPair = Union[int, Tuple[int, int]]
Pair = Tuple[int, int]

def force_pair(v: IntOrPair) -> Pair:
    '''Convert v to a pair of int, if it isn't already.'''
    if isinstance(v, tuple):
        if len(v) != 2:
            raise ValueError(v)
        return (int(v[0]), int(v[1]))
    elif isinstance(v, int):
        return (v, v)
    raise ValueError(v)

# Examples of how this function can be used:

for v in [(1, 2), 2, (1, 2, 3)]:
    try:
        print(f"{v!r:9} -> {force_pair(v)!r}")
    except ValueError:
        print(f"{v!r:9} -> ValueError")


# %%

def conv2d(
    x: Float[Tensor, "b ic h w"], 
    weights: Float[Tensor, "oc ic kh kw"], 
    stride: IntOrPair = 1, 
    padding: IntOrPair = 0
) -> Float[Tensor, "b oc oh ow"]:
    '''
    Like torch's conv2d using bias=False

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    '''
    stride_h , stride_w = force_pair(stride)
    padding_h, padding_w = force_pair(padding)
    x = pad2d(x, padding_w, padding_w, padding_h, padding_h, 0.0)
    b, i_c, h, w = x.shape
    o_c, i_c, k_h, k_w = weights.shape
    o_h = (h - k_h)//stride_h + 1
    o_w = (w - k_w)//stride_w + 1
    x_s, w_s = x.stride(), weights.stride()

    m_shape = (b     , o_c   , o_h            , o_w            , i_c   , k_h   , k_w   )
    a_s =     (x_s[0], 0     , stride_h*x_s[2], stride_w*x_s[3], x_s[1], x_s[2], x_s[3])
    b_s =     (0     , w_s[0], 0              , 0              , w_s[1], w_s[2], w_s[3])
    a = x.as_strided(m_shape, a_s)
    b = weights.as_strided(m_shape, b_s)
    return (a*b).sum(dim = (-1,-2,-3))


tests.test_conv2d(conv2d)
# %%

def maxpool2d(
    x: Float[Tensor, "b ic h w"], 
    kernel_size: IntOrPair, 
    stride: Optional[IntOrPair] = None, 
    padding: IntOrPair = 0
) -> Float[Tensor, "b ic oh ow"]:
    '''
    Like PyTorch's maxpool2d.

    x: shape (batch, channels, height, width)
    stride: if None, should be equal to the kernel size

    Return: (batch, channels, output_height, output_width)
    '''
    if stride is None:
        stride = kernel_size
    stride_h , stride_w = force_pair(stride)
    padding_h, padding_w = force_pair(padding)
    x = pad2d(x, padding_w, padding_w, padding_h, padding_h, pad_value = -float('inf'))
    b, i_c, h, w = x.shape
    k_h, k_w = kernel_size
    o_h = (h - k_h)//stride_h + 1
    o_w = (w - k_w)//stride_w + 1

    x_s = x.stride()
    m_shape = (b     , i_c   , o_h            , o_w            , k_h   , k_w   )
    a_s =     (x_s[0], x_s[1], stride_h*x_s[2], stride_w*x_s[3], x_s[2], x_s[3])
    a = x.as_strided(m_shape, a_s)
    res = einops.reduce(a, 'b i_c o_h o_w k_h k_w -> b i_c o_h o_w', 'max')
    # print(res)
    return res

tests.test_maxpool2d(maxpool2d)
# %%

layer0, layer1 = nn.Linear(3, 4), nn.Linear(4, 5)

layer0.requires_grad_(False) # generic code to set `param.requires_grad = False` recursively for a module (or entire model)

x = t.randn(3)
out = layer1(layer0(x)).sum()
out.backward()

assert layer0.weight.grad is None
assert layer1.weight.grad is not None
# %%

def get_resnet_for_feature_extraction(n_classes: int) -> ResNet34:
    '''
    Creates a ResNet34 instance, replaces its final linear layer with a classifier
    for `n_classes` classes, and freezes all weights except the ones in this layer.

    Returns the ResNet model.
    '''
    my_resnet = ResNet34()
    pretrained_resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    my_resnet = copy_weights(my_resnet, pretrained_resnet)
    my_resnet.requires_grad_(False)
    og_out  = my_resnet.out_layers
    n_in = og_out._modules['1'].weight.shape[1]
    my_resnet.out_layers = Sequential(
            AveragePool(),
            Linear(n_in, n_classes),
        )
    return my_resnet

tests.test_get_resnet_for_feature_extraction(get_resnet_for_feature_extraction)
# %%

def get_cifar(subset: int):
    cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=IMAGENET_TRANSFORM)
    cifar_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=IMAGENET_TRANSFORM)

    if subset > 1:
        cifar_trainset = Subset(cifar_trainset, indices=range(0, len(cifar_trainset), subset))
        cifar_testset = Subset(cifar_testset, indices=range(0, len(cifar_testset), subset))

    return cifar_trainset, cifar_testset


@dataclass
class ResNetTrainingArgs():
    batch_size: int = 128
    epochs: int = 6
    learning_rate: float = 1e-3
    n_classes: int = 10
    subset: int = 10
# %%

# YOUR CODE HERE - write your `ResNetTrainer` class
class ResNetTrainer():
    def __init__(self, args : ResNetTrainingArgs):
        self.model = get_resnet_for_feature_extraction(args.n_classes)
        self.args = args
        self.trainset , self.testset= get_cifar(10)
        self.train_loader = DataLoader(self.trainset, args.batch_size, True )
        self.test_loader  = DataLoader(self.testset,  args.batch_size, False) 
        self.loss = F.cross_entropy
        self.optimizer = t.optim.Adam
    def train(self):
        model = self.model.to(device)
        losses = []
        accuracies = []
        optimizer = self.optimizer(model.parameters(), self.args.learning_rate)
        for epoch in tqdm(range(self.args.epochs)):

            model.eval()
            correct = 0
            for x, y in self.test_loader:
                
                with t.inference_mode():
                    logits = model(x.to(device))
                preds = logits.argmax(dim = 1)
                correct +=  (y.to(device) == preds).sum().item()
            accuracies.append(correct/len(self.testset))

            model.train()
            for x, y in tqdm(self.train_loader, desc = f'Epoch{epoch+1}'):
                logits = model(x.to(device))
                loss = self.loss(logits, y.to(device))
                loss.backward()
                optimizer.step()
                model.zero_grad()
                losses.append(loss.item())
        self.losses = losses
        self.accuracies = accuracies
        line(
            losses, 
            yaxis_range=[0, max(losses) + 0.1],
            labels={"x": "Num batches seen", "y": "Cross entropy loss"}, 
            title="SimpleMLP training on MNIST",
            width=700,
        )
        line(
            accuracies, 
            yaxis_range=[0, 1],
            labels={"x": "Num epochs", "y": "Accuracy"}, 
            title="SimpleMLP test accuracy on MNIST",
            width=700,
        )
        

args = ResNetTrainingArgs()
trainer = ResNetTrainer(args)
trainer.train()

# %%
