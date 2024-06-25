#%%
import os
import sys
import torch as t
from torch import nn, optim
import einops
from einops.layers.torch import Rearrange
from tqdm import tqdm
from dataclasses import dataclass, field
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from typing import Optional, Tuple, List, Literal, Union
import plotly.express as px
import torchinfo
import time
import wandb
from PIL import Image
import pandas as pd
from pathlib import Path
from datasets import load_dataset

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part5_gans_and_vaes"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

device = t.device("cuda" if t.cuda.is_available() else "cpu")
device_str = str(device)


from part2_cnns.utils import print_param_count
import part5_gans_and_vaes.tests as tests
import part5_gans_and_vaes.solutions as solutions
from plotly_utils import imshow

from part2_cnns.solutions import (
    Linear,
    ReLU,
    Sequential,
    BatchNorm2d,
)
from part2_cnns.solutions_bonus import (
    pad1d,
    pad2d,
    conv1d_minimal,
    conv2d_minimal,
    Conv2d,
    Pair,
    IntOrPair,
    force_pair,
)

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"
# %%
class ConvTranspose2d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: IntOrPair, stride: IntOrPair = 1, padding: IntOrPair = 0
    ):
        '''
        Same as torch.nn.ConvTranspose2d with bias=False.
        '''
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        kernel_size = force_pair(kernel_size)
        sf = 1 / (self.out_channels * kernel_size[0] * kernel_size[1]) ** 0.5

        self.weight = nn.Parameter(sf * (2 * t.rand(in_channels, out_channels, *kernel_size) - 1))

    def forward(self, x: t.Tensor) -> t.Tensor:
        return solutions.conv_transpose2d(x, self.weight, self.stride, self.padding)

    def extra_repr(self) -> str:
        return ", ".join([
            f"{key}={getattr(self, key)}"
            for key in ["in_channels", "out_channels", "kernel_size", "stride", "padding"]
        ])
# %%
class Tanh(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        a = x.exp()
        b = (-x).exp()
        return (a-b)/(a+b)

tests.test_Tanh(Tanh)
# %%
class LeakyReLU(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: t.Tensor) -> t.Tensor:
        return t.maximum(x, self.negative_slope*x)

    def extra_repr(self) -> str:
        return f'LeakyRelu(neg_slope = {self.neg_slope})'

tests.test_LeakyReLU(LeakyReLU)
# %%

class Sigmoid(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return 1/(1 + (-x).exp())

tests.test_Sigmoid(Sigmoid)
# %%
from collections import OrderedDict


class GeneratorBlock(nn.Module):
    def __init__(
            self,
            with_batchnorm: bool,
            in_feats: int,
            out_feats: int
    ):
        super().__init__()
        self.with_batchnorm = with_batchnorm
        self.conv = ConvTranspose2d(in_feats, out_feats, 4, 2, 1, )
        if with_batchnorm:
            self.batchnorm = nn.BatchNorm2d(out_feats)
        self.activ = nn.ReLU()

    def forward(self,x: t.Tensor):
        x = self.conv(x)
        if self.with_batchnorm:
            x = self.batchnorm(x)
        x = self.activ(x)
        return x
    

class Generator(nn.Module):

    def __init__(
        self,
        latent_dim_size: int = 100,
        img_size: int = 64,
        img_channels: int = 3,
        hidden_channels: List[int] = [128, 256, 512],
    ):
        '''
        Implements the generator architecture from the DCGAN paper (the diagram at the top
        of page 4). We assume the size of the activations doubles at each layer (so image
        size has to be divisible by 2 ** len(hidden_channels)).

        Args:
            latent_dim_size:
                the size of the latent dimension, i.e. the input to the generator
            img_size:
                the size of the image, i.e. the output of the generator
            img_channels:
                the number of channels in the image (3 for RGB, 1 for grayscale)
            hidden_channels:
                the number of channels in the hidden layers of the generator (starting from
                the smallest / closest to the generated images, and working backwards to the 
                latent vector).

        '''

        n_layers = len(hidden_channels)
        assert img_size % (2 ** n_layers) == 0, "activation size must double at each layer"
        super().__init__()

        in_channels = hidden_channels[::-1]
        out_channels = in_channels[1::] + [img_channels,]

        blocks = OrderedDict()
        for i,(in_feat,out_feats) in enumerate(zip(in_channels, out_channels)):
            if i == (n_layers-1):
                blocks[f'GenBlock{i}'] = GeneratorBlock(False, in_feat, out_feats)
            else:
                blocks[f'GenBlock{i}'] = GeneratorBlock(True, in_feat, out_feats)


        first_feature_size = int(img_size/2**n_layers)
        volume_first_block = (first_feature_size)**2*hidden_channels[-1]
        volume_first_block = int(volume_first_block)
        print(volume_first_block)

        prep = OrderedDict()
        prep['fc'] = nn.Linear(latent_dim_size,volume_first_block, bias=False)
        prep['reshape'] = Rearrange('b (c h w) -> b c h w', c = in_channels[0], h = first_feature_size, w = first_feature_size)
        prep['batchnorm'] = nn.BatchNorm2d(in_channels[0])
        prep['activ'] = nn.ReLU()

        self.prep = nn.Sequential(prep)
        self.main_path = nn.Sequential(blocks)
        

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.prep(x)
        x = self.main_path(x)
        return x

#%%
class DiscriminatorBlock(nn.Module):
    def __init__(self, with_batchnorm : bool, in_channels: int, out_channels : int):
        super().__init__()
        self.with_batchnorm = with_batchnorm
        self.conv = Conv2d(in_channels, out_channels, 4, 2, 1)
        if self.with_batchnorm:
            self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activ = LeakyReLU(0.01)

    def forward(self, x: t.Tensor):
        x = self.conv(x)
        if self.with_batchnorm:
            x = self.batchnorm(x)
        x = self.activ(x)
        return x
    
class Discriminator(nn.Module):

    def __init__(
        self,
        img_size: int = 64,
        img_channels: int = 3,
        hidden_channels: List[int] = [128, 256, 512],
    ):
        '''
        Implements the discriminator architecture from the DCGAN paper (the mirror image of
        the diagram at the top of page 4). We assume the size of the activations doubles at
        each layer (so image size has to be divisible by 2 ** len(hidden_channels)).

        Args:
            img_size:
                the size of the image, i.e. the input of the discriminator
            img_channels:
                the number of channels in the image (3 for RGB, 1 for grayscale)
            hidden_channels:
                the number of channels in the hidden layers of the discriminator (starting from
                the smallest / closest to the input image, and working forwards to the probability
                output).
        '''
        n_layers = len(hidden_channels)
        assert img_size % (2 ** n_layers) == 0, "activation size must double at each layer"
        super().__init__()

        in_channels = [img_channels,] + hidden_channels[:-1]
        out_channels = hidden_channels


        blocks = OrderedDict()
        for i, (in_chan, out_chan) in enumerate(zip(in_channels, out_channels)):
            if i==0:
                blocks[f'DisBLock{i}'] = DiscriminatorBlock(False, in_chan, out_chan)
            else:
                blocks[f'DisBLock{i}'] = DiscriminatorBlock(True, in_chan, out_chan)
    
        self.main_path = nn.Sequential(blocks)
        
        final_size = int(img_size/2**n_layers)
        final_volume = int(final_size**2*out_channels[-1])
        end = OrderedDict()
        end['flatten'] = nn.Flatten()
        end['fc'] = nn.Linear(final_volume, 1, bias = False)
        end['sigmoid'] = Sigmoid()

        self.end = nn.Sequential(end)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.main_path(x)
        x = self.end(x)
        return x
#%%

class DCGAN(nn.Module):
    netD: Discriminator
    netG: Generator

    def __init__(
        self,
        latent_dim_size: int = 100,
        img_size: int = 64,
        img_channels: int = 3,
        hidden_channels: List[int] = [128, 256, 512],
    ):
        '''
        Implements the DCGAN architecture from the DCGAN paper (i.e. a combined generator
        and discriminator).
        '''
        super().__init__()
        self.netG = Generator(latent_dim_size, img_size, img_channels, hidden_channels)
        self.netD = Discriminator(img_size, img_channels, hidden_channels)

print_param_count(Generator(), solutions.DCGAN().netG)
print_param_count(Discriminator(), solutions.DCGAN().netD)


# %%

def initialize_weights(model: nn.Module) -> None:
    '''
    Initializes weights according to the DCGAN paper, by modifying model weights in place.
    '''
    pass

tests.test_initialize_weights(initialize_weights, ConvTranspose2d, Conv2d, Linear, BatchNorm2d)