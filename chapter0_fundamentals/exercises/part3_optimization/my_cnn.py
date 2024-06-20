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


MAIN = __name__ == "__main__"

device = t.device("cuda" if t.cuda.is_available() else "cpu")
print(device)

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
    def __init__(self, in_feats: int, out_feats: int, first_stride: int):
        '''
        A single residual block with optional downsampling.

        For compatibility with the pretrained model, declare the left side branch first using a `Sequential`.

        If first_stride is > 1, this means the optional (conv + bn) should be present on the right branch. Declare it second using another `Sequential`.
        '''
        self.conv_size = 3
        self.first_stride = first_stride
        super().__init__()
        print('initalize residual block, ', in_feats, out_feats, first_stride)
        left = OrderedDict()
        
        left['conv1'] = nn.Conv2d(in_feats, out_feats, self.conv_size, first_stride, 1)
        left['bn1'] = nn.BatchNorm2d(out_feats)
        left['ReLU'] = nn.ReLU()
        left['conv2'] = nn.Conv2d(out_feats, out_feats, self.conv_size, 1, 1)
        left['bn2'] =nn.BatchNorm2d(out_feats)
        # print(f'left_blocks = {left_blocks}')
        self.left = nn.Sequential(left)
                                 
        if first_stride>1:
            self.right = nn.Sequential(nn.Conv2d(in_feats, out_feats, 1,first_stride, 0), nn.BatchNorm2d(out_feats))
        else:
            self.right = nn.Identity()
        self.relu = nn.ReLU()
    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / stride, width / stride)

        If no downsampling block is present, the addition should just add the left branch's output to the input.
        '''
        left = self.left(x)
        right = self.right(x)
        # print(left.shape, right.shape, self.first_stride)
        return self.relu(left+right)
# %%

class BlockGroup(nn.Module):
    def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride: int):
        super().__init__()
        
        '''An n_blocks-long sequence of ResidualBlock where only the first block uses the provided stride.'''
        blocks = OrderedDict()
        blocks['SBlock']= ResidualBlock(in_feats, out_feats, first_stride)
        for i in range(n_blocks - 1):
            blocks[f'Block_{i}'] = ResidualBlock(out_feats, out_feats, 1)
        blocks['dropout'] = nn.Dropout(0.1)
        self.blocks = nn.Sequential(blocks)
    def forward(self, x: t.Tensor):
        return self.blocks(x)
# %%

