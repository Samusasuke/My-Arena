#%%
import os
import sys
import math
import numpy as np
import einops
import torch as t
from pathlib import Path

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part0_prereqs"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow, line, bar
from part0_prereqs.utils import display_array_as_img
import part0_prereqs.tests as tests

MAIN = __name__ == "__main__"

#%%

arr = np.load(section_dir / "numbers.npy")
#%%

display_array_as_img(arr[0])
# %%
arr1 = einops.rearrange(arr[:6,...], 'num chan h w -> chan h (num w)')

display_array_as_img(arr1)
# %%

arr2 = einops.repeat(arr[0], 'chan h w -> chan (2 h) w')

display_array_as_img(arr2)
# %%

arr3 = einops.repeat(arr[:2],'num chan h w-> chan (num h) (2 w)')

display_array_as_img(arr3)
# %%

arr4 = einops.repeat(arr[0], 'chan h w -> chan (h 2) w')

display_array_as_img(arr4)
# %%

arr4 = einops.rearrange(arr[0], 'channel h w -> h (channel w)')

display_array_as_img(arr4)
# %%

arr6 = einops.rearrange(arr[:6], '(first second) channel h w-> channel (first h) (second w)', first = 2)

display_array_as_img(arr6)
# %%

arr7 = einops.reduce(arr[:6], 'num chan h w -> h (num w)', 'max')

display_array_as_img(arr7)
# %%

arr8 = einops.reduce(arr[:6], 'num chan h w-> h w','min')

display_array_as_img(arr8)

# %%

arr9 = einops.rearrange(arr[1], 'c h w -> c w h')

display_array_as_img(arr9)
# %%

def first(x, axes):
    return x.flatten()[0]
arr10 = einops.reduce( arr[:6], '(f s) c (h 2) (w 2)-> c (f h) (s w)', 'max', f= 2)

display_array_as_img(arr10)
# %%
def assert_all_equal(actual: t.Tensor, expected: t.Tensor) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    assert (actual == expected).all(), f"Value mismatch, got: {actual}"
    print("Passed!")

def assert_all_close(actual: t.Tensor, expected: t.Tensor, rtol=1e-05, atol=0.0001) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    assert t.allclose(actual, expected, rtol=rtol, atol=atol)
    print("Passed!")
# %%
def rearrange_1() -> t.Tensor:
    '''Return the following tensor using only torch.arange and einops.rearrange:

    [[3, 4],
     [5, 6],
     [7, 8]]
    '''
    return einops.rearrange(t.arange(3,9), '(h w) -> h w', h=3)
    pass


expected = t.tensor([[3, 4], [5, 6], [7, 8]])
assert_all_equal(rearrange_1(), expected)
# %%

def rearrange_2() -> t.Tensor:
    '''Return the following tensor using only torch.arange and einops.rearrange:

    [[1, 2, 3],
     [4, 5, 6]]
    '''
    return einops.rearrange(t.arange(1,7), '(h w) -> h w',h = 2)
    pass


assert_all_equal(rearrange_2(), t.tensor([[1, 2, 3], [4, 5, 6]]))
# %%

def einsum_trace(mat: np.ndarray):
    '''
    Returns the same as `np.trace`.
    '''
    return einops.einsum(mat, 'i i -> ')
    pass

def einsum_mv(mat: np.ndarray, vec: np.ndarray):
    '''
    Returns the same as `np.matmul`, when `mat` is a 2D array and `vec` is 1D.
    '''
    return einops.einsum(mat, vec, 'i j, j -> i')
    pass

def einsum_mm(mat1: np.ndarray, mat2: np.ndarray):
    '''
    Returns the same as `np.matmul`, when `mat1` and `mat2` are both 2D arrays.
    '''
    return einops.einsum(mat1, mat2, 'i j, j k -> i k')
    pass

def einsum_inner(vec1: np.ndarray, vec2: np.ndarray):
    '''
    Returns the same as `np.inner`.
    '''
    return einops.einsum(vec1, vec2, 'i, i -> ')
    pass

def einsum_outer(vec1: np.ndarray, vec2: np.ndarray):
    '''
    Returns the same as `np.outer`.
    '''
    return einops.einsum(vec1, vec2, 'i , j-> i j')
    pass



tests.test_einsum_trace(einsum_trace)
tests.test_einsum_mv(einsum_mv)
tests.test_einsum_mm(einsum_mm)
tests.test_einsum_inner(einsum_inner)
tests.test_einsum_outer(einsum_outer)
# %%

