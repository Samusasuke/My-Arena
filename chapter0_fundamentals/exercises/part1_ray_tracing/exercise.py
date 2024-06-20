#%%
import os
import sys
import torch as t
from torch import Tensor
import einops
from ipywidgets import interact
import plotly.express as px
from ipywidgets import interact
from pathlib import Path
from IPython.display import display
from jaxtyping import Float, Int, Bool, Shaped, jaxtyped
import typeguard

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part1_ray_tracing"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow
from part1_ray_tracing.utils import render_lines_with_plotly, setup_widget_fig_ray, setup_widget_fig_triangle
import part1_ray_tracing.tests as tests

MAIN = __name__ == "__main__"
# %%

def make_rays_1d(num_pixels: int, y_limit: float) -> t.Tensor:
    '''
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) where the num_points dimension contains (origin, direction) and the num_dim dimension contains xyz.

    Example of make_rays_1d(9, 1.0): [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    '''
    ans = t.zeros((num_pixels, 2,3))
    ans[:,1,0] = 1
    t.linspace(-y_limit, y_limit, num_pixels,out=ans[:,1,1])
    return ans
rays1d = make_rays_1d(9, 10.0)

fig = render_lines_with_plotly(rays1d)
# %%

fig = setup_widget_fig_ray()
display(fig)

@interact
def response(seed=(0, 10, 1), v=(-2.0, 2.0, 0.01)):
    t.manual_seed(seed)
    L_1, L_2 = t.rand(2, 2)
    P = lambda v: L_1 + v * (L_2 - L_1)
    x, y = zip(P(-2), P(2))
    with fig.batch_update(): 
        fig.data[0].update({"x": x, "y": y}) 
        fig.data[1].update({"x": [L_1[0], L_2[0]], "y": [L_1[1], L_2[1]]}) 
        fig.data[2].update({"x": [P(v)[0]], "y": [P(v)[1]]})
# %%

segments = t.tensor([
    [[1.0, -12.0, 0.0], [1, -6.0, 0.0]], 
    [[0.5, 0.1, 0.0], [0.5, 1.15, 0.0]], 
    [[2, 12.0, 0.0], [2, 21.0, 0.0]]
])

render_lines_with_plotly(rays1d, segments)
# %%

def intersect_ray_1d(ray: t.Tensor, segment: t.Tensor) -> bool:
    '''
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    '''
    ray = ray[...,:2]
    segment = segment[...,:2]

    O, D = ray
    L1, L2 = segment
    matrix = t.stack([D,L1-L2],dim = 1)
    vector = L1-O
    try:
        s,u = t.linalg.solve(matrix, vector)
    except t.linalg.LinAlgError:
        return False
    return (s>=0.0) & (0.0<= u ) & (u <= 1.0)

    matrix[:,0] = D
    matrix[:,1] = L1-L2

tests.test_intersect_ray_1d(intersect_ray_1d)
tests.test_intersect_ray_1d_special_case(intersect_ray_1d)
# %%

def intersect_rays_1d(rays: Float[Tensor, "nrays 2 3"], segments: Float[Tensor, "nsegments 2 3"]) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, return True if it intersects any segment.
    '''
    n_rays = rays.shape[0]
    n_segs = segments.shape[0]
    n_dim = 2
    O, D = einops.repeat(rays[...,:n_dim], 'ray point coord -> point ray seg coord', seg = n_segs)
    L1, L2 = einops.repeat(segments[...,:n_dim], 'seg point coord -> point ray seg coord', ray = n_rays)

    matrix = t.stack([D, L1-L2], dim=-1) # shape (ray seg 2 2)
    assert matrix.shape == (n_rays, n_segs, 2,2)
    vector = L1-O
    valid = (t.linalg.det(matrix).abs())>1e-8
    # print(matrix[0,0])
    # print(valid)
    matrix[~valid] = t.eye(n_dim)

    s, u = einops.rearrange(t.linalg.solve(matrix, vector), '... coord -> coord ...')
    intersect = (s>=0.0) & (0.0<= u ) & (u <= 1.0) & (valid)
    # print(s)
    # print(u)
    return intersect.any(dim = 1)
    print(intersect)


tests.test_intersect_rays_1d(intersect_rays_1d)
tests.test_intersect_rays_1d_special_case(intersect_rays_1d)
# %%

def make_rays_2d(num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float) -> Float[t.Tensor, "nrays 2 3"]:
    '''
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    '''
    ans = t.zeros((num_pixels_y , num_pixels_z, 2, 3))
    ans[...,1,0] = 1
    ans[...,1,1] = t.linspace(-y_limit, y_limit, num_pixels_y)[:,None]
    ans[...,1,2] = t.linspace(-z_limit, z_limit, num_pixels_z)[None,:]
    return einops.rearrange(ans,'y_change z_change point coord -> (y_change z_change) point coord')

rays_2d = make_rays_2d(10, 10, 0.3, 0.3)
render_lines_with_plotly(rays_2d)
# %%

one_triangle = t.tensor([[0, 0, 0], [3, 0.5, 0], [2, 3, 0]])
A, B, C = one_triangle
x, y, z = one_triangle.T

fig = setup_widget_fig_triangle(x, y, z)

@interact(u=(-0.5, 1.5, 0.01), v=(-0.5, 1.5, 0.01))
def response(u=0.0, v=0.0):
    P = A + u * (B - A) + v * (C - A)
    fig.data[2].update({"x": [P[0]], "y": [P[1]]})

display(fig)
# %%

Point = Float[Tensor, "points=3"]

@jaxtyped
@typeguard.typechecked
def triangle_ray_intersects(A: Point, B: Point, C: Point, O: Point, D: Point) -> bool:
    '''
    A: shape (3,), one vertex of the triangle
    B: shape (3,), second vertex of the triangle
    C: shape (3,), third vertex of the triangle
    O: shape (3,), origin point
    D: shape (3,), direction point

    Return True if the ray and the triangle intersect.
    '''

    matrix = t.stack([D,A-B, A-C], dim = -1)
    vector = A-O

    valid = t.linalg.det(matrix).abs() > 1e-8
    matrix[~valid] = t.eye(3)
    s,u,v = einops.rearrange(t.linalg.solve(matrix, vector), '... l-> l ...')

    return ((s>0) & (u>0.0) &(v>0.0) &(u+v<1.0) &valid).item()
    pass


tests.test_triangle_ray_intersects(triangle_ray_intersects)
# %%

def raytrace_triangle(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"]
) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, return True if the triangle intersects that ray.
    '''
    n_rays,_,_ = rays.shape
    A,B,C = einops.repeat(triangle, 'point coord -> point ray coord', ray = n_rays)
    O,D = einops.rearrange(rays, 'ray point coord -> point ray coord')
    matrix = t.stack([D,A-B, A-C], dim = -1)
    vector = A-O

    valid = t.linalg.det(matrix).abs() > 1e-8
    matrix[~valid] = t.eye(3)
    s,u,v = einops.rearrange(t.linalg.solve(matrix, vector), '... l-> l ...')

    return ((s>0) & (u>0.0) &(v>0.0) &(u+v<1.0) &valid)
    pass


A = t.tensor([1, 0.0, -0.5])
B = t.tensor([1, -0.5, 0.0])
C = t.tensor([1, 0.5, 0.5])
num_pixels_y = num_pixels_z = 15
y_limit = z_limit = 0.5

# Plot triangle & rays
test_triangle = t.stack([A, B, C], dim=0)
rays2d = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
triangle_lines = t.stack([A, B, C, A, B, C], dim=0).reshape(-1, 2, 3)
render_lines_with_plotly(rays2d, triangle_lines)

# Calculate and display intersections
intersects = raytrace_triangle(rays2d, test_triangle)
img = intersects.reshape(num_pixels_y, num_pixels_z).int()
imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")
# %%

def raytrace_triangle_with_bug(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"]
) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, return True if the triangle intersects that ray.
    '''
    NR = rays.shape[0]

    A, B, C = einops.repeat(triangle, "pts dims -> pts NR dims", NR=NR)

    O, D = rays.unbind(-2)

    mat = t.stack([- D, B - A, C - A], dim = -1)

    dets = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    vec = O - A

    sol = t.linalg.solve(mat, vec)
    s, u, v = sol.unbind(dim=-1)

    return ((u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular)


intersects = raytrace_triangle_with_bug(rays2d, test_triangle)
img = intersects.reshape(num_pixels_y, num_pixels_z).int()
imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")
# %%

with open(section_dir / "pikachu.pt", "rb") as f:
    triangles = t.load(f)
# %%

def raytrace_mesh(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"]
) -> Float[Tensor, "nrays"]:
    '''
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    '''
    NR = rays.shape[0]
    NT = triangles.shape[0]

    A, B, C = einops.repeat(triangles, "trigs pts dims -> pts rays trigs dims", rays=NR)

    O, D = einops.repeat(  rays,       'rays pts dims -> pts rays trigs dims', trigs = NT)

    mat = t.stack([- D, B - A, C - A], dim = -1)

    dets = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    vec = O - A

    sol = t.linalg.solve(mat, vec)
    s, u, v = sol.unbind(dim=-1)

    intersects =  ((u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular)
    s[~intersects] = float('inf')
    return s.amin(dim = 1)

num_pixels_y = 120
num_pixels_z = 120
y_limit = z_limit = 1

rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
rays[:, 0] = t.tensor([-2, 0.0, 0.0])
dists = raytrace_mesh(rays, triangles)
intersects = t.isfinite(dists).view(num_pixels_y, num_pixels_z)
dists_square = dists.view(num_pixels_y, num_pixels_z)
img = t.stack([intersects, dists_square], dim=0)

fig = px.imshow(img, facet_col=0, origin="lower", color_continuous_scale="magma", width=1000)
fig.update_layout(coloraxis_showscale=False)
for i, text in enumerate(["Intersects", "Distance"]): 
    fig.layout.annotations[i]['text'] = text
fig.show()
# %%
