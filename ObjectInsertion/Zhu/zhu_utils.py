import torch
import math
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def create_frame(n: torch.Tensor, eps: float = 1e-6):
    """
    Generate orthonormal coordinate system based on surface normal
    [Duff et al. 17] Building An Orthonormal Basis, Revisited. JCGT. 2017.
    :param: n (bn, 3, ...)
    """
    z = F.normalize(n, dim=1, eps=eps)
    sgn = torch.where(z[:, 2, ...] >= 0, 1.0, -1.0)
    a = -1.0 / (sgn + z[:, 2, ...])
    b = z[:, 0, ...] * z[:, 1, ...] * a
    x = torch.stack([1.0 + sgn * z[:, 0, ...] * z[:, 0, ...] * a, sgn * b, -sgn * z[:, 0, ...]], dim=1)
    y = torch.stack([b, sgn + z[:, 1, ...] * z[:, 1, ...] * a, -z[:, 1, ...]], dim=1)
    return x, y, z


def to_global(d, x, y, z):
    """
    d, x, y, z: (*, 3)
    """
    return d[:, 0:1] * x + d[:, 1:2] * y + d[:, 2:3] * z


def depth_to_vpos(depth: torch.Tensor, fov, permute=False) -> torch.Tensor:
    row, col = depth.shape
    fovx = math.radians(fov)
    fovy = 2 * math.atan(math.tan(fovx / 2) / (col / row))
    vpos = torch.zeros(row, col, 3, device=depth.device)
    dmax = torch.max(depth)
    depth = depth / dmax
    Y = 1 - (torch.arange(row, device=depth.device) + 0.5) / row
    Y = Y * 2 - 1
    X = (torch.arange(col, device=depth.device) + 0.5) / col
    X = X * 2 - 1
    Y, X = torch.meshgrid(Y, X)
    vpos[:, :, 0] = depth * X * math.tan(fovx / 2)
    vpos[:, :, 1] = depth * Y * math.tan(fovy / 2)
    vpos[:, :, 2] = -depth
    return vpos if not permute else vpos.permute(2, 0, 1)