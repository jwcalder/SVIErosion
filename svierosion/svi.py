"""Spherical volume invariants on a grid via FFT convolution.

For a solid ``Omega`` described by cell volume fractions ``f`` (one cell per
node) the interior ball volume at every node ``x`` is

    |B_r(x) cap Omega|  ~=  h^3 * sum_y w(x - y) f(y),

where ``w(z) = |B_r(0) cap cell(z)| / h^3`` is the *exact* fraction of the
grid cell centred at ``z`` covered by the ball.  The sum is a discrete
convolution, evaluated with real FFTs.  The normalised exterior volume
invariant is ``N_r = 1 - |B_r cap Omega| / |B_r|``.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .grid import Grid


def ball_cell_weights(radius: float, spacing: float, quad: int = 32) -> np.ndarray:
    """Exact fractions ``|B_r(0) cap cell(i,j,k)| / h^3`` for |i|,|j|,|k| <= m.

    Cells fully inside/outside the ball are classified exactly; for cut cells
    the z-extent of the ball is integrated analytically and the remaining 2-D
    integral over (x, y) is done with ``quad x quad`` Gauss-Legendre points.
    """
    R = radius / spacing                      # ball radius in cell units
    m = int(math.floor(R + 0.5)) + 1
    ax = np.arange(-m, m + 1, dtype=np.float64)
    I, J, K = np.meshgrid(ax, ax, ax, indexing="ij")
    c = np.stack([I, J, K], -1)
    near = np.sqrt((np.maximum(np.abs(c) - 0.5, 0) ** 2).sum(-1))
    far = np.sqrt(((np.abs(c) + 0.5) ** 2).sum(-1))
    w = np.zeros(I.shape)
    w[far <= R] = 1.0
    cut = (near < R) & (far > R)
    g, gw = np.polynomial.legendre.leggauss(quad)
    g, gw = 0.5 * g, 0.5 * gw
    cc = c[cut]
    vols = np.empty(len(cc))
    for s in range(0, len(cc), 4096):
        cs = cc[s:s + 4096]
        X = cs[:, 0, None, None] + g[None, :, None]
        Y = cs[:, 1, None, None] + g[None, None, :]
        zr = np.sqrt(np.maximum(R * R - X * X - Y * Y, 0.0))
        lo = cs[:, 2, None, None] - 0.5
        hi = cs[:, 2, None, None] + 0.5
        L = np.maximum(np.minimum(hi, zr) - np.maximum(lo, -zr), 0.0)
        vols[s:s + 4096] = (L * gw[None, :, None] * gw[None, None, :]).sum((1, 2))
    w[cut] = vols
    return w


class BallConvolution:
    """Cached FFT of the ball kernel on a grid; evaluates interior ball volumes.

    Uses circular convolution, which is exact provided the solid stays at least
    ``self.m`` cells away from the grid boundary (checked by the caller via the
    narrow band margin).
    """

    def __init__(self, grid: Grid, radius: float, device=None, quad: int = 32):
        self.grid = grid
        self.radius = float(radius)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        w = ball_cell_weights(radius, grid.spacing, quad)
        self.m = (w.shape[0] - 1) // 2
        self.ball_volume = 4.0 / 3.0 * math.pi * radius ** 3
        self.quadrature_volume = float(w.sum()) * grid.spacing ** 3
        if any(2 * self.m + 1 > n for n in grid.shape):
            raise ValueError("ball kernel larger than the grid")
        kern = torch.zeros(grid.shape, dtype=torch.float32, device=self.device)
        idx = torch.arange(-self.m, self.m + 1, device=self.device)
        ii, jj, kk = (idx % grid.shape[0], idx % grid.shape[1], idx % grid.shape[2])
        kern[ii[:, None, None], jj[None, :, None], kk[None, None, :]] = torch.as_tensor(
            w, dtype=torch.float32, device=self.device)
        self.kernel_hat = torch.fft.rfftn(kern)
        del kern

    def interior_volume(self, fractions: torch.Tensor) -> torch.Tensor:
        """Dense field |B_r(x) cap Omega| at every node."""
        fh = torch.fft.rfftn(fractions)
        fh *= self.kernel_hat
        return torch.fft.irfftn(fh, s=self.grid.shape).mul_(self.grid.spacing ** 3)

    def exterior_fraction(self, fractions: torch.Tensor) -> torch.Tensor:
        """Dense field N_r(x) = 1 - |B_r(x) cap Omega| / |B_r|."""
        g = self.interior_volume(fractions)
        return g.mul_(-1.0 / self.ball_volume).add_(1.0)


def sphere_exterior_fraction(radius_ball: float, radius_sphere: float) -> float:
    """Exact N_r at a point of a sphere of radius R (valid for r <= 2R)."""
    r, R = radius_ball, radius_sphere
    if r > 2 * R:
        return 1.0 - (R / r) ** 3
    return 0.5 + 3.0 * r / (16.0 * R)
