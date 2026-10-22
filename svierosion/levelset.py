"""Narrow-band level set functions on uniform grids (PyTorch; CPU or CUDA).

Conventions
-----------
* ``phi`` is a dense float32 tensor on the grid, negative inside the solid.
* Computations are restricted to a narrow band of nodes with
  ``|phi| < width * h`` stored as sorted flat indices; outside the band phi
  is clamped to ``+-width * h``.
* Band nodes are kept at least :data:`MARGIN` cells away from the grid
  boundary, so that stencil gathers (up to 3 cells) and tricubic
  interpolation never leave the array.
* The numerical kernels live in :mod:`svierosion.kernels`.
"""

from __future__ import annotations

import torch

from . import kernels as K
from .grid import Grid

MARGIN = 4  # cells between any band node and the grid boundary


def dilate(mask: torch.Tensor, steps: int = 1) -> torch.Tensor:
    """Binary dilation by the (2*steps+1)^3 cube, via separable 1-D passes."""
    out = mask.clone()
    for ax in range(3):
        n = out.shape[ax]
        for _ in range(steps):
            src = out.clone()
            out.narrow(ax, 1, n - 1).logical_or_(src.narrow(ax, 0, n - 1))
            out.narrow(ax, 0, n - 1).logical_or_(src.narrow(ax, 1, n - 1))
    return out


def cube_plane_fraction(s, a, b, c):
    """Volume fraction of the unit cube [-1/2, 1/2]^3 on the side n.xi < -s.

    ``s`` is the signed distance (in cell units, positive outside) from the
    cube centre to a plane with unit normal ``n``; ``a >= b >= c >= 0`` are the
    sorted absolute normal components.  The fraction is the distribution
    function of a sum of three independent uniform random variables; the
    piecewise formula used is algebraically equivalent to the classical
    inclusion-exclusion expression but has no cancellation when b or c -> 0,
    so it is accurate in single precision.
    """
    return K._cube_plane_fraction(s, a, b, c)


class LevelSet:
    """A level set function with an explicitly stored narrow band."""

    def __init__(self, phi: torch.Tensor, grid: Grid, width: float = 6.0):
        if tuple(phi.shape) != tuple(grid.shape):
            raise ValueError("phi shape does not match grid")
        self.grid = grid
        self.h = float(grid.spacing)
        self.width = float(width)
        self.phi = phi.to(torch.float32).contiguous()
        self.device = self.phi.device
        self.strides = tuple(int(s) for s in grid.strides)
        self.rebuild_band()

    # ------------------------------------------------------------ band
    @property
    def flat(self) -> torch.Tensor:
        return self.phi.view(-1)

    def rebuild_band(self) -> None:
        """Band = nodes with |phi| < width*h; clamp all other values."""
        cap = self.width * self.h
        self.phi.clamp_(-cap, cap)
        # strictly inside the cap: clamped values are float32(cap), which may round
        # to just below the float64 cap, so test against a slightly smaller value
        self.band = torch.nonzero(self.flat.abs() < cap * (1 - 1e-5)).squeeze(1)
        self._check_margin(self.band)

    def _check_margin(self, idx: torch.Tensor) -> None:
        if idx.numel() == 0:
            return
        shape = self.grid.shape
        for ax, s in enumerate(self.strides):
            coord = (idx // s) % shape[ax]
            if int(coord.min()) < MARGIN or int(coord.max()) > shape[ax] - 1 - MARGIN:
                raise RuntimeError("narrow band reaches the grid boundary; enlarge the padding")

    def interface_nodes(self, idx: torch.Tensor | None = None) -> torch.Tensor:
        """Band nodes having a 6-neighbour of the opposite sign."""
        idx = self.band if idx is None else idx
        return idx[K.interface_mask(self.flat, idx, *self.strides)]

    def has_interface(self) -> bool:
        b = self.flat[self.band]
        return bool((b < 0).any()) and bool((b > 0).any())

    # ---------------------------------------------------- derivatives
    def central_gradient(self, idx: torch.Tensor) -> torch.Tensor:
        """Second-order central gradient, in units of phi per cell, shape (m, 3)."""
        f = self.flat
        return torch.stack([0.5 * (f[idx + s] - f[idx - s]) for s in self.strides], 1)

    def godunov_norm_inward(self, idx: torch.Tensor) -> torch.Tensor:
        """Godunov/WENO5 |grad phi| for phi_t = V |grad phi| with V >= 0."""
        return K.godunov_inward(self.flat, idx, *self.strides, 1.0 / self.h)

    def principal_curvatures(self, idx: torch.Tensor):
        """Principal curvatures (k1 >= k2) of the level sets through band nodes."""
        return K.principal_curvatures(self.flat, idx, *self.strides, self.h)

    # ------------------------------------------------- volume fractions
    def cell_fractions(self, idx: torch.Tensor | None = None) -> torch.Tensor:
        """Fraction of each node's cell inside the solid (exact for the tangent plane)."""
        idx = self.band if idx is None else idx
        return K.cell_fractions(self.flat, idx, *self.strides)

    def dense_fractions(self) -> torch.Tensor:
        """Dense occupancy field (1 inside, 0 outside, cell fractions in the band)."""
        f = (self.phi < 0).to(torch.float32)
        f.view(-1)[self.band] = self.cell_fractions(self.band)
        return f

    def volume(self) -> float:
        f = self.flat
        outside_band_inside = int((f < 0).sum()) - int((f[self.band] < 0).sum())
        return (outside_band_inside + float(self.cell_fractions(self.band).double().sum())) * self.h ** 3

    # ----------------------------------------------------- projection
    def closest_points(self, idx: torch.Tensor, newton_steps: int = 1) -> torch.Tensor:
        """Projection of band nodes onto the zero level set, in index coordinates.

        The first step is ``x - phi grad(phi)/|grad(phi)|^2`` with the central
        gradient at the node (exact for a signed distance function).  Further
        steps use tricubic values and gradients at the current estimate.
        """
        nx, ny, nz = self.grid.shape
        hi = (nx - 3.001, ny - 3.001, nz - 3.001)
        p = K.closest_points(self.flat, idx, ny, nz, *hi, self.h)
        for _ in range(newton_steps - 1):
            val, grad = K.tricubic_with_gradient(self.flat, ny, nz, p)
            g2 = (grad * grad).sum(1).clamp_min(1e-30)
            length = val.abs() / g2.sqrt()
            cap = 1.25 * val.abs() / self.h + 0.25
            step = (val / g2 * (cap / length.clamp_min(1e-30)).clamp(max=1.0)).unsqueeze(1) * grad
            p = self.clamp_to_interior(p - step)
        return p

    def clamp_to_interior(self, p: torch.Tensor) -> torch.Tensor:
        """Clamp index coordinates to the region where tricubic stencils fit."""
        hi = torch.tensor([n - 3.001 for n in self.grid.shape], device=p.device, dtype=p.dtype)
        return torch.minimum(p.clamp_min(1.0), hi)

    # --------------------------------------------------- reinitialisation
    def reinitialize(self, iterations: int = 12, cfl: float = 0.45, dilation: int = 3) -> None:
        """Reinitialise to a signed distance function (Min 2010, J. Comput. Phys. 229).

        Solves ``phi_tau + sgn(phi0) (|grad phi| - 1) = 0`` on the band dilated
        by ``dilation`` cells, using second-order ENO differences, a TVD-RK2
        pseudo-time integrator with local time steps and the *subcell
        resolution* boundary treatment at nodes next to the interface: the
        interface location along each grid line is found from a quadratic ENO
        interpolant of the initial phi and imposed as phi = 0 inside the
        one-sided differences.  This keeps the zero level set fixed to
        second order.
        """
        mask = torch.zeros(self.grid.shape, dtype=torch.bool, device=self.device)
        mask.view(-1)[self.band] = True
        mask = dilate(mask, dilation)
        for ax in range(3):  # keep the work set inside the safe region
            n = mask.shape[ax]
            mask.narrow(ax, 0, MARGIN).zero_()
            mask.narrow(ax, n - MARGIN, MARGIN).zero_()
        idx = torch.nonzero(mask.view(-1)).squeeze(1)
        del mask
        f = self.flat
        geo = K.reinit_geometry(f, idx, *self.strides, self.h, cfl)
        sgn, dtau, rest = geo[0], geo[1], geo[2:]
        for _ in range(iterations):
            u0 = f[idx]
            f[idx] = u0 + dtau * K.reinit_rhs(f, idx, *self.strides, self.h, sgn, *rest)
            u1 = f[idx]
            f[idx] = 0.5 * (u0 + u1 + dtau * K.reinit_rhs(f, idx, *self.strides, self.h, sgn, *rest))
        self.rebuild_band()


# ------------------------------------------------------------- interpolation
def tricubic(flat: torch.Tensor, shape, p: torch.Tensor) -> torch.Tensor:
    """Tensor-product cubic Lagrange interpolation at index coordinates p (m, 3)."""
    return K.tricubic(flat, int(shape[1]), int(shape[2]), p)


def tricubic_with_gradient(flat: torch.Tensor, shape, p: torch.Tensor):
    """Tricubic value and gradient (per cell) at index coordinates p (m, 3)."""
    return K.tricubic_with_gradient(flat, int(shape[1]), int(shape[2]), p)


def trilinear(flat: torch.Tensor, shape, p: torch.Tensor) -> torch.Tensor:
    """Trilinear interpolation at index coordinates p (m, 3) (monotone, no overshoot)."""
    return K.trilinear(flat, int(shape[1]), int(shape[2]), p)
