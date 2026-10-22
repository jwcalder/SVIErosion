"""Uniform Cartesian grids and device selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def get_device(device=None) -> torch.device:
    """``None``/``"auto"`` selects CUDA when available, otherwise the CPU."""
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def next_fast_size(n: int) -> int:
    """Smallest m >= n whose prime factors are in {2, 3, 5, 7} (fast FFT sizes)."""
    m = max(int(n), 1)
    while True:
        k = m
        for p in (2, 3, 5, 7):
            while k % p == 0:
                k //= p
        if k == 1:
            return m
        m += 1


@dataclass(frozen=True)
class Grid:
    """Nodes ``origin + spacing * (i, j, k)`` for ``0 <= (i, j, k) < shape``.

    Arrays on the grid use C order with axes (x, y, z).  Node (i, j, k) also
    represents the cubic cell of side ``spacing`` centred on it.
    """

    origin: tuple
    spacing: float
    shape: tuple

    @classmethod
    def around(cls, lo, hi, spacing: float, padding: float, fast: bool = True) -> "Grid":
        """Grid covering the box [lo, hi] with at least ``padding`` on every side.

        The box is centred in the grid, and with ``fast=True`` the dimensions
        are rounded up to FFT-friendly sizes.
        """
        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)
        need = np.ceil((hi - lo + 2 * padding) / spacing).astype(int) + 1
        shape = tuple(next_fast_size(n) if fast else int(n) for n in need)
        center = 0.5 * (lo + hi)
        origin = center - 0.5 * (np.asarray(shape) - 1) * spacing
        return cls(tuple(float(o) for o in origin), float(spacing), tuple(int(s) for s in shape))

    # ------------------------------------------------------------ conversions
    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    @property
    def strides(self) -> tuple:
        nx, ny, nz = self.shape
        return (ny * nz, nz, 1)

    def axes(self) -> list:
        return [self.origin[a] + self.spacing * np.arange(self.shape[a]) for a in range(3)]

    def to_index(self, points) -> np.ndarray:
        """World coordinates -> continuous grid (index) coordinates."""
        return (np.asarray(points, dtype=np.float64) - np.asarray(self.origin)) / self.spacing

    def to_world(self, index) -> np.ndarray:
        return np.asarray(self.origin) + self.spacing * np.asarray(index, dtype=np.float64)

    def node_coords(self, flat_index: torch.Tensor, dtype=torch.float64) -> torch.Tensor:
        """World coordinates of flat node indices, shape (m, 3)."""
        idx = unravel(flat_index, self.shape)
        o = torch.tensor(self.origin, dtype=dtype, device=flat_index.device)
        return o + self.spacing * idx.to(dtype)

    def bounds(self) -> np.ndarray:
        o = np.asarray(self.origin)
        return np.stack([o, o + self.spacing * (np.asarray(self.shape) - 1)])

    def describe(self) -> str:
        return (f"grid {self.shape[0]}x{self.shape[1]}x{self.shape[2]} "
                f"({self.size / 1e6:.1f}M nodes), spacing {self.spacing:.6g}")


def unravel(flat: torch.Tensor, shape) -> torch.Tensor:
    """Flat C-order indices -> (m, 3) integer indices."""
    nx, ny, nz = shape
    k = flat % nz
    j = (flat // nz) % ny
    i = flat // (ny * nz)
    return torch.stack([i, j, k], dim=1)
