"""Extraction of the zero level set as a triangle mesh."""

from __future__ import annotations

import warnings

import numpy as np
import torch

from .grid import Grid
from . import kernels as K
from .levelset import LevelSet
from .meshio import TriMesh


def extract_surface(ls: LevelSet, project: bool = True, max_shift: float = 0.5) -> TriMesh:
    """Marching cubes on the band, followed by an optional Newton projection.

    Marching cubes (Lewiner et al.) places vertices by linear interpolation
    along grid edges, an O(h^2) approximation of the zero level set.  With
    ``project=True`` each vertex is then moved by one Newton step
    ``p - phi(p) grad phi(p) / |grad phi(p)|^2`` for the tricubic interpolant of
    phi (step length limited to ``max_shift`` cells), which places it on the
    zero set of the higher-order interpolant.
    """
    from skimage.measure import marching_cubes

    grid = ls.grid
    idx = ls.band
    if idx.numel() == 0 or not ls.has_interface():
        return TriMesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    shape = grid.shape
    ijk = torch.stack([idx // (shape[1] * shape[2]), (idx // shape[2]) % shape[1], idx % shape[2]], 1)
    lo = (ijk.min(0).values - 1).clamp_min(0).cpu().numpy()
    hi = (ijk.max(0).values + 2).cpu().numpy()
    hi = np.minimum(hi, np.asarray(shape))
    sub_t = ls.phi[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    near = (sub_t.abs() < 2.0 * ls.h).cpu().numpy()     # only cells near the zero set
    sub = sub_t.detach().cpu().numpy()
    with warnings.catch_warnings():  # numpy>=2.5 deprecation inside scikit-image
        warnings.simplefilter("ignore", DeprecationWarning)
        verts, faces, _, _ = marching_cubes(sub, level=0.0, allow_degenerate=False, mask=near)
    verts = verts.astype(np.float64) + lo
    if project and len(verts):
        p = torch.as_tensor(verts, dtype=torch.float32, device=ls.device)
        val, grad = K.tricubic_with_gradient.fn(ls.flat, shape[1], shape[2], p)  # eager: one-off
        g2 = (grad * grad).sum(1).clamp_min(1e-30)
        step = (val / g2).unsqueeze(1) * grad
        n = step.norm(dim=1, keepdim=True)
        step = step * (max_shift / n.clamp_min(max_shift)).clamp(max=1.0)
        verts = (p - step).double().cpu().numpy()
    world = np.asarray(grid.origin) + grid.spacing * verts
    mesh = TriMesh(world, faces.astype(np.int64))
    if mesh.signed_volume() < 0:
        mesh = mesh.flipped()
    return mesh


def sample_field(field: torch.Tensor, grid: Grid, points_world: np.ndarray) -> np.ndarray:
    """Tricubic interpolation of a dense grid field at world coordinates."""
    if len(points_world) == 0:
        return np.zeros(0)
    p = torch.as_tensor(grid.to_index(points_world), dtype=torch.float32, device=field.device)
    out = []
    for s in range(0, len(p), 1 << 20):
        out.append(K.tricubic.fn(field.reshape(-1), grid.shape[1], grid.shape[2], p[s:s + (1 << 20)]))
    return torch.cat(out).double().cpu().numpy()
