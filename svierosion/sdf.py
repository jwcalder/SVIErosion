"""Signed distance functions of triangle meshes on uniform grids.

Algorithm (in the spirit of Batty's SDFGen, but vectorised for GPU/CPU):

1. *Exact near field.*  Every triangle (after splitting large ones) is paired
   with all grid nodes in its bounding box dilated by ``exact_band`` cells, and
   the exact point-triangle distance is computed.  A packed 64-bit key
   ``(float bits of distance) << 32 | triangle id`` is scatter-minimised into
   the grid, giving the exact distance and the closest triangle of every node
   within ``exact_band`` cells of the surface.
2. *Closest-triangle propagation.*  Rings of nodes outside the exact zone
   receive the minimum distance to the closest triangles of their 26
   neighbours (repeated ring by ring until the requested band is covered).
   This gives the exact distance except in rare configurations, and always an
   upper bound.
3. *Sign.*  Inside/outside is decided for every node by scanline ray casting
   along grid lines: each triangle is rasterised onto the grid lines that
   pierce it with an exact Simulation-of-Simplicity tie-break, so that a line
   through a shared edge or vertex is counted exactly once.  For consistently
   oriented meshes the signed crossing count (winding number) is used; for
   other meshes the parity; for meshes with holes a majority vote over the
   three axis directions reduces the effect of rays leaking through holes.

The returned field is negative inside, positive outside, exact to rounding
error within the band and clamped to ``+-band * h`` outside it.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from . import kernels as K
from .grid import Grid, get_device, unravel
from .meshio import TriMesh

_KEY_EMPTY = torch.iinfo(torch.int64).max


# ------------------------------------------------------------ geometry kernels
def point_triangle_distance2(p, a, b, c):
    """Squared Euclidean distance from points ``p`` to triangles ``(a, b, c)``.

    All inputs are (..., 3) tensors (broadcastable).  Region classification
    follows Ericson, *Real-Time Collision Detection*, sec. 5.1.5; the
    implementation is :func:`svierosion.kernels._point_triangle_distance2`.
    """
    return K.point_triangle_distance2_compiled(p, a, b, c)


def split_long_triangles(tri: np.ndarray, max_edge: float) -> np.ndarray:
    """Midpoint-subdivide triangles (n, 3, 3) until every edge is <= max_edge."""
    out = []
    while len(tri):
        e = np.stack([np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
                      np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
                      np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)], 1).max(1)
        small = e <= max_edge
        out.append(tri[small])
        big = tri[~small]
        if not len(big):
            break
        a, b, c = big[:, 0], big[:, 1], big[:, 2]
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        tri = np.concatenate([np.stack([a, ab, ca], 1), np.stack([ab, b, bc], 1),
                              np.stack([ca, bc, c], 1), np.stack([ab, bc, ca], 1)])
    return np.concatenate(out) if out else np.zeros((0, 3, 3))


def bisect_long_triangles(tri: np.ndarray, max_edge: float) -> np.ndarray:
    """Bisect the longest edge of triangles (n, 3, 3) until every edge is <= max_edge.

    Unlike 1-to-4 midpoint subdivision, longest-edge bisection also improves the
    aspect ratio of skinny triangles, so the pieces are well represented by
    their centroids.
    """
    out = []
    while len(tri):
        e = np.stack([np.linalg.norm(tri[:, 1] - tri[:, 2], axis=1),    # edge opposite vertex 0
                      np.linalg.norm(tri[:, 2] - tri[:, 0], axis=1),
                      np.linalg.norm(tri[:, 0] - tri[:, 1], axis=1)], 1)
        small = e.max(1) <= max_edge
        out.append(tri[small])
        big = tri[~small]
        if not len(big):
            break
        j = e[~small].argmax(1)                   # vertex opposite the longest edge
        a = big[np.arange(len(big)), j]
        b = big[np.arange(len(big)), (j + 1) % 3]
        c = big[np.arange(len(big)), (j + 2) % 3]
        m = 0.5 * (b + c)
        tri = np.concatenate([np.stack([a, b, m], 1), np.stack([a, m, c], 1)])
    return np.concatenate(out) if out else np.zeros((0, 3, 3))


# --------------------------------------------------------------- near field
def _exact_near_field(tri_idx: np.ndarray, grid: Grid, exact_band: float, device,
                      max_pairs: int) -> torch.Tensor:
    """Packed (distance, triangle) keys for all nodes within exact_band cells."""
    shape = np.asarray(grid.shape)
    keys = torch.full((grid.size,), _KEY_EMPTY, dtype=torch.int64, device=device)
    lo = np.ceil(tri_idx.min(1) - exact_band).astype(np.int64)
    hi = np.floor(tri_idx.max(1) + exact_band).astype(np.int64)
    lo = np.clip(lo, 0, shape - 1)
    hi = np.clip(hi, 0, shape - 1)
    dims = np.maximum(hi - lo + 1, 0)
    counts = dims.prod(1)
    order = np.argsort(counts, kind="stable")
    tri_dev = torch.as_tensor(tri_idx, dtype=torch.float32, device=device)
    strides = torch.tensor(grid.strides, dtype=torch.int64, device=device)
    csum = np.cumsum(counts[order])
    start = 0
    eb = float(exact_band) * (1 + 1e-6)
    while start < len(order):
        base = csum[start - 1] if start else 0
        stop = int(np.searchsorted(csum, base + max_pairs, side="right"))
        stop = max(stop, start + 1)
        sel = order[start:stop]
        start = stop
        cnt = torch.as_tensor(counts[sel], device=device)
        if int(cnt.sum()) == 0:
            continue
        lo_t = torch.as_tensor(lo[sel], device=device)
        dim_t = torch.as_tensor(dims[sel], device=device)
        sel_t = torch.as_tensor(sel, device=device)
        local = torch.repeat_interleave(torch.arange(len(sel), device=device), cnt)
        first = torch.cumsum(cnt, 0) - cnt
        q = torch.arange(int(cnt.sum()), device=device) - first[local]
        nyz = dim_t[local, 1] * dim_t[local, 2]
        ii = lo_t[local, 0] + q // nyz
        rem = q % nyz
        jj = lo_t[local, 1] + rem // dim_t[local, 2]
        kk = lo_t[local, 2] + rem % dim_t[local, 2]
        del q, rem, nyz
        p = torch.stack([ii, jj, kk], 1).to(torch.float32)
        t = tri_dev[sel_t[local]]
        d = point_triangle_distance2(p, t[:, 0], t[:, 1], t[:, 2]).sqrt()
        keep = d <= eb
        flat = (ii * strides[0] + jj * strides[1] + kk)[keep]
        key = (d[keep].view(torch.int32).to(torch.int64) << 32) | sel_t[local][keep]
        keys.scatter_reduce_(0, flat, key, reduce="amin")
    return keys


def _dilate26(mask: torch.Tensor) -> torch.Tensor:
    """Binary dilation by the 3x3x3 cube (three separable 1-D passes)."""
    out = mask.clone()
    for ax in range(3):
        src = out.clone()
        n = src.shape[ax]
        out.narrow(ax, 1, n - 1).logical_or_(src.narrow(ax, 0, n - 1))
        out.narrow(ax, 0, n - 1).logical_or_(src.narrow(ax, 1, n - 1))
    return out


_OFFSETS26 = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
              if (i, j, k) != (0, 0, 0)]


def _propagate(dist: torch.Tensor, tid: torch.Tensor, tri_dev: torch.Tensor, grid: Grid,
               rings: int, chunk: int = 1 << 22) -> None:
    """Extend (dist, tid) outward ring by ring using neighbours' closest triangles."""
    shape = grid.shape
    known = (tid >= 0).view(shape)
    sx, sy, sz = grid.strides
    offs = [o[0] * sx + o[1] * sy + o[2] * sz for o in _OFFSETS26]
    n = grid.size
    prev = None
    for _ in range(rings):
        ring = _dilate26(known) & ~known
        # Interior safety: never touch the outermost grid layer.
        for ax in range(3):
            ring.narrow(ax, 0, 1).zero_()
            ring.narrow(ax, shape[ax] - 1, 1).zero_()
        upd = torch.nonzero(ring.view(-1)).squeeze(1)
        if prev is not None:
            upd = torch.cat([upd, prev])
        if upd.numel() == 0:
            break
        for c0 in range(0, upd.numel(), chunk):
            u = upd[c0:c0 + chunk]
            p = unravel(u, shape).to(torch.float32)
            best = dist[u].clone()
            best_t = tid[u].clone()
            for o in offs:
                nb = (u + o).clamp(0, n - 1)
                t = tid[nb]
                ok = t >= 0
                if not bool(ok.any()):
                    continue
                tt = tri_dev[t.clamp_min(0)]
                d = point_triangle_distance2(p, tt[:, 0], tt[:, 1], tt[:, 2]).sqrt()
                better = ok & (d < best)
                best = torch.where(better, d, best)
                best_t = torch.where(better, t, best_t)
            dist[u] = best
            tid[u] = best_t
        known |= ring
        prev = torch.nonzero(ring.view(-1)).squeeze(1)


def _vertex_ids(tri: np.ndarray) -> np.ndarray:
    """Vertex ids (n, 3) for a triangle soup, merging exactly coincident vertices."""
    _, inv = np.unique(tri.reshape(-1, 3), axis=0, return_inverse=True)
    return inv.reshape(-1)[: 3 * len(tri)].reshape(-1, 3)


def _refine_with_stars(dist: torch.Tensor, tid: torch.Tensor, tri_dev: torch.Tensor,
                       tri_vid: np.ndarray, grid: Grid, nodes: torch.Tensor,
                       passes: int = 2, chunk: int = 1 << 18, max_pairs: int = 1 << 22) -> None:
    """Re-test, for each node, every triangle incident to a vertex of its current
    closest triangle (the "vertex star"), and keep the minimum distance.

    Work is split so that at most ``max_pairs`` node-triangle pairs are formed at
    once (vertex valences can be large, e.g. fan-triangulated caps)."""
    dev = dist.device
    vid = torch.as_tensor(tri_vid, device=dev)
    flat_v = vid.reshape(-1)
    order = torch.argsort(flat_v)
    star = order // 3                                   # triangle ids sorted by vertex
    deg = torch.bincount(flat_v, minlength=int(flat_v.max()) + 1)
    first = torch.cumsum(deg, 0) - deg
    for _ in range(passes):
        changed = 0
        for c0 in range(0, nodes.numel(), chunk):
            u_all = nodes[c0:c0 + chunk]
            t_all = tid[u_all]
            ok = t_all >= 0
            u_all, t_all = u_all[ok], t_all[ok]
            if u_all.numel() == 0:
                continue
            cnt = deg[vid[t_all]].sum(1)
            csum = torch.cumsum(cnt, 0).cpu().numpy()
            s0 = 0
            while s0 < u_all.numel():
                base = csum[s0 - 1] if s0 else 0
                s1 = max(int(np.searchsorted(csum, base + max_pairs, side="right")), s0 + 1)
                u, t = u_all[s0:s1], t_all[s0:s1]
                s0 = s1
                v = vid[t].reshape(-1)                         # 3 vertices per node
                d = deg[v]
                pair_node = torch.repeat_interleave(
                    torch.arange(u.numel(), device=dev).repeat_interleave(3), d)
                start = torch.repeat_interleave(first[v], d)
                offs = torch.arange(int(d.sum()), device=dev) - torch.repeat_interleave(
                    torch.cumsum(d, 0) - d, d)
                cand = star[start + offs]
                p = unravel(u, grid.shape).to(torch.float32)[pair_node]
                tt = tri_dev[cand]
                dd = point_triangle_distance2(p, tt[:, 0], tt[:, 1], tt[:, 2]).sqrt()
                key = (dd.view(torch.int32).to(torch.int64) << 32) | cand
                del p, tt, dd, start, offs
                best = torch.full((u.numel(),), _KEY_EMPTY, dtype=torch.int64, device=dev)
                best.scatter_reduce_(0, pair_node, key, reduce="amin")
                bd = (best >> 32).to(torch.int32).view(torch.float32)
                bt = best & 0xFFFFFFFF
                better = bd < dist[u]
                changed += int(better.sum())
                dist[u[better]] = bd[better]
                tid[u[better]] = bt[better]
        if changed == 0:
            break


# ---------------------------------------------------------------------- sign
def _orient_sos(x1, y1, x2, y2):
    """Sign of the oriented area of (0,0),(x1,y1),(x2,y2) with SoS tie-breaking.

    Antisymmetric under swapping the two points, so the triangles sharing an
    edge classify a point on that edge consistently (Batty, SDFGen).
    """
    a = x1 * y2 - y1 * x2
    s = np.sign(a).astype(np.int8)
    zero = s == 0
    if zero.any():
        tie = np.where(y2 > y1, -1, np.where(y2 < y1, 1,
                       np.where(x1 > x2, -1, np.where(x1 < x2, 1, 0)))).astype(np.int8)
        s = np.where(zero, tie, s)
    return s, a


def _crossings_along_axis(tri_idx: np.ndarray, shape, axis: int, signed: bool,
                          max_candidates: int = 1 << 23) -> np.ndarray:
    """Inside mask from crossings of grid lines parallel to ``axis``.

    ``tri_idx`` holds triangles in index coordinates.  Returns a boolean array
    in (x, y, z) order.
    """
    a, b, c = axis, (axis + 1) % 3, (axis + 2) % 3
    na, nb, nc = shape[a], shape[b], shape[c]
    U, W, Z = tri_idx[:, :, b], tri_idx[:, :, c], tri_idx[:, :, a]
    j0 = np.clip(np.ceil(U.min(1)), 0, nb).astype(np.int64)
    j1 = np.clip(np.floor(U.max(1)), -1, nb - 1).astype(np.int64)
    k0 = np.clip(np.ceil(W.min(1)), 0, nc).astype(np.int64)
    k1 = np.clip(np.floor(W.max(1)), -1, nc - 1).astype(np.int64)
    dj = np.maximum(j1 - j0 + 1, 0)
    dk = np.maximum(k1 - k0 + 1, 0)
    cnt = dj * dk
    tri_ids = np.nonzero(cnt)[0]
    acc = np.zeros((na + 1) * nb * nc, dtype=np.int32)
    csum = np.cumsum(cnt[tri_ids])
    start = 0
    while start < len(tri_ids):
        base = csum[start - 1] if start else 0
        stop = max(int(np.searchsorted(csum, base + max_candidates, side="right")), start + 1)
        t = tri_ids[start:stop]
        start = stop
        n_t = cnt[t]
        rep = np.repeat(np.arange(len(t)), n_t)
        q = np.arange(int(n_t.sum())) - np.repeat(np.cumsum(n_t) - n_t, n_t)
        jj = j0[t][rep] + q // dk[t][rep]
        kk = k0[t][rep] + q % dk[t][rep]
        tt = t[rep]
        x1 = U[tt, 0] - jj; x2 = U[tt, 1] - jj; x3 = U[tt, 2] - jj
        y1 = W[tt, 0] - kk; y2 = W[tt, 1] - kk; y3 = W[tt, 2] - kk
        sa, A = _orient_sos(x2, y2, x3, y3)
        sb, B = _orient_sos(x3, y3, x1, y1)
        sc, C = _orient_sos(x1, y1, x2, y2)
        hit = (sa != 0) & (sb == sa) & (sc == sa)
        if not hit.any():
            continue
        A, B, C, sa = A[hit], B[hit], C[hit], sa[hit]
        tt, jj, kk = tt[hit], jj[hit], kk[hit]
        S = A + B + C
        good = S != 0
        z = (A * Z[tt, 0] + B * Z[tt, 1] + C * Z[tt, 2])[good] / S[good]
        i0 = np.clip(np.floor(z).astype(np.int64) + 1, 0, na)
        flat = (i0 * nb + jj[good]) * nc + kk[good]
        w = -sa[good].astype(np.int64) if signed else np.ones(len(flat), dtype=np.int64)
        acc += np.bincount(flat, weights=w, minlength=acc.size).astype(np.int32)
    acc = acc.reshape(na + 1, nb, nc)
    if signed:
        inside = np.cumsum(acc, axis=0, dtype=np.int32)[:na] != 0
    else:
        inside = (np.cumsum(acc, axis=0, dtype=np.int32)[:na] & 1).astype(bool)
    inv = np.argsort([a, b, c])
    return np.ascontiguousarray(np.transpose(inside, inv))


def inside_mask(mesh: TriMesh, grid: Grid, axes=None, signed=None) -> np.ndarray:
    """Boolean inside/outside classification of every grid node.

    ``axes`` defaults to ``(0,)`` for closed meshes and ``(0, 1, 2)`` (majority
    vote) otherwise.  ``signed`` (winding-number rule) defaults to true for
    consistently oriented meshes, otherwise the parity rule is used.
    """
    topo = mesh.edge_topology()
    closed = topo["boundary_edges"] == 0 and topo["nonmanifold_edges"] == 0
    if signed is None:
        signed = topo["misoriented_edges"] == 0
    if axes is None:
        axes = (0,) if closed else (0, 1, 2)
    tri_idx = (mesh.vertices[mesh.faces] - np.asarray(grid.origin)) / grid.spacing
    votes = None
    for ax in axes:
        m = _crossings_along_axis(tri_idx, grid.shape, ax, signed).astype(np.uint8)
        votes = m if votes is None else votes + m
    return votes * 2 > len(axes)


# ------------------------------------------------------------------- driver
def mesh_to_sdf(mesh: TriMesh, grid: Grid, band: float = 6.0, exact_band: float = 2.0,
                device=None, sign_axes=None, max_pairs: int | None = None,
                verbose: bool = False) -> torch.Tensor:
    """Signed distance of ``mesh`` sampled on ``grid`` (negative inside).

    Parameters
    ----------
    band : half-width (in cells) of the band where the distance is computed;
        values outside are clamped to ``+-band * h``.
    exact_band : half-width (cells) of the brute-force exact zone; the rest of
        the band is filled by closest-triangle propagation.
    sign_axes : axes used for ray casting (see :func:`inside_mask`).

    Returns a dense float32 tensor of shape ``grid.shape`` on ``device``.
    """
    dev = get_device(device)
    if max_pairs is None:
        max_pairs = (1 << 21) if dev.type == "cuda" else (1 << 20)
    t0 = time.time()
    h = grid.spacing
    tri_idx = (mesh.vertices[mesh.faces] - np.asarray(grid.origin)) / h
    tri_split = split_long_triangles(tri_idx, max_edge=8.0)
    keys = _exact_near_field(tri_split, grid, exact_band, dev, max_pairs)
    found = keys != _KEY_EMPTY
    dist = torch.where(found, (keys >> 32).to(torch.int32).view(torch.float32),
                       torch.full((), float("inf"), device=dev))
    tid = torch.where(found, (keys & 0xFFFFFFFF), torch.full((), -1, device=dev,
                                                             dtype=torch.int64))
    del keys, found
    t1 = time.time()
    tri_dev = torch.as_tensor(tri_split, dtype=torch.float32, device=dev)
    rings = max(int(math.ceil(band - exact_band)) + 1, 0)
    _propagate(dist, tid, tri_dev, grid, rings)
    if len(tri_split) == len(mesh.faces):           # nothing was split: same order
        _, inv = np.unique(mesh.vertices, axis=0, return_inverse=True)
        tri_vid = inv.reshape(-1)[mesh.faces]
    else:
        tri_vid = _vertex_ids(tri_split)
    outer = torch.nonzero((dist > exact_band) & (dist <= band + 1.0)).squeeze(1)
    _refine_with_stars(dist, tid, tri_dev, tri_vid, grid, outer)
    del tid, tri_dev, outer
    t2 = time.time()
    inside = torch.as_tensor(inside_mask(mesh, grid, axes=sign_axes).reshape(-1), device=dev)
    t3 = time.time()
    dist = torch.clamp(dist, max=band) * h
    phi = torch.where(inside, -dist, dist).view(grid.shape).to(torch.float32)
    if verbose:
        print(f"[sdf] {len(mesh.faces)} triangles ({len(tri_split)} after splitting), "
              f"{grid.describe()}: near field {t1 - t0:.2f}s, propagation {t2 - t1:.2f}s, "
              f"sign {t3 - t2:.2f}s")
    return phi
