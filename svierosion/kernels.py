"""Hot numerical kernels as plain PyTorch functions on flattened grids.

Every kernel is wrapped in :class:`Kernel`, which compiles it with
``torch.compile(dynamic=True)`` on first use.  Compilation fuses the long
chains of element-wise operations and gathers into a few kernels (5-25x
faster than eager PyTorch for these stencils, on both CPU and GPU).  If
compilation is unavailable or fails, the eager function is used instead.
Compilation can be disabled with ``svierosion.kernels.set_compile(False)`` or
the environment variable ``SVIEROSION_COMPILE=0``.

Conventions: ``flat`` is the C-ordered flattened grid, ``idx`` flat node
indices, ``(sx, sy, sz)`` the flat strides of the three axes, points ``p``
are in index (cell) coordinates.
"""

from __future__ import annotations

import os
import warnings

import torch

_COMPILE = os.environ.get("SVIEROSION_COMPILE", "1") != "0"


def set_compile(flag: bool) -> None:
    """Enable or disable torch.compile for all kernels (takes effect on next call)."""
    global _COMPILE
    _COMPILE = bool(flag)


def compile_enabled() -> bool:
    return _COMPILE


class Kernel:
    """Lazily compiled function with eager fallback."""

    def __init__(self, fn):
        self.fn = fn
        self.__doc__ = fn.__doc__
        self.__name__ = fn.__name__
        self._compiled = None
        self._failed = False

    def __call__(self, *args):
        # Python scalars become 0-d tensors: as compile-time constants they can be
        # baked into cached graphs (observed with torch 2.13: the on-disk FX graph
        # cache reused a graph compiled for a different float argument of
        # full_like/clamp), whereas tensors are always runtime inputs.
        ref = next((a for a in args if isinstance(a, torch.Tensor)), None)
        if ref is not None:
            fdt = ref.dtype if ref.is_floating_point() else torch.float32
            args = tuple(_as_tensor(a, ref.device, fdt) for a in args)
        if _COMPILE and not self._failed:
            try:
                if self._compiled is None:
                    self._compiled = torch.compile(self.fn, dynamic=True)
                return self._compiled(*args)
            except Exception as exc:  # pragma: no cover - platform dependent
                self._failed = True
                self.error = exc
                warnings.warn(f"torch.compile failed for {self.__name__} ({type(exc).__name__}: "
                              f"{str(exc)[:200]}); using eager PyTorch", RuntimeWarning)
        return self.fn(*args)


def _as_tensor(a, device, fdt):
    if isinstance(a, bool) or isinstance(a, torch.Tensor):
        return a
    if isinstance(a, int):
        return torch.tensor(a, dtype=torch.int64, device=device)
    if isinstance(a, float):
        return torch.tensor(a, dtype=fdt, device=device)
    return a


def kernel(fn):
    return Kernel(fn)


# ------------------------------------------------------------------ helpers
def _weno5(v1, v2, v3, v4, v5):
    """Jiang-Peng WENO5 approximation of the derivative from five differences."""
    p1 = v1 / 3 - 7 * v2 / 6 + 11 * v3 / 6
    p2 = -v2 / 6 + 5 * v3 / 6 + v4 / 3
    p3 = v3 / 3 + 5 * v4 / 6 - v5 / 6
    s1 = 13.0 / 12 * (v1 - 2 * v2 + v3) ** 2 + 0.25 * (v1 - 4 * v2 + 3 * v3) ** 2
    s2 = 13.0 / 12 * (v2 - 2 * v3 + v4) ** 2 + 0.25 * (v2 - v4) ** 2
    s3 = 13.0 / 12 * (v3 - 2 * v4 + v5) ** 2 + 0.25 * (3 * v3 - 4 * v4 + v5) ** 2
    vmax = torch.maximum(torch.maximum(torch.maximum(v1 * v1, v2 * v2),
                                       torch.maximum(v3 * v3, v4 * v4)), v5 * v5)
    eps = 1e-6 * vmax + 1e-10  # (s + eps)^2 must not underflow in float32
    a1 = 0.1 / (s1 + eps) ** 2
    a2 = 0.6 / (s2 + eps) ** 2
    a3 = 0.3 / (s3 + eps) ** 2
    return (a1 * p1 + a2 * p2 + a3 * p3) / (a1 + a2 + a3)


def _minmod(a, b):
    return torch.where(a * b > 0, torch.where(a.abs() < b.abs(), a, b), torch.zeros_like(a))


def _cube_plane_fraction(s, a, b, c):
    M = 0.5 * (a + b + c)
    y = M - s
    flip = y > M
    y = torch.where(flip, 2 * M - y, y).clamp_min(0)
    tiny = 1e-12  # keeps b*c a normal float32 (no flush-to-zero 0/0)
    bs = b.clamp_min(tiny)
    cs = c.clamp_min(tiny)
    q = 3 * y * y - 3 * y * c + c * c
    H1 = y ** 3 / (bs * cs)
    H2 = q / bs
    u = (y - b).clamp_min(0)
    H3 = (q - u ** 3 / cs) / bs
    H4 = 3 * (2 * y - b - c)
    H = torch.where(y <= c, H1, torch.where(y <= b, H2, torch.where(y <= b + c, H3, H4)))
    ua = (y - a).clamp_min(0)
    F = H / (6 * a) - ua ** 3 / (6 * a * bs * cs)
    F = torch.where(flip, 1 - F, F)
    return F.clamp(0, 1)


def _sort3_abs(gx, gy, gz):
    """Sorted absolute values a >= b >= c of three tensors."""
    x, y, z = gx.abs(), gy.abs(), gz.abs()
    hi = torch.maximum(torch.maximum(x, y), z)
    lo = torch.minimum(torch.minimum(x, y), z)
    mid = x + y + z - hi - lo
    return hi, mid, lo


def _cubic_w(t):
    return (-t * (t - 1) * (t - 2) / 6, (t + 1) * (t - 1) * (t - 2) / 2,
            -(t + 1) * t * (t - 2) / 2, (t + 1) * t * (t - 1) / 6)


def _cubic_dw(t):
    return (-(3 * t * t - 6 * t + 2) / 6, (3 * t * t - 4 * t - 1) / 2,
            -(3 * t * t - 2 * t - 2) / 2, (3 * t * t - 1) / 6)


# ------------------------------------------------------------------ kernels
@kernel
def godunov_inward(flat, idx, sx, sy, sz, inv_h):
    """Godunov |grad phi| for phi_t = V|grad phi|, V >= 0, with WENO5 one-sided derivatives.

    |grad phi|^2 = sum_axes max( min(D^- phi, 0)^2, max(D^+ phi, 0)^2 ).
    """
    c = flat[idx]
    g2 = torch.zeros_like(c)
    for s in (sx, sy, sz):
        m3 = flat[idx - 3 * s]
        m2 = flat[idx - 2 * s]
        m1 = flat[idx - s]
        p1 = flat[idx + s]
        p2 = flat[idx + 2 * s]
        p3 = flat[idx + 3 * s]
        d0 = (m2 - m3) * inv_h
        d1 = (m1 - m2) * inv_h
        d2 = (c - m1) * inv_h
        d3 = (p1 - c) * inv_h
        d4 = (p2 - p1) * inv_h
        d5 = (p3 - p2) * inv_h
        dm = _weno5(d0, d1, d2, d3, d4)
        dp = _weno5(d5, d4, d3, d2, d1)
        g2 = g2 + torch.maximum(torch.clamp(dm, max=0) ** 2, torch.clamp(dp, min=0) ** 2)
    return torch.sqrt(g2)


@kernel
def cell_fractions(flat, idx, sx, sy, sz):
    """Exact volume fraction of each node's cell below the tangent plane of phi."""
    c = flat[idx]
    gx = 0.5 * (flat[idx + sx] - flat[idx - sx])
    gy = 0.5 * (flat[idx + sy] - flat[idx - sy])
    gz = 0.5 * (flat[idx + sz] - flat[idx - sz])
    gn = torch.sqrt(gx * gx + gy * gy + gz * gz).clamp_min(1e-30)
    a, b, cc = _sort3_abs(gx / gn, gy / gn, gz / gn)
    return _cube_plane_fraction(c / gn, a, b, cc)


@kernel
def closest_points(flat, idx, ny, nz, hi_x, hi_y, hi_z, h):
    """First Newton projection of nodes onto the zero set, (m, 3) index coordinates.

    Step ``phi g/|g|^2`` with the central gradient g, its length limited to
    ``1.25 |phi|/h + 0.25`` cells (no effect for a signed distance function),
    then clamped to the region where tricubic stencils fit.
    """
    sx = ny * nz
    sy = nz
    c = flat[idx]
    gx = 0.5 * (flat[idx + sx] - flat[idx - sx])
    gy = 0.5 * (flat[idx + sy] - flat[idx - sy])
    gz = 0.5 * (flat[idx + 1] - flat[idx - 1])
    g2 = (gx * gx + gy * gy + gz * gz).clamp_min(1e-30)
    length = c.abs() / torch.sqrt(g2)
    cap = 1.25 * c.abs() / h + 0.25
    scale = (c / g2) * torch.clamp(cap / length.clamp_min(1e-30), max=1.0)
    i = (idx // sx).to(flat.dtype)
    j = ((idx // sy) % ny).to(flat.dtype)
    k = (idx % nz).to(flat.dtype)
    px = torch.minimum((i - scale * gx).clamp_min(1.0), hi_x)
    py = torch.minimum((j - scale * gy).clamp_min(1.0), hi_y)
    pz = torch.minimum((k - scale * gz).clamp_min(1.0), hi_z)
    return torch.stack([px, py, pz], 1)


@kernel
def tricubic(flat, ny, nz, p):
    """Tensor-product cubic Lagrange interpolation at index coordinates p (m, 3)."""
    fl = torch.floor(p)
    t = p - fl
    base = fl.to(torch.int64) - 1
    b0 = base[:, 0] * (ny * nz) + base[:, 1] * nz + base[:, 2]
    wx = _cubic_w(t[:, 0])
    wy = _cubic_w(t[:, 1])
    wz = _cubic_w(t[:, 2])
    out = torch.zeros_like(t[:, 0])
    for a in range(4):
        for b in range(4):
            wab = wx[a] * wy[b]
            off = a * ny * nz + b * nz
            for c in range(4):
                out = out + (wab * wz[c]) * flat[b0 + (off + c)]
    return out


@kernel
def tricubic_with_gradient(flat, ny, nz, p):
    """Tricubic value and gradient (per cell) at index coordinates p (m, 3)."""
    fl = torch.floor(p)
    t = p - fl
    base = fl.to(torch.int64) - 1
    b0 = base[:, 0] * (ny * nz) + base[:, 1] * nz + base[:, 2]
    wx, wy, wz = _cubic_w(t[:, 0]), _cubic_w(t[:, 1]), _cubic_w(t[:, 2])
    dx, dy, dz = _cubic_dw(t[:, 0]), _cubic_dw(t[:, 1]), _cubic_dw(t[:, 2])
    val = torch.zeros_like(t[:, 0])
    gx = torch.zeros_like(val)
    gy = torch.zeros_like(val)
    gz = torch.zeros_like(val)
    for a in range(4):
        for b in range(4):
            off = a * ny * nz + b * nz
            for c in range(4):
                v = flat[b0 + (off + c)]
                val = val + wx[a] * wy[b] * wz[c] * v
                gx = gx + dx[a] * wy[b] * wz[c] * v
                gy = gy + wx[a] * dy[b] * wz[c] * v
                gz = gz + wx[a] * wy[b] * dz[c] * v
    return val, torch.stack([gx, gy, gz], 1)


@kernel
def trilinear(flat, ny, nz, p):
    """Trilinear interpolation at index coordinates p (m, 3)."""
    fl = torch.floor(p)
    t = p - fl
    base = fl.to(torch.int64)
    b0 = base[:, 0] * (ny * nz) + base[:, 1] * nz + base[:, 2]
    out = torch.zeros_like(t[:, 0])
    for a in range(2):
        wa = t[:, 0] if a else 1 - t[:, 0]
        for b in range(2):
            wb = wa * (t[:, 1] if b else 1 - t[:, 1])
            for c in range(2):
                out = out + wb * (t[:, 2] if c else 1 - t[:, 2]) * flat[b0 + (a * ny * nz + b * nz + c)]
    return out


@kernel
def principal_curvatures(flat, idx, sx, sy, sz, h):
    """Principal curvatures (k1 >= k2) of the level sets through nodes (k = 1/R on a sphere)."""
    c = flat[idx]
    xp, xm = flat[idx + sx], flat[idx - sx]
    yp, ym = flat[idx + sy], flat[idx - sy]
    zp, zm = flat[idx + sz], flat[idx - sz]
    px, py, pz = (xp - xm) / (2 * h), (yp - ym) / (2 * h), (zp - zm) / (2 * h)
    pxx = (xp - 2 * c + xm) / (h * h)
    pyy = (yp - 2 * c + ym) / (h * h)
    pzz = (zp - 2 * c + zm) / (h * h)

    def mixed(a, b):
        return (flat[idx + a + b] - flat[idx + a - b] - flat[idx - a + b] + flat[idx - a - b]) / (4 * h * h)

    pxy, pxz, pyz = mixed(sx, sy), mixed(sx, sz), mixed(sy, sz)
    g2 = (px * px + py * py + pz * pz).clamp_min(1e-30)
    gn = torch.sqrt(g2)
    ksum = ((pyy + pzz) * px * px + (pxx + pzz) * py * py + (pxx + pyy) * pz * pz
            - 2 * (px * py * pxy + px * pz * pxz + py * pz * pyz)) / (g2 * gn)
    K = (px * px * (pyy * pzz - pyz * pyz) + py * py * (pxx * pzz - pxz * pxz)
         + pz * pz * (pxx * pyy - pxy * pxy)
         + 2 * px * py * (pxz * pyz - pxy * pzz) + 2 * py * pz * (pxy * pxz - pyz * pxx)
         + 2 * px * pz * (pxy * pyz - pxz * pyy)) / (g2 * g2)
    H = 0.5 * ksum
    r = torch.sqrt((H * H - K).clamp_min(0))
    return H + r, H - r


@kernel
def interface_mask(flat, idx, sx, sy, sz):
    """True for nodes with a 6-neighbour of the opposite (or zero) sign."""
    c = flat[idx]
    hit = torch.zeros_like(c, dtype=torch.bool)
    for s in (sx, sy, sz):
        hit = hit | (flat[idx + s] * c <= 0) | (flat[idx - s] * c <= 0)
    return hit


def _interface_distance(p0, p1, dd, h):
    """Distance from a node (value p0) to the zero of phi towards its neighbour (p1).

    Quadratic ENO interpolation (Min 2010): phi(xi) = c0 + c1 xi + c2 xi^2 on
    the cell [-h/2, h/2] between the two nodes with c2 = dd / (2 h^2), where
    ``dd`` is the (minmod-limited) second difference.  Falls back to linear
    interpolation when the quadratic has no admissible root.
    """
    den = p0 - p1
    lin = h * p0 / torch.where(den != 0, den, torch.full_like(den, 1e-30))
    c2 = 0.5 * dd / (h * h)
    c1 = (p1 - p0) / h
    c0 = 0.5 * (p0 + p1) - c2 * h * h / 4
    disc = c1 * c1 - 4 * c2 * c0
    sq = torch.sqrt(disc.clamp_min(0))
    q = -0.5 * (c1 + torch.where(c1 >= 0, sq, -sq))
    xi = c0 / torch.where(q != 0, q, torch.full_like(q, 1e-30))
    quad = h / 2 + xi
    ok = (disc >= 0) & (dd.abs() > 1e-10 * h * h) & (quad >= 0) & (quad <= h)
    return torch.minimum(torch.where(ok, quad, lin).clamp_min(0), h)


@kernel
def reinit_geometry(flat, idx, sx, sy, sz, h, cfl):
    """Per-axis interface crossings and subcell distances from the initial phi.

    Returns (sgn, dtau, cp_x, cm_x, xp_x, xm_x, cp_y, ..., xm_z) where cp/cm
    flag a sign change towards the +/- neighbour and xp/xm are the distances
    to the interface along the axis (Min 2010).
    """
    c = flat[idx]
    sgn = torch.sign(c)
    hmin = torch.zeros_like(c) + h
    out = []
    for s in (sx, sy, sz):
        m2, m1, p1, p2 = flat[idx - 2 * s], flat[idx - s], flat[idx + s], flat[idx + 2 * s]
        d_m1 = c - 2 * m1 + m2
        d_0 = p1 - 2 * c + m1
        d_p1 = p2 - 2 * p1 + c
        cp = (c * p1) < 0
        cm = (c * m1) < 0
        xp = _interface_distance(c, p1, _minmod(d_0, d_p1), h)
        xm = _interface_distance(c, m1, _minmod(d_0, d_m1), h)
        hmin = torch.minimum(hmin, torch.where(cp, xp, hmin))
        hmin = torch.minimum(hmin, torch.where(cm, xm, hmin))
        out += [cp, cm, torch.maximum(xp, 1e-6 * h), torch.maximum(xm, 1e-6 * h)]
    dtau = cfl * torch.maximum(hmin, 1e-3 * h)
    return (sgn, dtau, *out)


@kernel
def reinit_rhs(flat, idx, sx, sy, sz, h, sgn,
               cpx, cmx, xpx, xmx, cpy, cmy, xpy, xmy, cpz, cmz, xpz, xmz):
    """-sgn(phi0) (|grad phi| - 1) with ENO2 + subcell-resolution one-sided differences."""
    c = flat[idx]
    g2 = torch.zeros_like(c)
    outside = sgn > 0
    for s, cp, cm, xp, xm in ((sx, cpx, cmx, xpx, xmx), (sy, cpy, cmy, xpy, xmy),
                              (sz, cpz, cmz, xpz, xmz)):
        m2, m1, p1, p2 = flat[idx - 2 * s], flat[idx - s], flat[idx + s], flat[idx + 2 * s]
        mm_p = _minmod(p1 - 2 * c + m1, p2 - 2 * p1 + c)
        mm_m = _minmod(p1 - 2 * c + m1, c - 2 * m1 + m2)
        Dp = torch.where(cp, -c / xp - 0.5 * xp * mm_p / (h * h), (p1 - c - 0.5 * mm_p) / h)
        Dm = torch.where(cm, c / xm + 0.5 * xm * mm_m / (h * h), (c - m1 + 0.5 * mm_m) / h)
        pos = torch.maximum(torch.clamp(Dm, min=0) ** 2, torch.clamp(Dp, max=0) ** 2)
        neg = torch.maximum(torch.clamp(Dm, max=0) ** 2, torch.clamp(Dp, min=0) ** 2)
        g2 = g2 + torch.where(outside, pos, neg)
    return -sgn * (torch.sqrt(g2) - 1.0)


def _point_triangle_distance2(p, a, b, c):
    ab = b - a
    ac = c - a
    ap = p - a
    bp = p - b
    cp = p - c
    d1 = (ab * ap).sum(-1)
    d2 = (ac * ap).sum(-1)
    d3 = (ab * bp).sum(-1)
    d4 = (ac * bp).sum(-1)
    d5 = (ab * cp).sum(-1)
    d6 = (ac * cp).sum(-1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    tiny = 1e-30
    denom = va + vb + vc
    denom = torch.where(denom.abs() > tiny, denom, torch.full_like(denom, tiny))
    v = vb / denom
    w = vc / denom
    # regions in reverse priority order, so that higher priority overrides
    e1 = d4 - d3
    e2 = d5 - d6
    in_bc = (va <= 0) & (e1 >= 0) & (e2 >= 0)
    wbc = e1 / (e1 + e2).clamp_min(tiny)
    v = torch.where(in_bc, 1 - wbc, v)
    w = torch.where(in_bc, wbc, w)
    in_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    wac = d2 / (d2 - d6).clamp_min(tiny)
    v = torch.where(in_ac, torch.zeros_like(v), v)
    w = torch.where(in_ac, wac, w)
    in_c = (d6 >= 0) & (d5 <= d6)
    v = torch.where(in_c, torch.zeros_like(v), v)
    w = torch.where(in_c, torch.ones_like(w), w)
    in_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    vab = d1 / (d1 - d3).clamp_min(tiny)
    v = torch.where(in_ab, vab, v)
    w = torch.where(in_ab, torch.zeros_like(w), w)
    in_b = (d3 >= 0) & (d4 <= d3)
    v = torch.where(in_b, torch.ones_like(v), v)
    w = torch.where(in_b, torch.zeros_like(w), w)
    in_a = (d1 <= 0) & (d2 <= 0)
    v = torch.where(in_a, torch.zeros_like(v), v)
    w = torch.where(in_a, torch.zeros_like(w), w)
    q = a + v.unsqueeze(-1) * ab + w.unsqueeze(-1) * ac
    return ((p - q) ** 2).sum(-1)


point_triangle_distance2_compiled = kernel(_point_triangle_distance2)
