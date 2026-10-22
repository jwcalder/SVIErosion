"""Watertight, outward-oriented triangle meshes of simple test solids.

Every generator returns a :class:`~svierosion.meshio.TriMesh`.  Prismatic
solids (boxes, triangular prisms, L-blocks, cylinders, broken-tube fragments)
are built by extruding a simple polygon, so their faces are exactly planar or
exactly the faceted surface that is requested.
"""

from __future__ import annotations

import numpy as np

from .meshio import TriMesh


# --------------------------------------------------------------------- helpers
def _polygon_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _ear_clip(poly: np.ndarray) -> np.ndarray:
    """Triangulate a simple counter-clockwise polygon; returns (n-2, 3) indices."""
    n = len(poly)
    idx = list(range(n))
    tris = []

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    guard = 0
    while len(idx) > 3:
        guard += 1
        if guard > 10 * n * n:
            raise ValueError("ear clipping failed; polygon is not simple")
        m = len(idx)
        for k in range(m):
            i0, i1, i2 = idx[(k - 1) % m], idx[k], idx[(k + 1) % m]
            a, b, c = poly[i0], poly[i1], poly[i2]
            if cross(a, b, c) <= 1e-14:
                continue  # reflex or degenerate vertex
            ear = True
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                p = poly[j]
                if cross(a, b, p) >= 0 and cross(b, c, p) >= 0 and cross(c, a, p) >= 0:
                    ear = False
                    break
            if ear:
                tris.append((i0, i1, i2))
                idx.pop(k)
                break
    tris.append(tuple(idx))
    return np.asarray(tris, dtype=np.int64)


def extrude_polygon(poly2d, height: float, z0: float | None = None) -> TriMesh:
    """Extrude a simple polygon in the (x, y) plane along z.

    The polygon may be given in either orientation; the result is a closed,
    outward-oriented prism occupying ``z0 <= z <= z0 + height`` (centred on
    z = 0 by default).
    """
    p = np.asarray(poly2d, dtype=np.float64)
    if _polygon_area(p) < 0:
        p = p[::-1]
    n = len(p)
    z0 = -0.5 * height if z0 is None else z0
    bottom = np.column_stack([p, np.full(n, z0)])
    top = np.column_stack([p, np.full(n, z0 + height)])
    verts = np.vstack([bottom, top])
    cap = _ear_clip(p)
    faces = [cap[:, ::-1], cap + n]  # bottom faces down, top faces up
    i = np.arange(n)
    j = (i + 1) % n
    faces.append(np.column_stack([i, j, j + n]))
    faces.append(np.column_stack([i, j + n, i + n]))
    return TriMesh(verts, np.vstack(faces))


def _check(mesh: TriMesh) -> TriMesh:
    if mesh.signed_volume() < 0:
        mesh = mesh.flipped()
    return mesh


# ------------------------------------------------------------------ polyhedra
def box(size=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0)) -> TriMesh:
    """Axis-aligned box with the given edge lengths (12 triangles)."""
    sx, sy, sz = np.asarray(size, dtype=np.float64) / 2
    m = extrude_polygon([(-sx, -sy), (sx, -sy), (sx, sy), (-sx, sy)], 2 * sz)
    m.vertices += np.asarray(center, dtype=np.float64)
    return _check(m)


def cube(edge: float = 1.0, center=(0.0, 0.0, 0.0)) -> TriMesh:
    return box((edge, edge, edge), center)


def triangular_prism(side: float = 1.0, length: float = 1.0) -> TriMesh:
    """Prism over an equilateral triangle (side ``side``, centroid at origin) along z."""
    ang = np.pi / 2 + 2 * np.pi * np.arange(3) / 3
    rad = side / np.sqrt(3.0)
    return _check(extrude_polygon(np.column_stack([rad * np.cos(ang), rad * np.sin(ang)]), length))


def l_block(a: float = 1.0, b: float = 0.5, thickness: float = 0.5) -> TriMesh:
    """L-shaped block: an a x a square minus its (a-b) x (a-b) upper-right corner.

    It has one concave (re-entrant) edge, which an exposure-driven flow should
    erode much more slowly than the convex edges.
    """
    h = a / 2
    poly = [(-h, -h), (h, -h), (h, -h + b), (-h + b, -h + b), (-h + b, h), (-h, h)]
    return _check(extrude_polygon(poly, thickness))


def cylinder(radius: float = 0.5, height: float = 1.0, segments: int = 256) -> TriMesh:
    t = 2 * np.pi * np.arange(segments) / segments
    return _check(extrude_polygon(np.column_stack([radius * np.cos(t), radius * np.sin(t)]), height))


def tube_fragment(r_in: float = 0.6, r_out: float = 1.0, angle: float = 2.2,
                  length: float = 1.6, segments: int = 160) -> TriMesh:
    """A piece of a thick-walled tube (a crude "long-bone shaft fragment").

    Annular sector ``r_in <= rho <= r_out``, ``|theta| <= angle/2``, extruded
    over ``|z| <= length/2``.  It has convex outer surface, concave inner
    (medullary) surface and sharp break edges.
    """
    t = np.linspace(-angle / 2, angle / 2, segments + 1)
    outer = np.column_stack([r_out * np.cos(t), r_out * np.sin(t)])
    inner = np.column_stack([r_in * np.cos(t[::-1]), r_in * np.sin(t[::-1])])
    m = extrude_polygon(np.vstack([outer, inner]), length)
    m.vertices[:, 0] -= 0.5 * (r_in + r_out)
    return _check(m)


# ------------------------------------------------------------- smooth solids
def icosphere(subdivisions: int = 5, radius: float = 1.0) -> TriMesh:
    """Subdivided icosahedron projected to the sphere (20 * 4**s triangles)."""
    p = (1 + 5 ** 0.5) / 2
    v = np.array([[-1, p, 0], [1, p, 0], [-1, -p, 0], [1, -p, 0],
                  [0, -1, p], [0, 1, p], [0, -1, -p], [0, 1, -p],
                  [p, 0, -1], [p, 0, 1], [-p, 0, -1], [-p, 0, 1]], dtype=np.float64)
    f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                  [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                  [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                  [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]], dtype=np.int64)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    for _ in range(subdivisions):
        e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
        uniq, inv = np.unique(e, axis=0, return_inverse=True)
        mid = v[uniq[:, 0]] + v[uniq[:, 1]]
        mid /= np.linalg.norm(mid, axis=1, keepdims=True)
        m = len(f)
        a = inv[:m] + len(v)
        b = inv[m:2 * m] + len(v)
        c = inv[2 * m:] + len(v)
        v = np.vstack([v, mid])
        f = np.vstack([np.column_stack([f[:, 0], a, c]), np.column_stack([f[:, 1], b, a]),
                       np.column_stack([f[:, 2], c, b]), np.column_stack([a, b, c])])
    return _check(TriMesh(radius * v, f))


def ellipsoid(radii=(1.0, 0.7, 0.5), subdivisions: int = 5) -> TriMesh:
    m = icosphere(subdivisions, 1.0)
    m.vertices *= np.asarray(radii, dtype=np.float64)
    return _check(m)


def torus(major: float = 1.0, minor: float = 0.35, n_major: int = 192, n_minor: int = 96) -> TriMesh:
    u = 2 * np.pi * np.arange(n_major) / n_major
    w = 2 * np.pi * np.arange(n_minor) / n_minor
    U, W = np.meshgrid(u, w, indexing="ij")
    x = (major + minor * np.cos(W)) * np.cos(U)
    y = (major + minor * np.cos(W)) * np.sin(U)
    z = minor * np.sin(W)
    verts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    i, j = np.meshgrid(np.arange(n_major), np.arange(n_minor), indexing="ij")
    a = i * n_minor + j
    b = ((i + 1) % n_major) * n_minor + j
    c = ((i + 1) % n_major) * n_minor + (j + 1) % n_minor
    d = i * n_minor + (j + 1) % n_minor
    faces = np.vstack([np.column_stack([a.ravel(), b.ravel(), c.ravel()]),
                       np.column_stack([a.ravel(), c.ravel(), d.ravel()])])
    return _check(TriMesh(verts, faces))


def subdivide(mesh: TriMesh, times: int = 1) -> TriMesh:
    """Midpoint (1-to-4) subdivision of an indexed mesh; keeps it watertight."""
    v, f = mesh.vertices.copy(), mesh.faces.copy()
    for _ in range(times):
        e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
        uniq, inv = np.unique(e, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        mid = 0.5 * (v[uniq[:, 0]] + v[uniq[:, 1]])
        m = len(f)
        a, b, c = inv[:m] + len(v), inv[m:2 * m] + len(v), inv[2 * m:] + len(v)
        v = np.vstack([v, mid])
        f = np.vstack([np.column_stack([f[:, 0], a, c]), np.column_stack([f[:, 1], b, a]),
                       np.column_stack([f[:, 2], c, b]), np.column_stack([a, b, c])])
    return TriMesh(v, f)


def vertex_normals(mesh: TriMesh) -> np.ndarray:
    n = mesh.face_normals(normalize=False)
    vn = np.zeros_like(mesh.vertices)
    for i in range(3):
        np.add.at(vn, mesh.faces[:, i], n)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-300)


def perturb(mesh: TriMesh, amplitude: float, wavelength: float, modes: int = 48,
            seed: int = 0) -> TriMesh:
    """Displace vertices along their normals by a smooth random field.

    The field is a sum of ``modes`` random plane waves with wavelengths around
    ``wavelength``, scaled to maximum ``amplitude``; useful to emulate the
    surface texture of scanned objects.
    """
    rng = np.random.default_rng(seed)
    k = rng.normal(size=(modes, 3))
    k *= (2 * np.pi / wavelength) * rng.uniform(0.6, 1.6, size=(modes, 1)) / np.linalg.norm(k, axis=1, keepdims=True)
    ph = rng.uniform(0, 2 * np.pi, modes)
    field = np.cos(mesh.vertices @ k.T + ph).sum(1)
    field *= amplitude / np.abs(field).max()
    return TriMesh(mesh.vertices + field[:, None] * vertex_normals(mesh), mesh.faces.copy())


def bone_like(subdivisions: int = 4, seed: int = 0) -> TriMesh:
    """A finely tessellated tube fragment with a rough surface (a synthetic 'scan')."""
    m = tube_fragment(segments=96)
    lo, hi = m.bounds
    L = float((hi - lo).max())
    m = subdivide(m, subdivisions)
    return _check(perturb(m, 0.006 * L, 0.06 * L, seed=seed))


SHAPES = {
    "cube": lambda: cube(1.0),
    "box": lambda: box((1.6, 1.0, 0.6)),
    "prism": lambda: triangular_prism(1.2, 1.2),
    "ellipsoid": lambda: ellipsoid((0.8, 0.55, 0.4), 6),
    "sphere": lambda: icosphere(6, 0.6),
    "lblock": lambda: l_block(1.2, 0.6, 0.6),
    "cylinder": lambda: cylinder(0.45, 1.2),
    "torus": lambda: torus(0.6, 0.25),
    "tube_fragment": lambda: tube_fragment(),
}


# -------------------------------------------------------- analytic distances
def sphere_sdf(points, center=(0.0, 0.0, 0.0), radius: float = 1.0):
    p = np.asarray(points, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    return np.linalg.norm(p, axis=-1) - radius


def box_sdf(points, size=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0)):
    """Exact signed distance to an axis-aligned box (negative inside)."""
    q = np.abs(np.asarray(points, dtype=np.float64) - np.asarray(center)) - np.asarray(size) / 2
    outside = np.linalg.norm(np.maximum(q, 0), axis=-1)
    inside = np.minimum(q.max(axis=-1), 0)
    return outside + inside
