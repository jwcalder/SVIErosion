"""Reading and writing triangle meshes in the PLY format.

Self-contained (numpy only).  Supports ``ascii``, ``binary_little_endian`` and
``binary_big_endian`` files, arbitrary extra vertex/face properties (ignored on
read unless requested), and polygonal faces (fan-triangulated on read).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_PLY_TYPES = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4",
    "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
}


@dataclass
class _Property:
    name: str
    dtype: str                      # numpy type code without byte order
    count_dtype: str | None = None  # set for list properties


@dataclass
class _Element:
    name: str
    count: int
    properties: list[_Property] = field(default_factory=list)


@dataclass
class TriMesh:
    """A triangle mesh: ``vertices`` (n, 3) float64 and ``faces`` (m, 3) int64.

    ``vertex_data`` holds optional per-vertex scalar arrays (e.g. colours,
    curvature, exposure) keyed by PLY property name.
    """

    vertices: np.ndarray
    faces: np.ndarray
    vertex_data: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        self.vertices = np.ascontiguousarray(self.vertices, dtype=np.float64)
        self.faces = np.ascontiguousarray(self.faces, dtype=np.int64)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (n, 3)")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("faces must have shape (m, 3)")

    # ------------------------------------------------------------------ geometry
    @property
    def bounds(self) -> np.ndarray:
        """(2, 3) array with the minimum and maximum corner."""
        return np.stack([self.vertices.min(0), self.vertices.max(0)])

    def triangles(self) -> np.ndarray:
        return self.vertices[self.faces]

    def face_normals(self, normalize: bool = True) -> np.ndarray:
        t = self.triangles()
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        if normalize:
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
        return n

    def area(self) -> float:
        return 0.5 * float(np.linalg.norm(self.face_normals(False), axis=1).sum())

    def signed_volume(self) -> float:
        """Enclosed volume (divergence theorem); positive for outward orientation."""
        t = self.triangles() - self.vertices.mean(0)
        return float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0)

    def volume(self) -> float:
        return abs(self.signed_volume())

    def edge_topology(self) -> dict:
        """Counts of boundary, manifold and non-manifold edges, and orientation consistency."""
        f = self.faces
        e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        key_sorted = np.sort(e, axis=1)
        n = max(int(self.vertices.shape[0]), 1)
        keys = key_sorted[:, 0] * n + key_sorted[:, 1]
        uniq, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
        # orientation: each interior edge must appear once in each direction
        direction = np.where(e[:, 0] < e[:, 1], 1, -1)
        dir_sum = np.bincount(inv, weights=direction, minlength=len(uniq))
        manifold = counts == 2
        return {
            "edges": int(len(uniq)),
            "boundary_edges": int(np.sum(counts == 1)),
            "nonmanifold_edges": int(np.sum(counts > 2)),
            "misoriented_edges": int(np.sum(manifold & (dir_sum != 0))),
        }

    def is_watertight(self) -> bool:
        t = self.edge_topology()
        return t["boundary_edges"] == 0 and t["nonmanifold_edges"] == 0

    def is_consistently_oriented(self) -> bool:
        return self.edge_topology()["misoriented_edges"] == 0

    def remove_unreferenced_vertices(self) -> "TriMesh":
        used = np.unique(self.faces)
        remap = -np.ones(len(self.vertices), dtype=np.int64)
        remap[used] = np.arange(len(used))
        data = {k: v[used] for k, v in self.vertex_data.items()}
        return TriMesh(self.vertices[used], remap[self.faces], data)

    def remove_degenerate_faces(self) -> "TriMesh":
        f = self.faces
        ok = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
        return TriMesh(self.vertices, f[ok], dict(self.vertex_data))

    def flipped(self) -> "TriMesh":
        return TriMesh(self.vertices, self.faces[:, ::-1].copy(), dict(self.vertex_data))

    def copy(self) -> "TriMesh":
        return TriMesh(self.vertices.copy(), self.faces.copy(),
                       {k: v.copy() for k, v in self.vertex_data.items()})


# ---------------------------------------------------------------------- reading
def _parse_header(fh):
    first = fh.readline().strip()
    if first != b"ply":
        raise ValueError("not a PLY file (missing 'ply' magic)")
    fmt = None
    elements: list[_Element] = []
    while True:
        raw = fh.readline()
        if not raw:
            raise ValueError("unexpected end of file in PLY header")
        line = raw.decode("ascii", errors="replace").strip()
        if not line or line.startswith(("comment", "obj_info")):
            continue
        tok = line.split()
        if tok[0] == "format":
            fmt = tok[1]
        elif tok[0] == "element":
            elements.append(_Element(tok[1], int(tok[2])))
        elif tok[0] == "property":
            if not elements:
                raise ValueError("PLY property before any element")
            if tok[1] == "list":
                elements[-1].properties.append(
                    _Property(tok[4], _PLY_TYPES[tok[3]], _PLY_TYPES[tok[2]]))
            else:
                elements[-1].properties.append(_Property(tok[2], _PLY_TYPES[tok[1]]))
        elif tok[0] == "end_header":
            break
        else:
            raise ValueError(f"unrecognised PLY header line: {line!r}")
    if fmt not in ("ascii", "binary_little_endian", "binary_big_endian"):
        raise ValueError(f"unsupported PLY format {fmt!r}")
    return fmt, elements


def _read_binary_element(buf: memoryview, pos: int, el: _Element, endian: str):
    """Return ({prop: array or list-of-arrays}, new_pos)."""
    props = el.properties
    if all(p.count_dtype is None for p in props):
        dt = np.dtype([(p.name, endian + p.dtype) for p in props])
        arr = np.frombuffer(buf, dtype=dt, count=el.count, offset=pos)
        return {p.name: arr[p.name] for p in props}, pos + el.count * dt.itemsize
    # Fast path: every list has a constant length equal to the first one's.
    try:
        lengths = []
        offset = pos
        fields = []
        for p in props:
            if p.count_dtype is None:
                fields.append((p.name, endian + p.dtype))
                offset += np.dtype(p.dtype).itemsize
            else:
                n0 = int(np.frombuffer(buf, dtype=endian + p.count_dtype, count=1, offset=offset)[0])
                lengths.append(n0)
                fields.append((p.name + "__n", endian + p.count_dtype))
                fields.append((p.name, endian + p.dtype, (n0,)))
                offset += np.dtype(p.count_dtype).itemsize + n0 * np.dtype(p.dtype).itemsize
        dt = np.dtype(fields)
        if pos + el.count * dt.itemsize <= len(buf):
            arr = np.frombuffer(buf, dtype=dt, count=el.count, offset=pos)
            ok = all(np.all(arr[p.name + "__n"] == n)
                     for p, n in zip([q for q in props if q.count_dtype], lengths))
            if ok:
                out = {p.name: arr[p.name] for p in props}
                return out, pos + el.count * dt.itemsize
    except (ValueError, IndexError):
        pass
    # General (slow) path for variable-length lists.
    out = {p.name: [] for p in props}
    for _ in range(el.count):
        for p in props:
            if p.count_dtype is None:
                val = np.frombuffer(buf, dtype=endian + p.dtype, count=1, offset=pos)[0]
                pos += np.dtype(p.dtype).itemsize
                out[p.name].append(val)
            else:
                n = int(np.frombuffer(buf, dtype=endian + p.count_dtype, count=1, offset=pos)[0])
                pos += np.dtype(p.count_dtype).itemsize
                val = np.frombuffer(buf, dtype=endian + p.dtype, count=n, offset=pos)
                pos += n * np.dtype(p.dtype).itemsize
                out[p.name].append(val)
    for p in props:
        if p.count_dtype is None:
            out[p.name] = np.asarray(out[p.name])
    return out, pos


def _read_ascii_element(lines, el: _Element):
    props = el.properties
    out = {p.name: [] for p in props}
    simple = all(p.count_dtype is None for p in props)
    if simple and el.count > 0:
        block = np.array([next(lines).split() for _ in range(el.count)], dtype=np.float64)
        for i, p in enumerate(props):
            out[p.name] = block[:, i].astype(p.dtype)
        return out
    for _ in range(el.count):
        tok = next(lines).split()
        k = 0
        for p in props:
            if p.count_dtype is None:
                out[p.name].append(float(tok[k]))
                k += 1
            else:
                n = int(float(tok[k]))
                out[p.name].append(np.array(tok[k + 1:k + 1 + n], dtype=np.float64).astype(p.dtype))
                k += 1 + n
    for p in props:
        if p.count_dtype is None:
            out[p.name] = np.asarray(out[p.name], dtype=p.dtype)
    return out


def _faces_from_lists(lists) -> np.ndarray:
    if isinstance(lists, np.ndarray) and lists.ndim == 2:
        polys = lists.astype(np.int64)
        if polys.shape[1] == 3:
            return polys
        tris = [np.stack([polys[:, 0], polys[:, i], polys[:, i + 1]], 1)
                for i in range(1, polys.shape[1] - 1)]
        return np.concatenate(tris)
    tris = []
    for poly in lists:
        poly = np.asarray(poly, dtype=np.int64)
        for i in range(1, len(poly) - 1):
            tris.append((poly[0], poly[i], poly[i + 1]))
    return np.asarray(tris, dtype=np.int64).reshape(-1, 3)


def read_ply(path, vertex_properties: bool = True) -> TriMesh:
    """Read a PLY triangle mesh (polygons are fan-triangulated).

    Extra scalar vertex properties are returned in ``mesh.vertex_data`` when
    ``vertex_properties`` is true.
    """
    path = Path(path)
    with open(path, "rb") as fh:
        fmt, elements = _parse_header(fh)
        data_start = fh.tell()
    data: dict[str, dict] = {}
    if fmt == "ascii":
        with open(path, "rb") as fh:
            fh.seek(data_start)
            text = fh.read().decode("ascii", errors="replace")
        lines = iter([ln for ln in text.splitlines() if ln.strip()])
        for el in elements:
            data[el.name] = _read_ascii_element(lines, el)
    else:
        endian = "<" if fmt == "binary_little_endian" else ">"
        raw = path.read_bytes()
        buf = memoryview(raw)
        pos = data_start
        for el in elements:
            data[el.name], pos = _read_binary_element(buf, pos, el, endian)
    if "vertex" not in data:
        raise ValueError("PLY file has no vertex element")
    v = data["vertex"]
    verts = np.stack([np.asarray(v[c], dtype=np.float64) for c in ("x", "y", "z")], 1)
    faces = np.zeros((0, 3), dtype=np.int64)
    if "face" in data:
        fd = data["face"]
        key = "vertex_indices" if "vertex_indices" in fd else (
            "vertex_index" if "vertex_index" in fd else None)
        if key is None:
            raise ValueError("PLY face element has no vertex_indices/vertex_index list")
        faces = _faces_from_lists(fd[key])
    extra = {}
    if vertex_properties:
        for name, arr in v.items():
            if name in ("x", "y", "z"):
                continue
            arr = np.asarray(arr)
            if arr.ndim == 1 and arr.dtype != object:
                extra[name] = arr.copy()
    return TriMesh(verts, faces, extra)


# ---------------------------------------------------------------------- writing
def write_ply(path, mesh: TriMesh, binary: bool = True, precision: str = "float",
              vertex_data: dict | None = None, comments: list[str] | None = None) -> None:
    """Write a triangle mesh to PLY.

    Parameters
    ----------
    precision : ``"float"`` (32-bit, most compatible) or ``"double"``.
    vertex_data : optional extra per-vertex scalars written as float properties
        (merged with ``mesh.vertex_data``).  A property named ``quality`` is
        displayed as a colour map by MeshLab.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = dict(mesh.vertex_data)
    if vertex_data:
        extra.update(vertex_data)
    nv, nf = len(mesh.vertices), len(mesh.faces)
    vtype, vcode = ("double", "f8") if precision == "double" else ("float", "f4")
    header = ["ply", "format " + ("binary_little_endian 1.0" if binary else "ascii 1.0")]
    for c in comments or []:
        header.append("comment " + c.replace("\n", " "))
    header += [f"element vertex {nv}"] + [f"property {vtype} {c}" for c in "xyz"]
    names = []
    for name, arr in extra.items():
        arr = np.asarray(arr)
        if arr.shape != (nv,):
            raise ValueError(f"vertex property {name!r} must have shape ({nv},)")
        names.append(name)
        header.append(f"property float {name}")
    header += [f"element face {nf}", "property list uchar int vertex_indices", "end_header"]
    with open(path, "wb") as fh:
        fh.write(("\n".join(header) + "\n").encode("ascii"))
        if binary:
            vdt = np.dtype([(c, "<" + vcode) for c in "xyz"] + [(n, "<f4") for n in names])
            va = np.empty(nv, dtype=vdt)
            for i, c in enumerate("xyz"):
                va[c] = mesh.vertices[:, i]
            for n in names:
                va[n] = np.asarray(extra[n], dtype=np.float32)
            fh.write(va.tobytes())
            fdt = np.dtype([("n", "u1"), ("v", "<i4", (3,))])
            fa = np.empty(nf, dtype=fdt)
            fa["n"] = 3
            fa["v"] = mesh.faces.astype(np.int32)
            fh.write(fa.tobytes())
        else:
            cols = [mesh.vertices] + [np.asarray(extra[n], dtype=np.float64)[:, None] for n in names]
            np.savetxt(fh, np.hstack(cols), fmt="%.17g" if precision == "double" else "%.9g")
            np.savetxt(fh, np.hstack([np.full((nf, 1), 3), mesh.faces]), fmt="%d")

