"""Rendering of meshes (VTK offscreen, matplotlib fallback) and cross sections."""

from __future__ import annotations

import numpy as np

from .meshio import TriMesh


# ------------------------------------------------------------------ cameras
def default_camera(mesh: TriMesh, azimuth: float = -55.0, elevation: float = 25.0,
                   distance: float = 2.6) -> dict:
    lo, hi = mesh.bounds
    c = 0.5 * (lo + hi)
    ext = float(np.linalg.norm(hi - lo))
    a, e = np.radians(azimuth), np.radians(elevation)
    d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    return {"focal": c, "position": c + distance * ext * d / 2 * 1.4, "up": (0.0, 0.0, 1.0),
            "scale": ext}


# --------------------------------------------------------------------- VTK
def _vtk_available() -> bool:
    try:
        import vtk  # noqa: F401
        return True
    except Exception:  # pragma: no cover
        return False


def _to_polydata(mesh: TriMesh, scalars=None):
    import vtk
    from vtk.util import numpy_support as ns
    pts = vtk.vtkPoints()
    pts.SetData(ns.numpy_to_vtk(np.ascontiguousarray(mesh.vertices, dtype=np.float64), deep=True))
    f = mesh.faces.astype(np.int64)
    cells = np.hstack([np.full((len(f), 1), 3, dtype=np.int64), f]).ravel()
    ca = vtk.vtkCellArray()
    ca.SetCells(len(f), ns.numpy_to_vtkIdTypeArray(cells, deep=True))
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetPolys(ca)
    if scalars is not None:
        arr = ns.numpy_to_vtk(np.ascontiguousarray(scalars, dtype=np.float64), deep=True)
        arr.SetName("s")
        pd.GetPointData().SetScalars(arr)
    return pd


def render_mesh(mesh: TriMesh, scalars=None, cmap: str = "YlOrRd", clim=None, camera=None,
                size=(900, 700), color=(0.86, 0.84, 0.78), edges: bool = False) -> np.ndarray:
    """Render one mesh to an RGB array (VTK offscreen; matplotlib if VTK fails)."""
    if camera is None:
        camera = default_camera(mesh)
    if _vtk_available():
        try:
            return _render_vtk(mesh, scalars, cmap, clim, camera, size, color, edges)
        except Exception:  # pragma: no cover - e.g. no OpenGL context
            pass
    return _render_mpl(mesh, scalars, cmap, clim, camera, size, color)


def _render_vtk(mesh, scalars, cmap, clim, camera, size, color, edges):
    import vtk
    from vtk.util import numpy_support as ns
    import matplotlib

    pd = _to_polydata(mesh, scalars)
    nrm = vtk.vtkPolyDataNormals()
    nrm.SetInputData(pd)
    nrm.SetFeatureAngle(50.0)
    nrm.SplittingOn()
    nrm.ConsistencyOn()
    nrm.Update()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(nrm.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetAmbient(0.12)
    prop.SetDiffuse(0.85)
    prop.SetSpecular(0.15)
    prop.SetSpecularPower(25)
    if scalars is not None:
        lo, hi = clim if clim is not None else (float(np.min(scalars)), float(np.max(scalars)))
        if hi <= lo:
            hi = lo + 1e-12
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        cm = matplotlib.colormaps[cmap]
        for i in range(256):
            r, g, b, _ = cm(i / 255)
            lut.SetTableValue(i, r, g, b, 1.0)
        lut.SetRange(lo, hi)
        lut.Build()
        mapper.SetLookupTable(lut)
        mapper.SetScalarRange(lo, hi)
        mapper.ScalarVisibilityOn()
    else:
        mapper.ScalarVisibilityOff()
        prop.SetColor(*color)
    if edges:
        prop.EdgeVisibilityOn()
        prop.SetEdgeColor(0.2, 0.2, 0.2)
    ren = vtk.vtkRenderer()
    ren.AddActor(actor)
    ren.SetBackground(1, 1, 1)
    kit = vtk.vtkLightKit()           # key/fill/back/head lights relative to the camera
    kit.SetKeyLightIntensity(0.85)
    kit.AddLightsToRenderer(ren)
    rw = vtk.vtkRenderWindow()
    rw.SetOffScreenRendering(1)
    rw.SetMultiSamples(8)
    rw.AddRenderer(ren)
    rw.SetSize(*size)
    cam = ren.GetActiveCamera()
    cam.SetFocalPoint(*camera["focal"])
    cam.SetPosition(*camera["position"])
    cam.SetViewUp(*camera["up"])
    cam.SetViewAngle(30)
    ren.ResetCameraClippingRange()
    rw.Render()
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(rw)
    w2i.ReadFrontBufferOff()
    w2i.Update()
    img = w2i.GetOutput()
    w, h, _ = img.GetDimensions()
    arr = ns.vtk_to_numpy(img.GetPointData().GetScalars()).reshape(h, w, -1)[::-1, :, :3]
    rw.Finalize()
    return np.ascontiguousarray(arr)


def _render_mpl(mesh, scalars, cmap, clim, camera, size, color):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(size[0] / 100, size[1] / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    tri = mesh.vertices[mesh.faces]
    n = mesh.face_normals()
    view = np.asarray(camera["position"]) - np.asarray(camera["focal"])
    view /= np.linalg.norm(view)
    shade = 0.35 + 0.65 * np.clip(n @ view, 0, 1)
    if scalars is not None:
        lo, hi = clim if clim is not None else (np.min(scalars), np.max(scalars))
        fc = matplotlib.colormaps[cmap]((np.asarray(scalars)[mesh.faces].mean(1) - lo) / max(hi - lo, 1e-12))
        fc[:, :3] *= shade[:, None]
    else:
        fc = np.tile(np.r_[color, 1.0], (len(tri), 1))
        fc[:, :3] *= shade[:, None]
    ax.add_collection3d(Poly3DCollection(tri, facecolors=fc, linewidths=0))
    lo, hi = mesh.bounds
    c = 0.5 * (lo + hi)
    r = 0.5 * float((hi - lo).max())
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    az = np.degrees(np.arctan2(view[1], view[0]))
    el = np.degrees(np.arcsin(view[2]))
    ax.view_init(elev=el, azim=az)
    ax.set_axis_off()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return img


# ------------------------------------------------------------ cross sections
def slice_mesh(mesh: TriMesh, origin, normal) -> np.ndarray:
    """Segments (n, 2, 3) of the intersection of a mesh with a plane."""
    n = np.asarray(normal, dtype=np.float64)
    n /= np.linalg.norm(n)
    d = (mesh.vertices - np.asarray(origin, dtype=np.float64)) @ n
    f = mesh.faces
    df = d[f]
    s = np.sign(df)
    cross = ~((s[:, 0] == s[:, 1]) & (s[:, 1] == s[:, 2]))
    segs = []
    for t, dd in zip(f[cross], df[cross]):
        pts = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if (dd[a] > 0) != (dd[b] > 0):
                w = dd[a] / (dd[a] - dd[b])
                pts.append(mesh.vertices[t[a]] + w * (mesh.vertices[t[b]] - mesh.vertices[t[a]]))
        if len(pts) == 2:
            segs.append(pts)
    return np.asarray(segs).reshape(-1, 2, 3)


def plot_sections(ax, meshes, labels, colors, axis: int = 2, value: float | None = None,
                  linewidths=None):
    """Draw the cross sections of several meshes by the plane x_axis = value."""
    from matplotlib.collections import LineCollection

    normal = np.zeros(3)
    normal[axis] = 1.0
    if value is None:
        lo, hi = meshes[0].bounds
        value = 0.5 * (lo[axis] + hi[axis])
    keep = [i for i in range(3) if i != axis]
    lws = linewidths or [1.6] * len(meshes)
    for m, lab, col, lw in zip(meshes, labels, colors, lws):
        seg = slice_mesh(m, normal * value, normal)
        if len(seg):
            ax.add_collection(LineCollection(seg[:, :, keep], colors=col, linewidths=lw, label=lab))
    ax.set_aspect("equal")
    ax.autoscale()
    names = "xyz"
    ax.set_xlabel(names[keep[0]])
    ax.set_ylabel(names[keep[1]])
    ax.set_title(f"section {names[axis]} = {value:.3g}", fontsize=9)
    return value
