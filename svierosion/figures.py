"""Composite figures: renders with a shared camera and colour scale, plus cross sections."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .meshio import TriMesh
from .viz import default_camera, plot_sections, render_mesh

PALETTE = ["0.55", "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b",
           "#e377c2", "#17becf", "#bcbd22"]


def render_comparison(meshes: list, titles: list, out_png, scalar: str | None = "depth",
                      cmap: str = "YlOrRd", clim=None, sections=True,
                      camera: dict | None = None, ncols: int | None = None,
                      suptitle: str | None = None, panel_size=(4.0, 3.4)) -> None:
    """Render meshes side by side (first mesh defines the camera) with cross sections.

    ``scalar`` names a per-vertex array in ``mesh.vertex_data`` used for colouring
    (meshes without it are drawn in a neutral colour).  The colour limits are
    shared across panels.  ``sections`` is True (default views), False, or a list
    of ``(axis, value, window)`` with ``window = (u0, u1, v0, v1)`` or None.
    ``camera`` may be a camera dict or ``(azimuth, elevation)`` in degrees.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(meshes)
    ncols = ncols or min(n, 4)
    nrows = int(np.ceil(n / ncols))
    if isinstance(camera, (tuple, list)):
        cam = default_camera(meshes[0], azimuth=camera[0], elevation=camera[1])
    else:
        cam = camera or default_camera(meshes[0])
    vals = [m.vertex_data.get(scalar) for m in meshes] if scalar else [None] * n
    have = [v for v in vals if v is not None and len(v)]
    if clim is None and have:
        clim = (0.0, float(max(np.percentile(v, 99.5) for v in have)))
    n_sec = (3 if sections is True else len(sections)) if sections else 0
    fig = plt.figure(figsize=(panel_size[0] * max(ncols, n_sec), panel_size[1] * (nrows + (1 if sections else 0))))
    gs = fig.add_gridspec(nrows + (1 if sections else 0), max(ncols, n_sec))
    im_for_bar = None
    for i, (m, t, v) in enumerate(zip(meshes, titles, vals)):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        img = render_mesh(m, v, cmap=cmap, clim=clim, camera=cam, size=(640, 520))
        ax.imshow(img)
        ax.set_title(t, fontsize=9)
        ax.axis("off")
        if v is not None:
            im_for_bar = (cmap, clim)
    if im_for_bar is not None:
        import matplotlib.cm as cm
        from matplotlib.colors import Normalize
        sm = cm.ScalarMappable(norm=Normalize(*im_for_bar[1]), cmap=im_for_bar[0])
        cax = fig.add_axes([0.92, 0.55 if sections else 0.15, 0.012, 0.3 if sections else 0.7])
        fig.colorbar(sm, cax=cax, label=scalar)
    if sections:
        lo, hi = meshes[0].bounds
        c = 0.5 * (lo + hi)
        ext = float((hi - lo).max())
        colors = PALETTE[:n] if n <= len(PALETTE) else None
        lws = [1.8] + [1.2] * (n - 1)
        if sections is True:
            # full mid-plane section, then two zoomed views of corner regions
            views = []
            for axis, keep in ((2, None), (2, (0, 1)), (1, (0, 2))):
                w = None if keep is None else section_zoom(meshes[0], axis, float(c[axis]), keep, 0.16 * ext)
                views.append((axis, float(c[axis]), w))
        else:
            views = list(sections)
        for j, (axis, value, w) in enumerate(views):
            ax = fig.add_subplot(gs[nrows, j])
            plot_sections(ax, meshes, titles, colors, axis=axis, value=value, linewidths=lws)
            if w is not None:
                ax.set_xlim(w[0], w[1])
                ax.set_ylim(w[2], w[3])
                ax.set_title(ax.get_title() + " (zoom)", fontsize=9)
            if j == 0:
                ax.legend(fontsize=7, loc="center", framealpha=0.85)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.subplots_adjust(left=0.03, right=0.9, top=0.92 if suptitle else 0.96, bottom=0.05,
                        wspace=0.15, hspace=0.25)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def section_zoom(mesh: TriMesh, axis: int, value: float, keep, size: float):
    """Square window of side ``size`` at the section point farthest along (+1, +1)."""
    from .viz import slice_mesh
    normal = np.zeros(3)
    normal[axis] = 1.0
    seg = slice_mesh(mesh, normal * value, normal)
    if not len(seg):
        return None
    pts = seg.reshape(-1, 3)[:, list(keep)]
    corner = pts[np.argmax(pts[:, 0] + pts[:, 1])]
    pad = 0.15 * size
    return (corner[0] - size + pad, corner[0] + pad, corner[1] - size + pad, corner[1] + pad)


def mesh_distance_stats(a: TriMesh, b: TriMesh, device=None) -> dict:
    """Symmetric surface distances between two meshes."""
    from .api import point_mesh_distance
    dab = point_mesh_distance(a.vertices, b, device=device)
    dba = point_mesh_distance(b.vertices, a, device=device)
    return {"mean": 0.5 * (dab.mean() + dba.mean()),
            "rms": float(np.sqrt(0.5 * ((dab ** 2).mean() + (dba ** 2).mean()))),
            "hausdorff": float(max(dab.max(), dba.max()))}
