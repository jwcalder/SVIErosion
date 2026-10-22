"""Exposure maps: the normalised exterior volume invariant N_r on a surface.

    python examples/exposure_map.py [--out examples/output]

Computes N_r on the initial surface of a few solids (and on the eroded tube
fragment) and renders it; N_r = 1/2 on flat parts, > 1/2 on exposed convex
parts, < 1/2 in concavities.
"""

import argparse
from pathlib import Path

import numpy as np

from svierosion import ErosionParams, erode, shapes
from svierosion.api import make_grid
from svierosion.evolve import ErosionSimulation
from svierosion.flows import SVIPowerLaw
from svierosion.levelset import LevelSet
from svierosion.sdf import mesh_to_sdf
from svierosion.surface import extract_surface, sample_field


def exposure(mesh, radius, resolution=200):
    p = ErosionParams(law="svi", radius=radius, resolution=resolution)
    g = make_grid(mesh, p)
    ls = LevelSet(mesh_to_sdf(mesh, g), g)
    sim = ErosionSimulation(ls, SVIPowerLaw(radius))
    surf = extract_surface(ls)
    surf.vertex_data["N"] = sample_field(sim.svi_field(), g, surf.vertices).astype(np.float32)
    return surf


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize

    from svierosion.viz import default_camera, render_mesh

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="examples/output")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tube = shapes.tube_fragment()
    bone = shapes.bone_like(3)
    L = float((tube.bounds[1] - tube.bounds[0]).max())
    items = [
        ("L-block, r = 0.15 L", shapes.l_block(1.2, 0.6, 0.6), 0.18, (-120, 30)),
        ("tube fragment, r = 0.15 L", tube, 0.15 * L, (150, 25)),
        ("rough tube fragment, r = 0.15 L", bone, 0.15 * L, (150, 25)),
        ("rough tube fragment, r = 0.05 L", bone, 0.05 * L, (150, 25)),
    ]
    panels = []
    for title, m, r, cam in items:
        panels.append((title, exposure(m, r), cam))
    eroded = erode(bone, ErosionParams(law="svi", radius=0.15 * L, k=1, s=1, t=0.93, resolution=200))
    panels.append(("rough tube fragment eroded to 90%\n(V = N/(0.93-N), r = 0.15 L)", eroded.mesh, (150, 25)))
    clim = (0.25, 0.85)
    fig, axs = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.2))
    for ax, (title, m, cam) in zip(axs, panels):
        img = render_mesh(m, m.vertex_data["N"], cmap="coolwarm", clim=clim,
                          camera=default_camera(m, *cam), size=(620, 620))
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    sm = cm.ScalarMappable(norm=Normalize(*clim), cmap="coolwarm")
    fig.colorbar(sm, ax=axs, fraction=0.012, pad=0.01, label="$N_r$")
    fig.savefig(out / "exposure_maps.png", dpi=130, bbox_inches="tight")
    print("wrote", out / "exposure_maps.png")


if __name__ == "__main__":
    main()
