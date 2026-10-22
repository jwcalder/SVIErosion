"""Run several erosion laws on the same solid to the same remaining volume."""

from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path

import numpy as np

from .api import ErosionParams, erode
from .figures import mesh_distance_stats, render_comparison
from .meshio import TriMesh, write_ply


def default_comparison_laws(radius: float, t: float = 0.93) -> list:
    """(label, ErosionParams keyword arguments) for a standard comparison."""
    return [
        ("grassfire V=1", dict(law="svi", radius=radius, k=0, s=0)),
        ("SVI V=N", dict(law="svi", radius=radius, k=1, s=0)),
        ("SVI V=N^4", dict(law="svi", radius=radius, k=4, s=0)),
        (f"SVI V=N/({t:g}-N)", dict(law="svi", radius=radius, k=1, s=1, t=t)),
        ("mean curv. V=H+", dict(law="mean", alpha=1.0)),
        # K^(1/4) is impractical on sharp-edged solids with explicit steps (dt ~ h^3);
        # add ("Gauss V=K+^(1/4)", dict(law="gauss", alpha=0.25)) for smooth solids
        ("Gauss curv. V=K+", dict(law="gauss", alpha=1.0)),
    ]


def slug(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower()


def run_comparison(mesh: TriMesh, laws: list, outdir, target: float = 0.9,
                   resolution: int = 160, spacing: float | None = None, device=None,
                   verbose: bool = True, max_steps: int = 200_000, title: str | None = None) -> list:
    """Erode ``mesh`` with every law; write meshes, a summary table and a figure.

    Returns a list of (label, ErosionResult).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    write_ply(outdir / "initial.ply", mesh)
    results = []
    rows = []
    for label, kw in laws:
        params = ErosionParams(**kw, target_fraction=target, resolution=resolution, spacing=spacing,
                               device=device, max_steps=max_steps)
        t0 = time.time()
        res = erode(mesh, params, verbose=False)
        wall = time.time() - t0
        name = slug(label)
        write_ply(outdir / f"{name}.ply", res.mesh,
                  vertex_data={"quality": res.mesh.vertex_data.get("depth", np.zeros(len(res.mesh.vertices)))})
        depth = res.mesh.vertex_data.get("depth", np.zeros(1))
        row = {"label": label, "stop": res.stop_reason, "steps": len(res.history),
               "model_time": res.time, "volume_fraction": res.fraction,
               "mesh_volume_fraction": (res.mesh.volume() / mesh.volume()) if len(res.mesh.faces) else 0.0,
               "depth_mean": float(depth.mean()), "depth_max": float(depth.max()),
               "wall_seconds": wall, "grid": "x".join(map(str, res.grid.shape))}
        rows.append(row)
        results.append((label, res))
        if verbose:
            print(f"[compare] {label:24s} {res.stop_reason:13s} steps {row['steps']:6d}  "
                  f"V/V0 {row['volume_fraction']:.4f}  depth mean {row['depth_mean']:.4g} "
                  f"max {row['depth_max']:.4g}  ({wall:.1f}s)", flush=True)
    with open(outdir / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # pairwise distances between final shapes
    finals = [r.mesh for _, r in results]
    labels = [lab for lab, _ in results]
    D = np.zeros((len(finals), len(finals)))
    for i in range(len(finals)):
        for j in range(i + 1, len(finals)):
            if len(finals[i].faces) and len(finals[j].faces):
                D[i, j] = D[j, i] = mesh_distance_stats(finals[i], finals[j], device=device)["mean"]
    (outdir / "pairwise_mean_distance.json").write_text(
        json.dumps({"labels": labels, "mean_distance": D.tolist()}, indent=2) + "\n")
    render_comparison([mesh] + finals, ["initial"] + labels, outdir / "comparison.png",
                      scalar="depth", suptitle=title or f"all runs stopped at V/V0 = {target:g}")
    return results
