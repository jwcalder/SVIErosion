"""Timing of the full pipeline on a synthetic high-resolution 'scan'.

    python benchmarks/performance.py [--subdivisions 5] [--resolutions 192 256 384] [--devices cuda cpu]

The test object is :func:`svierosion.shapes.bone_like`: a tube fragment
(thick-walled shaft piece with sharp break edges) subdivided to ~1.3M
triangles and given a rough surface (random displacement, 0.6% of its size).
Each run erodes it to 90% volume with the SVI law V = N/(0.93 - N),
r = 0.1 x largest extent.  Writes benchmarks/output/performance.csv.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

from svierosion import ErosionParams, erode, shapes, write_ply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subdivisions", type=int, default=5)
    ap.add_argument("--resolutions", type=int, nargs="*", default=[192, 256, 384])
    ap.add_argument("--cpu-resolutions", type=int, nargs="*", default=[192, 256])
    ap.add_argument("--devices", nargs="*", default=["cuda", "cpu"])
    ap.add_argument("--out", default="benchmarks/output")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    mesh = shapes.bone_like(a.subdivisions)
    print(f"bone-like mesh: {len(mesh.faces)} triangles, built in {time.time() - t0:.1f}s")
    write_ply(out / "bone_like.ply", mesh)
    lo, hi = mesh.bounds
    L = float((hi - lo).max())
    rows = []
    torch.set_num_threads(max(torch.get_num_threads(), 1))
    for dev in a.devices:
        if dev == "cuda" and not torch.cuda.is_available():
            continue
        for res in (a.resolutions if dev == "cuda" else a.cpu_resolutions):
            p = ErosionParams(law="svi", radius=0.1 * L, k=1, s=1, t=0.93, resolution=res,
                              target_fraction=0.9, device=dev)
            if dev == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t1 = time.time()
            r = erode(mesh, p)
            wall = time.time() - t1
            steps = len(r.history)
            row = {"device": dev, "resolution": res, "grid": "x".join(map(str, r.grid.shape)),
                   "nodes_M": r.grid.size / 1e6, "triangles_in": len(mesh.faces), "steps": steps,
                   "sdf_s": r.timings["sdf"], "evolve_s": r.timings["evolve"], "mesh_s": r.timings["mesh"],
                   "per_step_ms": 1000 * r.timings["evolve"] / max(steps, 1), "total_s": wall,
                   "gpu_peak_GB": torch.cuda.max_memory_allocated() / 2 ** 30 if dev == "cuda" else 0.0,
                   "triangles_out": len(r.mesh.faces), "fraction": r.fraction}
            rows.append(row)
            print(row, flush=True)
            if dev == "cuda" and res == max(a.resolutions):
                write_ply(out / f"bone_like_eroded_{res}.ply", r.mesh)
    with open(out / "performance.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
