"""Toy examples: SVI erosion of simple solids and comparison with curvature flows.

    python examples/toy_examples.py [--resolution 160] [--out examples/output] [--quick]

For every toy solid the script

1. erodes it to 90% of its volume under several members of the SVI family
   V = N^k / (t - N)^s (ball radius r = 0.15 x largest extent), and
2. compares a representative SVI law with mean- and Gauss-curvature flows,

writing PLY meshes (with per-vertex erosion depth, N_r, speed), CSV tables
and PNG figures (renders coloured by erosion depth + cross sections).
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from svierosion import ErosionParams, erode, shapes, write_ply
from svierosion.compare import slug
from svierosion.figures import render_comparison

SHAPES = ["cube", "prism", "ellipsoid", "lblock", "tube_fragment"]

# per-shape figure settings: camera (azimuth, elevation) and sections (axis, value, zoom window)
FIGURE = {
    "cube": dict(camera=(-55, 25), sections=[(2, 0.0, None), (2, 0.0, (0.36, 0.52, 0.36, 0.52)),
                                             (1, 0.3, (0.36, 0.52, 0.36, 0.52))]),
    "prism": dict(camera=(-60, 25), sections=[(2, 0.0, None), (2, 0.0, (-0.09, 0.09, 0.56, 0.72)),
                                              (0, 0.0, (0.55, 0.71, 0.46, 0.62))]),
    "ellipsoid": dict(camera=(-55, 25), sections=[(2, 0.0, None), (2, 0.0, (0.6, 0.82, -0.11, 0.11)),
                                                  (1, 0.0, None)]),
    "lblock": dict(camera=(-120, 30), sections=[(2, 0.0, None), (2, 0.0, (-0.1, 0.1, -0.1, 0.1)),
                                                (2, 0.0, (0.46, 0.62, -0.62, -0.46))]),
    "tube_fragment": dict(camera=(150, 25), sections=[(2, 0.0, None), (2, 0.0, (-0.6, -0.44, 0.47, 0.63)),
                                                      (2, 0.0, (-0.43, -0.27, 0.80, 0.96))]),
}
T_SING = 0.93          # > max N_r of every toy solid (prism corners: 11/12)

SVI_FAMILY = [
    ("k=0 (grassfire)", dict(k=0, s=0)),
    ("k=1", dict(k=1, s=0)),
    ("k=4", dict(k=4, s=0)),
    (f"k=0, s=1, t={T_SING}", dict(k=0, s=1, t=T_SING)),
    (f"k=1, s=1, t={T_SING}", dict(k=1, s=1, t=T_SING)),
]

COMPARISON = [
    (f"SVI k=1, s=1, t={T_SING}", dict(law="svi", k=1, s=1, t=T_SING)),
    ("mean curvature H+", dict(law="mean", alpha=1.0)),
    ("Gauss curvature K+", dict(law="gauss", alpha=1.0)),
]
# K^(1/4) needs ~1e5 explicit steps on solids with sharp edges (dt ~ h^3 / F'(0));
# run it on smooth solids only.
SMOOTH_ONLY = [("Gauss K+^(1/4)", dict(law="gauss", alpha=0.25))]
SMOOTH = {"ellipsoid", "sphere", "torus"}


def run_case(mesh, label, kw, radius, args, outdir: Path, rows: list, shape: str):
    p = ErosionParams(**{"law": "svi", "radius": radius, **kw}, target_fraction=args.target,
                      resolution=args.resolution, max_steps=args.max_steps)
    if p.law != "svi":
        p.radius = None
    t0 = time.time()
    res = erode(mesh, p)
    wall = time.time() - t0
    name = f"{shape}_{slug(label)}"
    write_ply(outdir / f"{name}.ply", res.mesh,
              vertex_data={"quality": res.mesh.vertex_data.get("depth", np.zeros(len(res.mesh.vertices)))})
    d = res.mesh.vertex_data.get("depth", np.zeros(1))
    row = dict(shape=shape, label=label, law=p.law, stop=res.stop_reason, steps=len(res.history),
               model_time=res.time, fraction=res.fraction,
               mesh_fraction=res.mesh.volume() / mesh.volume() if len(res.mesh.faces) else 0,
               depth_mean=float(d.mean()), depth_max=float(d.max()), wall=wall,
               grid="x".join(map(str, res.grid.shape)))
    rows.append(row)
    print(f"  {label:26s} {res.stop_reason:13s} steps {row['steps']:6d}  V/V0 {res.fraction:.4f}  "
          f"depth mean {row['depth_mean']:.4f} max {row['depth_max']:.4f}  {wall:6.1f}s", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=160)
    ap.add_argument("--gauss-resolution", type=int, default=96,
                    help="resolution for Gauss-curvature runs (explicit steps scale like h^3)")
    ap.add_argument("--target", type=float, default=0.9)
    ap.add_argument("--out", default="examples/output")
    ap.add_argument("--shapes", nargs="*", default=SHAPES)
    ap.add_argument("--max-steps", type=int, default=60000)
    ap.add_argument("--skip-family", action="store_true")
    ap.add_argument("--skip-comparison", action="store_true")
    ap.add_argument("--figures-only", action="store_true", help="re-render figures from saved PLY files")
    args = ap.parse_args()
    if args.figures_only:
        return figures_only(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for shape in args.shapes:
        mesh = shapes.SHAPES[shape]()
        lo, hi = mesh.bounds
        L = float((hi - lo).max())
        radius = 0.15 * L
        write_ply(out / f"{shape}_initial.ply", mesh)
        if not args.skip_family:
            print(f"[{shape}] SVI family, r = {radius:.3g}, resolution {args.resolution}")
            results = [run_case(mesh, lab, kw, radius, args, out, rows, shape) for lab, kw in SVI_FAMILY]
            render_comparison([mesh] + [r.mesh for r in results], ["initial"] + [lab for lab, _ in SVI_FAMILY],
                              out / f"{shape}_svi_family.png", scalar="depth", ncols=3, **FIGURE[shape],
                              suptitle=f"{shape}: SVI flows V = N^k/(t-N)^s, r = {radius:.3g}, "
                                       f"stopped at V/V0 = {args.target}")
        if not args.skip_comparison:
            print(f"[{shape}] comparison with curvature flows")
            results = []
            laws = COMPARISON + (SMOOTH_ONLY if shape in SMOOTH else [])
            for lab, kw in laws:
                if kw["law"] == "gauss":
                    a2 = argparse.Namespace(**{**vars(args), "resolution": args.gauss_resolution})
                    results.append(run_case(mesh, lab, kw, radius, a2, out, rows, shape))
                else:
                    results.append(run_case(mesh, lab, kw, radius, args, out, rows, shape))
            labels = [f"{lab}" + ("" if r.stop_reason == "target_volume" else f" [{r.stop_reason}, V/V0={r.fraction:.3f}]")
                      for (lab, _), r in zip(laws, results)]
            render_comparison([mesh] + [r.mesh for r in results], ["initial"] + labels,
                              out / f"{shape}_comparison.png", scalar="depth", ncols=3, **FIGURE[shape],
                              suptitle=f"{shape}: SVI vs curvature flows, stopped at V/V0 = {args.target}")
        with open(out / "toy_summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def figures_only(args):
    from svierosion import read_ply
    out = Path(args.out)
    for shape in args.shapes:
        mesh = read_ply(out / f"{shape}_initial.ply")
        L = float((mesh.bounds[1] - mesh.bounds[0]).max())
        fam = [read_ply(out / f"{shape}_{slug(lab)}.ply") for lab, _ in SVI_FAMILY]
        render_comparison([mesh] + fam, ["initial"] + [lab for lab, _ in SVI_FAMILY],
                          out / f"{shape}_svi_family.png", scalar="depth", ncols=3, **FIGURE[shape],
                          suptitle=f"{shape}: SVI flows V = N^k/(t-N)^s, r = {0.15 * L:.3g}, "
                                   f"stopped at V/V0 = {args.target}")
        laws = COMPARISON + (SMOOTH_ONLY if shape in SMOOTH else [])
        cmp_ = [read_ply(out / f"{shape}_{slug(lab)}.ply") for lab, _ in laws]
        render_comparison([mesh] + cmp_, ["initial"] + [lab for lab, _ in laws],
                          out / f"{shape}_comparison.png", scalar="depth", ncols=3, **FIGURE[shape],
                          suptitle=f"{shape}: SVI vs curvature flows, stopped at V/V0 = {args.target}")
        print("figures", shape)


if __name__ == "__main__":
    main()
