"""Command line interface: ``python -m svierosion <command> ...``.

Commands
--------
erode     erode a PLY mesh under an SVI or curvature law until a volume fraction
compare   run several laws on one mesh to the same volume fraction
distance  vertex-wise distance from one mesh to another (comparison with scans)
shapes    write the built-in toy solids as PLY files
info      print basic statistics of a mesh
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np


def _add_law_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("flow")
    g.add_argument("--law", choices=["svi", "mean", "gauss"], default="svi",
                   help="speed law: svi = N^k/(t-N)^s, mean = (H_+)^alpha, gauss = (k1_+ k2_+)^alpha")
    g.add_argument("--radius", "-r", type=float, help="SVI ball radius r (mesh units)")
    g.add_argument("--radius-rel", type=float,
                   help="SVI radius as a fraction of the largest bounding-box side")
    g.add_argument("-k", type=int, default=1, help="SVI exponent k (default 1)")
    g.add_argument("-s", type=int, default=0, choices=[0, 1], help="SVI geometric term switch s")
    g.add_argument("-t", type=float, help="SVI singularity location t (needed for s=1)")
    g.add_argument("--eps", type=float, default=0.0, help="additive constant speed (grassfire)")
    g.add_argument("--alpha", type=float, default=1.0, help="curvature-law exponent")
    g.add_argument("--delta", type=float, default=None,
                   help="regularisation of x^alpha near 0 for alpha < 1 (default: automatic)")
    g.add_argument("--max-speed-ratio", type=float, default=100.0,
                   help="cap of the s=1 SVI speed relative to the flat-surface speed")


def _add_numerics_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("numerics")
    g.add_argument("--target", type=float, default=0.9, help="final volume fraction (default 0.9)")
    g.add_argument("--snapshots", type=float, nargs="*", default=[],
                   help="additional volume fractions at which meshes are written")
    g.add_argument("--spacing", type=float, help="grid spacing h (mesh units)")
    g.add_argument("--resolution", type=int, default=200,
                   help="cells across the largest bounding-box side (if --spacing not given)")
    g.add_argument("--band", type=float, default=6.0, help="narrow band half width (cells)")
    g.add_argument("--cfl", type=float, default=0.5)
    g.add_argument("--device", default=None, help="cuda, cpu (default: cuda if available)")
    g.add_argument("--no-compile", action="store_true", help="disable torch.compile kernels")
    g.add_argument("--no-project", action="store_true",
                   help="do not project marching-cubes vertices onto the tricubic zero set")


def _params_from_args(args, mesh):
    from .api import ErosionParams
    radius = args.radius
    if args.law == "svi" and radius is None:
        if args.radius_rel is None:
            raise SystemExit("error: the SVI law needs --radius or --radius-rel")
        lo, hi = mesh.bounds
        radius = args.radius_rel * float((hi - lo).max())
    return ErosionParams(law=args.law, radius=radius, k=args.k, s=args.s, t=args.t, eps=args.eps,
                         max_speed_ratio=args.max_speed_ratio, alpha=args.alpha, delta=args.delta,
                         target_fraction=args.target, snapshots=tuple(args.snapshots),
                         spacing=args.spacing, resolution=args.resolution, band=args.band,
                         cfl=args.cfl, device=args.device, project_vertices=not args.no_project)


def _progress(verbose):
    state = {"t": time.time()}

    def cb(sim, rec):
        if not verbose:
            return
        now = time.time()
        if now - state["t"] > 2.0 or rec.step == 1:
            state["t"] = now
            q = f" N=[{rec.qmin:.3f},{rec.qmax:.3f}]" if sim.law.requires_svi else \
                f" H=[{rec.qmin:.3g},{rec.qmax:.3g}]"
            print(f"  step {rec.step:6d}  t={rec.time:.5g}  dt={rec.dt:.2e}  V/V0={rec.fraction:.5f}  "
                  f"vmax={rec.vmax:.3g}{q}" + (f" capped={rec.capped}" if rec.capped else ""),
                  flush=True)
    return cb


def _write_history(path: Path, history) -> None:
    if not history:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(asdict(history[0])))
        w.writeheader()
        for rec in history:
            w.writerow(asdict(rec))


def _write_result(res, out: Path, quality: str) -> list:
    from .meshio import write_ply
    written = []
    stem = out.with_suffix("")
    for frac, t, m in res.snapshots:
        p = Path(f"{stem}_v{frac:.3f}.ply")
        _write_mesh(write_ply, p, m, quality)
        written.append(p)
    _write_mesh(write_ply, out, res.mesh, quality)
    written.append(out)
    _write_history(Path(f"{stem}_history.csv"), res.history)
    meta = {"summary": res.summary(), "params": res.params, "stop_reason": res.stop_reason,
            "time": res.time, "volume_fraction": res.fraction,
            "grid": {"origin": res.grid.origin, "spacing": res.grid.spacing, "shape": res.grid.shape},
            "timings": res.timings,
            "initial_mesh_volume": res.initial_volume_mesh,
            "final_mesh_volume": res.mesh.volume() if len(res.mesh.faces) else 0.0,
            "snapshots": [{"fraction": f, "time": t} for f, t, _ in res.snapshots]}
    Path(f"{stem}.json").write_text(json.dumps(meta, indent=2, default=float) + "\n")
    return written


def _write_mesh(write_ply, path, mesh, quality):
    extra = {}
    if quality and quality in mesh.vertex_data:
        extra["quality"] = mesh.vertex_data[quality]
    write_ply(path, mesh, vertex_data=extra)


# ------------------------------------------------------------------ commands
def cmd_erode(args) -> int:
    from . import kernels
    from .api import erode, volume_fraction_report
    from .meshio import read_ply
    if args.no_compile:
        kernels.set_compile(False)
    mesh = read_ply(args.input)
    params = _params_from_args(args, mesh)
    out = Path(args.output) if args.output else Path(args.input).with_name(Path(args.input).stem + "_eroded.ply")
    if not args.quiet:
        print(f"[svierosion] {args.input}: {len(mesh.vertices)} vertices, {len(mesh.faces)} triangles, "
              f"volume {mesh.volume():.6g}, watertight {mesh.is_watertight()}")
    res = erode(mesh, params, callback=_progress(not args.quiet), verbose=not args.quiet)
    files = _write_result(res, out, args.quality)
    if not args.quiet:
        print("[svierosion] " + res.summary())
        print("[svierosion] " + volume_fraction_report(res))
        for f in files:
            print(f"[svierosion] wrote {f}")
    if args.render:
        from .figures import render_comparison
        render_comparison([mesh, res.mesh], ["initial", f"eroded ({res.fraction:.3f})"],
                          args.render, scalar="depth")
        if not args.quiet:
            print(f"[svierosion] wrote {args.render}")
    return 0 if res.stop_reason == "target_volume" else 2


def cmd_compare(args) -> int:
    from . import kernels
    from .compare import default_comparison_laws, run_comparison
    from .meshio import read_ply
    if args.no_compile:
        kernels.set_compile(False)
    mesh = read_ply(args.input)
    lo, hi = mesh.bounds
    radius = args.radius if args.radius is not None else (args.radius_rel or 0.15) * float((hi - lo).max())
    laws = default_comparison_laws(radius, t=args.t)
    run_comparison(mesh, laws, Path(args.output), target=args.target, resolution=args.resolution,
                   spacing=args.spacing, device=args.device, verbose=not args.quiet)
    return 0


def cmd_distance(args) -> int:
    from .api import point_mesh_distance
    from .meshio import read_ply, write_ply
    a, b = read_ply(args.mesh_a), read_ply(args.mesh_b)
    d = point_mesh_distance(a.vertices, b, device=args.device, exact=True)
    db = point_mesh_distance(b.vertices, a, device=args.device, exact=True)
    print(f"A->B: mean {d.mean():.6g}  rms {np.sqrt((d ** 2).mean()):.6g}  max {d.max():.6g}")
    print(f"B->A: mean {db.mean():.6g}  rms {np.sqrt((db ** 2).mean()):.6g}  max {db.max():.6g}")
    print(f"Hausdorff {max(d.max(), db.max()):.6g}")
    if args.output:
        write_ply(args.output, a, vertex_data={"quality": d.astype(np.float32)})
        print(f"wrote {args.output} (vertex quality = distance to {args.mesh_b})")
    return 0


def cmd_shapes(args) -> int:
    from . import shapes
    from .meshio import write_ply
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for name, make in shapes.SHAPES.items():
        m = make()
        write_ply(out / f"{name}.ply", m)
        print(f"{name:14s} {len(m.faces):7d} triangles  volume {m.volume():.5f}  -> {out / (name + '.ply')}")
    return 0


def cmd_info(args) -> int:
    from .meshio import read_ply
    m = read_ply(args.input)
    lo, hi = m.bounds
    topo = m.edge_topology()
    print(f"vertices {len(m.vertices)}, triangles {len(m.faces)}")
    print(f"bounding box {lo} .. {hi} (extent {hi - lo})")
    print(f"area {m.area():.6g}, signed volume {m.signed_volume():.6g}")
    print(f"edges {topo['edges']}, boundary {topo['boundary_edges']}, non-manifold "
          f"{topo['nonmanifold_edges']}, misoriented {topo['misoriented_edges']}")
    print(f"vertex properties: {sorted(m.vertex_data)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="svierosion", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("erode", help="erode a mesh")
    p.add_argument("input")
    p.add_argument("-o", "--output", help="output PLY (default: <input>_eroded.ply)")
    _add_law_args(p)
    _add_numerics_args(p)
    p.add_argument("--quality", default="depth",
                   help="vertex scalar duplicated as 'quality' (MeshLab colour map): depth, N, speed, H")
    p.add_argument("--render", help="also write a PNG comparing initial and eroded meshes")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_erode)

    p = sub.add_parser("compare", help="compare SVI and curvature flows on one mesh")
    p.add_argument("input")
    p.add_argument("-o", "--output", required=True, help="output directory")
    p.add_argument("--radius", "-r", type=float)
    p.add_argument("--radius-rel", type=float, help="radius / largest bounding-box side (default 0.15)")
    p.add_argument("-t", type=float, default=0.93, help="t for the s=1 SVI laws")
    p.add_argument("--target", type=float, default=0.9)
    p.add_argument("--spacing", type=float)
    p.add_argument("--resolution", type=int, default=160)
    p.add_argument("--device", default=None)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("distance", help="distances between two meshes")
    p.add_argument("mesh_a")
    p.add_argument("mesh_b")
    p.add_argument("-o", "--output", help="write mesh A with per-vertex distance as quality")
    p.add_argument("--device", default=None)
    p.set_defaults(func=cmd_distance)

    p = sub.add_parser("shapes", help="write the toy solids")
    p.add_argument("output")
    p.set_defaults(func=cmd_shapes)

    p = sub.add_parser("info", help="mesh statistics")
    p.add_argument("input")
    p.set_defaults(func=cmd_info)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
