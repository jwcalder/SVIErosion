"""Accuracy and convergence studies used in the report.

    python benchmarks/convergence.py [--out benchmarks/output] [--only N sdf reinit sphere cube]

Writes CSV tables (and prints them).  Sections:

N       accuracy of N_r on a sphere for several discretisations (occupancy model x
        interpolation), including a binary-count / smoothed-Heaviside baseline
sdf     exactness of the mesh -> signed distance conversion
reinit  interface displacement caused by reinitialisation
sphere  convergence of the flows on a sphere against the exact radius ODE
cube    self-convergence of an SVI flow on the unit cube (sharp initial edges)
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp

from svierosion import ErosionParams, erode, shapes
from svierosion.evolve import ErosionSimulation
from svierosion.figures import mesh_distance_stats
from svierosion.flows import GaussCurvatureLaw, MeanCurvatureLaw, SVIPowerLaw
from svierosion.grid import Grid, get_device
from svierosion.levelset import LevelSet, tricubic, trilinear
from svierosion.sdf import mesh_to_sdf, point_triangle_distance2
from svierosion.svi import BallConvolution, sphere_exterior_fraction

DEV = get_device(None)
CTR = np.array([0.0123, -0.0311, 0.0217])


def write(rows, path, show=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if show:
        keys = list(rows[0])
        print(" | ".join(keys))
        for r in rows:
            print(" | ".join(f"{r[k]:.3e}" if isinstance(r[k], float) else str(r[k]) for k in keys))
    print(f"-> {path}\n")


def sphere_ls(R, h, pad, distort=False):
    g = Grid.around([-R] * 3, [R] * 3, h, padding=pad)
    X, Y, Z = [torch.tensor(a, device=DEV) for a in np.meshgrid(*g.axes(), indexing="ij")]
    phi = torch.sqrt((X - CTR[0]) ** 2 + (Y - CTR[1]) ** 2 + (Z - CTR[2]) ** 2) - R
    if distort:
        phi = phi * (1 + 0.4 * torch.sin(3 * X) * torch.cos(2 * Y)) + 0.3 * phi ** 2
    return LevelSet(phi.float(), g, width=6), g


def fib_sphere(n, R):
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    th = np.pi * (1 + 5 ** 0.5) * i
    rr = np.sqrt(1 - z * z)
    return CTR + R * np.stack([rr * np.cos(th), rr * np.sin(th), z], -1)


# ---------------------------------------------------------------------- N_r
def study_N(out):
    R, r = 1.0, 0.4
    exact = sphere_exterior_fraction(r, R)
    rows = []
    for ratio in (4, 6, 8, 12, 16, 24, 32):
        h = r / ratio
        ls, g = sphere_ls(R, h, r + 10 * h)
        conv = BallConvolution(g, r, device=DEV)
        pts = torch.tensor(g.to_index(fib_sphere(5000, R)), dtype=torch.float32, device=DEV)
        phi = ls.phi
        occ = {
            "binary": (phi < 0).float(),
            "heaviside": _heaviside(phi, h),
            "planar (ours)": ls.dense_fractions(),
        }
        for name, f in occ.items():
            N = conv.exterior_fraction(f)
            for interp, fn in (("trilinear", trilinear), ("tricubic", tricubic)):
                e = (fn(N.view(-1), g.shape, pts).double() - exact).cpu().numpy()
                rows.append({"r/h": ratio, "occupancy": name, "interp": interp,
                             "max_err": float(np.abs(e).max()), "mean_err": float(e.mean())})
        # baseline in the spirit of the original student code: sine-smoothed Heaviside,
        # 3^3-supersampled ball normalised by its own sum, value at the nearest node
        w3 = _supersampled_ball(r, h, 3)
        Nb = 1 - _conv_with(w3, _heaviside(phi, h), g)
        near = torch.round(pts).long()
        nb = Nb[near[:, 0], near[:, 1], near[:, 2]]
        e = (nb.double() - exact).cpu().numpy()
        rows.append({"r/h": ratio, "occupancy": "baseline (student-like)", "interp": "nearest node",
                     "max_err": float(np.abs(e).max()), "mean_err": float(e.mean())})
        # nodal speeds are effectively interpolated linearly by the level set update
        e = (trilinear(Nb.contiguous().view(-1), g.shape, pts).double() - exact).cpu().numpy()
        rows.append({"r/h": ratio, "occupancy": "baseline (student-like)", "interp": "trilinear",
                     "max_err": float(np.abs(e).max()), "mean_err": float(e.mean())})
    write(rows, out / "N_accuracy.csv")


def _heaviside(phi, h, width=1.0):
    s = torch.clamp(phi / (width * h), -1, 1)
    return 0.5 - 0.5 * (s + torch.sin(math.pi * s) / math.pi)


def _supersampled_ball(r, h, samples):
    m = int(math.ceil(r / h + 0.5))
    a = np.arange(-m, m + 1, dtype=np.float64)
    x, y, z = np.meshgrid(a, a, a, indexing="ij", sparse=True)
    off = (np.arange(samples) + 0.5) / samples - 0.5
    w = np.zeros((len(a),) * 3)
    for dx in off:
        for dy in off:
            for dz in off:
                w += ((x + dx) ** 2 + (y + dy) ** 2 + (z + dz) ** 2 <= (r / h) ** 2)
    return w / w.sum()


def _conv_with(w, f, g):
    m = (w.shape[0] - 1) // 2
    k = torch.zeros(g.shape, dtype=torch.float32, device=DEV)
    idx = torch.arange(-m, m + 1, device=DEV)
    k[(idx % g.shape[0])[:, None, None], (idx % g.shape[1])[None, :, None],
      (idx % g.shape[2])[None, None, :]] = torch.as_tensor(w, dtype=torch.float32, device=DEV)
    return torch.fft.irfftn(torch.fft.rfftn(f) * torch.fft.rfftn(k), s=g.shape)


# ---------------------------------------------------------------------- SDF
def study_sdf(out):
    rows = []
    m = shapes.cube(1.0)
    for h in (0.05, 0.02, 0.01):
        g = Grid.around(*m.bounds, spacing=h, padding=0.3)
        t0 = time.time()
        phi = mesh_to_sdf(m, g, band=6, device=DEV).cpu().numpy()
        t = time.time() - t0
        X, Y, Z = np.meshgrid(*g.axes(), indexing="ij")
        ex = np.clip(shapes.box_sdf(np.stack([X, Y, Z], -1)), -6 * h, 6 * h)
        rows.append({"mesh": "cube (12 tri)", "h": h, "grid": "x".join(map(str, g.shape)),
                     "max_err/h": float(np.abs(phi - ex).max() / h),
                     "sign_errors": int(np.sum((phi < 0) != (ex < 0))), "seconds": t})
    tor = shapes.torus(0.6, 0.25, 192, 96)
    for h in (0.02, 0.01):
        g = Grid.around(*tor.bounds, spacing=h, padding=0.2)
        t0 = time.time()
        phi = mesh_to_sdf(tor, g, band=6, device=DEV)
        t = time.time() - t0
        flat = phi.view(-1)
        band = torch.nonzero(flat.abs() < 5.99 * h).squeeze(1)
        sel = band[torch.randperm(band.numel(), device=band.device)[:20000]]
        pts = g.node_coords(sel)
        tri = torch.as_tensor(tor.triangles(), dtype=torch.float64, device=DEV)
        best = torch.full((len(pts),), float("inf"), dtype=torch.float64, device=DEV)
        for c0 in range(0, len(tri), 256):
            for p0 in range(0, len(pts), 4096):
                t3 = tri[c0:c0 + 256]
                d2 = point_triangle_distance2(pts[p0:p0 + 4096, None], t3[None, :, 0], t3[None, :, 1],
                                              t3[None, :, 2])
                best[p0:p0 + 4096] = torch.minimum(best[p0:p0 + 4096], d2.min(1).values)
        err = (flat[sel].abs().double() - best.sqrt()).abs()
        rows.append({"mesh": f"torus ({len(tor.faces)} tri), 20k band nodes", "h": h,
                     "grid": "x".join(map(str, g.shape)), "max_err/h": float(err.max() / h),
                     "sign_errors": -1, "seconds": t})
    write(rows, out / "sdf_accuracy.csv")


# ------------------------------------------------------------------- reinit
def study_reinit(out):
    rows = []
    for h in (0.08, 0.04, 0.02, 0.01):
        ls, g = sphere_ls(1.0, h, 12 * h, distort=True)
        v0 = ls.volume()
        ls.reinitialize(40)
        idx = ls.band
        x = g.node_coords(idx).cpu().numpy()
        ph = ls.flat[idx].cpu().numpy()
        ex = np.linalg.norm(x - CTR, axis=1) - 1.0
        near = np.abs(ex) < 1.5 * h
        band = np.abs(ex) < 5 * h
        rows.append({"h": h, "max_err_near/h": float(np.abs(ph - ex)[near].max() / h),
                     "max_err_band/h": float(np.abs(ph - ex)[band].max() / h),
                     "rel_volume_change": float(ls.volume() / v0 - 1)})
    write(rows, out / "reinit_accuracy.csv")


# ------------------------------------------------------------------ spheres
def study_sphere(out):
    rows = []
    T = 0.15
    r = 0.4
    laws = [("SVI N", SVIPowerLaw(r, 1, 0)), ("SVI N^4", SVIPowerLaw(r, 4, 0)),
            ("SVI N^2/(0.8-N)", SVIPowerLaw(r, 2, 1, 0.8)), ("grassfire", SVIPowerLaw(r, 0, 0)),
            ("H+", MeanCurvatureLaw(1.0)), ("K+", GaussCurvatureLaw(1.0)),
            ("K+^(1/4)", GaussCurvatureLaw(0.25, delta=1e-4))]
    for name, law in laws:
        if law.requires_svi:
            F = lambda R, law=law: float(law.speed(torch.tensor(sphere_exterior_fraction(r, R),
                                                                dtype=torch.float64)))
            Rex = solve_ivp(lambda t, y: [-F(y[0])], (0, T), [1.0], rtol=1e-12, atol=1e-14).y[0, -1]
            hs = [r / q for q in (6, 8, 12, 16, 24)]
        else:
            Rex = {"H+": math.sqrt(1 - 2 * T), "K+": (1 - 3 * T) ** (1 / 3),
                   "K+^(1/4)": (1 - 1.5 * T) ** (2 / 3)}[name]
            hs = [0.08, 0.04, 0.02]
        prev = None
        for h in hs:
            ls, g = sphere_ls(1.0, h, (r if law.requires_svi else 0) + 10 * h)
            sim = ErosionSimulation(ls, law)
            t0 = time.time()
            res = sim.run(target_fraction=None, max_time=T)
            wall = time.time() - t0
            idx = ls.band
            x = g.node_coords(idx).cpu().numpy()
            ph = ls.flat[idx].cpu().numpy()
            near = np.abs(ph) < h
            Rn = np.linalg.norm(x[near] - CTR, axis=1) - ph[near]
            err = abs(Rn.mean() - Rex)
            order = math.log(prev[1] / err) / math.log(prev[0] / h) if prev and err > 0 else float("nan")
            prev = (h, err)
            rows.append({"law": name, "h": h, "steps": res.steps, "R_exact": Rex, "abs_err": err,
                         "order": order, "spread": float(Rn.max() - Rn.min()), "seconds": wall})
    write(rows, out / "sphere_convergence.csv")


# --------------------------------------------------------------------- cube
def study_cube(out):
    m = shapes.cube(1.0)
    rows = []
    ref = None
    results = {}
    for res_ in (256, 192, 128, 96, 64):
        p = ErosionParams(law="svi", radius=0.15, k=1, s=1, t=0.93, resolution=res_, target_fraction=0.9)
        t0 = time.time()
        r = erode(m, p)
        results[res_] = (r, time.time() - t0)
    ref = results[256][0].mesh
    for res_ in (64, 96, 128, 192):
        r, wall = results[res_]
        d = mesh_distance_stats(r.mesh, ref)
        rows.append({"resolution": res_, "h": 1.0 / res_, "steps": len(r.history), "mean_dist_to_256": d["mean"],
                     "hausdorff_to_256": d["hausdorff"], "seconds": wall})
    for i in range(1, len(rows)):
        rows[i]["order_mean"] = math.log(rows[i - 1]["mean_dist_to_256"] / rows[i]["mean_dist_to_256"]) / \
            math.log(rows[i - 1]["h"] / rows[i]["h"])
    rows[0]["order_mean"] = float("nan")
    write(rows, out / "cube_self_convergence.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmarks/output")
    ap.add_argument("--only", nargs="*", default=["N", "sdf", "reinit", "sphere", "cube"])
    a = ap.parse_args()
    out = Path(a.out)
    for name in a.only:
        print(f"===== {name} =====", flush=True)
        {"N": study_N, "sdf": study_sdf, "reinit": study_reinit, "sphere": study_sphere,
         "cube": study_cube}[name](out)


if __name__ == "__main__":
    main()
