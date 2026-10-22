"""High-level interface: erode a triangle mesh and return triangle meshes."""

from __future__ import annotations

import time
import warnings
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from .evolve import ErosionSimulation, StepRecord
from .flows import GaussCurvatureLaw, MeanCurvatureLaw, SVIPowerLaw, SpeedLaw
from .grid import Grid, get_device
from .levelset import MARGIN, LevelSet, tricubic
from .meshio import TriMesh
from .kernels import _point_triangle_distance2
from .sdf import bisect_long_triangles, mesh_to_sdf
from .surface import extract_surface, sample_field


@dataclass
class ErosionParams:
    """All parameters of an erosion run (lengths in mesh units).

    Exactly one of ``spacing`` (grid step h) and ``resolution`` (number of
    cells across the largest extent of the input) is used; ``spacing`` wins.
    """

    law: str = "svi"                  # "svi", "mean" or "gauss"
    radius: float | None = None       # SVI ball radius r
    k: int = 1                        # SVI exponent
    s: int = 0                        # SVI geometric term switch
    t: float | None = None            # SVI singularity location
    eps: float = 0.0                  # additive grassfire term
    max_speed_ratio: float = 100.0    # SVI speed cap relative to the flat speed (s = 1)
    alpha: float = 1.0                # curvature-law exponent
    delta: float | None = None        # regularisation of x^alpha (alpha < 1); None = automatic:
                                      # 1/L for H and 1/L^2 for K, L = largest extent
    target_fraction: float = 0.9      # stop at this fraction of the initial volume
    snapshots: tuple = ()             # extra volume fractions at which to save meshes
    spacing: float | None = None
    resolution: int = 200
    band: float = 6.0                 # narrow band half-width (cells)
    cfl: float = 0.5
    feedback_cfl: float = 1.0
    parabolic_cfl: float = 0.25
    reinit_distance: float = 1.5      # cells of motion between reinitialisations
    reinit_iterations: int = 20
    project_vertices: bool = True
    device: str | None = None
    max_steps: int = 1_000_000

    def make_law(self, length_scale: float = 1.0) -> SpeedLaw:
        """Build the speed law; ``length_scale`` (object size) sets the automatic delta."""
        if self.law == "svi":
            if self.radius is None:
                raise ValueError("the SVI law needs a radius")
            return SVIPowerLaw(self.radius, self.k, self.s, self.t, self.max_speed_ratio, self.eps)
        if self.law in ("mean", "mean_curvature"):
            d = self.delta if self.delta is not None else (1.0 / length_scale if self.alpha < 1 else 0.0)
            return MeanCurvatureLaw(self.alpha, self.eps, d)
        if self.law in ("gauss", "gauss_curvature"):
            d = self.delta if self.delta is not None else (length_scale ** -2 if self.alpha < 1 else 0.0)
            return GaussCurvatureLaw(self.alpha, self.eps, d)
        raise ValueError(f"unknown law {self.law!r}")


@dataclass
class ErosionResult:
    mesh: TriMesh
    snapshots: list = field(default_factory=list)     # [(fraction, time, TriMesh)]
    history: list = field(default_factory=list)       # [StepRecord]
    grid: Grid | None = None
    stop_reason: str = ""
    time: float = 0.0
    fraction: float = 1.0
    initial_volume_grid: float = 0.0
    initial_volume_mesh: float = 0.0
    timings: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    initial_mesh: TriMesh | None = None
    level_set: LevelSet | None = None
    simulation: ErosionSimulation | None = None

    def summary(self) -> str:
        t = self.timings
        return (f"stop: {self.stop_reason}, steps {len(self.history)}, time {self.time:.6g}, "
                f"volume fraction {self.fraction:.5f} | {self.grid.describe()} | "
                f"sdf {t.get('sdf', 0):.1f}s, evolution {t.get('evolve', 0):.1f}s, "
                f"meshing {t.get('mesh', 0):.1f}s")


def make_grid(mesh: TriMesh, params: ErosionParams) -> Grid:
    lo, hi = mesh.bounds
    ext = float((hi - lo).max())
    h = params.spacing if params.spacing else ext / params.resolution
    pad = (params.band + MARGIN + 2) * h
    if params.law == "svi":
        pad = max(pad, params.radius + 3 * h)
    return Grid.around(lo, hi, h, pad)


def point_mesh_distance(points: np.ndarray, mesh: TriMesh, max_edge: float | None = None,
                        k: int = 16, device=None, exact: bool = False) -> np.ndarray:
    """Exact unsigned distance from points to a triangle mesh.

    Triangles longer than ``max_edge`` (default: twice the median edge) are
    bisected along their longest edges, so the pieces are fat.  For each point
    the ``k`` nearest piece centroids (k-d tree) give candidate triangles and an
    upper bound ``d`` of the distance.  If ``rho`` bounds the distance from a
    centroid to the points of its triangle, the closest triangle has its
    centroid within ``d + rho``; points whose k-th nearest centroid is closer
    than that are re-queried with 4k candidates up to 256.  With ``exact=True``
    the remaining points get a ball query, so the result is exact; otherwise
    (default, much faster for points far from the mesh) their value is an upper
    bound exceeding the distance by at most ``d - (r_256 - rho)``, with r_256 the
    distance to the 256th centroid -- in practice a tiny relative error.
    """
    from scipy.spatial import cKDTree

    dev = get_device(device)
    points = np.asarray(points, dtype=np.float64)
    tri = mesh.triangles()
    e = np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1)
    max_edge = max(max_edge or 0.0, 2.0 * float(np.median(e)))
    tri = bisect_long_triangles(tri, max_edge)
    cen = tri.mean(1)
    rho = float(np.linalg.norm(tri - cen[:, None, :], axis=2).max())
    tree = cKDTree(cen)
    ref = points.mean(0) if len(points) else np.zeros(3)
    T = torch.as_tensor(tri - ref, dtype=torch.float32, device=dev)

    def evaluate(pidx, cand):
        """min distance from points[pidx] to triangles cand (n, kk), padded with -1."""
        res = np.empty(len(pidx))
        step = max(1, (1 << 22) // max(cand.shape[1], 1))
        for s0 in range(0, len(pidx), step):
            P = torch.as_tensor(points[pidx[s0:s0 + step]] - ref, dtype=torch.float32, device=dev)
            C = torch.as_tensor(cand[s0:s0 + step], device=dev)
            t = T[C.clamp_min(0)]
            d2 = _point_triangle_distance2(P[:, None, :], t[:, :, 0], t[:, :, 1], t[:, :, 2])
            d2 = torch.where(C >= 0, d2, torch.full_like(d2, float("inf")))
            res[s0:s0 + step] = d2.min(1).values.sqrt().double().cpu().numpy()
        return res

    out = np.full(len(points), np.inf)
    todo = np.arange(len(points))
    use_ball = exact
    kk = min(k, len(tri))
    while len(todo):
        dk, cand = tree.query(points[todo], k=kk, workers=-1)
        dk, cand = dk.reshape(len(todo), kk), cand.reshape(len(todo), kk)
        out[todo] = np.minimum(out[todo], evaluate(todo, cand))
        complete = (dk[:, -1] >= out[todo] + rho) | (kk >= len(tri))
        todo = todo[~complete]
        if kk >= 256 and len(todo):
            if not use_ball:
                break
            lists = tree.query_ball_point(points[todo], out[todo] + rho, workers=-1)
            width = max(len(l) for l in lists)
            cand = np.full((len(todo), width), -1, dtype=np.int64)
            for i, l in enumerate(lists):
                cand[i, :len(l)] = l
            out[todo] = np.minimum(out[todo], evaluate(todo, cand))
            break
        kk = min(4 * kk, len(tri))
    return out


def _annotate(sim: ErosionSimulation, mesh: TriMesh, initial: TriMesh, law: SpeedLaw) -> None:
    """Attach per-vertex depth (distance to the initial surface), N_r / curvature and speed."""
    if len(mesh.vertices) == 0:
        return
    grid = sim.ls.grid
    mesh.vertex_data["depth"] = point_mesh_distance(mesh.vertices, initial,
                                                    max_edge=4 * grid.spacing,
                                                    device=sim.ls.device).astype(np.float32)
    if law.requires_svi:
        N = sample_field(sim.svi_field(), grid, mesh.vertices)
        mesh.vertex_data["N"] = N.astype(np.float32)
        mesh.vertex_data["speed"] = law.speed(torch.as_tensor(N)).numpy().astype(np.float32)
    else:
        ls = sim.ls
        k1, k2 = ls.principal_curvatures(ls.band)
        dense = torch.zeros(grid.size, dtype=torch.float32, device=ls.device)
        dense[ls.band] = k1
        K1 = sample_field(dense, grid, mesh.vertices)
        dense[ls.band] = k2
        K2 = sample_field(dense, grid, mesh.vertices)
        mesh.vertex_data["H"] = (0.5 * (K1 + K2)).astype(np.float32)
        mesh.vertex_data["K"] = (K1 * K2).astype(np.float32)
        V, _ = law.speed_and_diffusivity(torch.as_tensor(K1), torch.as_tensor(K2))
        mesh.vertex_data["speed"] = V.numpy().astype(np.float32)


def erode(mesh: TriMesh, params: ErosionParams, callback=None, verbose: bool = False,
          keep_state: bool = False) -> ErosionResult:
    """Erode ``mesh`` under ``params.law`` until ``params.target_fraction`` of its volume remains."""
    dev = get_device(params.device)
    topo = mesh.edge_topology()
    if topo["boundary_edges"] or topo["nonmanifold_edges"]:
        warnings.warn(f"input mesh is not closed ({topo['boundary_edges']} boundary and "
                      f"{topo['nonmanifold_edges']} non-manifold edges); inside/outside is decided by a "
                      "majority vote of ray casts along the three axes, check the result", RuntimeWarning)
    lo, hi = mesh.bounds
    law = params.make_law(length_scale=float((hi - lo).max()))
    timings = {}
    t0 = time.time()
    grid = make_grid(mesh, params)
    phi = mesh_to_sdf(mesh, grid, band=params.band, device=dev, verbose=verbose)
    ls = LevelSet(phi, grid, width=params.band)
    timings["sdf"] = time.time() - t0
    if verbose:
        print(f"[erode] {grid.describe()}, device {dev}; law: {law.describe()}")
    sim = ErosionSimulation(ls, law, cfl=params.cfl, feedback_cfl=params.feedback_cfl,
                            parabolic_cfl=params.parabolic_cfl,
                            reinit_distance=params.reinit_distance,
                            reinit_iterations=params.reinit_iterations)
    if law.requires_svi and law.s == 1:
        n0 = sim.svi_field()
        nmax = float(tricubic(n0.view(-1), grid.shape, ls.closest_points(ls.interface_nodes())).max())
        del n0
        if nmax >= law.t:
            raise ValueError(f"max N_r on the initial surface is {nmax:.4f} >= t = {law.t}; "
                             "choose t closer to 1 (the flow needs t > N_r everywhere)")
        if verbose:
            print(f"[erode] max N_r on initial surface = {nmax:.4f} (t = {law.t})")
    targets = sorted({float(f) for f in params.snapshots if f > params.target_fraction},
                     reverse=True) + [params.target_fraction]
    snaps = []
    t1 = time.time()
    tmesh = 0.0
    res = None
    for frac in targets:
        res = sim.run(target_fraction=frac, max_steps=params.max_steps, callback=callback)
        tm = time.time()
        m = extract_surface(ls, project=params.project_vertices)
        _annotate(sim, m, mesh, law)
        tmesh += time.time() - tm
        snaps.append((res.fraction, sim.time, m))
        if res.stop_reason != "target_volume":
            break
    timings["evolve"] = time.time() - t1 - tmesh
    timings["mesh"] = tmesh
    final = snaps[-1][2]
    out = ErosionResult(final, snaps[:-1], sim.history, grid, res.stop_reason, sim.time,
                        res.fraction, sim.initial_volume, mesh.volume(), timings,
                        {**asdict(params), "law_description": law.describe(), **law.params()},
                        mesh)
    if keep_state:
        out.level_set, out.simulation = ls, sim
    return out


def volume_fraction_report(res: ErosionResult) -> str:
    v_in = res.initial_volume_mesh
    v_out = res.mesh.volume() if len(res.mesh.faces) else 0.0
    return (f"mesh volume {v_in:.6g} -> {v_out:.6g} (ratio {v_out / v_in:.5f}); "
            f"level-set volume ratio {res.fraction:.5f}")


__all__ = ["ErosionParams", "ErosionResult", "erode", "make_grid", "point_mesh_distance",
           "volume_fraction_report", "StepRecord"]
