"""Volume invariant accuracy, reinitialisation and flows with exact solutions."""

import math
import os
import tempfile
import unittest

import numpy as np
import torch
from scipy.integrate import solve_ivp

from svierosion import ErosionParams, erode, kernels, shapes
from svierosion.evolve import ErosionSimulation
from svierosion.flows import GaussCurvatureLaw, MeanCurvatureLaw, SVIPowerLaw
from svierosion.grid import Grid, get_device
from svierosion.levelset import LevelSet, tricubic
from svierosion.svi import BallConvolution, ball_cell_weights, sphere_exterior_fraction

DEVICE = get_device(os.environ.get("SVIEROSION_TEST_DEVICE"))
CTR = np.array([0.0123, -0.0311, 0.0217])


def sphere_levelset(R, h, pad, distort=False):
    g = Grid.around([-R] * 3, [R] * 3, h, padding=pad)
    X, Y, Z = [torch.tensor(a, device=DEVICE) for a in np.meshgrid(*g.axes(), indexing="ij")]
    phi = torch.sqrt((X - CTR[0]) ** 2 + (Y - CTR[1]) ** 2 + (Z - CTR[2]) ** 2) - R
    if distort:
        phi = phi * (1 + 0.4 * torch.sin(3 * X) * torch.cos(2 * Y)) + 0.3 * phi ** 2
    return LevelSet(phi.float(), g, width=6), g


def interface_radius(ls, g):
    idx = ls.band
    x = g.node_coords(idx).cpu().numpy()
    ph = ls.flat[idx].cpu().numpy()
    near = np.abs(ph) < g.spacing
    return np.linalg.norm(x[near] - CTR, axis=1) - ph[near]


class TestVolumeInvariant(unittest.TestCase):
    def test_ball_weights_sum_to_ball_volume(self):
        for ratio in (3.3, 8.0, 12.7):
            w = ball_cell_weights(1.0, 1.0 / ratio)
            self.assertAlmostEqual(w.sum() / ratio ** 3 / (4 * math.pi / 3), 1.0, places=4)

    def test_sphere_exterior_fraction(self):
        R, r = 1.0, 0.4
        for ratio, tol in ((8, 3e-4), (16, 6e-5)):
            h = r / ratio
            ls, g = sphere_levelset(R, h, r + 10 * h)
            conv = BallConvolution(g, r, device=DEVICE)
            N = conv.exterior_fraction(ls.dense_fractions())
            pts = ls.closest_points(ls.interface_nodes())
            vals = tricubic(N.view(-1), g.shape, pts)
            err = (vals - sphere_exterior_fraction(r, R)).abs().max().item()
            self.assertLess(err, tol, f"r/h={ratio}")

    def test_flat_plane_is_one_half(self):
        h, r = 0.05, 0.3
        g = Grid.around([-1] * 3, [1] * 3, h, padding=r + 10 * h)
        X, Y, Z = [torch.tensor(a, device=DEVICE) for a in np.meshgrid(*g.axes(), indexing="ij")]
        n = torch.tensor([0.3, -0.5, 0.81], dtype=torch.float64)
        n /= n.norm()
        # half ball of radius 0.9 cut by a tilted plane; test points on the flat face
        # farther than r from its rim, so every ball sees only the plane
        lin = n[0] * X + n[1] * Y + n[2] * Z - 0.0131
        phi = torch.maximum(lin, torch.sqrt(X * X + Y * Y + Z * Z) - 0.9).float()
        ls = LevelSet(phi, g)
        conv = BallConvolution(g, r, device=DEVICE)
        N = conv.exterior_fraction(ls.dense_fractions())
        idx = ls.interface_nodes()
        x = g.node_coords(idx).cpu()
        keep = ((x @ n - 0.0131).abs() < 2 * h) & (x.norm(dim=1) < 0.45)
        vals = tricubic(N.view(-1), g.shape, ls.closest_points(idx[keep.to(idx.device)]))
        self.assertLess((vals - 0.5).abs().max().item(), 2e-4)


class TestReinitialisation(unittest.TestCase):
    def test_interface_preserved(self):
        for h, tol in ((0.08, 0.01), (0.04, 0.005)):
            ls, g = sphere_levelset(1.0, h, 12 * h, distort=True)
            ls.reinitialize(40)
            idx = ls.band
            x = g.node_coords(idx).cpu().numpy()
            ph = ls.flat[idx].cpu().numpy()
            ex = np.linalg.norm(x - CTR, axis=1) - 1.0
            near = np.abs(ex) < 1.5 * h
            self.assertLess(np.abs(ph - ex)[near].max() / h, tol)


class TestFlowsOnSphere(unittest.TestCase):
    """The sphere stays a sphere; its radius obeys an ODE that we integrate exactly."""

    def _run(self, law, h, pad, T):
        ls, g = sphere_levelset(1.0, h, pad)
        ErosionSimulation(ls, law).run(target_fraction=None, max_time=T)
        return interface_radius(ls, g)

    def test_svi_laws(self):
        r, T = 0.4, 0.15
        for law, tol in ((SVIPowerLaw(r, 1, 0), 3e-5), (SVIPowerLaw(r, 2, 1, 0.8), 3e-4),
                         (SVIPowerLaw(r, 0, 0), 3e-5)):
            F = lambda R: float(law.speed(torch.tensor(sphere_exterior_fraction(r, R), dtype=torch.float64)))
            Rex = solve_ivp(lambda t, y: [-F(y[0])], (0, T), [1.0], rtol=1e-12, atol=1e-14).y[0, -1]
            Rn = self._run(law, r / 8, r + 10 * r / 8, T)
            self.assertLess(abs(Rn.mean() - Rex), tol, law.describe())
            self.assertLess(Rn.max() - Rn.min(), 10 * tol + 1e-4, law.describe())

    def test_curvature_laws(self):
        T, h = 0.1, 0.05
        cases = ((MeanCurvatureLaw(1.0), math.sqrt(1 - 2 * T)),
                 (GaussCurvatureLaw(1.0), (1 - 3 * T) ** (1 / 3)),
                 (GaussCurvatureLaw(0.25, delta=1e-4), (1 - 1.5 * T) ** (2 / 3)))
        for law, Rex in cases:
            Rn = self._run(law, h, 12 * h, T)
            self.assertLess(abs(Rn.mean() - Rex), 5e-4, law.describe())


class TestDriver(unittest.TestCase):
    def test_stops_exactly_at_target_volume(self):
        m = shapes.cube(1.0)
        res = erode(m, ErosionParams(law="svi", radius=0.2, k=1, s=1, t=0.9, resolution=60,
                                     target_fraction=0.93, device=DEVICE))
        self.assertEqual(res.stop_reason, "target_volume")
        self.assertAlmostEqual(res.fraction, 0.93, places=5)
        self.assertAlmostEqual(res.mesh.volume() / m.volume(), 0.93, delta=2e-3)
        self.assertTrue(res.mesh.is_watertight())
        for key in ("depth", "N", "speed"):
            self.assertIn(key, res.mesh.vertex_data)

    def test_rejects_t_below_max_n(self):
        with self.assertRaises(ValueError):
            erode(shapes.cube(1.0), ErosionParams(law="svi", radius=0.2, k=1, s=1, t=0.8,
                                                  resolution=40, device=DEVICE))

    def test_compiled_and_eager_kernels_agree(self):
        ls, g = sphere_levelset(1.0, 0.06, 0.8)
        idx = ls.band
        out = {}
        for flag in (True, False):
            kernels.set_compile(flag)
            try:
                out[flag] = (ls.godunov_norm_inward(idx), ls.cell_fractions(idx),
                             tricubic(ls.flat, g.shape, ls.closest_points(idx)))
            finally:
                kernels.set_compile(True)
        for a, b in zip(out[True], out[False]):
            self.assertLess((a - b).abs().max().item(), 1e-4)

    def test_cli_smoke(self):
        from svierosion.cli import main
        from svierosion.meshio import read_ply, write_ply
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "prism.ply")
            write_ply(src, shapes.triangular_prism(1.0, 1.0))
            out = os.path.join(d, "prism_eroded.ply")
            rc = main(["erode", src, "-o", out, "--radius-rel", "0.2", "-k", "1", "--resolution", "48",
                       "--target", "0.95", "--quiet"] + (["--device", str(DEVICE)]))
            self.assertEqual(rc, 0)
            m = read_ply(out)
            self.assertIn("quality", m.vertex_data)
            self.assertTrue(os.path.exists(os.path.join(d, "prism_eroded_history.csv")))


if __name__ == "__main__":
    unittest.main()
