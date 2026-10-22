"""Mesh I/O, signed distance construction and cell volume fractions."""

import os
import tempfile
import unittest

import numpy as np
import torch

from svierosion import shapes
from svierosion.grid import Grid, get_device
from svierosion.levelset import cube_plane_fraction
from svierosion.meshio import read_ply, write_ply
from svierosion.sdf import mesh_to_sdf, point_triangle_distance2

DEVICE = get_device(os.environ.get("SVIEROSION_TEST_DEVICE"))


class TestPly(unittest.TestCase):
    def test_roundtrip_binary_and_ascii(self):
        m = shapes.torus(1.0, 0.3, 32, 16)
        q = m.vertices[:, 2].copy()
        with tempfile.TemporaryDirectory() as d:
            for binary, prec in ((True, "float"), (True, "double"), (False, "double")):
                p = os.path.join(d, f"t_{binary}_{prec}.ply")
                write_ply(p, m, binary=binary, precision=prec, vertex_data={"quality": q})
                r = read_ply(p)
                tol = 1e-6 if prec == "float" else 1e-12
                self.assertTrue(np.allclose(r.vertices, m.vertices, atol=tol))
                self.assertTrue(np.array_equal(r.faces, m.faces))
                self.assertTrue(np.allclose(r.vertex_data["quality"], q, atol=1e-6))

    def test_quads_are_triangulated(self):
        txt = ("ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\n"
               "property float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n"
               "0 0 0\n1 0 0\n1 1 0\n0 1 0\n4 0 1 2 3\n")
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "q.ply")
            with open(p, "w") as fh:
                fh.write(txt)
            m = read_ply(p)
        self.assertEqual(m.faces.shape, (2, 3))

    def test_toy_shapes_are_closed_and_outward(self):
        for name, make in shapes.SHAPES.items():
            m = make()
            self.assertTrue(m.is_watertight(), name)
            self.assertTrue(m.is_consistently_oriented(), name)
            self.assertGreater(m.signed_volume(), 0, name)


class TestSDF(unittest.TestCase):
    def test_cube_is_exact(self):
        m = shapes.cube(1.0)
        for h in (0.05, 0.0137):
            g = Grid.around(*m.bounds, spacing=h, padding=0.3)
            phi = mesh_to_sdf(m, g, band=6, device=DEVICE).cpu().numpy()
            X, Y, Z = np.meshgrid(*g.axes(), indexing="ij")
            ex = np.clip(shapes.box_sdf(np.stack([X, Y, Z], -1)), -6 * h, 6 * h)
            self.assertLess(np.abs(phi - ex).max(), 2e-5 * h)

    def test_band_distances_match_brute_force(self):
        m = shapes.torus(0.6, 0.25, 48, 24)
        h = 0.03
        g = Grid.around(*m.bounds, spacing=h, padding=0.25)
        phi = mesh_to_sdf(m, g, band=6, device=DEVICE)
        flat = phi.view(-1)
        band = torch.nonzero(flat.abs() < 5.99 * h).squeeze(1)
        sel = band[torch.randperm(band.numel(), generator=torch.Generator().manual_seed(0))[:3000].to(band.device)]
        pts = g.node_coords(sel).to(torch.float64)
        tri = torch.as_tensor(m.triangles(), dtype=torch.float64, device=pts.device)
        best = torch.full((len(pts),), float("inf"), dtype=torch.float64, device=pts.device)
        for c in range(0, len(tri), 512):
            t = tri[c:c + 512]
            d2 = point_triangle_distance2(pts[:, None], t[None, :, 0], t[None, :, 1], t[None, :, 2])
            best = torch.minimum(best, d2.min(1).values)
        exact = best.sqrt()
        err = (flat[sel].abs().double() - exact).abs().max().item()
        self.assertLess(err, 1e-5 * h * 10)
        # sign: inside the tube iff distance to the core circle < minor radius
        P = pts.cpu().numpy()
        rho = np.hypot(P[:, 0], P[:, 1])
        inside = np.hypot(rho - 0.6, P[:, 2]) < 0.25
        far = np.abs(np.hypot(rho - 0.6, P[:, 2]) - 0.25) > 0.01
        self.assertTrue(np.array_equal((flat[sel].cpu().numpy() < 0)[far], inside[far]))

    def test_sphere_mesh_error_is_facet_sag(self):
        m = shapes.icosphere(4, 0.8)
        sag = 0.8 - np.min(np.linalg.norm(m.triangles().mean(1), axis=1))
        h = 0.04
        g = Grid.around(*m.bounds, spacing=h, padding=0.3)
        phi = mesh_to_sdf(m, g, band=6, device=DEVICE).cpu().numpy()
        X, Y, Z = np.meshgrid(*g.axes(), indexing="ij")
        ex = np.clip(np.sqrt(X ** 2 + Y ** 2 + Z ** 2) - 0.8, -6 * h, 6 * h)
        self.assertLess(np.abs(phi - ex).max(), 1.05 * sag)


class TestBand(unittest.TestCase):
    def test_clamped_values_are_outside_the_band(self):
        # float32(6 h) can round below the float64 cap 6 h (here h = 0.0075)
        from svierosion.levelset import LevelSet
        h = 0.0075
        g = Grid((0.0, 0.0, 0.0), h, (24, 24, 24))
        phi = torch.full(g.shape, 6.0, dtype=torch.float32) * h
        phi[12, 12, 12] = -h
        ls = LevelSet(phi, g, width=6.0)
        self.assertLess(ls.band.numel(), 30)


class TestCellFraction(unittest.TestCase):
    def test_against_inclusion_exclusion_float64(self):
        g = torch.Generator().manual_seed(1)
        n = torch.randn(20000, 3, generator=g, dtype=torch.float64)
        n /= n.norm(dim=1, keepdim=True)
        s = (torch.rand(20000, generator=g, dtype=torch.float64) - 0.5) * 2.0
        a = n.abs().sort(dim=1, descending=True).values
        got = cube_plane_fraction(s.float(), a[:, 0].float(), a[:, 1].float(), a[:, 2].float()).double()
        # classical formula in float64 (fine for generic normals)
        x = -s + a.sum(1) / 2
        tot = torch.zeros_like(x)
        for e0 in (0, 1):
            for e1 in (0, 1):
                for e2 in (0, 1):
                    v = x - e0 * a[:, 0] - e1 * a[:, 1] - e2 * a[:, 2]
                    tot += (-1) ** (e0 + e1 + e2) * v.clamp_min(0) ** 3
        ref = (tot / (6 * a.prod(1))).clamp(0, 1)
        ok = a[:, 2] > 1e-3
        self.assertLess((got - ref)[ok].abs().max().item(), 2e-5)

    def test_degenerate_normals(self):
        s = torch.linspace(-1, 1, 101)
        one, zero = torch.ones_like(s), torch.zeros_like(s)
        f = cube_plane_fraction(s, one, zero, zero)            # axis-aligned plane
        self.assertTrue(torch.allclose(f, (0.5 - s).clamp(0, 1), atol=1e-6))
        r = 2 ** -0.5 * one
        f2 = cube_plane_fraction(s, r, r, zero)               # 45 degree plane
        self.assertTrue(torch.isfinite(f2).all())
        self.assertAlmostEqual(float(cube_plane_fraction(torch.tensor([0.0]), *(torch.tensor([3 ** -0.5]),) * 3)), 0.5, 6)


if __name__ == "__main__":
    unittest.main()
