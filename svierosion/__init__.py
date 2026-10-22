"""svierosion: erosion of solids by spherical-volume-invariant (SVI) flows.

Level set implementation of the flows with inward normal velocity

    V = N_r(x)^k / (t - N_r(x))^s,

where N_r(x) is the normalised exterior spherical volume invariant
(|B_r(x) \\ Omega| / |B_r|), together with curvature-driven comparison flows
V = (H_+)^alpha and V = (kappa_1+ kappa_2+)^alpha.

Typical use::

    from svierosion import read_ply, write_ply, ErosionParams, erode
    mesh = read_ply("fragment.ply")
    res = erode(mesh, ErosionParams(law="svi", radius=2.0, k=1, s=1, t=0.9,
                                    target_fraction=0.9, resolution=300))
    write_ply("fragment_eroded.ply", res.mesh)
"""

from .api import ErosionParams, ErosionResult, erode, point_mesh_distance
from .evolve import ErosionSimulation
from .flows import GaussCurvatureLaw, MeanCurvatureLaw, SVIPowerLaw
from .grid import Grid, get_device
from .levelset import LevelSet
from .meshio import TriMesh, read_ply, write_ply
from .sdf import mesh_to_sdf
from .surface import extract_surface
from .svi import BallConvolution

__version__ = "0.1.0"

__all__ = ["ErosionParams", "ErosionResult", "erode", "point_mesh_distance", "ErosionSimulation",
           "GaussCurvatureLaw", "MeanCurvatureLaw", "SVIPowerLaw", "Grid", "get_device", "LevelSet",
           "TriMesh", "read_ply", "write_ply", "mesh_to_sdf", "extract_surface", "BallConvolution"]
