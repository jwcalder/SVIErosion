# SVIErosion

`svierosion` erodes triangulated solids (`.ply`) by **spherical-volume-invariant (SVI) flows** and by
curvature flows for comparison, using a narrow-band **level set method** on a uniform grid.
It is written in PyTorch and runs on a CUDA GPU or on the CPU.

The SVI family has inward normal velocity

$$ V(x) = \frac{N_r(x)^k}{(t - N_r(x))^s}, \qquad N_r(x) = \frac{|B_r(x)\setminus\Omega|}{|B_r|}, $$

with $k \ge 0$ an integer, $s\in\{0,1\}$, and $1/2 < t < 1$ (needed when $s=1$; $t$ must exceed
$N_r$ everywhere: $N_r = 1/2$ on flat parts, $3/4$ on a right-angled edge, $7/8$ at a cube corner).
The comparison flows are $V = (H_+)^\alpha$ (mean curvature) and $V = ((\kappa_1)_+(\kappa_2)_+)^\alpha$
(Gauss curvature at convex points). Every run stops when the solid has reached a prescribed fraction of
its initial volume (default 90%).

The implementation report, with all mathematical and numerical details, validation and toy examples, is in
[`report/report.pdf`](report/report.pdf).

## Installation

```bash
pip install -e .
```

Dependencies: `numpy`, `scipy`, `scikit-image`, `torch` (CUDA build recommended), `matplotlib`;
`vtk` is optional and used for rendering figures.

## Command line

```bash
python -m svierosion erode fragment.ply -o fragment_eroded.ply --radius 2.0 -k 1 -s 1 -t 0.93 --target 0.9 --resolution 400
```

* `--radius` is in mesh units (or `--radius-rel 0.1` for 10% of the largest bounding-box side).
* `--resolution N` puts N grid cells across the largest side of the object (or give `--spacing h`).
* `--snapshots 0.97 0.94` also writes meshes at those volume fractions.
* `--law mean --alpha 1` and `--law gauss --alpha 0.25` select the curvature flows.
* `--device cpu` forces the CPU; `--no-compile` disables `torch.compile` (slower, avoids compilation time).

Outputs: the eroded mesh (binary PLY, with per-vertex `depth` = distance to the original surface, `N`,
`speed`, and `quality` = a copy of `depth` that MeshLab displays as a colour map), a JSON file with the run
parameters and timings, and a CSV with the per-step history.

Other commands:

```bash
python -m svierosion compare fragment.ply -o comparison_dir --radius 2.0     # SVI vs curvature flows
python -m svierosion distance eroded.ply tumbled_scan.ply -o dist.ply        # surface distances
python -m svierosion shapes toy_dir                                           # write the toy solids
python -m svierosion info fragment.ply                                        # mesh statistics
```

## Python

```python
from svierosion import read_ply, write_ply, ErosionParams, erode

mesh = read_ply("fragment.ply")
res = erode(mesh, ErosionParams(law="svi", radius=2.0, k=1, s=1, t=0.93,
                                target_fraction=0.9, resolution=400, snapshots=(0.95,)))
write_ply("fragment_eroded.ply", res.mesh)
print(res.summary())
```

## Examples, tests, benchmarks

```bash
python examples/toy_examples.py          # cube, prism, ellipsoid, L-block, tube fragment
python -m unittest discover -s tests     # unit tests (set SVIEROSION_TEST_DEVICE=cpu for the CPU)
python benchmarks/convergence.py         # accuracy / convergence tables
python benchmarks/performance.py         # timings on a synthetic 1.3M-triangle "scan"
```

## Package layout

| module | content |
|---|---|
| `meshio.py` | PLY reader/writer (ascii/binary), `TriMesh` |
| `sdf.py` | exact signed distance of a mesh on a grid (GPU/CPU) |
| `levelset.py`, `kernels.py` | narrow band, WENO5/Godunov, reinitialisation, cell volume fractions, interpolation |
| `svi.py` | exact ball/cell overlap weights and FFT evaluation of $N_r$ |
| `flows.py` | speed laws |
| `evolve.py` | TVD-RK3 time integration, time-step control, stopping at a volume fraction |
| `surface.py` | marching cubes + projection to the zero set |
| `api.py`, `cli.py`, `compare.py` | high-level interface, command line, flow comparisons |
| `shapes.py`, `viz.py`, `figures.py` | toy solids, rendering, figures |
