"""Time integration of erosion flows by the level set method.

The level set equation for inward normal velocity ``V >= 0`` with ``phi < 0``
inside is

    phi_t = V |grad phi|.

Discretisation:

* ``|grad phi|``: Godunov flux with WENO5 one-sided derivatives.
* ``V``: evaluated at the closest point ``x - phi grad phi / |grad phi|^2``
  of every band node (velocity extension constant along normals), from a
  dense field of the driving geometric quantity using tricubic
  interpolation.  For SVI laws that field is ``N_r`` (FFT convolution); for
  curvature laws the principal curvatures.
* Time: three-stage TVD Runge-Kutta; the (nonlocal) velocity is recomputed
  at every stage, so the coupling is also third-order in time.
* Step size: the minimum of the CFL limit ``cfl h / max V``, for SVI laws a
  limit ``feedback_cfl (4 r / 3) / max dV/dN`` on the relaxation rate of the
  nonlocal feedback (a surface bump of height delta changes N by up to
  3 delta / (4 r)), and for curvature laws the parabolic limit
  ``parabolic_cfl h^2 / max(dV/dk1 + dV/dk2)``.
* The band is reinitialised (subcell-fixed PDE redistancing) whenever the
  interface may have moved ``reinit_distance`` cells since the last one.
* The final step is shortened (secant iteration on dt) so that the eroded
  volume hits the requested fraction exactly.
"""

from __future__ import annotations

import math
import time as _time
from dataclasses import dataclass, field

import torch

from .flows import SpeedLaw
from .levelset import LevelSet, tricubic, trilinear
from .svi import BallConvolution


@dataclass
class StepRecord:
    step: int
    time: float
    dt: float
    volume: float
    fraction: float
    vmax: float
    vmean: float
    qmin: float = float("nan")      # N_r (SVI) or mean curvature (curvature laws) on interface
    qmax: float = float("nan")
    capped: int = 0
    reinit: bool = False
    wall: float = 0.0


@dataclass
class RunResult:
    stop_reason: str
    steps: int
    time: float
    fraction: float
    history: list = field(default_factory=list)


class ErosionSimulation:
    """Evolve a :class:`~svierosion.levelset.LevelSet` under a speed law."""

    def __init__(self, levelset: LevelSet, law: SpeedLaw, cfl: float = 0.5,
                 feedback_cfl: float = 1.0, parabolic_cfl: float = 0.25,
                 reinit_distance: float = 1.5, reinit_iterations: int = 20,
                 cutoff=(4.0, 5.5), newton_steps: int = 1):
        self.ls = levelset
        self.law = law
        self.cfl = float(cfl)
        self.feedback_cfl = float(feedback_cfl)
        self.parabolic_cfl = float(parabolic_cfl)
        self.reinit_distance = float(reinit_distance)
        self.reinit_iterations = int(reinit_iterations)
        self.cutoff = tuple(float(c) for c in cutoff)
        if self.cutoff[1] > levelset.width:
            raise ValueError("cutoff must lie inside the band")
        self.newton_steps = int(newton_steps)
        self.conv = None
        if law.requires_svi:
            self.conv = BallConvolution(levelset.grid, law.radius, device=levelset.device)
            self._check_convolution_margin()
        self.initial_volume = levelset.volume()
        self.time = 0.0
        self.steps = 0
        self.history: list[StepRecord] = []
        self._displacement = 0.0

    # ------------------------------------------------------------- checks
    def _check_convolution_margin(self) -> None:
        """Circular convolution is exact if the solid stays m+1 cells from the boundary."""
        m = self.conv.m + 1
        shape = self.ls.grid.shape
        inside = torch.nonzero(self.ls.flat < 0).squeeze(1)
        for ax, s in enumerate(self.ls.strides):
            c = (inside // s) % shape[ax]
            if int(c.min()) < m or int(c.max()) > shape[ax] - 1 - m:
                raise ValueError("grid padding is smaller than the SVI ball radius; "
                                 "use a larger padding")

    # ---------------------------------------------------------- velocity
    def _cutoff_factor(self, phi_band: torch.Tensor) -> torch.Tensor:
        a, b = self.cutoff[0] * self.ls.h, self.cutoff[1] * self.ls.h
        x = phi_band.abs()
        t = ((b - x) / (b - a)).clamp(0, 1)
        return t * t * (3 - 2 * t)

    def svi_field(self) -> torch.Tensor:
        """Dense N_r field for the current shape."""
        return self.conv.exterior_fraction(self.ls.dense_fractions())

    def _velocity(self, stats: bool = False):
        ls = self.ls
        idx = ls.band
        phi_b = ls.flat[idx]
        cp = ls.closest_points(idx, self.newton_steps)
        info = {}
        near = phi_b.abs() < ls.h
        if self.law.requires_svi:
            Nd = self.svi_field()
            N = tricubic(Nd.view(-1), ls.grid.shape, cp)
            del Nd
            V = self.law.speed(N)
            if stats:
                Nn = N[near] if bool(near.any()) else N
                info["qmin"] = float(Nn.min())
                info["qmax"] = float(Nn.max())
                info["dvmax"] = float(self.law.dspeed(Nn).max())
                info["capped"] = int((Nn > getattr(self.law, "n_cap", 1.0)).sum())
        else:
            k1, k2 = ls.principal_curvatures(idx)
            dense = torch.zeros(ls.grid.size, dtype=torch.float32, device=ls.device)
            # trilinear: curvature fields are rough near sharp edges; avoid overshoot
            dense[idx] = k1
            k1c = trilinear(dense, ls.grid.shape, cp)
            dense[idx] = k2
            k2c = trilinear(dense, ls.grid.shape, cp)
            del dense
            V, D = self.law.speed_and_diffusivity(k1c, k2c)
            if stats:
                sel = near if bool(near.any()) else torch.ones_like(near)
                Hn = 0.5 * (k1c + k2c)[sel]
                info["qmin"] = float(Hn.min())
                info["qmax"] = float(Hn.max())
                info["dmax"] = float(D[sel].max())
        if stats:
            info["vmean"] = float(V[near].mean()) if bool(near.any()) else float(V.mean())
        V = V * self._cutoff_factor(phi_b)
        return V, info

    def _rhs(self, stats: bool = False):
        V, info = self._velocity(stats)
        G = self.ls.godunov_norm_inward(self.ls.band)
        return V * G, V, info

    def stable_dt(self, V: torch.Tensor, info: dict) -> float:
        h = self.ls.h
        vmax = float(V.max()) if V.numel() else 0.0
        dt = self.cfl * h / vmax if vmax > 0 else math.inf
        if self.law.requires_svi and info.get("dvmax", 0) > 0:
            dt = min(dt, self.feedback_cfl * (4.0 * self.law.radius / 3.0) / info["dvmax"])
        if self.law.requires_curvature and info.get("dmax", 0) > 0:
            dt = min(dt, self.parabolic_cfl * h * h / info["dmax"])
        return dt

    def _rk3(self, u0: torch.Tensor, L0: torch.Tensor, dt: float) -> None:
        f = self.ls.flat
        idx = self.ls.band
        f[idx] = u0 + dt * L0
        L1, _, _ = self._rhs()
        f[idx] = 0.75 * u0 + 0.25 * (f[idx] + dt * L1)
        L2, _, _ = self._rhs()
        f[idx] = u0 / 3.0 + (2.0 / 3.0) * (f[idx] + dt * L2)

    # ---------------------------------------------------------------- run
    def run(self, target_fraction: float | None = 0.9, max_steps: int = 100000,
            max_time: float | None = None, tol: float = 1e-6, callback=None) -> RunResult:
        """Evolve until the volume equals ``target_fraction`` of the initial volume
        (or ``max_time``/``max_steps`` is reached).  Can be called repeatedly with
        decreasing targets to take snapshots."""
        ls = self.ls
        V0 = self.initial_volume
        target = None if target_fraction is None else target_fraction * V0
        vol = ls.volume()
        reason = "max_steps"
        n_start = self.steps
        while self.steps - n_start < max_steps:
            if target is not None and vol <= target * (1 + tol):
                reason = "target_volume"
                break
            if max_time is not None and self.time >= max_time * (1 - 1e-12):
                reason = "max_time"
                break
            if not ls.has_interface():
                reason = "vanished"
                break
            t_wall = _time.time()
            idx = ls.band
            u0 = ls.flat[idx].clone()
            L0, Vb, info = self._rhs(stats=True)
            dt = self.stable_dt(Vb, info)
            if max_time is not None:
                dt = min(dt, max_time - self.time)
            if not math.isfinite(dt) or dt <= 0:
                reason = "stalled"
                break
            self._rk3(u0, L0, dt)
            vnew = ls.volume()
            if target is not None and vnew < target:
                # secant / regula falsi on dt for V(dt) = target
                lo_t, lo_v, hi_t, hi_v = 0.0, vol, dt, vnew
                for _ in range(8):
                    if hi_v >= lo_v:
                        break
                    dts = lo_t + (target - lo_v) * (hi_t - lo_t) / (hi_v - lo_v)
                    ls.flat[idx] = u0
                    self._rk3(u0, L0, dts)
                    v = ls.volume()
                    if abs(v - target) <= tol * V0:
                        break
                    if v > target:
                        lo_t, lo_v = dts, v
                    else:
                        hi_t, hi_v = dts, v
                dt, vnew = dts, v
            vmax = float(Vb.max())
            self.time += dt
            self.steps += 1
            self._displacement += dt * vmax
            did_reinit = False
            reached = target is not None and vnew <= target * (1 + tol)
            if self._displacement >= self.reinit_distance * ls.h and not reached:
                ls.reinitialize(self.reinit_iterations)
                self._displacement = 0.0
                did_reinit = True
                vnew = ls.volume()
            rec = StepRecord(self.steps, self.time, dt, vnew, vnew / V0, vmax, info.get("vmean", vmax),
                             info.get("qmin", float("nan")), info.get("qmax", float("nan")),
                             info.get("capped", 0), did_reinit, _time.time() - t_wall)
            self.history.append(rec)
            if callback is not None:
                callback(self, rec)
            vol = vnew
        return RunResult(reason, self.steps, self.time, vol / V0, self.history)
