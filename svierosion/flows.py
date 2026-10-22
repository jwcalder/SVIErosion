"""Erosion speed laws (inward normal velocities, always >= 0).

* :class:`SVIPowerLaw` -- the family ``V = N^k / (t - N)^s`` driven by the
  normalised exterior spherical volume invariant ``N = N_r(x)``.
* :class:`MeanCurvatureLaw` -- ``V = (H_+)^alpha``.
* :class:`GaussCurvatureLaw` -- ``V = ((kappa_1)_+ (kappa_2)_+)^alpha``.

Each law returns the speed and an estimate of the parameters that control
the explicit time step (see :mod:`svierosion.evolve`).
"""

from __future__ import annotations

import torch


class SpeedLaw:
    requires_svi = False
    requires_curvature = False
    name = "law"

    def describe(self) -> str:  # pragma: no cover - overridden
        return self.name

    def params(self) -> dict:  # pragma: no cover - overridden
        return {}


class SVIPowerLaw(SpeedLaw):
    """Inward speed ``V = N^k / (t - N)^s`` (+ optional constant ``eps``).

    Parameters
    ----------
    radius : ball radius r of the volume invariant (length units of the mesh).
    k : integer exponent >= 0.
    s : 0 or 1; s = 1 includes the "geometric" factor 1/(t - N) =
        (1/t) sum_j (N/t)^j, which makes the most exposed points erode
        dramatically faster.
    t : singularity location, 0 < t < 1, required when s = 1.  The flow is
        only defined where N < t.
    max_speed_ratio : when s = 1, N is capped at the value ``n_cap`` where
        V(n_cap) = max_speed_ratio * V(1/2) (V(1/2) being the speed of a flat
        surface).  This only acts where N comes very close to t (e.g. thin
        plates eroded from both sides, where N -> 1) and keeps the time step
        from collapsing; the number of capped nodes is reported.
    eps : additive constant (grassfire term), as in the paper's F(s) = s + eps.
    """

    requires_svi = True

    def __init__(self, radius: float, k: int = 1, s: int = 0, t: float | None = None,
                 max_speed_ratio: float = 100.0, eps: float = 0.0):
        if radius <= 0:
            raise ValueError("radius must be positive")
        if int(k) != k or k < 0:
            raise ValueError("k must be a nonnegative integer")
        if s not in (0, 1):
            raise ValueError("s must be 0 or 1")
        if s == 1:
            if t is None or not (0.5 < t < 1.0):
                raise ValueError("s = 1 requires 1/2 < t < 1 (t must exceed N on flat parts, N = 1/2)")
        if eps < 0:
            raise ValueError("eps must be >= 0")
        self.radius = float(radius)
        self.k = int(k)
        self.s = int(s)
        self.t = None if t is None else float(t)
        self.eps = float(eps)
        self.max_speed_ratio = float(max_speed_ratio)
        self.n_cap = 1.0
        if self.s == 1:
            target = self.max_speed_ratio * self._raw(0.5)
            lo, hi = 0.5, self.t
            for _ in range(200):          # bisection for V(n_cap) = target
                mid = 0.5 * (lo + hi)
                if self._raw(mid) < target:
                    lo = mid
                else:
                    hi = mid
            self.n_cap = lo
        self.name = f"svi_k{self.k}_s{self.s}" + (f"_t{self.t:g}" if self.s else "")

    def _raw(self, n: float) -> float:
        v = n ** self.k
        if self.s:
            v /= (self.t - n)
        return v + self.eps

    def speed(self, N: torch.Tensor) -> torch.Tensor:
        N = N.clamp(0.0, 1.0)
        if self.s:
            N = torch.clamp(N, max=self.n_cap)
            V = N ** self.k / (self.t - N)
        else:
            V = N ** self.k if self.k else torch.ones_like(N)
        return V + self.eps if self.eps else V

    def dspeed(self, N: torch.Tensor) -> torch.Tensor:
        """dV/dN (used for the stability restriction of the nonlocal feedback)."""
        N = N.clamp(1e-12, 1.0)
        if self.s:
            N = torch.clamp(N, max=self.n_cap)
            d = self.t - N
            V = N ** self.k / d
            return V * (self.k / N + 1.0 / d)
        if self.k == 0:
            return torch.zeros_like(N)
        return self.k * N ** (self.k - 1)

    def describe(self) -> str:
        if self.s:
            core = f"N^{self.k}/({self.t:g} - N)"
        else:
            core = f"N^{self.k}" if self.k else "1"
        if self.eps:
            core += f" + {self.eps:g}"
        return f"V = {core},  N = N_r with r = {self.radius:g}"

    def params(self) -> dict:
        return {"law": "svi", "radius": self.radius, "k": self.k, "s": self.s, "t": self.t,
                "eps": self.eps, "max_speed_ratio": self.max_speed_ratio, "n_cap": self.n_cap}


class _CurvatureLaw(SpeedLaw):
    requires_curvature = True

    def __init__(self, alpha: float = 1.0, eps: float = 0.0, delta: float = 0.0):
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if delta < 0:
            raise ValueError("delta must be >= 0")
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.delta = float(delta)
        if self.alpha < 1 and self.delta == 0:
            raise ValueError("alpha < 1 needs a regularisation delta > 0 (d/dx x^alpha is "
                             "infinite at 0, which makes explicit time steps collapse)")

    def _power(self, x: torch.Tensor) -> torch.Tensor:
        """x^alpha, regularised for alpha < 1 as x (x + delta)^(alpha - 1).

        The regularised function equals x^alpha (1 + O(delta/x)) for x >> delta,
        is linear with slope delta^(alpha-1) near 0 and has no constant offset.
        """
        if self.alpha == 1.0:
            return x
        if self.alpha < 1:
            return x * (x + self.delta) ** (self.alpha - 1)
        return x ** self.alpha

    def _dpower(self, x: torch.Tensor) -> torch.Tensor:
        a = self.alpha
        if a == 1.0:
            return torch.ones_like(x)
        if a < 1:
            y = x + self.delta
            return y ** (a - 2) * (a * x + self.delta)
        return a * x.clamp_min(0) ** (a - 1)


class MeanCurvatureLaw(_CurvatureLaw):
    """Inward speed ``V = ((H)_+)^alpha + eps`` with H = (kappa_1 + kappa_2)/2.

    Sign convention: H = 1/R > 0 on a sphere.  For alpha < 1 the power is
    regularised with ``delta`` (units of curvature), see ``_power``.
    """

    name = "mean"

    def speed_and_diffusivity(self, k1: torch.Tensor, k2: torch.Tensor):
        H = 0.5 * (k1 + k2)
        Hp = H.clamp_min(0)
        V = self._power(Hp) + self.eps
        # dV/dkappa_i = 0.5 * F'(H) for H > 0
        D = torch.where(H > 0, self._dpower(Hp), torch.zeros_like(H))  # = dV/dk1 + dV/dk2
        return V, D

    def describe(self) -> str:
        e = f" + {self.eps:g}" if self.eps else ""
        return f"V = (H_+)^{self.alpha:g}{e}"

    def params(self) -> dict:
        return {"law": "mean_curvature", "alpha": self.alpha, "eps": self.eps, "delta": self.delta}


class GaussCurvatureLaw(_CurvatureLaw):
    """Inward speed ``V = ((kappa_1)_+ (kappa_2)_+)^alpha + eps``.

    This is the Gauss curvature K at convex points and 0 at saddle points and
    concave points.  (Using K_+ = max(K, 0) instead would also erode concave
    elliptic points, e.g. the bottom of a pit, where the equation is backward
    parabolic and ill-posed.)  alpha = 1/4 is the affine-invariant flow of
    Rayleigh/Andrews; alpha = 1 is Firey's model.
    """

    name = "gauss"

    def speed_and_diffusivity(self, k1: torch.Tensor, k2: torch.Tensor):
        a = k1.clamp_min(0)
        b = k2.clamp_min(0)
        Kc = a * b
        V = self._power(Kc) + self.eps
        dF = self._dpower(Kc)
        D = dF * (a + b)                       # dV/dk1 + dV/dk2 = F'(K)(k2 + k1)
        D = torch.where(Kc > 0, D, dF * 0)
        return V, D

    def describe(self) -> str:
        e = f" + {self.eps:g}" if self.eps else ""
        return f"V = (k1_+ k2_+)^{self.alpha:g}{e}"

    def params(self) -> dict:
        return {"law": "gauss_curvature", "alpha": self.alpha, "eps": self.eps, "delta": self.delta}


def make_law(kind: str, **kw) -> SpeedLaw:
    kind = kind.lower()
    if kind == "svi":
        return SVIPowerLaw(**kw)
    if kind in ("mean", "mean_curvature", "h"):
        return MeanCurvatureLaw(**kw)
    if kind in ("gauss", "gauss_curvature", "k"):
        return GaussCurvatureLaw(**kw)
    raise ValueError(f"unknown law {kind!r}")


def flat_speed(law: SVIPowerLaw) -> float:
    """Speed of a flat surface (N = 1/2) under an SVI law."""
    return law._raw(0.5)


__all__ = ["SpeedLaw", "SVIPowerLaw", "MeanCurvatureLaw", "GaussCurvatureLaw", "make_law",
           "flat_speed"]
