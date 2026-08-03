"""
Offline regeneration of ``bessel_tab/bessel_tables.npz``.

The transfer-function integrals in :mod:`abcmb.spectrum` need three
spherical-Bessel kernels at every (l, x) they touch::

    phi0(l, x) = j_l(x)                      Sachs-Wolfe + ISW source
    phi1(l, x) = j_l'(x)                     Doppler source
    phi2(l, x) = (3 j_l''(x) + j_l(x)) / 2   quadrupole / polarization source

JAX has no spherical Bessel function (``jax.scipy.special.bessel_jn`` is
cylindrical, integer-order only, and returns every order up to ``v``), and a
recurrence-based evaluation would cost O(l) sequential steps at each of the
~10^8 points the integrals visit. So the kernels are tabulated once, here,
against SciPy, and interpolated at runtime by ``ABCMBTools.fast_interp``.

This module is *not* imported at runtime. It exists so the shipped tables have
a reproducible provenance rather than being opaque data, and it is pinned by
``pytests/test_spectrum.py`` against the tables actually shipped.

Regenerate with::

    python -m abcmb._generators.bessel_tables

Tabulation convention :

* Each column is one multipole from the ``l`` ladder; the ladder is
  non-uniform (step 1 at low l, up to 40 near l = 5000).
* ``x_hi`` is the fifth local maximum of the signed kernel, so each
  column covers the
  first five oscillations -- past that, ``spectrum.py`` switches to the
  large-x asymptotic expansion (``Q``/``J``/``j`` there).
* ``x_lo`` is where |f| first rises through ``1e-10``, or 0 for l <= 19,
  where the function is already appreciable at small x.
* The grid is ``linspace(x_lo, x_hi, 5000)`` -- uniform, which is what lets
  ``fast_interp`` index arithmetically instead of searching.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.special import spherical_jn

N_X = 5000  # grid points per column
FLOOR = 1e-10  # |f| threshold defining x_lo
N_MAX = 5  # tabulate out to the fifth local maximum
SMALL_L = 19  # at or below this, start the grid at x = 0


def phi0(ell: int, x):
    """Spherical Bessel j_l(x)."""
    return spherical_jn(ell, x)


def phi1(ell: int, x):
    """First derivative j_l'(x)."""
    return spherical_jn(ell, x, derivative=True)


def phi2(ell: int, x):
    """(3 j_l'' + j_l)/2, with j_l'' eliminated via the spherical Bessel ODE
    ``x^2 j'' + 2x j' + (x^2 - l(l+1)) j = 0``. Equivalent to the closed form
    ``((3l(l-1) - 2x^2) j_l + 6x j_{l+1}) / (2x^2)`` used by spectrum.py's
    large-x branch."""
    j = spherical_jn(ell, x)
    jp = spherical_jn(ell, x, derivative=True)
    return (-6.0 / x * jp + (3.0 * ell * (ell + 1) / x**2 - 2.0) * j) / 2.0


KERNELS = {"phi0": phi0, "phi1": phi1, "phi2": phi2}


def _nth_local_max(fn, ell: int, n: int = N_MAX) -> float:
    """x of the n-th local maximum of the *signed* ``fn(l, .)``."""
    # Oscillations live at x > l, but near the turning point x ~ l they are
    # stretched (spacing ~ l^(1/3)), so a fixed multiple of 2*pi undershoots at
    # large l -- l=5000's fifth maximum sits ~158 past l. Scale with l.
    lo_scan = max(ell - 5.0, 1e-6)
    hi_guess = ell * 1.15 + 20.0 * (n + 4)
    # Resolution, not a fixed point count: peaks are separated by >= ~pi (and
    # by ~l^(1/3) near the turning point), so 0.01 in x resolves them at every
    # l, while a fixed 800k points made high-l columns minutes-slow.
    npts = int(np.clip((hi_guess - lo_scan) / 0.01, 20_000, 300_000))
    scan = np.linspace(lo_scan, hi_guess, npts)
    v = fn(ell, scan)
    peaks = np.where((v[1:-1] > v[:-2]) & (v[1:-1] >= v[2:]))[0] + 1
    if len(peaks) < n:
        raise RuntimeError(f"only {len(peaks)} maxima found for l={ell}")
    i = peaks[n - 1]
    # Refine: f' changes sign at the peak.
    lo, hi = float(scan[i - 1]), float(scan[i + 1])
    h = (hi - lo) * 1e-6

    def d(x):
        return (fn(ell, x + h) - fn(ell, x - h)) / (2 * h)

    try:
        return float(brentq(d, lo, hi))
    except ValueError:
        return float(scan[i])


def _x_lo(fn, ell: int, x_hi: float) -> float:
    """Where |f| first rises through FLOOR; 0 for the small-l columns."""
    if ell <= SMALL_L:
        return 0.0
    return float(brentq(lambda x: abs(fn(ell, x)) - FLOOR, 1e-8, x_hi))


def build_tables(ells) -> dict[str, np.ndarray]:
    """Return ``{"l", "xphi0", "phi0", "xphi1", "phi1", "xphi2", "phi2"}``."""
    ells = np.asarray(ells, dtype=np.int64)
    out: dict[str, np.ndarray] = {"l": ells}
    for name, fn in KERNELS.items():
        xs = np.empty((N_X, ells.size))
        vs = np.empty((N_X, ells.size))
        for col, ell in enumerate(ells):
            ell = int(ell)
            hi = _nth_local_max(fn, ell)
            lo = _x_lo(fn, ell, hi)
            grid = np.linspace(lo, hi, N_X)
            xs[:, col] = grid
            with np.errstate(divide="ignore", invalid="ignore"):
                vals = np.asarray(fn(ell, grid), dtype=float)
            # phi2's closed form is 0/0 at x = 0 but the function is regular
            # there (1/5 at l=2, 0 above); evaluate the limit just off zero
            # rather than letting nan_to_num write a spurious 0.
            if grid[0] == 0.0 and not np.isfinite(vals[0]):
                vals[0] = float(fn(ell, 1e-6))
            vs[:, col] = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        out[f"x{name}"] = xs
        out[name] = vs
    return out


def main() -> None:
    import pathlib

    here = pathlib.Path(__file__).parent.parent / "bessel_tab" / "bessel_tables.npz"
    ells = np.load(here)["l"]  # preserve the shipped multipole ladder
    tables = build_tables(ells)
    np.savez(here, **tables)
    print(f"wrote {here} ({here.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
