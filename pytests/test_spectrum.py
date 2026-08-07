"""
Tests for SpectrumSolver's multipole-axis contracts.

The internal contiguous ell axis (``lensing_ells``) must start exactly at 2
and step by 1: the Wigner-d recurrences chain adjacent entries, the lensing
correlation sums must cover the full multipole range, and get_Cl slices
outputs with ``[ells - 2]``. ``l_min`` may only select which ells are
returned. The full-solve check that l_min=3 reproduces the l_min=2 spectra
on shared ells lives in accuracy_test.py.
"""

import jax.numpy as jnp
import pytest

from abcmb.inputs.schema import resolve_options
from abcmb.spectrum import SpectrumSolver


def _solver(**option_overrides):
    """A SpectrumSolver on schema defaults, with the k axis left unused.

    The multipole-axis contracts tested here depend only on ``l_min``,
    ``l_max`` and ``lensing``, so the k grid is a placeholder.
    """
    options = resolve_options(dict(option_overrides))
    return SpectrumSolver(jnp.array([0.1]), options)


def test_ellmin_below_2_raises():
    # resolve_options rejects l_min=1 on its own (an option bound is fatal),
    # so bypass it: this pins SpectrumSolver's *own* guard, which is what
    # stops jnp.where(bessel_l_tab <= 1)[0][-1] from dying on an empty array.
    options = resolve_options({"l_max": 100})
    options["l_min"] = 1
    with pytest.raises(ValueError, match="l_min"):
        SpectrumSolver(jnp.array([0.1]), options)


def test_ellmin_below_2_rejected_by_schema():
    with pytest.raises(ValueError, match=r"'l_min'.*below the minimum 2"):
        resolve_options({"l_min": 1, "l_max": 100})


@pytest.mark.parametrize("lensing", [False, True])
def test_internal_ell_axis_anchored_at_2(lensing):
    ss = _solver(l_min=30, l_max=100, lensing=lensing)
    assert int(ss.ells[0]) == 30
    assert int(ss.lensing_ells[0]) == 2
    assert bool(jnp.all(jnp.diff(ss.lensing_ells) == 1))


def test_bessel_tables_are_reproducible_from_scipy():
    # ties abcmb/_bessel_table_generation.py back to generate data
    # Columns are chosen to cover both x_lo branches -- l=2 starts the grid at
    # x=0 (and is the one column where phi2's 0/0 point has a nonzero limit,
    # 1/5), l=21 is the first column whose grid starts at the |f| = 1e-10
    # crossing. High-l columns are skipped: correctness there is the same code
    # path, but each costs ~1 min of scipy recurrences.
    import numpy as np

    from abcmb._generators import bessel_tables as gen
    from abcmb.spectrum import file_dir

    shipped = np.load(file_dir + "/data/bessel_tables.npz")
    ells = shipped["l"]

    for col in (0, 17):
        ell = int(ells[col])
        for name, fn in gen.KERNELS.items():
            x_hi = gen._nth_local_max(fn, ell)
            x_lo = gen._x_lo(fn, ell, x_hi)
            grid = np.linspace(x_lo, x_hi, gen.N_X)
            with np.errstate(divide="ignore", invalid="ignore"):
                vals = np.asarray(fn(ell, grid), dtype=float)
            if grid[0] == 0.0 and not np.isfinite(vals[0]):
                vals[0] = float(fn(ell, 1e-6))
            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)

            x_ship, v_ship = shipped["x" + name][:, col], shipped[name][:, col]
            # Endpoints: root-refinement tolerance, not storage precision.
            assert abs(x_lo - x_ship[0]) < 1e-5, f"l={ell} {name} x_lo"
            assert abs(x_hi - x_ship[-1]) < 1e-4, f"l={ell} {name} x_hi"
            # Values: the tables were written as ASCII to ~10 significant
            # digits, so ~1e-10 is the precision they were ever stored at.
            assert np.abs(vals - v_ship).max() < 1e-8, f"l={ell} {name} values"

    # The l=2 phi2 column is the one place the closed form is 0/0 while the
    # function is regular; a regression that zeroes it would be invisible in
    # the max-error check above (one point in 5000).
    assert shipped["phi2"][0, 0] == pytest.approx(0.2, abs=1e-9)


def test_bessel_tables_have_expected_structure():
    # Shape/dtype contract the interpolation path depends on: phi columns are
    # indexed by the same l ladder, and grids are uniform (fast_interp indexes
    # arithmetically rather than searching, so a non-uniform grid would be
    # silently misread).
    import numpy as np

    from abcmb.spectrum import file_dir

    z = np.load(file_dir + "/data/bessel_tables.npz")
    n_l = z["l"].size
    assert z["l"].dtype == np.int64
    for name in ("xphi0", "phi0", "xphi1", "phi1", "xphi2", "phi2"):
        assert z[name].shape == (5000, n_l), name
    for name in ("xphi0", "xphi1", "xphi2"):
        g = z[name]
        step = np.diff(g, axis=0)
        rel = np.abs(step / step[0] - 1.0).max()
        assert rel < 1e-5, f"{name} grid is not uniform (max dev {rel:.2e})"
