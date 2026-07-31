"""
Tests for SpectrumSolver's multipole-axis contracts.

The internal contiguous ell axis (``lensing_ells``) must start exactly at 2
and step by 1: the Wigner-d recurrences chain adjacent entries, the lensing
correlation sums must cover the full multipole range, and get_Cl slices
outputs with ``[ells - 2]``. ``ellmin`` may only select which ells are
returned. The full-solve check that l_min=3 reproduces the l_min=2 spectra
on shared ells lives in accuracy_test.py.
"""

import jax.numpy as jnp
import pytest

from abcmb.spectrum import SpectrumSolver


def test_ellmin_below_2_raises():
    with pytest.raises(ValueError, match="l_min"):
        SpectrumSolver(ellmin=1, ellmax=100)


@pytest.mark.parametrize("lensing", [False, True])
def test_internal_ell_axis_anchored_at_2(lensing):
    ss = SpectrumSolver(ellmin=30, ellmax=100, lensing=lensing)
    assert int(ss.ells[0]) == 30
    assert int(ss.lensing_ells[0]) == 2
    assert bool(jnp.all(jnp.diff(ss.lensing_ells) == 1))
