"""
Tests for the k-grid construction loops in :mod:`abcmb.model_setup`.

The grids are built by eager while-loops whose step size comes from user
options; a step that underflows relative to k would leave ``k + step == k``
and hang forever. CLASS guards its equivalent loops the same way
(perturbations.c, ``smallest_allowed_variation``); before the guard, the
only backstop here was overflowing a fixed-size buffer with an opaque
IndexError.
"""

import warnings

import pytest


def _options(**overrides):
    from abcmb import schema

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return schema.resolve_options(overrides)


def test_perturbation_k_step_underflow_raises():
    from abcmb.model_setup import get_k_axis_perturbations

    with pytest.raises(ValueError, match="k_step_super"):
        get_k_axis_perturbations(_options(k_step_super=0.0, k_step_sub=0.0))


def test_transfer_k_step_underflow_raises():
    from abcmb.model_setup import get_k_axis_transfer

    with pytest.raises(ValueError, match="k_transfer_linstep"):
        get_k_axis_transfer(_options(k_transfer_linstep=0.0), 1e-5, 0.3)


def test_large_k_grid_warns():
    # Declared successor to the old 2000-slot buffer: an LSS-scale k_max is
    # served (unlike the old opaque IndexError) but flagged as expensive,
    # since every grid point is a full Boltzmann solve and the linear
    # extension step does not scale like CLASS's log-per-decade sampling.
    from abcmb.model_setup import get_k_axis_perturbations

    with pytest.warns(UserWarning, match="perturbation k-grid has"):
        get_k_axis_perturbations(_options(k_max=12.0))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        get_k_axis_perturbations(_options())  # defaults: no warning
