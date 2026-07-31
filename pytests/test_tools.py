"""
Tests for the numerical helpers in :mod:`abcmb.ABCMBTools`.

``fast_interp`` exists as a uniform-grid replacement for ``jnp.interp``
(measured 7.4x faster at the spectrum.py transfer-integral shape: 3 interps
of the 1704-point transfer k-axis against 5000-row Bessel-table columns per
lna scan step; 4x on background-style scalar chains). Speed is not asserted
here -- CI timing is too noisy -- but the correctness contract that justifies
keeping it is.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from abcmb.ABCMBTools import d00, fast_interp, wigner_d_matrix

XP_MIN, XP_MAX, N = 0.0, 10.0, 101


def test_wigner_d00_matches_legendre():
    # Exact identity: d^l_00(beta) = P_l(cos beta). Running eagerly also
    # proves the l=0 boundary row (from d00's padding) generates no NaNs:
    # jax_debug_nans (on in conftest) checks every primitive here, which the
    # old blanket nan_to_num guard could not pass.
    mu = jnp.linspace(-0.95, 0.95, 9)
    ells = jnp.arange(2, 16)
    out = np.asarray(d00(mu, ells))
    for j, ell in enumerate(range(2, 16)):
        coeffs = np.zeros(ell + 1)
        coeffs[ell] = 1.0
        ref = np.polynomial.legendre.legval(np.asarray(mu), coeffs)
        assert np.allclose(out[:, j], ref, atol=1e-12)


def test_wigner_invalid_ells_fails_loudly():
    # ells below m is a usage error (sqrt(l^2 - m^2) goes NaN); the targeted
    # boundary guards must not scrub it. With jax_debug_nans on (conftest),
    # the NaN surfaces as an immediate FloatingPointError.
    with pytest.raises(FloatingPointError):
        wigner_d_matrix(jnp.linspace(-0.5, 0.5, 3), jnp.arange(1, 6), 3, 1)


def test_fast_interp_exact_on_linear_data():
    # Linear interpolation reproduces linear data exactly, at any query point.
    xp = jnp.linspace(XP_MIN, XP_MAX, N)
    fp = 3.0 * xp - 2.0
    x = jnp.array([0.017, 1.5, 4.9999, 7.03, 9.983])
    assert jnp.allclose(fast_interp(x, XP_MIN, XP_MAX, fp), 3.0 * x - 2.0, atol=1e-12)


def test_fast_interp_matches_jnp_interp_interior():
    # On a uniform grid the two implementations compute the same thing; the
    # only intended difference is index arithmetic vs searchsorted.
    xp = jnp.linspace(XP_MIN, XP_MAX, N)
    fp = jnp.sin(xp) * jnp.exp(-0.1 * xp)
    x = jnp.linspace(XP_MIN + 0.01, XP_MAX - 0.01, 1000)
    assert jnp.allclose(
        fast_interp(x, XP_MIN, XP_MAX, fp), jnp.interp(x, xp, fp), atol=1e-12
    )


def test_fast_interp_clamps_out_of_range():
    # Out-of-range queries clamp to the endpoint values, like jnp.interp.
    # Tolerance absorbs the eps=1e-6 index clip fast_interp uses to keep its
    # gather in bounds (it blends 1e-6 of the neighboring cell at the edges).
    xp = jnp.linspace(XP_MIN, XP_MAX, N)
    fp = jnp.cos(xp)
    x = jnp.array([XP_MIN - 5.0, XP_MIN, XP_MAX, XP_MAX + 5.0])
    expected = jnp.array([fp[0], fp[0], fp[-1], fp[-1]])
    assert jnp.allclose(fast_interp(x, XP_MIN, XP_MAX, fp), expected, atol=1e-5)


def test_fast_interp_scalar_query():
    fp = jnp.linspace(0.0, 1.0, N) ** 2
    out = fast_interp(2.5, XP_MIN, XP_MAX, fp)
    assert out.shape == ()
    assert jnp.isfinite(out)


def test_fast_interp_gradient_is_local_slope():
    # fast_interp runs inside jitted/differentiated graphs (transfer integrals,
    # background thermodynamics), so d/dx must be the segment slope.
    xp = jnp.linspace(XP_MIN, XP_MAX, N)
    fp = jnp.sin(xp)
    dx = (XP_MAX - XP_MIN) / (N - 1)
    x0 = 3.14  # mid-cell interior point
    i = int(x0 // dx)
    slope = (fp[i + 1] - fp[i]) / dx
    g = jax.grad(lambda x: fast_interp(x, XP_MIN, XP_MAX, fp))(x0)
    assert jnp.allclose(g, slope, atol=1e-12)
