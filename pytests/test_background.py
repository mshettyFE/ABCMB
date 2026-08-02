"""
Guards for the conformal-time tabulation (BackgroundPreRecomb).
"""

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from diffrax import (
    ForwardMode,
    Kvaerno5,
    ODETerm,
    PIDController,
    SaveAt,
    diffeqsolve,
)

import abcmb.constants as cnst
from abcmb.background import BackgroundPreRecomb

# kind of hacky, but fine for tests
CUT = BackgroundPreRecomb.lna_tau_cut


@pytest.fixture(scope="module")
def bg_setup():
    """Model, derived params, and a constructed BackgroundPreRecomb."""
    from abcmb.main import Model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Model(l_max=300, k_max=0.2)
        params = {
            "h": 0.6762,
            "omega_cdm": 0.1193,
            "omega_b": 0.0225,
            "A_s": 2.12424e-9,
            "n_s": 0.9709,
            "Neff": 3.044,
            "YHe": 0.245,
            "tau_reion": 0.0544,
        }
        full = model.add_derived_parameters(params)

    def _to_float(a):
        if eqx.is_array(a) and jnp.issubdtype(a.dtype, jnp.integer):
            return a.astype(jnp.float64)
        return a

    full = jax.tree_util.tree_map(_to_float, full)
    pre = model.get_BG_pre_recomb(full)

    def tau_approx(lna):
        # The RD closed form from _tabulate_conformal_time.
        return (
            jnp.exp(lna)
            / (cnst.H0_over_h / cnst.c_Mpc_over_s)
            / jnp.sqrt(full["omega_r"])
        )

    return model, full, pre, tau_approx


def test_early_window_respects_rtol(bg_setup):
    # Just above the cut both the ODE and the closed form are accurate, so
    # they must agree tightly (no reference solve needed).
    _, _, pre, tau_approx = bg_setup
    grid = np.asarray(pre.lna_tau_tab)
    mask = (grid > CUT) & (grid <= CUT + 1.0)
    expected = np.asarray(jax.vmap(tau_approx)(pre.lna_tau_tab[mask]))
    got = np.asarray(pre.tau_tab)[mask]
    rel = np.max(np.abs(got - expected) / expected)
    assert rel <= 1e-5, (
        f"early-window rel error {rel:.2e}: atol regime or cut regressed?"
    )


def test_rho_tot_scalar_contract(bg_setup):
    # Scalar-in, scalar-out; batching is the caller's explicit vmap.
    _, full, pre, _ = bg_setup
    assert jnp.shape(pre.rho_tot(-1.0, full)) == ()
    assert jnp.shape(pre.P_tot(-1.0, full)) == ()
    grid = jnp.linspace(-3.0, -1.0, 7)
    assert jnp.shape(jax.vmap(lambda l: pre.rho_tot(l, full))(grid)) == (7,)


def test_table_matches_converged_reference(bg_setup):
    # ONE deeper-start, tighter-tolerance solve
    _, full, pre, tau_approx = bg_setup
    grid = np.asarray(pre.lna_tau_tab)
    mask = grid >= -14.0
    t0 = CUT - 5.0
    ts = jnp.concatenate([jnp.array([CUT]), pre.lna_tau_tab[mask]])
    ref = diffeqsolve(
        ODETerm(pre._dtau_dlna),
        solver=Kvaerno5(),
        t0=t0,
        t1=0.0,
        dt0=1e-5,
        y0=tau_approx(t0),
        saveat=SaveAt(ts=ts),
        stepsize_controller=PIDController(rtol=1e-12, atol=1e-18),
        args=full,
        adjoint=ForwardMode(),
    ).ys

    seam_rel = float(jnp.abs(tau_approx(CUT) - ref[0]) / ref[0])
    assert seam_rel <= 5e-6, f"seam error {seam_rel:.2e}: was lna_tau_cut moved later?"

    got = np.asarray(pre.tau_tab)[mask]
    rel = np.max(np.abs(got - np.asarray(ref[1:])) / got)
    assert rel <= 3e-5, f"consumer-range tau error {rel:.2e} vs converged reference"


def test_tau0_omega_r_gradient_is_seed_only(bg_setup):
    # AD through the tabulation, including the static numpy split:
    # Proves that numpy usage doesn't break autodif. The only
    # omega_r consumer is the closed-form seed, so the derivative is
    # analytically -tau(cut)/(2*omega_r). Catches both broken tracing
    # (e.g. numpy touching traced values) and an accidental new omega_r
    # dependence in the background.
    model, full, pre, tau_approx = bg_setup

    def tau0_of(omega_r):
        p = dict(full)
        p["omega_r"] = omega_r
        return model.get_BG_pre_recomb(p).tau0

    x0 = jnp.asarray(full["omega_r"])
    ad = float(jax.jacfwd(tau0_of)(x0))
    analytic = float(-tau_approx(CUT) / (2.0 * x0))
    assert abs((ad - analytic) / analytic) <= 1e-8, (
        f"d tau0/d omega_r: AD {ad:+.6e} vs analytic seed-path {analytic:+.6e}"
    )


@pytest.fixture(scope="module")
def full_background(bg_setup):
    """A post-recombination Background (needs the HyRex solve)."""
    model, full, _pre, _ = bg_setup
    pre = model.get_BG_pre_recomb(full)
    recomb_inputs = pre.make_recomb_inputs(model.RecModel, full)
    recomb_out = eqx.filter_jit(model.RecModel)((recomb_inputs, full))
    return model.get_BG(full, pre, recomb_out), full


def test_drag_epoch_matches_known_cosmology(full_background):
    # Reference values for base-LCDM, from Planck Collaboration, "Planck 2018
    # results. VI. Cosmological parameters", A&A 641, A6 (2020)
    # [arXiv:1807.06209], TT,TE,EE+lowE+lensing:
    #     z_drag = 1059.94 +/- 0.30
    #     r_drag = 147.09 +/- 0.26 Mpc
    # The bands below are deliberately loose: this fixture's cosmology is
    # Planck-*like*, not the Planck best fit, so this is a "did the drag epoch
    # land in the right place" check, not a precision comparison.
    BG, full = full_background
    z_d = float(BG.z_d(full))
    rs_d = float(BG.rs_d(full))
    assert 1000.0 < z_d < 1120.0, f"drag redshift z_d = {z_d}"
    assert 130.0 < rs_d < 160.0, f"sound horizon at drag rs_d = {rs_d} Mpc"


def test_baryon_photon_ratio_scales_like_a(full_background):
    # both densities are known powers of a, so R must be exactly
    # linear in a.
    BG, full = full_background
    lna1, lna2 = -8.0, -10.0
    r1 = float(BG.R_ratio_lna(lna1, full))
    r2 = float(BG.R_ratio_lna(lna2, full))
    assert r1 > 0.0
    ratio = r2 / r1 / np.exp(lna2 - lna1)
    assert abs(ratio - 1.0) < 1e-12, f"R/a not constant: {ratio}"
