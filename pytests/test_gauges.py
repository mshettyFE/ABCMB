"""
Unit tests for the :mod:`abcmb.gauges` package and the vocabulary it shares
with the fluids (:mod:`abcmb.metric`).

Scope: whatever can be exercised without a cosmology. Nothing here builds a
``Model``, solves a background, or integrates anything -- those live in
``test_gauges_integration.py``. Keeping the split explicit is what makes it
visible when a part of the gauge package has no isolated coverage at all.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _options(**kwargs):
    from abcmb.inputs import schema

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return schema.resolve_options(kwargs)


def test_resolve_gauge_accepts_exactly_two_spellings():
    # One lowercase spelling per gauge, matching CLASS. No aliases and no case
    # folding: the config value, the options value, the GaugeName member and
    # the warning text are then all the same string.
    from abcmb.gauges import NewtonianGauge, SynchronousGauge, resolve_gauge
    from abcmb.species import GaugeName

    assert isinstance(resolve_gauge("synchronous"), SynchronousGauge)
    assert isinstance(resolve_gauge("newtonian"), NewtonianGauge)
    assert [str(g) for g in GaugeName] == ["synchronous", "newtonian"]

    for rejected in ("conformal", "longitudinal", "Newtonian", "SYNCHRONOUS"):
        with pytest.raises(ValueError, match="is not one of"):
            resolve_gauge(rejected)


def test_metric_structs_share_no_field_names():
    # The whole point of separate structs: reading PT.metric.eta on a
    # newtonian run must raise, not silently return a different quantity.
    from abcmb.gauges import NewtonianMetric, SynchronousMetric

    sync = set(SynchronousMetric.__annotations__)
    newt = set(NewtonianMetric.__annotations__)
    assert sync & newt == set(), (
        "a shared field name lets gauge-specific code read the wrong gauge "
        "silently instead of raising AttributeError"
    )


def test_fluids_cannot_see_the_gauge():
    # The core invariant of the design. A Fluid may declare which gauge its
    # own ICs are in; it must have no way to learn which gauge it is being
    # integrated in, since that is what makes one y_prime correct in both.
    from abcmb import model_setup, species

    fluids = model_setup.populate_species(None, _options())
    for f in fluids:
        attrs = set(dir(f))
        assert "gauge" not in attrs
        # ic_gauge is a claim about y_ini only, and never varies with the run.
        assert f.ic_gauge == species.GaugeName.SYNCHRONOUS
    # ...and it does not change when the model runs in the other gauge.
    newtonian_fluids = model_setup.populate_species(None, _options(gauge="newtonian"))
    for f in newtonian_fluids:
        assert f.ic_gauge == species.GaugeName.SYNCHRONOUS


def test_untransformable_ic_gauge_raises_at_construction():
    # A direct Fluid subclass declaring the other gauge's ICs but not saying
    # how to transform them must fail loudly at Model construction, not run
    # with initial conditions that mean something else.
    from abcmb import model_setup, species

    class Stubborn(species.Fluid):
        name = "Stubborn"
        num_equations = 1
        is_matter = False
        ic_gauge = species.GaugeName.NEWTONIAN

        def rho(self, lna, args):
            return 0.0

        def P(self, lna, args):
            return 0.0

    # Synchronous run: the declaration disagrees, so it must raise...
    with pytest.raises(NotImplementedError, match="does not implement y_ini_shift"):
        model_setup.populate_species((Stubborn,), _options())

    # ...but running in the gauge it declares needs no transformation.
    fluids = model_setup.populate_species((Stubborn,), _options(gauge="newtonian"))
    assert "Stubborn" in {f.name for f in fluids}


def test_standard_fluid_gets_the_shift_for_free():
    # The other half of the contract: a StandardFluid subclass declaring the
    # other gauge needs to write nothing, because the delta/theta/sigma shift
    # is implemented on the base class -- and the only fluid-specific piece,
    # the 1+w factor on the density, comes from the fluid's own w.
    from abcmb import model_setup, species
    from abcmb.species import GaugeShift

    class BorrowedMatter(species.StandardFluid):
        name = "BorrowedMatter"
        is_matter = True
        ic_gauge = species.GaugeName.NEWTONIAN
        num_equations = 3  # delta, theta, sigma

        def rho(self, lna, args):
            return 1e-9 / jnp.exp(lna) ** 3

        def P(self, lna, args):  # w = 0
            return 0.0

        def y_ini(self, k, tau_ini, args):
            return jnp.array([1.0, 2.0, 3.0])

    class BorrowedRadiation(BorrowedMatter):
        name = "BorrowedRadiation"

        def P(self, lna, args):  # w = 1/3
            return self.rho(lna, args) / 3.0

    fluids = model_setup.populate_species(
        (BorrowedMatter, BorrowedRadiation), _options()
    )
    by_name = {f.name: f for f in fluids}

    shift = GaugeShift(
        delta_per_one_plus_w=jnp.asarray(1.5),
        theta=jnp.asarray(-0.000625),
        lna=jnp.asarray(-12.0),
    )

    # Same shift, two equations of state: only the density entry differs, by
    # exactly 1+w, and the gauge-invariant shear is untouched in both.
    np.testing.assert_allclose(
        np.asarray(by_name["BorrowedMatter"].y_ini_shift(shift, {})),
        [1.5, -0.000625, 0.0],
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        np.asarray(by_name["BorrowedRadiation"].y_ini_shift(shift, {})),
        [4.0 / 3.0 * 1.5, -0.000625, 0.0],
        rtol=1e-14,
    )


def test_gauge_shift_carries_two_independent_shifts():
    # GaugeShift must not assume the density and velocity shifts share a
    # generator. That is true for the synchronous/newtonian pair (both are
    # MB95's alpha) but is a property of that pair, not of gauge
    # transformations, so the struct carries the two separately.
    from abcmb.species import GaugeShift

    fields = set(GaugeShift.__annotations__)
    assert "alpha" not in fields, (
        "a single generator field bakes the synchronous<->newtonian relation "
        "into the fluid-facing struct"
    )
    assert {"delta_per_one_plus_w", "theta"} <= fields

    # The pair-specific relation lives in exactly one place instead.
    from abcmb.gauges import NewtonianGauge, SynchronousGauge
    from abcmb.gauges.base import _shift_from_alpha

    k, lna, aH, alpha = 0.05, -12.0, 2.0, -0.25
    forward = NewtonianGauge().ic_shift(k, lna, aH, alpha)
    backward = SynchronousGauge().ic_shift(k, lna, aH, alpha)
    expected = _shift_from_alpha(k, lna, aH, alpha)
    for field in ("delta_per_one_plus_w", "theta"):
        assert float(getattr(forward, field)) == float(getattr(expected, field))
        # The reverse direction is the exact negation.
        assert float(getattr(backward, field)) == -float(getattr(expected, field))


def test_ic_gauge_declaration_is_required():
    # Any fluid that writes its own y_ini must write the ic_gauge
    from abcmb import model_setup, species

    class Sloppy(species.StandardFluid):
        name = "Sloppy"
        num_equations = 2
        is_matter = True

        def rho(self, lna, args):
            return 1e-9 / jnp.exp(lna) ** 3

        def P(self, lna, args):
            return 0.0

        def y_ini(self, k, tau_ini, args):
            return jnp.array([1.0, 2.0])

    with pytest.raises(TypeError, match="does not declare ic_gauge"):
        model_setup.populate_species((Sloppy,), _options())

    # Declaring it — either value — satisfies the check.
    class Declared(Sloppy):
        name = "Declared"
        ic_gauge = species.GaugeName.SYNCHRONOUS

    assert "Declared" in {
        f.name for f in model_setup.populate_species((Declared,), _options())
    }

    # ...and a subclass inherits the declaration rather than repeating it.
    class Child(species.ColdDarkMatter):
        name = "Child"

    assert "Child" in {
        f.name for f in model_setup.populate_species((Child,), _options())
    }

    # Every built-in declares it explicitly rather than leaning on the default.
    for fluid in model_setup.populate_species(None, _options()):
        if fluid.num_equations == 0:
            continue
        assert any(
            "ic_gauge" in c.__dict__
            for c in type(fluid).__mro__
            if c is not species.Fluid
        ), f"{fluid.name} relies on the inherited ic_gauge default"


def _grav_reference(a):
    """
    ``4 pi G a^2 / c^2``, written out a second time.

    ``_random_state`` also calls this, but only to scale the random totals,
    where being order-of-magnitude right is all that is needed.
    """
    from abcmb import constants as cnst

    return 4.0 * np.pi * cnst.G * a**2 / cnst.c_Mpc_over_s**2


def _random_state(key):
    """
    A random state for the Einstein constraints, with each total scaled so its
    own term lands alongside the others.

    The scaling carries the content. Stress-energy enters through a
    ``4 pi G a^2 / c^2`` prefactor of order 1e-17, so O(1) densities leave every
    gravitational term far below ``k^2 eta`` and the assertions stop touching
    them. Each total therefore carries the powers of k and aH that put its
    contribution at the same magnitude as ``k^2 eta``.
    """
    from abcmb.gauges import FluidTotals

    k_key, a_key, aH_key, y_key, t_key = jax.random.split(key, 5)
    k = float(10 ** jax.random.uniform(k_key, minval=-3.0, maxval=0.0))
    a = float(10 ** jax.random.uniform(a_key, minval=-5.0, maxval=0.0))
    aH = float(10 ** jax.random.uniform(aH_key, minval=-2.0, maxval=1.0))
    metric_y = float(jax.random.normal(y_key)) or 1.0
    ref = k**2 * abs(metric_y) / _grav_reference(a)
    rd, rt, rs = (float(v) for v in jax.random.normal(t_key, (3,)))
    return {
        "k": k,
        "a": a,
        "aH": aH,
        "metric_y": metric_y,
        "totals": FluidTotals(rd * ref, rt * ref * k**2 / aH, rs * ref),
    }


def _call(gauge, st, metric_y=None):
    return gauge.sources(
        st["k"],
        st["a"],
        st["aH"],
        st["metric_y"] if metric_y is None else metric_y,
        st["totals"],
    )


def _gauges():
    from abcmb.gauges import NewtonianGauge, SynchronousGauge

    return SynchronousGauge(), NewtonianGauge()


# --- 1. reference-free -----------------------------------------------------


def test_sources_are_linear_in_the_state():
    # The Einstein constraints are linear in (metric_y, totals), so the metric
    # derivative and all three source slots must be too.

    import jax

    for key in jax.random.split(jax.random.PRNGKey(11), 6):
        k1, k2, kc = jax.random.split(key, 3)
        s1, s2 = _random_state(k1), _random_state(k2)
        s2.update({name: s1[name] for name in ("k", "a", "aH")})  # shared background
        c1, c2 = (float(v) for v in jax.random.normal(kc, (2,)))
        mixed = dict(s1)
        mixed["metric_y"] = c1 * s1["metric_y"] + c2 * s2["metric_y"]
        mixed["totals"] = jax.tree.map(
            lambda x, y: c1 * x + c2 * y, s1["totals"], s2["totals"]
        )
        for gauge in _gauges():
            (d1, m1), (d2, m2), (dm, mm) = (
                _call(gauge, s1),
                _call(gauge, s2),
                _call(gauge, mixed),
            )
            assert float(dm) == pytest.approx(c1 * float(d1) + c2 * float(d2), rel=1e-9)
            for got, a1, a2 in zip(
                jax.tree.leaves(mm),
                jax.tree.leaves(m1),
                jax.tree.leaves(m2),
                strict=True,
            ):
                assert float(got) == pytest.approx(
                    c1 * float(a1) + c2 * float(a2), rel=1e-9, abs=1e-300
                )


def test_each_gauge_has_exactly_one_identically_vanishing_slot():
    # euler vanishes in synchronous gauge, shear in newtonian -- identically,
    # for any state.
    from abcmb.gauges import NewtonianGauge, SynchronousGauge

    for key in jax.random.split(jax.random.PRNGKey(0), 8):
        st = _random_state(key)
        _, sync = _call(SynchronousGauge(), st)
        assert float(sync.euler) == 0.0
        assert float(sync.continuity) != 0.0 and float(sync.shear) != 0.0
        _, newt = _call(NewtonianGauge(), st)
        assert float(newt.shear) == 0.0
        assert float(newt.continuity) != 0.0 and float(newt.euler) != 0.0


# --- 2. cross-checks between independently computed outputs ----------------


def test_newtonian_potentials_match_the_synchronous_generator():
    """
    MB95 Eq. (18) relates the gauges: ``phi = eta - aH alpha`` and
    ``psi = alpha_dot + aH alpha``, i.e. ``aH (alpha' + alpha)`` in d/dlna.

    Follows from ``(rho+P)sigma`` is gauge invariant, so one set of
    totals serves both gauges.
    """
    from abcmb.gauges import NewtonianGauge, SynchronousGauge

    for key in jax.random.split(jax.random.PRNGKey(3), 8):
        st = _random_state(key)
        sync = SynchronousGauge().metric_history(
            st["k"], st["a"], st["aH"], st["metric_y"], st["totals"]
        )
        phi = NewtonianGauge().metric_y_ini(st["aH"], st["metric_y"], sync.alpha)
        newt = NewtonianGauge().metric_history(
            st["k"], st["a"], st["aH"], phi, st["totals"]
        )
        assert float(newt.psi) == pytest.approx(
            st["aH"] * (float(sync.alpha_prime) + float(sync.alpha)), rel=1e-8
        )


def test_synchronous_shear_slot_matches_the_generator():
    """
    ``sources.shear == k^2 alpha / aH``.
    """
    from abcmb.gauges import SynchronousGauge

    for key in jax.random.split(jax.random.PRNGKey(5), 8):
        st = _random_state(key)
        _, src = _call(SynchronousGauge(), st)
        m = SynchronousGauge().metric_history(
            st["k"], st["a"], st["aH"], st["metric_y"], st["totals"]
        )
        assert float(src.shear) == pytest.approx(
            st["k"] ** 2 * float(m.alpha) / st["aH"], rel=1e-9
        )


# --- 3. transcription pins -------------------------------------------------


def test_constraint_coefficients_are_pinned_to_mb95():
    """
    The irreducible part. ``h'`` and ``eta'`` (MB95 Eq. 21b, 21c) and
    ``psi``, ``phi'`` (Eq. 23b, 23d) can only be checked by writing them down
    a second time, because everything else that knows these relations is
    derived from these lines.

    A shared misreading of MB95 passes this. It is a fault localiser, not
    evidence of correctness.
    """
    from abcmb.gauges import NewtonianGauge, SynchronousGauge

    for key in jax.random.split(jax.random.PRNGKey(7), 8):
        st = _random_state(key)
        k, a, aH, y = st["k"], st["a"], st["aH"], st["metric_y"]
        grav = _grav_reference(a)
        rd, rt, rs = (float(v) for v in jax.tree.leaves(st["totals"]))

        h_prime = 2.0 / aH**2 * (k**2 * y + grav * rd)
        eta_prime = grav * rt / aH / k**2
        dy, src = _call(SynchronousGauge(), st)
        assert float(dy) == pytest.approx(eta_prime, rel=1e-10)
        assert float(src.continuity) == pytest.approx(h_prime / 2.0, rel=1e-10)

        psi = y - 3.0 * grav * rs / k**2
        dy, src = _call(NewtonianGauge(), st)
        assert float(dy) == pytest.approx(-psi + grav * rt / aH / k**2, rel=1e-10)
        assert float(src.euler) == pytest.approx(k**2 * psi / aH, rel=1e-10)


def test_ic_shift_values_are_the_mb95_generator():
    # Transcription pin. Deliberately not compared against _shift_from_alpha:
    # routing the expectation through the helper the implementation calls would
    # establish only that the two gauges differ by a sign.
    from abcmb.gauges import NewtonianGauge, SynchronousGauge

    for key in jax.random.split(jax.random.PRNGKey(13), 6):
        v_key, lna_key = jax.random.split(key)
        k, aH, alpha = (
            float(10**v)
            for v in jax.random.uniform(v_key, (3,), minval=-2.0, maxval=0.0)
        )
        lna = float(jax.random.uniform(lna_key, minval=-15.0, maxval=-5.0))
        fwd = NewtonianGauge().ic_shift(k, lna, aH, alpha)
        assert float(fwd.delta_per_one_plus_w) == pytest.approx(
            -3.0 * aH * alpha, rel=1e-13
        )
        assert float(fwd.theta) == pytest.approx(k**2 * alpha, rel=1e-13)
        assert float(fwd.lna) == lna
        back = SynchronousGauge().ic_shift(k, lna, aH, alpha)
        assert float(back.delta_per_one_plus_w) == pytest.approx(
            3.0 * aH * alpha, rel=1e-13
        )
        assert float(back.theta) == pytest.approx(-(k**2) * alpha, rel=1e-13)


def test_cmb_sources_from_each_metric():
    # Transcription pin for CLASS's integrated-by-parts source forms. The
    # Doppler offset is alpha in synchronous gauge (theta_b + k^2 alpha is the
    # Newtonian velocity) and zero in newtonian, where theta_b already is it.
    from abcmb.gauges import NewtonianMetric, SynchronousMetric

    for key in jax.random.split(jax.random.PRNGKey(17), 5):
        k_key, bg_key, s_key, n_key = jax.random.split(key, 4)
        k = float(10 ** jax.random.uniform(k_key, minval=-2.0, maxval=0.0))
        aH, aH_dot, g, g_prime, emk = (
            float(v) for v in jax.random.normal(bg_key, (5,))
        )

        def col(v):
            return jnp.full((2, 1), v)

        args = (jnp.array([k]), col(aH), col(aH_dot), col(g), col(g_prime), col(emk))
        eta, eta_prime, alpha, alpha_prime = (
            float(v) for v in jax.random.normal(s_key, (4,))
        )
        s = SynchronousMetric(
            eta=col(eta),
            h_prime=col(0.0),
            eta_prime=col(eta_prime),
            alpha=col(alpha),
            alpha_prime=col(alpha_prime),
        ).cmb_sources(*args)
        np.testing.assert_allclose(s.sw_potential, aH * alpha_prime, rtol=1e-12)
        np.testing.assert_allclose(s.theta_offset, alpha, rtol=1e-12)
        np.testing.assert_allclose(s.theta_offset_prime, alpha_prime, rtol=1e-12)
        np.testing.assert_allclose(
            s.isw_T0,
            g * (eta - aH * alpha_prime - 2.0 * aH * alpha)
            + 2.0 * emk * (aH * eta_prime - aH_dot * alpha - aH**2 * alpha_prime),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            s.isw_T1, emk * (aH * alpha_prime + 2.0 * aH * alpha - eta) * k, rtol=1e-12
        )

        phi, psi, phi_prime = (float(v) for v in jax.random.normal(n_key, (3,)))
        n = NewtonianMetric(
            phi=col(phi), psi=col(psi), phi_prime=col(phi_prime)
        ).cmb_sources(*args)
        np.testing.assert_allclose(n.sw_potential, psi, rtol=1e-12)
        np.testing.assert_allclose(n.theta_offset, 0.0)
        np.testing.assert_allclose(n.theta_offset_prime, 0.0)
        np.testing.assert_allclose(
            n.isw_T0, g * (phi - psi) + 2.0 * emk * aH * phi_prime, rtol=1e-12
        )
        np.testing.assert_allclose(n.isw_T1, emk * (psi - phi) * k, rtol=1e-12)
