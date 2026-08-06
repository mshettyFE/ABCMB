"""
Integration tests for gauge selection

Everything here builds a ``Model``, a ``Background``, or both, and exercises
the gauge through the machinery that consumes it -- initial conditions, the
species stack, the perturbation evolver. The isolated tests of the gauge
package itself are in ``test_gauges.py``.

Two things are being defended. First, that the conformal Newtonian gauge is
implemented consistently -- checked against the Einstein constraint that is
*redundant* in that gauge, so it holds only if the initial conditions and the
metric transformation are right. Second, that the traps around gauges are hard
to fall into: a forgotten ``sources.euler``, a mis-declared ``ic_gauge``, or
code reading one gauge's metric variables off another gauge's run.
"""

import warnings

import jax.numpy as jnp
import numpy as np


def _options(**kwargs):
    from abcmb.inputs import schema

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return schema.resolve_options(kwargs)


class _ForgetfulFluid:
    """Builder for a fluid whose Euler equation drops ``sources.euler``."""

    @staticmethod
    def make():
        from abcmb import species

        class Forgetful(species.StandardFluid):
            name = "Forgetful"
            num_equations = 2
            is_matter = True

            def rho(self, lna, args):
                return 1e-6 * args["omega_cdm"] / jnp.exp(lna) ** 3

            def P(self, lna, args):
                return 0.0

            def y_ini(self, k, tau_ini, args):
                return jnp.array([0.0, 0.0])

            def y_prime(self, k, lna, sources, y, args):
                aH = args.BG.aH(lna, args.params)
                theta = y[self.first_idx + 1]
                # The bug, and only this bug: the continuity equation is
                # complete, but the Euler equation drops sources.euler.
                return jnp.array([-(theta / aH + sources.continuity), -theta])

        return Forgetful


def test_euler_omission_is_invisible_to_synchronous_but_caught(full_background_pair):
    # sources.euler is identically
    # zero in synchronous gauge, so a fluid that never reads it is exactly
    # correct there -- no synchronous test can distinguish it.
    from abcmb.species import gauge_source_omissions, metric_source_dependence

    BG, params, species_list = full_background_pair

    # Baseline: no built-in fluid forgets any of them
    assert gauge_source_omissions(species_list, BG, params) == {}

    # Every perturbed built-in reads continuity and euler; only the ones
    # carrying a quadrupole read shear.
    dep = metric_source_dependence(species_list, BG, params)
    for name in ("Baryon", "Photon", "ColdDarkMatter", "MasslessNeutrino"):
        assert dep[name]["continuity"], f"{name} ignores sources.continuity"
        assert dep[name]["euler"], f"{name} ignores sources.euler"
    assert dep["Photon"]["shear"] and dep["MasslessNeutrino"]["shear"]
    assert not dep["ColdDarkMatter"]["shear"]
    assert not dep["Baryon"]["shear"]
    # A fluid with no perturbations reads nothing.
    assert dep["DarkEnergy"] == dict.fromkeys(("continuity", "euler", "shear"), False)

    # Negative control: the bug is detected, and named.
    forgetful = _ForgetfulFluid.make()(1, _options())
    flagged = gauge_source_omissions((*species_list, forgetful), BG, params)
    assert flagged.get("Forgetful") == ["sources.euler"], flagged


def test_dropped_velocity_in_continuity_is_caught(full_background_pair):
    # The other synchronous-invisible term: cold dark matter's delta' = -(theta/aH + continuity).
    from abcmb import species
    from abcmb.species import gauge_source_omissions

    BG, params, species_list = full_background_pair

    class DroppedVelocity(species.StandardFluid):
        name = "DroppedVelocity"
        num_equations = 2
        is_matter = True

        def rho(self, lna, args):
            return 1e-6 * args["omega_cdm"] / jnp.exp(lna) ** 3

        def P(self, lna, args):
            return 0.0

        def y_ini(self, k, tau_ini, args):
            return jnp.array([0.0, 0.0])

        def y_prime(self, k, lna, sources, y, args):
            theta = y[self.first_idx + 1]
            # The bug: no theta in the continuity equation.
            return jnp.array([-sources.continuity, -theta + sources.euler])

    dropped = DroppedVelocity(1, _options())
    flagged = gauge_source_omissions((*species_list, dropped), BG, params)
    assert flagged.get("DroppedVelocity") == [
        "own velocity in the continuity equation"
    ], flagged


def test_newtonian_ics_satisfy_the_redundant_energy_constraint(full_background_pair):
    r"""
    In conformal Newtonian gauge phi is integrated through the *momentum*
    constraint, which leaves the energy constraint (MB95 Eq. 23a)

        k^2 phi + 3 aH G (rho+P)theta_tot / k^2 + G rho_delta_tot = 0,
        G = 4 pi G a^2 / c^2,

    as a redundant relation that only holds if the initial conditions are
    right."""
    from abcmb.gauges import AllSpeciesTotals, NewtonianGauge
    from abcmb.gauges.base import _grav
    from abcmb.perturbations import PerturbationEvolver
    from abcmb.species import PerturbationContext

    BG, params, species_list = full_background_pair
    gauge = NewtonianGauge()
    evolver = PerturbationEvolver(
        species_list, jnp.array([0.05]), _options(gauge="newtonian"), gauge=gauge
    )
    ctx = PerturbationContext(BG, params, species_list)

    for k in (1e-4, 1e-3, 1e-2, 0.1, 1.0):
        lna_ini = -14.0
        y = evolver.initial_conditions_one_k(k, lna_ini, (BG, params))
        totals = AllSpeciesTotals.from_species(species_list, lna_ini, y, ctx)
        a = jnp.exp(lna_ini)
        aH = BG.aH(lna_ini, params)
        grav = _grav(a)

        phi = y[0]
        residual = (
            k**2 * phi
            + 3.0 * aH * grav * totals.rho_plus_P_theta / k**2
            + grav * totals.rho_delta
        )
        # The identity is a near-total cancellation between terms of order 10,
        # so "zero" only means anything relative to their size
        scale = sum(
            (
                abs(float(k**2 * phi)),
                abs(float(grav * totals.rho_delta)),
                abs(float(3.0 * aH * grav * totals.rho_plus_P_theta / k**2)),
            )
        )
        rel = abs(float(residual)) / scale
        assert rel < 1e-12, f"energy constraint violated at k={k}: {rel:.3e}"


def test_synchronous_ics_are_unshifted(full_background_pair):
    # Every built-in fluid declares synchronous ICs, so a synchronous run must
    # apply no transformation at all -- the fluid pieces are y_ini verbatim.
    from abcmb.gauges import SynchronousGauge
    from abcmb.perturbations import PerturbationEvolver

    BG, params, species_list = full_background_pair
    evolver = PerturbationEvolver(
        species_list, jnp.array([0.05]), _options(), gauge=SynchronousGauge()
    )
    k, lna_ini = 0.05, -14.0
    y = evolver.initial_conditions_one_k(k, lna_ini, (BG, params))
    tau_ini = BG.tau(lna_ini)
    expected = jnp.concatenate(
        [jnp.asarray(s.y_ini(k, tau_ini, params)) for s in species_list]
    )
    np.testing.assert_allclose(np.asarray(y[1:]), np.asarray(expected), rtol=0, atol=0)


def test_cdm_velocity_stays_exactly_zero_in_synchronous(full_background_pair):
    # ColdDarkMatter carries theta in both gauges; synchronous gauge is
    # *defined* by theta_c = 0, so the extra component must be identically
    # zero there or the gauge condition is being violated.
    from abcmb.gauges import SynchronousGauge
    from abcmb.perturbations import PerturbationEvolver
    from abcmb.species import find_species

    BG, params, species_list = full_background_pair
    evolver = PerturbationEvolver(
        species_list, jnp.array([0.05]), _options(), gauge=SynchronousGauge()
    )
    cdm = find_species(species_list, "ColdDarkMatter")
    k, lna = 0.05, -14.0

    y = evolver.initial_conditions_one_k(k, lna, (BG, params))
    assert float(y[cdm.first_idx + 1]) == 0.0

    # ...and nothing drives it: the theta slot of its derivative is zero too.
    dy = evolver.get_derivatives(lna, y, (k, BG, params))
    assert float(dy[cdm.first_idx + 1]) == 0.0


def test_mis_declared_ic_gauge_is_caught_by_the_adiabatic_check(lcdm_model):
    # A fluid whose ICs really are synchronous but which declares them
    # newtonian. The Einstein constraints stay satisfied under a per-species
    # gauge slip, so only the adiabaticity relation can catch this.
    import jax.numpy as jnp

    from abcmb import species
    from abcmb.species import adiabatic_ic_residuals

    params = lcdm_model.add_derived_parameters({})

    def _cdm_like(ic_gauge):
        class Clone(species.StandardFluid):
            name = "Clone"
            num_equations = 2
            is_matter = True

            def rho(self, lna, args):
                return args["omega_cdm"] / jnp.exp(lna) ** 3

            def P(self, lna, args):
                return 0.0

            def y_ini(self, k, tau_ini, args):
                # Genuinely synchronous: the same series ColdDarkMatter uses.
                return jnp.array(
                    [0.75 * species.adiabatic_ics.delta_gamma(k, tau_ini, args), 0.0]
                )

        Clone.ic_gauge = ic_gauge
        return Clone(1, lcdm_model.options)

    photon = species.find_species(lcdm_model.species_list, "Photon")

    honest = adiabatic_ic_residuals(
        [photon, _cdm_like(species.GaugeName.SYNCHRONOUS)], params
    )
    assert honest["Clone.delta"] < 1e-12, honest

    lying = adiabatic_ic_residuals(
        [photon, _cdm_like(species.GaugeName.NEWTONIAN)], params
    )
    assert lying["Clone.delta"] > 0.1, (
        f"a mis-declared ic_gauge slipped through: {lying}"
    )


def test_newtonian_gauge_warns_about_synchronous_tuned_tolerances():
    # ABCMB's solver defaults are converged in synchronous gauge (tightening
    # them moves P(k) by ~1e-4) but not in newtonian gauge (~1.4%). Switching
    # gauge must therefore not quietly cost accuracy.
    from abcmb.gauges import NewtonianGauge
    from abcmb.main import Model

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Model(gauge="newtonian")
    assert any("tuned for the synchronous gauge" in str(w.message) for w in caught)

    # Silent once the recommendations are taken...
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Model(
            gauge="newtonian",
            **NewtonianGauge.recommended_tolerances,
            max_steps_PE=NewtonianGauge.recommended_max_steps,
        )
    assert not any("tuned for the synchronous gauge" in str(w.message) for w in caught)

    # ...and silent in the gauge the defaults were tuned against.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Model()
    assert not any("tuned for the synchronous gauge" in str(w.message) for w in caught)


def test_fluid_totals_constructors_agree(full_background_pair):
    # AllSpeciesTotals has two constructors -- one for the ODE (a single lna) and
    # one batched over the output grid -- and the gauge equations are evaluated
    # through both. They sum the same species in the same order, so they must
    # agree exactly

    import jax

    from abcmb.gauges import AllSpeciesTotals
    from abcmb.species import PerturbationContext

    BG, params, species_list = full_background_pair
    ctx = PerturbationContext(BG, params, species_list)

    lna = jnp.linspace(-14.0, -5.0, 5)
    n_y = 1 + sum(s.num_equations for s in species_list)
    modes = jax.random.normal(jax.random.PRNGKey(0), (n_y, lna.size, 3))

    grid = AllSpeciesTotals.from_species_on_grid(species_list, lna, modes, ctx)
    scalar = [
        AllSpeciesTotals.from_species(species_list, lna[i], modes[:, i, j], ctx)
        for i in range(lna.size)
        for j in range(modes.shape[2])
    ]
    # Compared leaf-wise rather than by naming fields: a field added to
    # AllSpeciesTotals is then covered automatically instead of silently skipped,
    # which is the failure this test exists to prevent.
    stacked = jax.tree.map(
        lambda *v: np.reshape(np.array(v), (lna.size, modes.shape[2])), *scalar
    )
    leaves = list(zip(jax.tree.leaves(grid), jax.tree.leaves(stacked), strict=True))
    assert len(leaves) == len(AllSpeciesTotals.__annotations__)
    for got, ref in leaves:
        np.testing.assert_array_equal(np.asarray(got), ref)
