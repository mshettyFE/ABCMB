"""
Species-construction tests
"""

import warnings

import pytest


def _options():
    from abcmb.inputs import schema

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return schema.resolve_options({})


def test_duplicate_species_name_raises():
    # Coupling lookups and the perturbation output tables are keyed by name; a
    # duplicate would silently shadow the earlier fluid in every coupling
    # lookup, so registration must fail loudly instead.
    from abcmb import model_setup, species

    with pytest.raises(ValueError, match="duplicate species name 'ColdDarkMatter'"):
        model_setup.populate_species((species.ColdDarkMatter,), _options())


def test_missing_required_role_raises():
    # The baryon-photon coupling references fluids named 'Baryon' and 'Photon';
    # a model without them must fail at construction with a clear message, not
    # with a KeyError mid-trace on the first call.
    import warnings

    from abcmb import model_setup, species
    from abcmb.inputs import schema

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        options = schema.resolve_options({"use_LCDM_species": False})
    with pytest.raises(ValueError, match="no fluid named 'Baryon'"):
        model_setup.populate_species((species.ColdDarkMatter,), options)
    # ...and present roles satisfy the check (full LCDM set builds fine).
    species_list = model_setup.populate_species(None, _options())
    assert {"Baryon", "Photon"} <= {s.name for s in species_list}


def test_user_species_non_fluid_class_raises():
    # A class that isn't a Fluid subclass used to pass the classes-not-
    # instances guard and detonate downstream with unrelated errors
    # (AttributeError on .name, or a constructor TypeError). It must fail
    # here, with a message naming the actual mistake.
    from abcmb import model_setup

    class NotAFluid:
        def __init__(self, first_idx, options):
            pass

    with pytest.raises(TypeError, match="does not inherit from Fluid"):
        model_setup.populate_species((NotAFluid,), _options())


def test_role_impostor_raises_at_construction():
    # A fluid merely *named* 'Baryon' that isn't a species.Baryon used to be
    # accepted here and only failed later, mid-trace, at the coupling
    # isinstance asserts. The coupling requires the genuine subclass either
    # way, so the check belongs at Model construction.
    import warnings

    from abcmb import model_setup, species
    from abcmb.inputs import schema

    class FakeBaryon(species.BackgroundFluid):
        name = "Baryon"
        is_matter = True

        def rho(self, lna, params):
            return params["omega_b"]

        def P(self, lna, params):
            return 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        options = schema.resolve_options({"use_LCDM_species": False})
    with pytest.raises(TypeError, match="'Baryon' is a FakeBaryon"):
        model_setup.populate_species((FakeBaryon, species.Photon), options)


def test_user_species_instance_raises():
    # user_species takes Fluid *classes*: ABCMB instantiates them itself to
    # assign first_idx. Passing an instance fails with an instructive message
    # instead of "'X' object is not callable".
    from abcmb import model_setup, species

    instance = species.MassiveNeutrino(0, _options())
    with pytest.raises(TypeError, match="classes, not instances"):
        model_setup.populate_species((instance,), _options())


def test_scalar_contract_probe_rejects_array_rho(lcdm_model):
    # Construction-time probe (jax.eval_shape, no numerics): a fluid whose
    # rho returns an array for scalar lna would break species stacks and
    # [:, None] promotions far from the culprit -- it must fail at Model
    # construction with the fluid's name in the message.
    import jax.numpy as jnp

    from abcmb import model_setup, species

    class ArrayRho(species.BackgroundFluid):
        name = "ArrayRho"

        def rho(self, lna, args):
            return jnp.zeros(3)  # violates scalar-in, scalar-out

        def P(self, lna, args):
            return 0.0

    with pytest.raises(TypeError, match=r"ArrayRho\.rho returned shape \(3,\)"):
        model_setup.populate_species((ArrayRho,), _options())


def test_scalar_contract_probe_skips_unprobeable(lcdm_model):
    # Best-effort: a fluid the abstract probe cannot evaluate (concrete
    # float() on a param) is warned about and admitted, never rejected.
    from abcmb import model_setup, species

    class Unprobeable(species.BackgroundFluid):
        name = "Unprobeable"

        def rho(self, lna, args):
            return float(args["custom_thing"]) * 1.0  # defeats eval_shape

        def P(self, lna, args):
            return 0.0

    with pytest.warns(UserWarning, match="Unprobeable.rho could not be probed"):
        species_list = model_setup.populate_species((Unprobeable,), _options())
    assert any(s.name == "Unprobeable" for s in species_list)


def test_abstract_name_is_matter_enforced():
    # name/is_matter are AbstractVars: a species that forgets them fails at
    # instantiation with a clear message, not deep inside derive_parameters.
    from abcmb import species

    class Incomplete(species.StandardFluid):  # no name, no is_matter
        def rho(self, lna, args):
            return 0.0

        def P(self, lna, args):
            return 0.0

    with pytest.raises(TypeError, match="abstract attributes"):
        Incomplete(0, _options())

    # BackgroundFluid declares is_matter = False (a background-only fluid
    # cannot be matter), so its subclasses need only a name.
    class BgOnlyName(species.BackgroundFluid):
        name = "BgOnlyName"

        def rho(self, lna, args):
            return 0.0

        def P(self, lna, args):
            return 0.0

    assert BgOnlyName(0, _options()).is_matter is False


def test_missing_num_equations_raises():
    # num_equations has no default: a perturbed fluid that silently inherited
    # 0 would be allocated no slots while its y_ini/y_prime still contribute
    # entries, misaligning every later fluid's slice of y. Forgetting to
    # declare it now fails at instantiation.
    from abcmb import species

    class Forgetful(species.StandardFluid):  # declares no num_equations
        name = "Forgetful"
        is_matter = False

        def rho(self, lna, args):
            return 0.0

        def P(self, lna, args):
            return 0.0

    with pytest.raises(TypeError, match="num_equations"):
        Forgetful(0, _options())


def test_y_ini_layout_mismatch_raises():
    # Trace-time layout check: y_ini must return exactly num_equations
    # entries, else the declared slices misalign against the concatenated
    # vector (self-consistently -- nothing else would error).
    import types

    import jax.numpy as jnp

    from abcmb.gauges import SynchronousGauge
    from abcmb.perturbations import PerturbationEvolver
    from abcmb.species import GaugeName

    class _P:
        # Enough of the Fluid surface for IC assembly: the layout check, the
        # stress-energy sums that fix the gauge generator, and the IC-gauge
        # declaration.
        ic_gauge = GaugeName.SYNCHRONOUS

        def __init__(self, name, num_equations, ini_len):
            self.name, self.num_equations = name, num_equations
            self._ini_len = ini_len

        def y_ini(self, k, tau_ini, params):
            return jnp.zeros(self._ini_len)

        def rho_delta(self, lna, y, ctx):
            return 0.0

        def rho_plus_P_theta(self, lna, y, ctx):
            return 0.0

        def rho_plus_P_sigma(self, lna, y, ctx):
            return 0.0

    params = {"om": jnp.asarray(1e-3), "R_nu": jnp.asarray(0.4)}
    bg = types.SimpleNamespace(tau=lambda lna: 1.0, aH=lambda lna, p: 1.0)

    good = types.SimpleNamespace(
        species_list=(_P("A", 2, 2), _P("B", 1, 1)), gauge=SynchronousGauge()
    )
    out = PerturbationEvolver.initial_conditions_one_k(good, 0.1, -14.0, (bg, params))
    assert out.shape == (4,)  # 1 metric slot + 2 + 1

    bad = types.SimpleNamespace(
        species_list=(_P("A", 0, 2),), gauge=SynchronousGauge()
    )  # claims 0, returns 2
    with pytest.raises(ValueError, match="declares num_equations=0"):
        PerturbationEvolver.initial_conditions_one_k(bad, 0.1, -14.0, (bg, params))


def test_options_key_set_is_stable(lcdm_model):
    # options = exactly what resolve_options returned: model construction must
    # not stash computed keys into it (the old k_min/k_max_cmb pattern), while
    # user passthrough extras (custom-species knobs) must survive resolution.
    # (The shared lcdm_model is built with custom_knob=1 for exactly this.)
    from abcmb.inputs import schema

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resolved = set(schema.resolve_options({"custom_knob": 1}))
    assert set(lcdm_model.options) == resolved
    assert "custom_knob" in lcdm_model.options  # escape hatch stays open


def test_k_batch_strategy_option():
    # The k-mode batching strategy is a declared option, not a backend sniff:
    # explicit values are honored verbatim; 'auto' resolves by the JAX
    # default backend.

    from jax import default_backend

    from abcmb import perturbations
    from abcmb.inputs import schema

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        options = schema.resolve_options({})
    Strategy = perturbations.KBatchStrategy
    assert options["k_batch_strategy"] == "auto"
    assert perturbations._k_batch_strategy("scan") is Strategy.SCAN
    assert perturbations._k_batch_strategy("VMAP") is Strategy.VMAP
    expected_auto = Strategy.VMAP if default_backend() == "gpu" else Strategy.SCAN
    assert perturbations._k_batch_strategy("auto") is expected_auto
    # Schema warns at resolution (non-fatal choices)...
    with pytest.warns(UserWarning, match=r"not one of.*Did you mean"):
        schema.resolve_options({"k_batch_strategy": "vamp"})
    # ...and an uninterpretable value fails loudly at use time.
    with pytest.raises(ValueError, match="not one of 'auto', 'scan', 'vmap'"):
        perturbations._k_batch_strategy("vamp")


def test_rho_P_scalar_contract(lcdm_model):
    # Contract: background rho/P return arrays of lna's shape
    import warnings

    import jax.numpy as jnp

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params = lcdm_model.add_derived_parameters(
            {
                "h": 0.6762,
                "omega_cdm": 0.1193,
                "omega_b": 0.0225,
                "A_s": 2.12424e-9,
                "n_s": 0.9709,
                "Neff": 3.044,
                "YHe": 0.245,
                "tau_reion": 0.0544,
            }
        )
    # Scalar contract: every species' rho/P at one lna is a scalar, so the
    # species stack is always a clean (n_species,) -- ragged stacks and
    # batch-axis collapse are unrepresentable. Batch via explicit vmap.
    for s in lcdm_model.species_list:
        assert jnp.shape(s.rho(-1.0, params)) == (), f"{s.name}.rho"
        assert jnp.shape(s.P(-1.0, params)) == (), f"{s.name}.P"
    stack = jnp.asarray([s.rho(-1.0, params) for s in lcdm_model.species_list])
    assert stack.shape == (len(lcdm_model.species_list),)


def test_massive_nu_quadrature_stencils():
    # The stencils are CAMB's tuned arrays, exactly, carried as static
    # instance fields (hashable, so they participate in the jit cache key),
    # and they size the ODE state.
    from abcmb import species
    from abcmb.species.massive_neutrino import (
        _CAMB_Q_BG,
        _CAMB_Q_PERT,
        _CAMB_W_BG,
        _CAMB_W_PERT,
    )

    mn = species.MassiveNeutrino(1, _options())
    assert mn.q_pert == _CAMB_Q_PERT and mn.q_bg == _CAMB_Q_BG
    assert mn.num_equations == 3 * mn.num_ells_per_bin

    # Provenance pins: the vendored generator (arXiv:1201.3654 Appendix A
    # moment matching) must reproduce both carried stencils to their
    # published digits (truncation radius ~2e-6, thresholds ~3x).
    from abcmb._generators.camb_stencils import (
        camb_five_point_rule,
        camb_three_point_rule,
    )

    gen_q, gen_w = camb_three_point_rule()
    assert all(abs(a / b - 1) < 5e-6 for a, b in zip(gen_q, _CAMB_Q_PERT))
    assert all(abs(a / b - 1) < 5e-6 for a, b in zip(gen_w, _CAMB_W_PERT))
    gen_q, gen_w = camb_five_point_rule()
    assert all(abs(a / b - 1) < 5e-6 for a, b in zip(gen_q, _CAMB_Q_BG))
    assert all(abs(a / b - 1) < 5e-6 for a, b in zip(gen_w, _CAMB_W_BG))


def test_continuity_relation(lcdm_model):
    # Metamorphic cross-check: every background fluid must satisfy
    # d(rho)/dlna = -3(rho+P) identically, tying its rho and P
    # implementations together (for MassiveNeutrino, the two quadrature
    # integrals).

    import jax.numpy as jnp

    from abcmb import species
    from abcmb.species.validation import continuity_residuals

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params = lcdm_model.add_derived_parameters(
            {
                "h": 0.6762,
                "omega_cdm": 0.1193,
                "omega_b": 0.0225,
                "A_s": 2.12424e-9,
                "n_s": 0.9709,
                "Neff": 3.044,
                "YHe": 0.245,
                "tau_reion": 0.0544,
            }
        )

    res = continuity_residuals(lcdm_model.species_list, params)
    for name, r in res.items():
        assert r < 1e-13, f"{name}: continuity residual {r:.2e}"

    # MassiveNeutrino is not in the LCDM stack; check it standalone with a
    # nonzero count (rho scales with N_nu_massive, so 0 would be vacuous).
    mnu_params = dict(params)
    mnu_params["N_nu_massive"] = jnp.asarray(1.0)
    mn = species.MassiveNeutrino(1, lcdm_model.options)
    r = continuity_residuals([mn], mnu_params)["MassiveNeutrino"]
    assert r < 1e-13, f"MassiveNeutrino: continuity residual {r:.2e}"

    # Negative control: a fluid whose P is inconsistent with its rho (matter
    # dilution but radiation pressure) must be caught with an O(1) residual.
    class BrokenFluid(species.BackgroundFluid):
        name = "BrokenFluid"

        def rho(self, lna, args):
            return args["omega_cdm"] * jnp.exp(-3.0 * lna)

        def P(self, lna, args):
            return self.rho(lna, args) / 3.0  # wrong: implies a^-4 dilution

    r = continuity_residuals([BrokenFluid(1, lcdm_model.options)], params)
    assert r["BrokenFluid"] > 0.1, "validator failed to flag an inconsistent fluid"


def test_adiabatic_ic_relations(lcdm_model):
    # Metamorphic IC checks: (1) adiabaticity ties every species' delta to
    # the photon's (3/4 for matter, 1 for radiation) and theta_b to theta_g,
    # (2) the k-tau-om scaling degeneracy fixes
    # the individual powers of k and tau that the combined k*tau forms hide;
    # (3) the massive-nu Psi bins must encode the massless-nu (delta, theta,
    # sigma) through dlnf0/dlnq. Since the adiabatic_ics extraction, (1) and
    # (3) hold largely by construction for the built-ins (shared series);
    # what they still pin: the composition factors (3/4, the Eq. 97 map,
    # bin striding), and the validators remain the diagnostic for custom
    # fluids. Measured residuals are 0.0; thresholds leave rounding room.
    import jax.numpy as jnp
    import numpy as np

    from abcmb import species
    from abcmb.species import adiabatic_ic_residuals, ic_scaling_residuals

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params = lcdm_model.add_derived_parameters(
            {
                "h": 0.6762,
                "omega_cdm": 0.1193,
                "omega_b": 0.0225,
                "A_s": 2.12424e-9,
                "n_s": 0.9709,
                "Neff": 3.044,
                "YHe": 0.245,
                "tau_reion": 0.0544,
            }
        )

    for name, r in adiabatic_ic_residuals(lcdm_model.species_list, params).items():
        assert r < 1e-13, f"{name}: adiabatic relation residual {r:.2e}"
    for name, r in ic_scaling_residuals(lcdm_model.species_list, params).items():
        assert r < 1e-13, f"{name}: k-tau scaling residual {r:.2e}"

    # Massive-nu bins vs the massless-nu ICs they must encode.
    mn = species.MassiveNeutrino(1, lcdm_model.options)
    ml = species.find_species(lcdm_model.species_list, "MasslessNeutrino")
    k, tau = 0.05, 0.5
    y_ml = np.asarray(ml.y_ini(k, tau, params))[:3]
    y_mn = np.asarray(mn.y_ini(k, tau, params))
    dlnf0 = np.asarray(mn._dlnf0_dlnq_pert())
    for i in range(len(mn.q_pert)):
        got = y_mn[i * mn.num_ells_per_bin : i * mn.num_ells_per_bin + 3]
        expected = -np.array([y_ml[0] / 4.0, y_ml[1] / 3.0, y_ml[2] / 2.0]) * dlnf0[i]
        r = float(np.max(np.abs(got / expected - 1.0)))
        assert r < 1e-13, f"massive-nu bin {i}: {r:.2e}"

    # Negative controls: a matter fluid with the radiation delta amplitude
    # (missing 3/4) and a wrong k-power in theta must be flagged by both.
    class WrongIC(species.StandardFluid):
        name = "WrongIC"
        num_equations = 2
        is_matter = True

        def y_ini(self, k, tau_ini, args):
            delta = -((k * tau_ini) ** 2) / 3.0  # missing the 3/4
            theta = -(k**3) * tau_ini**3 / 36.0  # k^3, not k^4
            return jnp.array([delta, theta])

    photon = species.find_species(lcdm_model.species_list, "Photon")
    wrong = WrongIC(1, lcdm_model.options)
    a = adiabatic_ic_residuals([photon, wrong], params)
    assert a["WrongIC.delta"] > 0.1, "adiabaticity validator missed a wrong ratio"
    s = ic_scaling_residuals([wrong], params)
    assert s["WrongIC"] > 0.1, "scaling validator missed a wrong k power"


def test_dimensional_scaling(lcdm_model):
    # Lambda-scaling (metamorphic homogeneity): scale the dimensionful
    # params and demand the known power laws.
    import jax.numpy as jnp

    from abcmb import species

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params = lcdm_model.add_derived_parameters(
            {
                "h": 0.6762,
                "omega_cdm": 0.1193,
                "omega_b": 0.0225,
                "A_s": 2.12424e-9,
                "n_s": 0.9709,
                "Neff": 3.044,
                "YHe": 0.245,
                "tau_reion": 0.0544,
            }
        )

    def scaled(pp, **kv):
        q = dict(pp)
        q.update({k: v * q[k] for k, v in kv.items()})
        return q

    lam, lna = 2.0, -3.0
    sl = lcdm_model.species_list
    photon = species.find_species(sl, "Photon")
    ml = species.find_species(sl, "MasslessNeutrino")
    baryon = species.find_species(sl, "Baryon")

    checks = {
        "photon rho ~ TCMB0^4": photon.rho(lna, scaled(params, TCMB0=lam))
        / photon.rho(lna, params)
        / lam**4,
        "massless rho ~ TCMB0^4": ml.rho(lna, scaled(params, TCMB0=lam))
        / ml.rho(lna, params)
        / lam**4,
        "massless rho ~ N": ml.rho(lna, scaled(params, N_nu_massless=2.0))
        / ml.rho(lna, params)
        / 2.0,
        "baryon rho ~ omega_b": baryon.rho(lna, scaled(params, omega_b=2.0))
        / baryon.rho(lna, params)
        / 2.0,
    }
    # Massive nu: (m, TCMB0) -> (lam m, lam TCMB0) leaves x = m/T invariant,
    # so the quadrature integral is unchanged and rho, P scale as T^4.
    mn = species.MassiveNeutrino(1, lcdm_model.options)
    pm = dict(params)
    pm["N_nu_massive"] = jnp.asarray(1.0)
    pm_scaled = scaled(pm, m_nu_massive=lam, TCMB0=lam)
    checks["massive rho ~ lam^4"] = mn.rho(lna, pm_scaled) / mn.rho(lna, pm) / lam**4
    checks["massive P ~ lam^4"] = mn.P(lna, pm_scaled) / mn.P(lna, pm) / lam**4

    for name, ratio in checks.items():
        assert abs(float(ratio) - 1.0) < 1e-14, f"{name}: ratio {float(ratio):.6f}"


def test_massive_nu_relativistic_limit(lcdm_model):
    # Cross-module oracle: as m -> 0 a massive neutrino IS a massless one,
    # so with matched count and temperature ratio the two rho
    # implementations must agree.

    import jax.numpy as jnp

    from abcmb import species

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params = lcdm_model.add_derived_parameters(
            {
                "h": 0.6762,
                "omega_cdm": 0.1193,
                "omega_b": 0.0225,
                "A_s": 2.12424e-9,
                "n_s": 0.9709,
                "Neff": 3.044,
                "YHe": 0.245,
                "tau_reion": 0.0544,
            }
        )
    p = dict(params)
    p["N_nu_massive"] = jnp.asarray(1.0)
    p["m_nu_massive"] = jnp.asarray(1e-7)
    p["N_nu_massless"] = jnp.asarray(1.0)
    p["T_nu_massless"] = params["T_nu_massive"]

    mn = species.MassiveNeutrino(1, lcdm_model.options)
    ml = species.find_species(lcdm_model.species_list, "MasslessNeutrino")
    ratio = float(mn.rho(-8.0, p) / ml.rho(-8.0, p))
    assert abs(ratio - 1.0) < 2e-3, f"relativistic-limit rho ratio {ratio:.6f}"
    w = float(mn.P(-8.0, p) / mn.rho(-8.0, p))
    assert abs(w - 1.0 / 3.0) < 1e-9, f"relativistic-limit w {w:.2e}"


def test_is_neutrino_flags():
    # Opt-in trait: False unless a species declares itself neutrino-like. The
    # two neutrino classes opt in; nothing else does.
    from abcmb import species

    assert species.Fluid.is_neutrino is False
    assert species.MasslessNeutrino.is_neutrino is True
    assert species.MassiveNeutrino.is_neutrino is True
    assert species.Photon.is_neutrino is False
    assert species.ColdDarkMatter.is_neutrino is False
    assert species.DarkEnergy.is_neutrino is False
    assert species.Baryon.is_neutrino is False
