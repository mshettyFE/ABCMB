"""
Species-construction tests
"""

import warnings

import pytest


def _options():
    from abcmb import schema

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return schema.resolve_options({})


def test_duplicate_species_name_raises():
    # species_dict and the perturbation output tables are keyed by name; a
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

    from abcmb import model_setup, schema, species

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        options = schema.resolve_options({"use_LCDM_species": False})
    with pytest.raises(ValueError, match="no fluid named 'Baryon'"):
        model_setup.populate_species((species.ColdDarkMatter,), options)
    # ...and present roles satisfy the check (full LCDM set builds fine).
    species_list, species_dict = model_setup.populate_species(None, _options())
    assert {"Baryon", "Photon"} <= set(species_dict)


def test_user_species_instance_raises():
    # user_species takes Fluid *classes*: ABCMB instantiates them itself to
    # assign first_idx. Passing an instance fails with an instructive message
    # instead of "'X' object is not callable".
    from abcmb import model_setup, species

    instance = species.MassiveNeutrino(0, _options())
    with pytest.raises(TypeError, match="classes, not instances"):
        model_setup.populate_species((instance,), _options())


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

    from abcmb.perturbations import PerturbationEvolver

    class _P:
        def __init__(self, name, num_equations, ini_len):
            self.name, self.num_equations = name, num_equations
            self._ini_len = ini_len

        def y_ini(self, k, tau_ini, params):
            return jnp.zeros(self._ini_len)

    params = {"om": jnp.asarray(1e-3), "R_nu": jnp.asarray(0.4)}
    bg = types.SimpleNamespace(tau=lambda lna: 1.0)

    good = types.SimpleNamespace(species_list=(_P("A", 2, 2), _P("B", 1, 1)))
    out = PerturbationEvolver.initial_conditions_one_k(good, 0.1, -14.0, (bg, params))
    assert out.shape == (4,)  # 1 metric slot + 2 + 1

    bad = types.SimpleNamespace(species_list=(_P("A", 0, 2),))  # claims 0, returns 2
    with pytest.raises(ValueError, match="declares num_equations=0"):
        PerturbationEvolver.initial_conditions_one_k(bad, 0.1, -14.0, (bg, params))


def test_options_key_set_is_stable(lcdm_model):
    # options = exactly what resolve_options returned: model construction must
    # not stash computed keys into it (the old k_min/k_max_cmb pattern), while
    # user passthrough extras (custom-species knobs) must survive resolution.
    # (The shared lcdm_model is built with custom_knob=1 for exactly this.)
    from abcmb import schema

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resolved = set(schema.resolve_options({"custom_knob": 1}))
    assert set(lcdm_model.options) == resolved
    assert "custom_knob" in lcdm_model.options  # escape hatch stays open


def test_k_batch_strategy_option():
    # The k-mode batching strategy is a declared option, not a backend sniff:
    # explicit values are honored verbatim; 'auto' resolves by JAX backend
    # (CI pins JAX_PLATFORM_NAME=cpu, so auto -> scan here).
    from abcmb import perturbations, schema

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        options = schema.resolve_options({})
    Strategy = perturbations.KBatchStrategy
    assert options["k_batch_strategy"] == "auto"
    assert perturbations._k_batch_strategy("scan") is Strategy.SCAN
    assert perturbations._k_batch_strategy("VMAP") is Strategy.VMAP
    assert perturbations._k_batch_strategy("auto") is Strategy.SCAN
    # Schema warns at resolution (non-fatal choices)...
    with pytest.warns(UserWarning, match=r"not one of.*Did you mean"):
        schema.resolve_options({"k_batch_strategy": "vamp"})
    # ...and an uninterpretable value fails loudly at use time.
    with pytest.raises(ValueError, match="not one of 'auto', 'scan', 'vmap'"):
        perturbations._k_batch_strategy("vamp")


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
