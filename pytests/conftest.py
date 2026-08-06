import os
import warnings

import jax
import pytest

# Make CI runs consistent. JAX_PLATFORMS (the modern variable) makes CPU the
# *only* backend, so backend sniffs (k_batch_strategy='auto') can't see an
# installed GPU; the deprecated JAX_PLATFORM_NAME alone no longer hides it.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

# Enable double precision
jax.config.update("jax_enable_x64", True)

# Uncomment if you want failing fast on NaNs/Infs
jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_debug_infs", True)


@pytest.fixture(scope="session")
def lcdm_model():
    """
    One shared default-options Model; treat it as read-only.

    Built with a single inert passthrough kwarg (``custom_knob``) so tests can
    also assert the custom-options escape hatch against a real construction.
    """
    from abcmb.main import Model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Model(custom_knob=1)


@pytest.fixture(scope="session")
def full_background_pair(lcdm_model):
    """
    ``(Background, derived params, species_list)`` for the shared LCDM model.

    Session-scoped because building it runs the HyRex recombination solve;
    treat the Background as read-only.
    """
    import equinox as eqx

    params = lcdm_model.add_derived_parameters({})
    pre = lcdm_model.get_BG_pre_recomb(params)
    recomb_inputs = pre.make_recomb_inputs(lcdm_model.RecModel, params)
    recomb_out = eqx.filter_jit(lcdm_model.RecModel)((recomb_inputs, params))
    BG = lcdm_model.get_BG(params, pre, recomb_out)
    return BG, params, lcdm_model.species_list
