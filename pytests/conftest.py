import os
import warnings

import jax
import pytest

# Make CI runs consistent
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
