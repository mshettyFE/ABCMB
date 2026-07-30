"""
ABCMB.

A fully differentiable Boltzmann solver for the CMB.
"""

from typing import TYPE_CHECKING

from .version import __version__

__author__ = "Zilu Zhou, Cara Giovanetti, and Hongwan Liu"

# The public API, re-exported lazily (PEP 562) so that `import abcmb` stays
# JAX-free (pinned by test_import_abcmb_stays_jax_free): the submodule is only
# imported when the attribute is first touched.
_EXPORTS = {
    # The model front door.
    "Model": ".main",
    "Output": ".main",
    # File-driven entry points (config front door).
    "load_config": ".config",
    "model_from_config": ".config",
    "save_run": ".config",
    "dump_defaults": ".config",
    # The custom-species extension API.
    "Fluid": ".species",
    "StandardFluid": ".species",
    "BackgroundFluid": ".species",
}

# Literal (not computed from _EXPORTS) so static checkers can read it; kept in
# sync with _EXPORTS by test_lazy_public_api.
__all__ = [
    "__version__",
    "__author__",
    "Model",
    "Output",
    "load_config",
    "model_from_config",
    "save_run",
    "dump_defaults",
    "Fluid",
    "StandardFluid",
    "BackgroundFluid",
]

if TYPE_CHECKING:
    # Static-only mirror of _EXPORTS so pyright/IDEs see the lazy attributes
    # (the `X as X` aliases mark them as deliberate re-exports).
    from .config import dump_defaults as dump_defaults
    from .config import load_config as load_config
    from .config import model_from_config as model_from_config
    from .config import save_run as save_run
    from .main import Model as Model
    from .main import Output as Output
    from .species import BackgroundFluid as BackgroundFluid
    from .species import Fluid as Fluid
    from .species import StandardFluid as StandardFluid


def __getattr__(name):
    if name in _EXPORTS:
        from importlib import import_module

        return getattr(import_module(_EXPORTS[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
