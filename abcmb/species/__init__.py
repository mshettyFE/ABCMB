"""
Fluid species for ABCMB, one module per species (base classes in ``base``).

Concrete species never import each other: coupled fluids (Baryon <-> Photon,
and user IDM/IDR-style pairs) find their partners at runtime through
``species_dict``, so adding a species is one new module plus a re-export here.
The public surface of the former single-module ``abcmb.species`` is preserved.
"""

from .baryon import Baryon
from .base import (
    BackgroundFluid,
    Fluid,
    FluidParams,
    OutputArgs,
    PerturbationContext,
    StandardFluid,
)
from .cdm import ColdDarkMatter
from .dark_energy import DarkEnergy
from .massive_neutrino import MassiveNeutrino
from .massless_neutrino import MasslessNeutrino
from .photon import Photon
from .validation import (
    adiabatic_ic_residuals,
    continuity_residuals,
    ic_scaling_residuals,
)

__all__ = [
    "BackgroundFluid",
    "Baryon",
    "ColdDarkMatter",
    "DarkEnergy",
    "Fluid",
    "FluidParams",
    "MasslessNeutrino",
    "MassiveNeutrino",
    "OutputArgs",
    "Photon",
    "StandardFluid",
    "PerturbationContext",
    "continuity_residuals",
    "adiabatic_ic_residuals",
    "ic_scaling_residuals",
]
