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
    StandardFluid,
    YPrimeArgs,
)
from .cdm import ColdDarkMatter
from .dark_energy import DarkEnergy
from .neutrinos import MassiveNeutrino, MasslessNeutrino
from .photon import Photon

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
    "YPrimeArgs",
]
