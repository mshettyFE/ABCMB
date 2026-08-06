"""
Fluid species for ABCMB, one module per species (base classes in ``base``).

Concrete species never import each other: coupled fluids (Baryon <-> Photon,
and user IDM/IDR-style pairs) find their partners at runtime by name
(``args.find`` / ``find_species``), so adding a species is one new module plus
a re-export here. The public surface of the former single-module
``abcmb.species`` is preserved.
"""

from ..metric import GaugeName, GaugeShift, MetricSources
from . import adiabatic_ics
from .baryon import Baryon
from .base import (
    BackgroundFluid,
    Fluid,
    FluidParams,
    OutputArgs,
    PerturbationContext,
    StandardFluid,
    find_species,
)
from .cdm import ColdDarkMatter
from .dark_energy import DarkEnergy
from .massive_neutrino import MassiveNeutrino
from .massless_neutrino import MasslessNeutrino
from .photon import Photon
from .validation import (
    adiabatic_ic_residuals,
    continuity_residuals,
    gauge_source_omissions,
    ic_scaling_residuals,
    metric_source_dependence,
)

__all__ = [
    "adiabatic_ics",
    "BackgroundFluid",
    "Baryon",
    "ColdDarkMatter",
    "DarkEnergy",
    "Fluid",
    "FluidParams",
    "GaugeName",
    "GaugeShift",
    "MasslessNeutrino",
    "MassiveNeutrino",
    "MetricSources",
    "OutputArgs",
    "Photon",
    "StandardFluid",
    "PerturbationContext",
    "find_species",
    "continuity_residuals",
    "adiabatic_ic_residuals",
    "ic_scaling_residuals",
    "metric_source_dependence",
    "gauge_source_omissions",
]
