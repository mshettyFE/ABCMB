"""
Cold dark matter.
    Follows Ma & Bertschinger (1995), ApJ 455, 7 (arXiv:astro-ph/9506072).
    CDM defines the synchronous gauge (theta_c = 0), leaving the single
    evolution equation."""

import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array

from .. import constants as cnst
from . import adiabatic_ics
from .base import FluidParams, OutputArgs, PerturbationContext, StandardFluid


class ColdDarkMatter(StandardFluid):
    """
    Cold dark matter fluid species implementation.

    Non-relativistic, pressureless dark matter with density
    perturbations but no velocity or shear modes.

    """

    name = "ColdDarkMatter"
    num_equations = 1  # CDM only receives density perturbation in synchronous gauge.
    is_matter = True

    def rho(self, lna: ArrayLike, args: FluidParams) -> Array | float:
        """
        Compute cold dark matter density.
        Returns:
            Cold dark matter density (units: eV cm^{-3})
        """
        params = args
        return (
            params["omega_cdm"]
            * (3.0 * cnst.H0_over_h**2 / 8.0 / jnp.pi / cnst.G)
            / jnp.exp(lna) ** 3
        )

    def P(self, lna: ArrayLike, args: FluidParams) -> Array | float:
        """
        Compute cold dark matter pressure.

        Returns:
            Cold dark matter pressure (units: eV cm^{-3})
        """
        return 0.0

    def y_ini(self, k: ArrayLike, tau_ini: ArrayLike, args: FluidParams) -> Array:
        """
        Adiabatic initial condition: delta_c = (3/4) delta_gamma (matter
        counts particles, radiation counts T^4). Series and citations in
        :mod:`.adiabatic_ics`.

        Returns:
            Initial density perturbation (units: dimensionless)
        """
        params = args
        delta = 0.75 * adiabatic_ics.delta_gamma(k, tau_ini, params)
        return jnp.array([delta])

    def y_prime(
        self,
        k: ArrayLike,
        lna: ArrayLike,
        metric_h_prime: ArrayLike,  # Derivative of h metric
        metric_eta_prime: ArrayLike,  # Derivative of eta metric
        y: Array,
        args: PerturbationContext,
    ) -> Array:
        """
        Compute time derivatives of cold dark matter perturbations.
        Eq. 42 in Ma and Bertschinger (1995).
        Returns:
            Time derivative of density perturbation (units: dimensionless)
        """
        return jnp.array([-0.5 * metric_h_prime])

    def output_perturbations(
        self, lna: ArrayLike, modes: Array, args: OutputArgs
    ) -> dict[str, Array]:
        """Output keys: ``delta``."""
        return {"delta": modes[self.first_idx]}
