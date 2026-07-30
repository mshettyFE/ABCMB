"""
Cold dark matter.
"""

import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array

from .. import constants as cnst
from .base import FluidParams, OutputArgs, PerturbationContext, StandardFluid


class ColdDarkMatter(StandardFluid):
    """
    Cold dark matter fluid species implementation.

    Non-relativistic, pressureless dark matter with density
    perturbations but no velocity or shear modes.

    Methods:
    --------
    rho : Compute cold dark matter density (units: eV cm^{-3})
    P : Compute cold dark matter pressure (units: eV cm^{-3})
    y_ini : Compute initial perturbation conditions
    y_prime : Compute perturbation time derivatives
    """

    name = "ColdDarkMatter"
    num_equations = 1  # CDM only receives density perturbation in synchronous gauge.
    is_matter = True

    def __init__(self, first_idx, options):
        super().__init__(first_idx, options)

    def rho(self, lna: ArrayLike, args: FluidParams) -> Array | float:
        """
        Compute cold dark matter density.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        args : mapping
            Cosmological parameters (params)

        Returns:
        --------
        float
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

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        args : mapping
            Cosmological parameters (params)

        Returns:
        --------
        float
            Cold dark matter pressure (units: eV cm^{-3})

        Notes:
        ------
        Cold dark matter is pressureless, so this always returns zero.
        """
        return 0.0

    def y_ini(self, k: ArrayLike, tau_ini: ArrayLike, args: FluidParams) -> Array:
        """
        Compute initial conditions for cold dark matter perturbations.

        Parameters:
        -----------
        k : float
            Wavenumber (units: Mpc^{-1})
        tau_ini : float
            Initial conformal time (units: Mpc)
        args : mapping
            Cosmological parameters (params)

        Returns:
        --------
        array
            Initial density perturbation (units: dimensionless)
        """
        params = args
        delta = -((k * tau_ini) ** 2) / 4.0 * (1.0 - params["om"] * tau_ini / 5.0)
        return jnp.array([delta])

    def y_prime(
        self,
        k: ArrayLike,
        lna: ArrayLike,
        metric_h_prime: ArrayLike,
        metric_eta_prime: ArrayLike,
        y: Array,
        args: PerturbationContext,
    ) -> Array:
        """
        Compute time derivatives of cold dark matter perturbations.

        Parameters:
        -----------
        k : float
            Wavenumber (units: Mpc^{-1})
        lna : float
            Logarithm of scale factor
        metric_h_prime : float
            Derivative of metric h
        metric_eta_prime : float
            Derivative of metric eta
        y : array
            Current perturbation mode values
        args : PerturbationContext
            Background cosmology, cosmological parameters, and the species
            registry for coupled fluids (use ``args.BG``, ``args.params``,
            ``args.species_list``, ``args.species_dict``)
            -- BG is unused in this implementation

        Returns:
        --------
        array
            Time derivative of density perturbation (units: dimensionless)
        """
        return jnp.array([-0.5 * metric_h_prime])

    def output_perturbations(
        self, lna: ArrayLike, modes: Array, args: OutputArgs
    ) -> dict[str, Array]:
        return {"delta": modes[self.first_idx]}
