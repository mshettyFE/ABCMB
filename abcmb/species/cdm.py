"""
Cold dark matter.
    Follows Ma & Bertschinger (1995), ApJ 455, 7 (arXiv:astro-ph/9506072).
    CDM defines the synchronous gauge (theta_c = 0), leaving the single
    evolution equation."""

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

    """

    name = "ColdDarkMatter"
    num_equations = 1  # CDM only receives density perturbation in synchronous gauge.
    is_matter = True

    def __init__(self, first_idx, options):
        super().__init__(first_idx, options)

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
        Compute initial conditions for cold dark matter perturbations.
        The adiabatic initial condition is their Eq. (96): delta_c =
        (3/4)*delta_g = -(C/2)*(k*tau)^2, where C = 1/2 is fixed by the
        unit-curvature normalization eta_ini -> 2C = 1 (see
        initial_conditions_one_k), plus the next-order (1 - om*tau/5)
        correction used by CLASS (perturbations.c, adiabatic ICs;
        om = a*rho_m/sqrt(rho_r); first-order series of Cyr-Racine &
        Sigurdson, arXiv:1012.0569).

        Returns:
            Initial density perturbation (units: dimensionless)
        """
        params = args
        delta = -((k * tau_ini) ** 2) / 4.0 * (1.0 - params["om"] * tau_ini / 5.0)
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
