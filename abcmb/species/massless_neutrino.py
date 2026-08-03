"""
Massless neutrinos.
"""

import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Float

from .. import constants as cnst
from . import adiabatic_ics
from .base import FluidParams, OutputArgs, PerturbationContext, StandardFluid


class MasslessNeutrino(StandardFluid):
    """
    Represents relativistic neutrinos with multiple angular momentum modes.
    """

    name = "MasslessNeutrino"
    is_matter = False
    is_neutrino = True

    def __init__(self, first_idx, options, **kwargs):
        super().__init__(first_idx, options, **kwargs)
        self.num_equations = options["l_max_massless_nu"] + 1

    def rho(self, lna: ArrayLike, args: FluidParams) -> Array | float:
        """
        Compute neutrino density.

        Returns:
            Neutrino density (units: eV cm^{-3})
        """
        params = args

        a = jnp.exp(lna)
        rho = (
            params["N_nu_massless"]
            * 2.0
            * 7.0
            / 8.0
            * jnp.pi**2
            / 30.0
            * params["T_nu_massless"] ** 4
            * params["TCMB0"] ** 4
            / a**4
        )  # eV^4
        rho = rho / (cnst.c * cnst.hbar) ** 3  # Convert to eV cm^{-3}
        return rho

    def P(self, lna: ArrayLike, args: FluidParams) -> Array | float:
        """
        Compute neutrino pressure.

        Returns:
           Neutrino pressure (units: eV cm^{-3})
        """
        params = args
        return self.rho(lna, params) / 3.0

    def y_ini(self, k: ArrayLike, tau_ini: ArrayLike, args: FluidParams) -> Array:
        """
        Adiabatic initial conditions: the radiation delta (= delta_gamma)
        plus the collisionless theta and sigma -- free streaming leaves the
        shear unsuppressed, unlike the photon's. Series and citations in
        :mod:`.adiabatic_ics`.

         Returns:
           Initial perturbation mode values (units: 1/Mpc for theta, else dimensionless)
        """
        params = args
        delta = adiabatic_ics.delta_gamma(k, tau_ini, params)
        theta = adiabatic_ics.theta_nu(k, tau_ini, params)
        sigma = adiabatic_ics.sigma_nu(k, tau_ini, params)

        # Return the three non-zero ell modes; all higher ell-modes start at zero.
        return jnp.concatenate(
            (jnp.array([delta, theta, sigma]), jnp.zeros(self.num_equations - 3))
        )

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
        Compute time derivatives of massless neutrino perturbations.
        Follows Ma & Bertschinger (1995), ApJ 455, 7 (arXiv:astro-ph/9506072).
        The collisionless hierarchy is their Eq. (49), truncated at l_max with
        their Eq. (51).

        Returns:
            Time derivatives of perturbation modes (units: 1/Mpc for theta, else dimensionless)
        """
        BG, params = args.BG, args.params
        aH = BG.aH(lna, params)
        tau = BG.tau(lna)

        L = jnp.arange(self.num_equations) + self.first_idx
        F = y[L]
        delta = F[0]
        theta = F[1]
        sigma = F[2]

        # density, velocity, shear perturbations
        delta_prime = -4.0 / 3.0 / aH * theta - 2.0 / 3.0 * metric_h_prime
        theta_prime = k**2 / aH * (delta / 4.0 - sigma)
        sigma_prime = (
            4.0 / 15.0 / aH * theta
            - 3.0 / 10.0 * k / aH * F[3]
            + 2.0 / 15.0 * metric_h_prime
            + 4.0 / 5.0 * metric_eta_prime
        )
        F3_prime = 1.0 / 7.0 * k / aH * (6.0 * sigma - 4.0 * F[4])

        # Rest of the Boltzmann Hierarchy
        lmax = self.num_equations - 1
        L = jnp.arange(4, lmax)
        Fl_prime = 1.0 / (2.0 * L + 1.0) * k / aH * (L * F[L - 1] - (L + 1) * F[L + 1])
        Flmax_prime = k / aH * F[lmax - 1] - (lmax + 1) / aH / tau * F[lmax]

        return jnp.concatenate(
            (
                jnp.array([delta_prime, theta_prime, sigma_prime, F3_prime]),
                Fl_prime,
                jnp.array([Flmax_prime]),
            )
        )

    def output_perturbations(
        self,
        lna: Float[Array, " n_lna"],
        modes: Float[Array, "n_y n_lna n_k"],
        args: OutputArgs,
    ) -> dict[str, Float[Array, "n_lna n_k"]]:
        """Output keys: ``delta``, ``theta``, ``sigma``."""
        return {
            "delta": modes[self.first_idx],
            "theta": modes[self.first_idx + 1],
            "sigma": modes[self.first_idx + 2],
        }
