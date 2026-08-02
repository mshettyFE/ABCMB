"""
Photons (temperature + polarization hierarchies). Coupled to the Baryon
fluid at runtime via a name lookup (``args.find``), deliberately not an
import.
"""

import equinox as eqx
import jax.numpy as jnp
from jax import lax
from jax.typing import ArrayLike
from jaxtyping import Array

from .. import constants as cnst
from . import adiabatic_ics
from .base import FluidParams, OutputArgs, PerturbationContext, StandardFluid


class Photon(StandardFluid):
    """

    Relativistic photons with temperature and polarization Boltzmann hierarchies.

    Notes
    -----
    """

    # Number of temperature multipole moments in Boltzmann hierarchy
    num_F_ell_modes: int = eqx.field(default=0, static=True)
    # Number of polarization multipole moments in Boltzmann hierarchy
    num_G_ell_modes: int = eqx.field(default=0, static=True)
    name = "Photon"
    is_matter = False

    def __init__(self, first_idx, options):
        super().__init__(first_idx, options)
        self.num_F_ell_modes = options["l_max_g"] + 1
        self.num_G_ell_modes = options["l_max_pol_g"] + 1
        self.num_equations = self.num_F_ell_modes + self.num_G_ell_modes

    def rho(self, lna: ArrayLike, args: FluidParams) -> Array | float:
        """
        Compute photon density.

        Returns:
            Photon density (units: eV cm^{-3})
        """
        params = args
        a = jnp.exp(lna)
        return (
            jnp.pi**2 / 15.0 * params["TCMB0"] ** 4 / a**4 / (cnst.c * cnst.hbar) ** 3
        )

    def P(self, lna: ArrayLike, args: FluidParams) -> Array | float:
        """
        Compute photon pressure.

        Returns:
            Photon pressure (units: eV cm^{-3})
        """
        params = args
        return self.rho(lna, params) / 3.0

    def y_ini(self, k: ArrayLike, tau_ini: ArrayLike, args: FluidParams) -> Array:
        """
        Adiabatic initial conditions (see :mod:`.adiabatic_ics` for the
        series and citations). Shear and all higher F/G moments start at
        zero: Thomson scattering isotropizes the photons (tight coupling).

        Returns:
            Initial perturbation mode values (units: 1/Mpc for theta, else dimensionless)
        """
        params = args
        delta = adiabatic_ics.delta_gamma(k, tau_ini, params)
        theta = adiabatic_ics.theta_tight_coupled(k, tau_ini, params)
        return jnp.concatenate(
            (jnp.array([delta, theta]), jnp.zeros(self.num_equations - 2))
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
        Compute time derivatives of photon perturbations.
        Follows Ma & Bertschinger (1995), ApJ 455, 7 (arXiv:astro-ph/9506072).
        The temperature + polarization hierarchies with Thomson coupling are
        their Eq. (63) (the [1, 0, 0.2] polarization source is
        delta_l0 + delta_l2/5), truncated at l_max with their Eq. (65).

        Returns:
            Time derivatives of perturbation modes (units: 1/Mpc for theta, else dimensionless)
        """
        BG, params = args.BG, args.params
        # Coupled partner: same structural requirement in the other direction.
        baryon = args.find("Baryon", StandardFluid)

        aH = BG.aH(lna, params)
        tau_c = BG.tau_c(lna, params)
        tau = BG.tau(lna)

        Flmax = self.num_F_ell_modes - 1
        Glmax = self.num_G_ell_modes - 1
        F = lax.dynamic_slice(y, (self.first_idx,), (self.num_F_ell_modes,))
        G = lax.dynamic_slice(
            y, (self.first_idx + self.num_F_ell_modes,), (self.num_G_ell_modes,)
        )
        delta = F[0]
        theta = F[1]
        sigma = F[2]
        theta_b = baryon.get_theta(lna, y, args)

        delta_prime = -4.0 / 3.0 / aH * theta - 2.0 / 3.0 * metric_h_prime
        theta_prime = k**2 / aH * (delta / 4.0 - sigma) + (theta_b - theta) / aH / tau_c
        sigma_prime = (
            4.0 / 15.0 / aH * theta
            - 3.0 / 10.0 * k / aH * F[3]
            + 2.0 / 15.0 * metric_h_prime
            + 4.0 / 5.0 * metric_eta_prime
            - 9.0 / 10.0 / aH / tau_c * sigma
            + (G[0] + G[2]) / 20.0 / aH / tau_c
        )
        F3_prime = k / 7.0 / aH * (6.0 * sigma - 4.0 * F[4]) - F[3] / aH / tau_c

        # Temperature Boltzmann Hierarchy
        L = jnp.arange(4, Flmax)  # Excludes the lmax mode
        Fl_prime = (
            1.0 / (2.0 * L + 1.0) * k / aH * (L * F[L - 1] - (L + 1) * F[L + 1])
            - F[L] / aH / tau_c
        )
        Flmax_prime = (
            k / aH * F[Flmax - 1]
            - (Flmax + 1) / aH / tau * F[Flmax]
            - F[Flmax] / aH / tau_c
        )

        # Polarization Boltzmann Hierarchy
        L = jnp.arange(0, Glmax)  # Excludes the lmax mode
        Gl_prime = (
            1.0 / (2.0 * L + 1.0) * k / aH * (L * G[L - 1] - (L + 1) * G[L + 1])
            - G[L] / aH / tau_c
            + (2.0 * sigma + G[0] + G[2])
            / 2.0
            / aH
            / tau_c
            * jnp.concatenate((jnp.array([1.0, 0.0, 0.2]), jnp.zeros(Glmax - 3)))
        )

        Glmax_prime = (
            k / aH * G[Glmax - 1]
            - (Glmax + 1) / aH / tau * G[Glmax]
            - G[Glmax] / aH / tau_c
        )
        return jnp.concatenate(
            (
                jnp.array([delta_prime, theta_prime, sigma_prime, F3_prime]),
                Fl_prime,
                jnp.array([Flmax_prime]),
                Gl_prime,
                jnp.array([Glmax_prime]),
            )
        )

    def output_perturbations(
        self, lna: ArrayLike, modes: Array, args: OutputArgs
    ) -> dict[str, Array]:
        """Output keys: ``delta``, ``theta``, ``sigma``, plus the polarization
        moments ``G0`` and ``G2``."""
        return {
            "delta": modes[self.first_idx],
            "theta": modes[self.first_idx + 1],
            "sigma": modes[self.first_idx + 2],
            "G0": modes[self.first_idx + self.num_F_ell_modes],
            "G2": modes[self.first_idx + self.num_F_ell_modes + 2],
        }
