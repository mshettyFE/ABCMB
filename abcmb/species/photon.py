"""
Photons (temperature + polarization hierarchies). Coupled to the Baryon
fluid at runtime via species_dict (a name lookup, deliberately not an import).
"""

import equinox as eqx
import jax.numpy as jnp
from jax import lax
from jax.typing import ArrayLike
from jaxtyping import Array

from .. import constants as cnst
from .base import FluidParams, OutputArgs, PerturbationContext, StandardFluid


class Photon(StandardFluid):
    """
    Photon fluid species implementation.

    Relativistic photons with temperature and polarization Boltzmann hierarchies.

    Attributes:
    -----------
    num_F_ell_modes : int
        Number of temperature multipole moments in Boltzmann hierarchy
    num_G_ell_modes : int
        Number of polarization multipole moments in Boltzmann hierarchy

    Methods:
    --------
    rho : Compute photon density (units: eV cm^{-3})
    P : Compute photon pressure (units: eV cm^{-3})
    y_ini : Compute initial perturbation conditions
    y_prime : Compute perturbation time derivatives
    """

    num_F_ell_modes: int = eqx.field(default=0, static=True)
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

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        args : mapping
            Cosmological parameters (params)

        Returns:
        --------
        float
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

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        args : mapping
            Cosmological parameters (params)

        Returns:
        --------
        float
            Photon pressure (units: eV cm^{-3})
        """
        params = args
        return self.rho(lna, params) / 3.0

    def y_ini(self, k: ArrayLike, tau_ini: ArrayLike, args: FluidParams) -> Array:
        """
        Compute initial conditions for photon perturbations.

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
            Initial perturbation mode values (units: 1/Mpc for theta, else dimensionless)
        """
        params = args
        delta = -((k * tau_ini) ** 2) / 3.0 * (1.0 - params["om"] * tau_ini / 5.0)
        theta = (
            -(k**4)
            * tau_ini**3
            / 36.0
            * (
                1.0
                - 3.0
                * (1.0 + 5.0 * params["R_b"] - params["R_nu"])
                / 20.0
                / (1.0 - params["R_nu"])
                * params["om"]
                * tau_ini
            )
        )
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

        Returns:
        --------
        array
            Time derivatives of perturbation modes (units: 1/Mpc for theta, else dimensionless)
        """
        BG, params = args.BG, args.params
        # Get Baryon from list
        baryon = args.species_list[args.species_dict["Baryon"]]
        # Same structural requirement in the other direction.
        assert isinstance(baryon, StandardFluid)

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
