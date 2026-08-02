"""
Baryons. Coupled to the Photon fluid at runtime via a name lookup
(``args.find``), deliberately not an import.
    Follows Ma & Bertschinger (1995), ApJ 455, 7 (arXiv:astro-ph/9506072).
"""

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array

from .. import constants as cnst
from . import adiabatic_ics
from .base import FluidParams, OutputArgs, PerturbationContext, StandardFluid

if TYPE_CHECKING:
    from ..background import Background


class Baryon(StandardFluid):
    """
    Non-relativistic baryons with density and velocity perturbations.
    """

    name = "Baryon"
    num_equations = 2
    is_matter = True

    def rho(self, lna: ArrayLike, args: FluidParams) -> Array | float:
        """
        Compute baryon density.

        Returns:
            Baryon density (units: eV cm^{-3})
        """
        unit_factor = 3.0 * cnst.H0_over_h**2 / (8.0 * jnp.pi * cnst.G)
        return args["omega_b"] * unit_factor / (jnp.exp(lna) ** 3)

    def P(self, lna: ArrayLike, args: FluidParams) -> Array | float:
        """
        Baryon pressure is neglected for SM baryons;

        Returns:
           Baryon pressure (units: eV cm^{-3})

        """
        return 0.0

    def cs2(self, lna: ArrayLike, args: PerturbationContext) -> Array:
        """
        Adiabatic sound speed squared, from Eq. (68) of Ma & Bertschinger
        (1995), arXiv:astro-ph/9506072, with the baryon-temperature derivative
        substituted analytically from the Compton-heating evolution equation
        (their Eq. 69).

         Returns:
            Sound speed squared (units: dimensionless)

        """
        BG, params = args.BG, args.params
        # Coupled partner: the baryon-photon coupling needs the
        # delta/theta/sigma layout, so narrow to StandardFluid.
        photon = args.find("Photon", StandardFluid)

        Tm = BG.Tm(lna, params)  # Baryon temp
        Tg = BG.TCMB(lna, params)  # Photon temp
        mu = self.mean_mass(lna, (BG, params))
        R = 4.0 * photon.rho(lna, params) / 3.0 / self.rho(lna, params)
        dlnTm_dlna = -2.0 + 2.0 * mu * R / cnst.me * (Tg / Tm - 1.0) / (
            BG.aH(lna, params) * BG.tau_c(lna, params)
        )

        return (Tm / mu) * (1.0 - dlnTm_dlna / 3.0)

    def mean_mass(
        self, lna: ArrayLike, args: "tuple[Background, FluidParams]"
    ) -> Array:
        """
        Compute mean baryon mass at given redshift.
        Defined to be mu = rho_b / n_b = rho_b / (nH + nHe + ne)
        We expect rho_b to drop out of final calculation

        Returns:
            Mean baryon mass (units: eV)

        """
        BG, params = args
        f_H = 1.0 - params["YHe"]  # H fraction = H number in units rho_N/mH
        nHe = cnst.mH / cnst.mHe * params["YHe"]
        ne = f_H * BG.xe(lna)
        n = f_H + ne + nHe
        return cnst.mH / n

    def y_ini(self, k: ArrayLike, tau_ini: ArrayLike, args: FluidParams) -> Array:
        r"""
        Adiabatic initial conditions: delta_b = (3/4) delta_gamma (CRS B4)
        and theta_b = theta_gamma (tight coupling at zeroth order; the B5
        slip is dropped). Series and citations in :mod:`.adiabatic_ics`.

        Returns:
            Initial perturbation mode values (units: 1/Mpc for theta, else dimensionless)
        """
        params = args
        delta = 0.75 * adiabatic_ics.delta_gamma(k, tau_ini, params)
        theta = adiabatic_ics.theta_tight_coupled(k, tau_ini, params)
        return jnp.array([delta, theta])

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
        Compute time derivatives of baryon perturbations.

        Follows Ma & Bertschinger (1995), ApJ 455, 7 (arXiv:astro-ph/9506072).
        Evolution is their Eq. (66) with R = 4*rho_g/(3*rho_b) and
        tau_c = 1/(a*n_e*sigma_T)

        Returns:
            Time derivatives of perturbation modes (units: 1/Mpc for theta, else dimensionless)
        """
        BG, params = args.BG, args.params
        # Coupled partner: the baryon-photon coupling needs the
        # delta/theta/sigma layout, so narrow to StandardFluid.
        photon = args.find("Photon", StandardFluid)

        aH = BG.aH(lna, params)
        cs2 = self.cs2(lna, args)
        R = 4.0 * photon.rho(lna, params) / 3.0 / self.rho(lna, params)
        tau_c = BG.tau_c(lna, params)

        delta = y[self.first_idx]
        theta = y[self.first_idx + 1]
        theta_g = photon.get_theta(lna, y, args)
        delta_prime = -theta / aH - metric_h_prime / 2.0
        theta_prime = (
            -theta + cs2 * k**2 * delta / aH + R / tau_c / aH * (theta_g - theta)
        )

        return jnp.array([delta_prime, theta_prime])

    def output_perturbations(
        self, lna: ArrayLike, modes: Array, args: OutputArgs
    ) -> dict[str, Array]:
        """Output keys: ``delta``, ``theta``."""
        return {
            "delta": modes[self.first_idx],
            "theta": modes[self.first_idx + 1],
        }
