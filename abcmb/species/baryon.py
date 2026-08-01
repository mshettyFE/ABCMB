"""
Baryons. Coupled to the Photon fluid at runtime via species_dict
(a name lookup, deliberately not an import).
"""

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array

from .. import constants as cnst
from .base import FluidParams, OutputArgs, PerturbationContext, StandardFluid

if TYPE_CHECKING:
    from ..background import Background


class Baryon(StandardFluid):
    """
    Baryon fluid species implementation.

    Non-relativistic baryons with density and velocity perturbations.

    Methods:
    --------
    rho : Compute baryon density (units: eV cm^{-3})
    P : Compute baryon pressure (units: eV cm^{-3})
    cs2 : Compute sound speed squared (units: dimensionless)
    mean_mass : Compute mean baryon mass (units: eV)
    y_ini : Compute initial perturbation conditions
    y_prime : Compute perturbation time derivatives
    """

    name = "Baryon"
    num_equations = 2
    is_matter = True

    def __init__(self, first_idx, options):
        super().__init__(first_idx, options)

    def rho(self, lna: ArrayLike, args: FluidParams) -> Array | float:
        """
        Compute baryon density.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        args : mapping
            Cosmological parameters (params)

        Returns:
        --------
        float
            Baryon density (units: eV cm^{-3})
        """
        return (
            args["omega_b"]
            * (3.0 * cnst.H0_over_h**2 / 8.0 / jnp.pi / cnst.G)
            / jnp.exp(lna) ** 3
        )

    def P(self, lna: ArrayLike, args: FluidParams) -> Array | float:
        """
        Compute baryon pressure.

        Returns:
           Baryon pressure (units: eV cm^{-3})

        Notes:
        ------
        Baryon pressure is neglected, standard practice for SM baryons;
        """
        return 0.0

    def cs2(self, lna: ArrayLike, args: PerturbationContext) -> Array:
        """
        Compute sound speed squared.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        args : PerturbationContext
            Background cosmology, cosmological parameters, and the species
            registry for coupled fluids (use ``args.BG``, ``args.params``,
            ``args.species_list``, ``args.species_dict``)

        Returns:
        --------
        float
            Sound speed squared (units: dimensionless)

        Notes:
        ------
        Adiabatic sound speed squared, from Eq. (68) of Ma & Bertschinger
        (1995), arXiv:astro-ph/9506072, with the baryon-temperature derivative
        substituted analytically from the Compton-heating evolution equation.
        Although we can neglect the pressure, this term is important for perturbation growth
        during recombination. During reionization this cs2 is negative. This is not physical
        but it should not matter for cosmology.
        """
        BG, params = args.BG, args.params
        # Get photon class from list
        photon = args.species_list[args.species_dict["Photon"]]
        # The baryon-photon coupling needs the delta/theta/sigma layout;
        # narrows the type and fails loudly if a Photon replacement isn't one.
        assert isinstance(photon, StandardFluid)

        Tm = BG.Tm(lna, params)  # Baryon temp
        Tg = BG.TCMB(lna, params)  # Photon temp
        mu = self.mean_mass(lna, (BG, params))
        R = 4.0 * photon.rho(lna, params) / 3.0 / self.rho(lna, params)

        return (
            Tm
            / mu
            * (
                5.0 / 3.0
                - 2.0
                / 3.0
                * mu
                * R
                / cnst.me
                / BG.aH(lna, params)
                / BG.tau_c(lna, params)
                * (Tg / Tm - 1.0)
            )
        )

    def mean_mass(
        self, lna: ArrayLike, args: "tuple[Background, FluidParams]"
    ) -> Array:
        """
        Compute mean baryon mass at given redshift.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        args : tuple
            Background cosmology and cosmological parameters (BG, params)

        Returns:
        --------
        float
            Mean baryon mass (units: eV)

        Notes:
        ------
        Defined to be mu = rho_b / n_b = rho_b / (nH + nHe + ne)
        """
        BG, params = args
        denom = (1.0 + BG.xe(lna)) * (
            1.0 - params["YHe"]
        ) + cnst.mH / cnst.mHe * params["YHe"]
        return cnst.mH / denom

    def y_ini(self, k: ArrayLike, tau_ini: ArrayLike, args: FluidParams) -> Array:
        """
        Compute initial conditions for baryon perturbations.

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
        delta = -((k * tau_ini) ** 2) / 4.0 * (1.0 - params["om"] * tau_ini / 5.0)
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
        # Get photon class from list
        photon = args.species_list[args.species_dict["Photon"]]
        # The baryon-photon coupling needs the delta/theta/sigma layout;
        # narrows the type and fails loudly if a Photon replacement isn't one.
        assert isinstance(photon, StandardFluid)

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
