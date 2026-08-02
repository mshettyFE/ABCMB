"""
The adiabatic growing mode's initial-condition series, shared by every
standard species.

Series from Cyr-Racine & Sigurdson (arXiv:1012.0569), Appendix B, with
beta_1 = 1/2 (fixed by ABCMB's eta_ini = 1 normalization, eta -> 2*beta_1)
and at zeroth order in tight coupling (epsilon = tau_c/tau -> 0):
``delta_gamma`` is their B1, ``theta_tight_coupled`` is B2 at eps^0 (the
B5 slip is dropped, hence theta_b = theta_gamma), and ``theta_nu`` /
``sigma_nu`` are the collisionless counterparts (as used by CLASS,
perturbations.c adiabatic ICs: theta_ur, shear_ur). The leading orders
reproduce Ma & Bertschinger (1995), Eq. (96).
"""

from jax.typing import ArrayLike
from jaxtyping import Array

from .base import FluidParams


def delta_gamma(k: ArrayLike, tau_ini: ArrayLike, params: FluidParams) -> Array:
    """Photon (and any adiabatic radiation) density IC; matter scales by 3/4."""
    return -((k * tau_ini) ** 2) / 3.0 * (1.0 - params["om"] * tau_ini / 5.0)


def theta_tight_coupled(k: ArrayLike, tau_ini: ArrayLike, params: FluidParams) -> Array:
    """Common photon-baryon velocity IC (tight coupling at zeroth order)."""
    prefactor = -(k**4) * tau_ini**3 / 36.0
    slope = (1.0 + 5.0 * params["R_b"] - params["R_nu"]) / (1.0 - params["R_nu"])
    correction = 3.0 / 20.0 * slope * params["om"] * tau_ini
    return prefactor * (1.0 - correction)


def theta_nu(k: ArrayLike, tau_ini: ArrayLike, params: FluidParams) -> Array:
    """Collisionless (neutrino) velocity IC."""
    R_nu = params["R_nu"]
    prefactor = -k * (k * tau_ini) ** 3 / 36.0 / (4.0 * R_nu + 15.0)
    leading = 4.0 * R_nu + 11.0 + 12.0
    slope = 3.0 * (8.0 * R_nu**2 + 50.0 * R_nu + 275.0) / 20.0 / (2.0 * R_nu + 15.0)
    correction = slope * params["om"] * tau_ini
    return prefactor * (leading - correction)


def sigma_nu(k: ArrayLike, tau_ini: ArrayLike, params: FluidParams) -> Array:
    """Collisionless (neutrino) shear IC -- nonzero because nothing
    isotropizes a free-streaming species (the photon's is wiped by Thomson
    scattering and starts at zero)."""
    R_nu = params["R_nu"]
    prefactor = 2.0 * (k * tau_ini) ** 2 / (45.0 + 12.0 * R_nu)
    slope = (4.0 * R_nu - 5.0) / 4.0 / (2.0 * R_nu + 15.0)
    correction = slope * params["om"] * tau_ini
    return prefactor * (1.0 + correction)
