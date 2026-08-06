"""
Cold dark matter.
    Follows Ma & Bertschinger (1995), ApJ 455, 7 (arXiv:astro-ph/9506072)."""

import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Float

from .. import constants as cnst
from ..metric import GaugeName, MetricSources
from . import adiabatic_ics
from .base import (
    FluidParams,
    OutputArgs,
    PerturbationContext,
    StandardFluid,
)


class ColdDarkMatter(StandardFluid):
    """
    Cold dark matter fluid species implementation.

    Non-relativistic, pressureless dark matter with density and velocity
    perturbations and no shear.

    Notes
    -----
    The velocity is carried in both gauges even though synchronous gauge is
    *defined* by ``theta_c = 0``. That costs one identically-zero component of
    the state vector there (it starts at zero and its only source,
    ``sources.euler``, vanishes in that gauge), and buys a fluid that is the
    same code in every gauge -- the alternative is a species that has to be
    told which gauge it is in, which is the one thing the ``MetricSources``
    abstraction exists to prevent.
    """

    name = "ColdDarkMatter"
    num_equations = 2
    is_matter = True
    # y_ini is built from the shared series in adiabatic_ics, which are
    # synchronous; declared rather than inherited so the fact sits with
    # the initial conditions it describes.
    ic_gauge = GaugeName.SYNCHRONOUS

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

        The velocity starts at zero, which is the defining condition of the
        synchronous gauge these series are written in; the evolver's gauge
        transformation is what gives it the right nonzero start elsewhere.

        Returns:
            Initial perturbation mode values (units: 1/Mpc for theta, else dimensionless)
        """
        params = args
        delta = 0.75 * adiabatic_ics.delta_gamma(k, tau_ini, params)
        return jnp.array([delta, 0.0])

    def y_prime(
        self,
        k: ArrayLike,
        lna: ArrayLike,
        sources: MetricSources,
        y: Array,
        args: PerturbationContext,
    ) -> Array:
        """
        Compute time derivatives of cold dark matter perturbations.
        Eq. 42 in Ma and Bertschinger (1995), with w = cs2 = 0 -- identical to
        the baryon pair minus the Thomson coupling.

        Note the ``theta / aH`` in the continuity equation. It vanishes in
        synchronous gauge along with ``sources.euler``, so dropping it is
        invisible there and silently rescales the matter power spectrum
        everywhere else.

        Returns:
            Time derivatives of perturbation modes (units: 1/Mpc for theta, else dimensionless)
        """
        aH = args.BG.aH(lna, args.params)
        theta = y[self.first_idx + 1]
        delta_prime = -(theta / aH + sources.continuity)
        theta_prime = -theta + sources.euler
        return jnp.array([delta_prime, theta_prime])

    def output_perturbations(
        self,
        lna: Float[Array, " n_lna"],
        modes: Float[Array, "n_y n_lna n_k"],
        args: OutputArgs,
    ) -> dict[str, Float[Array, "n_lna n_k"]]:
        """Output keys: ``delta``, ``theta``."""
        return {
            "delta": modes[self.first_idx],
            "theta": modes[self.first_idx + 1],
        }
