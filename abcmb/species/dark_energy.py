"""
Cosmological-constant dark energy (background only).
"""

import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array

from .. import constants as cnst
from .base import BackgroundFluid, FluidParams


class DarkEnergy(BackgroundFluid):
    """
    Represents a constant energy density fluid with negative pressure.
    """

    name = "DarkEnergy"

    def rho(self, lna: ArrayLike, args: FluidParams) -> Array | float:
        """
        Compute dark energy density.

        Returns:
            Dark energy density (units: eV cm^{-3})
        """
        params = args
        return params["omega_Lambda"] * (
            3.0 * cnst.H0_over_h**2 / 8.0 / jnp.pi / cnst.G
        )

    def P(self, lna: ArrayLike, args: FluidParams) -> Array | float:
        """
        Compute dark energy pressure.

        Returns:
            Dark energy pressure (units: eV cm^{-3})
        """
        params = args
        return -self.rho(lna, params)
