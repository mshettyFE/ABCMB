"""
Cosmological-constant dark energy (background only).
"""

import jax.numpy as jnp

from .. import constants as cnst
from .base import BackgroundFluid


class DarkEnergy(BackgroundFluid):
    """
    Dark energy fluid species implementation.

    Represents a constant energy density fluid with negative pressure.

    Methods:
    --------
    rho : Compute dark energy density (units: eV cm^{-3})
    P : Compute dark energy pressure (units: eV cm^{-3})
    """

    name = "DarkEnergy"
    is_matter = False

    def __init__(self, first_idx, options):
        super().__init__(first_idx, options)

    def rho(self, lna, args):
        """
        Compute dark energy density.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        args : dict
            Cosmological parameters (params)

        Returns:
        --------
        float
            Dark energy density (units: eV cm^{-3})
        """
        params = args
        return params["omega_Lambda"] * (
            3.0 * cnst.H0_over_h**2 / 8.0 / jnp.pi / cnst.G
        )

    def P(self, lna, args):
        """
        Compute dark energy pressure.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        args : dict
            Cosmological parameters (params)

        Returns:
        --------
        float
            Dark energy pressure (units: eV cm^{-3})
        """
        params = args
        return -self.rho(lna, params)
