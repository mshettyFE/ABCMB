"""
``BackgroundPreRecomb.make_recomb_inputs`` samples the background quantities
HyRex needs (TCMB, nH, H) onto the recombination model's lna grid and hands
them across the jit/device boundary as a :class:`RecombInputs` bundle; the
vendored HyRex solvers interpolate from it instead of computing their own
cosmology (see abcmb/hyrex/VENDORED.md). This class is the prototype of the
injection API that would let HyRex become an external dependency.
"""

import equinox as eqx
from jax import config
from jaxtyping import Array, Float

from .ABCMBTools import fast_interp

config.update("jax_enable_x64", True)


class RecombInputs(eqx.Module):
    """
    Background quantities sampled on the recombination model's uniform lna
    grid, with linear interpolation accessors (the form HyRex consumes).
    """

    lna_grid: Float[Array, " n_rec"]
    TCMB_arr: Float[Array, " n_rec"]  # Photon-bath temperature (eV)
    nH_arr: Float[Array, " n_rec"]  # Hydrogen number density (cm^-3)
    H_arr: Float[Array, " n_rec"]  # Hubble parameter (s^-1)

    def TCMB(self, lna: Float[Array, ""] | float) -> Float[Array, ""]:
        return fast_interp(lna, self.lna_grid[0], self.lna_grid[-1], self.TCMB_arr)

    def nH(self, lna: Float[Array, ""] | float) -> Float[Array, ""]:
        return fast_interp(lna, self.lna_grid[0], self.lna_grid[-1], self.nH_arr)

    def H(self, lna: Float[Array, ""] | float) -> Float[Array, ""]:
        return fast_interp(lna, self.lna_grid[0], self.lna_grid[-1], self.H_arr)
