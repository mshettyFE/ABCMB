import numpy as np
import jax.numpy as jnp
from jax import config
import equinox as eqx
from diffrax import Kvaerno3, ForwardMode

from .hydrogen import hydrogen_model
from .helium import helium_model
from .array_with_padding import array_with_padding
from ..ABCMBTools import fast_interp
config.update("jax_enable_x64", True)

# RecombInputs -- the bundle of background quantities this model consumes --
# is ABCMB's host-side handoff type and lives in abcmb.recomb_interface
# (it was an ABCMB fork addition, not upstream HyRex code).


class recomb_model(eqx.Module):
    """
    Complete recombination model implementation.

    Combines helium and hydrogen recombination calculations with
    reionization modeling to compute full ionization history.

    Methods:
    --------
    get_history : Compute complete recombination and reionization history (units: dimensionless)
    """

    integration_spacing : jnp.float64
    lna_axis_full : jnp.array

    z1 : jnp.float64

    twog_redshift : jnp.float64
    He4equil_redshift : jnp.float64
    idx_4He_equil : jnp.array

    adjoint : "diffrax.adjoint" = eqx.field(static=True)

    def __init__(self, integration_spacing = 5.0e-4, z0=8000., z1=0., adjoint = ForwardMode):
        """
        Initialize complete recombination model.

        Sets up time grids and parameters for helium recombination,
        hydrogen recombination, and reionization phases.

        Parameters:
        -----------
        integration_spacing : float, optional
            Step size for integration (default: 5.0e-4)
        z0 : float, optional
            Initial redshift (default: 8000.)
        z1 : float, optional
            Final redshift (default: 0.)
        adjoint : diffrax.adjoint
            Adjoint mode for diffrax solves (static field).  Defaults
            to ForwardMode.
        """
        self.integration_spacing = integration_spacing
        self.adjoint = adjoint
        self.z1 = z1

        # Define time axes
        self.lna_axis_full  = jnp.arange(-jnp.log(1+z0), -jnp.log(1+z1), self.integration_spacing)

        self.twog_redshift = 701.
        self.He4equil_redshift = 3601. # generous

        self.idx_4He_equil = jnp.where(self.lna_axis_full <= -jnp.log(self.He4equil_redshift))[0]

    def __call__(self, args, rtol=1e-6, atol=1e-9,solver=Kvaerno3(),max_steps=1024):
        """
        Compute complete recombination and reionization history.

        Parameters:
        -----------
        args : tuple
            Recombination input arrays and cosmological parameters
            (recomb_inputs, params).
        rtol : float, optional
            Relative tolerance for ODE solver (default: 1e-6)
        atol : float, optional
            Absolute tolerance for ODE solver (default: 1e-9)
        solver : diffrax.Solver, optional
            ODE solver instance (default: Kvaerno3())
        max_steps : int, optional
            Maximum solver steps (default: 1024)

        Returns:
        --------
        tuple
            (xe_full_reion, lna_full, Tm, lna_Tm) - complete ionization history
            with reionization, log scale factor, matter temperature, and temperature grid
        """
        return self.get_history(args, rtol, atol, solver, max_steps)

    def get_history(self, args, rtol=1e-6, atol=1e-9,solver=Kvaerno3(),max_steps=1024):
        """
        Compute complete recombination and reionization history.

        Combines helium recombination, hydrogen recombination, and
        reionization to produce complete ionization fraction evolution.

        Parameters:
        -----------
        args : tuple
            Recombination input arrays and cosmological parameters
            (recomb_inputs, params).
        rtol : float, optional
            Relative tolerance for ODE solver (default: 1e-6)
        atol : float, optional
            Absolute tolerance for ODE solver (default: 1e-9)
        solver : diffrax.Solver, optional
            ODE solver instance (default: Kvaerno3())
        max_steps : int, optional
            Maximum solver steps (default: 1024)

        Returns:
        --------
        tuple
            (xe_full_reion, lna_full, Tm, lna_Tm) containing complete ionization
            fraction evolution with reionization, log scale factor grid,
            matter temperature, and temperature grid
        """

        recomb_inputs, params = args
        lna_axis_4Heequil  = self.lna_axis_full[self.idx_4He_equil]

        xe_4He, lna_4He = helium_model(lna_axis_4Heequil, adjoint=self.adjoint)(args)
        xe_full, lna_full, Tm, lna_Tm = hydrogen_model(xe_4He,lna_4He,-jnp.log(1+self.z1),lna_4He.lastval,self.twog_redshift, adjoint=self.adjoint)(args)

        # Containment boundary (ABCMB fork addition): sanitize the inf
        # padding and resample every history onto the static lna_axis_full
        # grid, so array_with_padding -- its sentinel and traced-int
        # metadata -- never leaves this package. Outside a history's valid
        # range the resample clamps to its endpoint values (benign fill; no
        # inf can reach downstream AD). Tm's validity start is returned as a
        # scalar so the caller can gate its early-time approximation.
        grid = self.lna_axis_full

        def _resample(vals_awp, axis_awp):
            vals = jnp.where(jnp.isinf(vals_awp.arr), vals_awp.lastval, vals_awp.arr)
            x0 = axis_awp.arr[0]
            dx = axis_awp.arr[1] - x0
            n = axis_awp.arr.shape[0]
            return fast_interp(grid, x0, x0 + (n - 1) * dx, vals)

        xe_grid = _resample(xe_full, lna_full)
        Tm_grid = _resample(Tm, lna_Tm)

        return (xe_grid, grid, Tm_grid, lna_Tm.arr[0])
