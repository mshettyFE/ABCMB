"""
Fluid base classes and the fluid-interface type aliases.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar, NamedTuple

import equinox as eqx
import jax.numpy as jnp
from jax import config
from jax.typing import ArrayLike
from jaxtyping import Array

if TYPE_CHECKING:
    from ..background import Background

config.update("jax_enable_x64", True)

# The params mapping seen by fluid methods. An Open set (a Mapping, not the schema's
# closed Params TypedDict) on purpose: custom species read passthrough keys the
# schema does not declare, so the extension surface must not close the dict.
# When adding fluids to library,  refer to docs/promoting_a_fluid.rst
# Params is assignable to this, so library internals keep key-name checking.
FluidParams = Mapping[str, Array]


class PerturbationContext(NamedTuple):
    """
    Everything a fluid's ``y_prime`` may need beyond its own state: the
    background cosmology, the params mapping, and the species registry for
    coupled fluids.

    Prefer attribute access (``args.params``, ``args.species_dict``); the
    context also unpacks positionally as ``(BG, params, species_list,
    species_dict)`` for backward compatibility, but positional unpacking is
    arity-coupled if fields are ever added.
    """

    BG: "Background"
    params: FluidParams
    species_list: "tuple[Fluid, ...]"
    species_dict: dict[str, int]


# output_perturbations receives only the background half of the context.
OutputArgs = tuple["Background", FluidParams]


class Fluid(eqx.Module):
    """
    Base class for fluid species.

    Defines fluid properties.

    Fields:
    -------
    first_idx : int
        Position of the first perturbation equation
        in the Diffrax vector. For most fluids this is the density perturbation
        mode "delta". Note slot 0 of the vector is reserved for the metric
        perturbation eta, so fluid blocks start at index 1: in an assembled
        model first_idx is never 0 (0 is only natural when testing a fluid
        standalone, against a hand-built y with no metric slot).
    num_equations : int
        Number of equations that need to be simultaneously evolved in the
        perturbations module.
    name : str
        Name of the fluid, used to find fluid and refer to it later
        in the computation using species_dict["name"].
    is_matter : bool
        Whether the fluid is non-relativistic today and contributes
        towards the total matter power spectrum.
    is_neutrino : bool
        Default = False
        Sector flag, like is_matter: whether this species is counted in the
        neutrino sector for the Neff / R_nu accounting in derive_parameters
        (the derivation reads this flag, never the species' name).

    Methods:
    --------
        rho : Compute energy density (units: eV cm^{-3})
        P   : Compute pressure (units: eV cm^{-3})
        w   : Compute equation of state parameter (units: dimensionless)
        y_ini   : Adiabatic initial conditions, in synchronous gauge
        y_prime : Perturbation derivatives, in synchronous gauge
        rho_delta        : Perturbed density function δρ (units: eV cm^{-3})
        rho_plus_P_theta : Velocity perturbation  (units: eV cm^{-3} Mpc^{-1})
        rho_plus_P_sigma : Compute standard shear perturbation (units: eV cm^{-3})
    """

    # Every concrete species must provide these (catch early instead of at runtime)
    name: eqx.AbstractVar[str]
    is_matter: eqx.AbstractVar[bool]

    # Sector flag like is_matter, but optional: neutrino-like species opt in.
    is_neutrino: ClassVar[bool] = False

    first_idx: int = eqx.field(static=True)
    num_equations: int = eqx.field(static=True)

    def __init__(self, first_idx, options):
        self.first_idx = first_idx

    def rho(self, lna: ArrayLike, args: FluidParams) -> Array:
        """
         Calculates the energy density of the fluid species at a given
         cosmological epoch using the logarithm of the scale factor.

        Returns:
            Energy density (units: eV cm^{-3})
        """
        raise NotImplementedError(
            "Fluid species must implement an energy density function."
        )

    def P(self, lna: ArrayLike, args: FluidParams) -> Array:
        """
        Calculates the pressure of the fluid species at a given
        cosmological epoch using the logarithm of the scale factor.

        Returns:
            Pressure (units: eV cm^{-3})
        """
        raise NotImplementedError("Fluid species must implement a pressure function.")

    def w(self, lna: ArrayLike, args: FluidParams) -> Array | float:
        """
        Compute equation of state parameter.

        Calculates the ratio of pressure to energy density, representing
        the equation of state for the fluid species.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        args : mapping
            Cosmological parameters (params)

        Returns:
        --------
        float
            Equation of state parameter (units: dimensionless)
        """
        return self.P(lna, args) / self.rho(lna, args)

    def y_ini(self, k: ArrayLike, tau_ini: ArrayLike, args: FluidParams) -> Array:
        """
        Calculates the initial state of perturbation modes at early cosmological times.

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
            Initial perturbation mode values
        """
        raise NotImplementedError(
            "Fluid species must implement the initial conditions of their perturbation modes."
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
        Compute time derivatives of perturbation modes.

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
            registry for coupled fluids. Prefer ``args.params`` /
            ``args.species_dict`` attribute access; unpacks positionally as
            ``(BG, params, species_list, species_dict)`` too.

        Returns:
        --------
        array
            Time derivatives of perturbation modes
        """
        raise NotImplementedError(
            "Fluid species must implement a perturbation derivative function."
        )

    def rho_delta(self, lna: ArrayLike, y: Array, args: FluidParams) -> Array | float:
        """
        Compute density perturbation.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        y : array
            Perturbation mode values
        args : mapping
            Cosmological parameters (params)

        Returns:
        --------
        float
            Density perturbation (units: eV cm^{-3})
        """
        raise NotImplementedError(
            "Fluid species must implement a perturbation derivative function."
        )

    def rho_plus_P_theta(
        self, lna: ArrayLike, y: Array, args: FluidParams
    ) -> Array | float:
        """
        Compute velocity perturbation.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        y : array
            Perturbation mode values
        args : mapping
            Cosmological parameters (params)

        Returns:
        --------
        float
            Velocity perturbation (units: eV cm^{-3} Mpc^{-1})
        """
        raise NotImplementedError(
            "Fluid species must implement a perturbation derivative function."
        )

    def rho_plus_P_sigma(
        self, lna: ArrayLike, y: Array, args: FluidParams
    ) -> Array | float:
        """
        Compute shear perturbation.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        y : array
            Perturbation mode values
        args : mapping
            Cosmological parameters (params)

        Returns:
        --------
        float
            Shear perturbation (units: eV cm^{-3})
        """
        raise NotImplementedError(
            "Fluid species must implement a perturbation derivative function."
        )

    def output_perturbations(
        self, lna: ArrayLike, modes: Array, args: OutputArgs
    ) -> dict[str, Array]:
        """
        Return named perturbation arrays for storage in PerturbationTable.

        Each concrete species overrides this to select the physically
        meaningful subset of its modes. Species with no perturbations
        (e.g. dark energy) return an empty dict via this base implementation.

        Parameters:
        -----------
        lna : array, shape (Nlna,)
            Logarithm of scale factor grid
        modes : array, shape (Ny, Nlna, Nk)
            Full perturbation state, already transposed
        args : tuple
            (BG, params) — background cosmology and cosmological parameters

        Returns:
        --------
        dict
            {quantity_name: array(Nlna, Nk)}. Empty for background-only species.
        """
        return {}


class StandardFluid(Fluid):
    """
    Standard implementation of perturbation methods for fluid species.

    Provides default computations for perturbation-related methods
    used in this code.

    Methods:
    --------
    rho_delta : Compute standard density perturbation (units: eV cm^{-3})
    rho_plus_P_theta : Compute standard velocity perturbation (units: eV cm^{-3} Mpc^{-1})
    rho_plus_P_sigma : Compute standard shear perturbation (units: eV cm^{-3})
    """

    def __init__(self, first_idx, options):
        super().__init__(first_idx, options)

    def get_delta(self, lna: ArrayLike, y: Array, args: FluidParams) -> Array:
        """
        Getter method for density perturbation from perturbation equations vector

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        y : array
            Perturbation mode values
        args : mapping
            Cosmological parameters (params)

        Returns:
        --------
        float
            Dimensionless density perturbation (units: None)
        """
        return y[self.first_idx]

    def get_theta(self, lna: ArrayLike, y: Array, args: FluidParams) -> Array:
        """
        Getter method for velocity divergence perturbation from perturbation equations vector

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        y : array
            Perturbation mode values
        args : mapping
            Cosmological parameters (params)

        Returns:
        --------
        float
            Velocity divergence perturbation (units: 1/Mpc)
        """
        if self.num_equations > 1:
            return y[self.first_idx + 1]
        return jnp.zeros_like(y[self.first_idx])

    def get_sigma(self, lna: ArrayLike, y: Array, args: FluidParams) -> Array:
        """
        Getter method for shear perturbation from perturbation equations vector

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        y : array
            Perturbation mode values
        args : mapping
            Cosmological parameters (params)

        Returns:
        --------
        float
            Dimensionless shear perturbation (units: None)
        """
        if self.num_equations > 2:
            return y[self.first_idx + 2]
        return jnp.zeros_like(y[self.first_idx])

    # Called by diffrax, child classes should never override. Okay to implement here.
    def rho_delta(self, lna: ArrayLike, y: Array, args: FluidParams) -> Array | float:
        """
        Compute energy density perturbation, contribution to metric perturbation evolution.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        y : array
            Perturbation mode values
        args : mapping
            Cosmological parameters (params)

        Returns:
        --------
        float
            Energy density perturbation (units: eV cm^{-3})
        """
        params = args
        return self.rho(lna, params) * self.get_delta(lna, y, args)

    def rho_plus_P_theta(
        self, lna: ArrayLike, y: Array, args: FluidParams
    ) -> Array | float:
        """
        Compute velocity perturbation times the sum of energy density and pressure. {0, i} component
        of the perturbed stress energy tensor.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        y : array
            Perturbation mode values
        args : mapping
            Cosmological parameters (params)

        Returns:
        --------
        float
            Velocity perturbation (units: eV cm^{-3} Mpc^{-1})
        """
        params = args
        return (self.rho(lna, params) + self.P(lna, params)) * self.get_theta(
            lna, y, args
        )

    def rho_plus_P_sigma(
        self, lna: ArrayLike, y: Array, args: FluidParams
    ) -> Array | float:
        """
        Compute shear stress perturbation, needed for CMB

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        y : array
            Perturbation mode values
        args : mapping
            Cosmological parameters (params)

        Returns:
        --------
        float
           Shear stress perturbation (units: eV cm^{-3})
        """
        params = args
        return (self.rho(lna, params) + self.P(lna, params)) * self.get_sigma(
            lna, y, args
        )


class BackgroundFluid(Fluid):
    num_equations = 0
    # Forced, not a default: P(k) membership is consumed through rho_delta,
    # which is hard-wired to zero below -- a "matter" fluid with no density
    # perturbation would be incoherent. Subclasses need only declare `name`.
    is_matter = False

    def __init__(self, first_idx, options):
        super().__init__(first_idx, options)

    def y_ini(self, k: ArrayLike, tau_ini: ArrayLike, args: FluidParams) -> Array:
        """
        Trivial initial condition vector for background.
        """
        return jnp.array([])

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
        Trivial derivative vector for background.
        """
        return jnp.array([])

    def rho_delta(self, lna: ArrayLike, y: Array, args: FluidParams) -> Array | float:
        return 0.0

    def rho_plus_P_theta(
        self, lna: ArrayLike, y: Array, args: FluidParams
    ) -> Array | float:
        return 0.0

    def rho_plus_P_sigma(
        self, lna: ArrayLike, y: Array, args: FluidParams
    ) -> Array | float:
        return 0.0
