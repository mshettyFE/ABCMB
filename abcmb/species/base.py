"""
Fluid base classes and the fluid-interface type aliases.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar, TypeVar, cast

import equinox as eqx
import jax.numpy as jnp
from jax import config
from jax.typing import ArrayLike
from jaxtyping import Array, Float

if TYPE_CHECKING:
    from ..background import Background

config.update("jax_enable_x64", True)

# The params mapping seen by fluid methods. An Open set (a Mapping, not the schema's
# closed Params TypedDict) on purpose: custom species read passthrough keys the
# schema does not declare, so the extension surface must not close the dict.
# When adding fluids to library,  refer to docs/promoting_a_fluid.rst
# Params is assignable to this, so library internals keep key-name checking.
FluidParams = Mapping[str, Array]

_F = TypeVar("_F", bound="Fluid")


class MetricSources(eqx.Module):
    """
    The metric's contribution to a fluid's equations, in the three slots it can
    occupy. Written once by the evolver, read by every ``y_prime``.

    Deliberately carries no gauge tag: a fluid must not be able to ask which
    gauge it is in, since that is exactly what this abstraction buys.
    """

    continuity: Array
    euler: Array
    shear: Array


class PerturbationContext(eqx.Module):
    """
    Everything a fluid's ``y_prime`` may need beyond its own state: the
    background cosmology, the params mapping, and the species tuple for
    coupled fluids.

    Access by attribute (``args.BG``, ``args.params``, ``args.species_list``)
    and look up coupled fluids with ``args.find(name, cls)``.
    """

    BG: "Background"
    params: FluidParams
    species_list: "tuple[Fluid, ...]"

    def find(self, name: str, cls: "type[_F] | None" = None) -> "_F":
        """
        Return the fluid named ``name``, optionally narrowed to ``cls`` --
        the coupled-fluid lookup (see :func:`find_species`).
        """
        return find_species(self.species_list, name, cls)


# output_perturbations receives only the background half of the context.
OutputArgs = tuple["Background", FluidParams]


class Fluid(eqx.Module):
    """
    Base class for fluid species.
    """

    # Every concrete species must provide these (catch early instead of at runtime)
    name: eqx.AbstractVar[str]
    # flag fluid as non-relativistic and that it contributes to the matter power spectrum
    is_matter: eqx.AbstractVar[bool]
    # Sector flag like is_matter, but optional: neutrino-like species opt in.
    # used for the Neff / R_nu accounting in derive_parameters
    is_neutrino: ClassVar[bool] = False
    # Position of first perturbation equation in Diffrax vector. Slot 0 is reserved
    # for metric perturbations, so fluid block actually starts at 1.
    first_idx: int = eqx.field(static=True)
    # Number of equations to be evolved in the perturbations module
    num_equations: int = eqx.field(static=True)

    def __init__(self, first_idx, options):
        self.first_idx = first_idx

    def rho(self, lna: ArrayLike, args: FluidParams) -> Array:
        """
         Calculates the energy density of the fluid species at a given
         cosmological epoch using the logarithm of the scale factor.

        Contract: scalar-in, scalar-out -- write for a single lna and let
        callers batch with jax.vmap (constants may return plain scalars).

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

        Contract: scalar-in, scalar-out (see rho).

        Returns:
            Pressure (units: eV cm^{-3})
        """
        raise NotImplementedError("Fluid species must implement a pressure function.")

    def w(self, lna: ArrayLike, args: FluidParams) -> Array:
        """
        Compute equation of state parameter.

        Calculates the ratio of pressure to energy density, representing
        the equation of state for the fluid species.

        Returns:
           Equation of state parameter (units: dimensionless)
        """
        return self.P(lna, args) / self.rho(lna, args)

    def y_ini(self, k: ArrayLike, tau_ini: ArrayLike, args: FluidParams) -> Array:
        """
        Calculates the initial state of perturbation modes at early cosmological times.

        Two conventions are implicit in this signature, so they are stated here:

        * **The adiabatic mode.** Returning a single state leaves no room for the
          isocurvature modes, so "adiabatic" is what this method means. Supporting
          the others would need the mode as an argument.
        * **Synchronous-gauge variables.** The shared series in
          :mod:`.adiabatic_ics` are normalized to ``eta = 1`` superhorizon, which
          is the synchronous metric perturbation.

        Returns:
           Initial perturbation mode values
        """
        raise NotImplementedError(
            "Fluid species must implement the initial conditions of their perturbation modes."
        )

    def y_prime(
        self,
        k: ArrayLike,  # wavenumber
        lna: ArrayLike,
        sources: MetricSources,  # metric contribution, in its three slots
        y: Array,
        args: PerturbationContext,
    ) -> Array:
        """
        Compute time derivatives of perturbation modes.

        Write the equations against ``sources.continuity`` / ``sources.euler`` /
        ``sources.shear`` rather than against the metric variables of a
        particular gauge -- see :class:`MetricSources` for the substitution
        table. A fluid written that way is correct in every gauge ABCMB
        supports, with no branching.

        Returns:
           Time derivatives of perturbation modes
        """
        raise NotImplementedError(
            "Fluid species must implement a perturbation derivative function."
        )

    def rho_delta(
        self, lna: ArrayLike, y: Array, args: PerturbationContext
    ) -> Array | float:
        """
        Compute density perturbation.

        Returns:
           Density perturbation (units: eV cm^{-3})
        """
        raise NotImplementedError(
            "Fluid species must implement a perturbation derivative function."
        )

    def rho_plus_P_theta(
        self, lna: ArrayLike, y: Array, args: PerturbationContext
    ) -> Array | float:
        """
        Compute velocity perturbation.

        Returns:
           Velocity perturbation (units: eV cm^{-3} Mpc^{-1})
        """
        raise NotImplementedError(
            "Fluid species must implement a perturbation derivative function."
        )

    def rho_plus_P_sigma(
        self, lna: ArrayLike, y: Array, args: PerturbationContext
    ) -> Array | float:
        """
        Compute shear perturbation.

        Returns:
           Shear perturbation (units: eV cm^{-3})
        """
        raise NotImplementedError(
            "Fluid species must implement a perturbation derivative function."
        )

    def output_perturbations(
        self,
        lna: Float[Array, " n_lna"],
        modes: Float[Array, "n_y n_lna n_k"],
        args: OutputArgs,
    ) -> dict[str, Float[Array, "n_lna n_k"]]:
        """
        Return named perturbation arrays for storage in PerturbationTable.

        Each concrete species overrides this to select the physically
        meaningful subset of its modes. Species with no perturbations
        (e.g. dark energy) return an empty dict via this base implementation.
        """
        return {}


def find_species(
    species_list: "tuple[Fluid, ...]", name: str, cls: "type[_F] | None" = None
) -> "_F":
    """
    Return the fluid named ``name`` from ``species_list``. Passing ``cls``
    narrows the static type and raises ``TypeError`` if the named fluid does
    not implement that interface -- pass the interface the caller actually
    relies on (e.g. ``StandardFluid`` for the delta/theta layout).

    A linear scan on purpose: fluid lookups run at trace time (or eagerly,
    once), never inside compiled code, and a cosmology holds a handful of
    fluids, so a name->index cache would be pure bookkeeping.
    """
    for s in species_list:
        if s.name != name:
            continue
        if cls is not None and not isinstance(s, cls):
            raise TypeError(
                f"the fluid named '{name}' is a {type(s).__name__}, not a "
                f"{cls.__name__} subclass: the caller requires that "
                f"interface. To customize it, subclass species.{cls.__name__} "
                "(see docs/promoting_a_fluid.rst)."
            )
        return cast("_F", s)
    raise ValueError(
        f"no fluid named '{name}'; this model has {[s.name for s in species_list]}."
    )


class StandardFluid(Fluid):
    """
    Standard implementation of perturbation methods for fluid species.

    Provides default computations for perturbation-related methods
    used in this code.

    """

    def get_delta(self, lna: ArrayLike, y: Array, args: PerturbationContext) -> Array:
        """
        Getter method for density perturbation from perturbation equations vector

        Returns:
           Dimensionless density perturbation (units: None)
        """
        return y[self.first_idx]

    def get_theta(self, lna: ArrayLike, y: Array, args: PerturbationContext) -> Array:
        """
        Getter method for velocity divergence perturbation from perturbation equations vector

        Returns:
           Velocity divergence perturbation (units: 1/Mpc)
        """
        if self.num_equations > 1:
            return y[self.first_idx + 1]
        return jnp.zeros_like(y[self.first_idx])

    def get_sigma(self, lna: ArrayLike, y: Array, args: PerturbationContext) -> Array:
        """
        Getter method for shear perturbation from perturbation equations vector

        Returns:
           Dimensionless shear perturbation (units: None)
        """
        if self.num_equations > 2:
            return y[self.first_idx + 2]
        return jnp.zeros_like(y[self.first_idx])

    # Consumed by the Einstein-equation sources (get_derivatives, inside the
    # diffrax solve) and the output-table sums. Subclasses customize the
    # getters above, not these aggregates -- the formulas rho*delta and
    # (rho+P)*theta are layout-independent. A fluid where the factorization
    # itself fails (e.g. momentum-binned integrals) implements the Fluid
    # interface directly, as MassiveNeutrino does.
    def rho_delta(
        self, lna: ArrayLike, y: Array, args: PerturbationContext
    ) -> Array | float:
        """
        Compute energy density perturbation, contribution to metric perturbation evolution.

        Returns:
            Energy density perturbation (units: eV cm^{-3})
        """
        params = args.params
        return self.rho(lna, params) * self.get_delta(lna, y, args)

    def rho_plus_P_theta(
        self, lna: ArrayLike, y: Array, args: PerturbationContext
    ) -> Array | float:
        """
        Compute velocity perturbation times the sum of energy density and pressure. {0, i} component
        of the perturbed stress energy tensor.

        Returns:
            Velocity perturbation (units: eV cm^{-3} Mpc^{-1})
        """
        params = args.params
        return (self.rho(lna, params) + self.P(lna, params)) * self.get_theta(
            lna, y, args
        )

    def rho_plus_P_sigma(
        self, lna: ArrayLike, y: Array, args: PerturbationContext
    ) -> Array | float:
        """
        Compute shear stress perturbation, needed for CMB

        Returns:
           Shear stress perturbation (units: eV cm^{-3})
        """
        params = args.params
        return (self.rho(lna, params) + self.P(lna, params)) * self.get_sigma(
            lna, y, args
        )


class BackgroundFluid(Fluid):
    num_equations = 0
    # Forced, not a default: P(k) membership is consumed through rho_delta,
    # which is hard-wired to zero below -- a "matter" fluid with no density
    # perturbation would be incoherent. Subclasses need only declare `name`.
    is_matter = False

    def y_ini(self, k: ArrayLike, tau_ini: ArrayLike, args: FluidParams) -> Array:
        """
        Trivial initial condition vector for background.
        """
        return jnp.array([])

    def y_prime(
        self,
        k: ArrayLike,
        lna: ArrayLike,
        sources: MetricSources,
        y: Array,
        args: PerturbationContext,
    ) -> Array:
        """
        Trivial derivative vector for background.
        """
        return jnp.array([])

    def rho_delta(
        self, lna: ArrayLike, y: Array, args: PerturbationContext
    ) -> Array | float:
        return 0.0

    def rho_plus_P_theta(
        self, lna: ArrayLike, y: Array, args: PerturbationContext
    ) -> Array | float:
        return 0.0

    def rho_plus_P_sigma(
        self, lna: ArrayLike, y: Array, args: PerturbationContext
    ) -> Array | float:
        return 0.0
