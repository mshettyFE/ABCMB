"""
Cosmological perturbation evolution module.

Integrates linear perturbation equations for scalar modes across
cosmic time using background cosmology and species interactions.
"""

import os
from typing import TYPE_CHECKING, cast

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax, vmap
from jax.typing import ArrayLike
from jaxtyping import Array, Float

from . import constants as cnst
from .inputs.schema import KBatchStrategy
from .species import Baryon, Fluid, PerturbationContext, StandardFluid, find_species

if TYPE_CHECKING:
    from .background import Background
    from .inputs._schema_types import Options, Params

file_dir = os.path.dirname(__file__)
jax.config.update("jax_enable_x64", True)

# The (BG, params) pair threaded through the PerturbationEvolver methods.
EvolverArgs = tuple["Background", "Params"]
# What the diffrax vector field (get_derivatives) receives: the evolver pair
# with the mode's wavenumber prepended (built as ``(k, *args)`` in
# evolution_one_k).
DerivativeArgs = tuple[ArrayLike, "Background", "Params"]


def _k_batch_strategy(value: str) -> KBatchStrategy:
    """
    Resolve the ``k_batch_strategy`` option to a concrete strategy.

    'auto' picks by the JAX default backend: VMAP on GPU (lockstep batching
    saturates throughput hardware; the wasted lanes are free), SCAN otherwise
    (sequential modes each take exactly their own adaptive steps, minimizing
    total work -- vmapping on CPU would make every mode pay for the stiffest
    one's step count).
    """
    value = value.lower()
    if value == "auto":
        return (
            KBatchStrategy.VMAP
            if jax.default_backend() == "gpu"
            else KBatchStrategy.SCAN
        )
    try:
        return KBatchStrategy(value)
    except ValueError:
        raise ValueError(
            f"k_batch_strategy={value!r} is not one of 'auto', 'scan', 'vmap'."
        ) from None


def _einstein_constraints(
    k: ArrayLike,
    a: ArrayLike,
    aH: ArrayLike,
    metric_eta: ArrayLike,
    sum_rho_delta: ArrayLike,
    sum_rho_plus_P_theta: ArrayLike,
) -> tuple[Array, Array]:
    """
    The synchronous-gauge Einstein constraints closing the fluid system:
    ``h'`` from the energy constraint and ``eta'`` from the momentum
    constraint, given the summed fluid sources.
    """
    grav = 4.0 * jnp.pi * cnst.G * a**2
    metric_h_prime = (
        2.0 / aH**2 * (k**2 * metric_eta + grav / cnst.c_Mpc_over_s**2 * sum_rho_delta)
    )
    metric_eta_prime = grav / aH / k**2 * sum_rho_plus_P_theta / cnst.c_Mpc_over_s**2
    return metric_h_prime, metric_eta_prime


class PerturbationEvolver(eqx.Module):
    """
    Linear scalar perturbation evolution solver.

    Evolves perturbations for all fluid species using Einstein-Boltzmann
    equations in synchronous gauge.

    """

    # A list of all fluids in the cosmology; coupled fluids are looked
    # up in it by name (``species.find_species``).
    species_list: tuple[Fluid, ...]
    # The wavenumbers k at which to compute perturbations. Required, no
    # default: build with model_setup.get_k_axis_perturbations, whose hybrid
    # grid samples the acoustic oscillations linearly -- a plausible-looking
    # log grid would quietly undersample them at high k.
    k_axis_perturbations: Float[Array, " n_k"]
    options: "Options" = eqx.field(default_factory=dict)
    adjoint: type[diffrax.AbstractAdjoint] = eqx.field(
        default=diffrax.ForwardMode, static=True
    )

    def full_evolution(self, args: EvolverArgs) -> "PerturbationTable":
        """
        Evolve perturbations for multiple wavenumber modes.

        Integrates perturbation equations for a range of k modes,
        then interpolates results onto common time grid.

        Notes:
        ------
        Time grid: ``lna_output_points`` points from the transfer start (set by
        the ``transfer_start_threshold`` option via the background) to today.
        The k grid is ``self.k_axis_perturbations``, built by
        ``model_setup.get_k_axis_perturbations``.
        """
        BG, params = args
        lna = jnp.linspace(
            BG.lna_transfer_start, 0.0, self.options["lna_output_points"]
        )

        def scan_fun(_, ki):
            # evolution_one_k returns shape (Nlna, Ny)
            y = self.evolution_one_k(ki, lna, args)
            return None, y

        if _k_batch_strategy(self.options["k_batch_strategy"]) is KBatchStrategy.VMAP:
            res = vmap(self.evolution_one_k, in_axes=[0, None, None])(
                self.k_axis_perturbations, lna, args
            )
        else:
            _, res = lax.scan(
                scan_fun, None, self.k_axis_perturbations
            )  # res has shape (Nk, Nlna, Ny)

        # Transpose so the shape is (Ny, Nlna, Nk), easier for vmapping over in PT
        res = res.transpose(2, 1, 0)
        PT = self.make_output_table(lna, res, args)
        return PT

    def get_starting_time(self, k: ArrayLike, args: EvolverArgs) -> Float[Array, ""]:
        """
        Determine the integration starting time for one mode.
        Thresholds: τc/τh < ``R_tc`` (tight coupling) and τh/τk < ``R_large``
        (superhorizon); the earlier of the two crossings wins.
        """
        BG, params = args

        # 1) Starting lna
        lna_start_range = jnp.linspace(-20.0, -10.0, 10000)

        # a) τc/τh  →  f1(lna) = BG.tau_c * BG.aH
        f1 = vmap(lambda l: BG.tau_c(l, params) * BG.aH(l, params))(lna_start_range)
        # invert f1(lna) = thr1  →  lna = interp(thr1, f1, lna_range)
        # jnp.interp, not ABCMBTools.fast_interp: the inversion abscissae
        # (f1, f2) are non-uniform over ~4 decades, so fast_interp's
        # uniform-grid indexing would misplace crossings by several e-folds
        # in lna
        lna1 = jnp.interp(self.options["R_tc"], f1, lna_start_range)
        # b) τh/τk  →  f2(lna) = k / BG.aH
        f2 = k / vmap(lambda l: BG.aH(l, params))(lna_start_range)
        # invert f2(lna) = thr2
        lna2 = jnp.interp(self.options["R_large"], f2, lna_start_range)

        lna_ini = jnp.minimum(lna1, lna2)

        return lna_ini

    def initial_conditions_one_k(
        self, k: ArrayLike, lna_ini: ArrayLike, args: EvolverArgs
    ) -> Float[Array, " n_y"]:
        """
        Compute initial conditions for perturbation evolution.

        Sets up initial values for metric and fluid perturbations at early times
        using adiabatic initial conditions.

        Returns:
           Initial perturbation state vector

        Notes:
        ------
        Uses CLASS-style initial conditions with metric perturbations h and η.
        Assumes adiabatic initial conditions with vanishing isocurvature modes.
        """
        BG, params = args
        # Metric eta adiabatic IC: the CRS series with beta_1 = 1/2, i.e.
        # eta -> 1 superhorizon (the normalization that fixes every species'
        # amplitude; the species' own series live in species/adiabatic_ics).
        tau_ini = BG.tau(lna_ini)

        om = params["om"]
        R_nu = params["R_nu"]

        prefactor = k**2 * tau_ini**2 / 12.0 / (15.0 + 4.0 * R_nu)
        leading = 5.0 + 4.0 * R_nu
        slope = (16.0 * R_nu * R_nu + 280.0 * R_nu + 325.0) / 10.0 / (2.0 * R_nu + 15.0)
        correction = slope * tau_ini * om
        metric_eta_ini = 1.0 - prefactor * (leading - correction)

        # Static  layout check: a species whose y_ini size
        # disagrees with its declared num_equations would silently shift every
        # later fluid's slice of y (the totals still add up, so nothing else
        # errors)
        pieces = []
        for p in self.species_list:
            piece = p.y_ini(k, tau_ini, params)
            if piece.shape != (p.num_equations,):
                raise ValueError(
                    f"species '{p.name}' declares num_equations="
                    f"{p.num_equations} but its y_ini returned shape "
                    f"{piece.shape}; the perturbation vector layout would be "
                    "misaligned."
                )
            pieces.append(piece)
        y_ini = jnp.concatenate([jnp.array([metric_eta_ini])] + pieces)

        return y_ini

    def get_derivatives(
        self, lna: ArrayLike, y: Float[Array, " n_y"], args: DerivativeArgs
    ) -> Float[Array, " n_y"]:
        """
        Compute time derivatives for perturbation evolution.

        Assembles the full system of Einstein-Boltzmann equations for
        metric and fluid perturbations in synchronous gauge.

        y : array
            Current perturbation state vector
        Returns:
           Time derivatives of perturbation state
        """
        k, BG, params = args
        a = jnp.exp(lna)
        aH = BG.aH(lna, params)
        metric_eta = y[0]

        # The single fluid-facing context, shared by the rho_* aggregates
        # here and the y_prime calls below.
        ctx = PerturbationContext(BG, params, self.species_list)

        # Metric perturbation derivatives
        sum_rho_delta = 0.0
        sum_rho_plus_P_theta = 0.0

        for i in range(len(self.species_list)):
            species = self.species_list[i]
            # If species has density perturbation, add to total.
            sum_rho_delta += species.rho_delta(lna, y, ctx)
            # If species has velocity perturbation, add to total.
            sum_rho_plus_P_theta += species.rho_plus_P_theta(lna, y, ctx)

        metric_h_prime, metric_eta_prime = _einstein_constraints(
            k, a, aH, metric_eta, sum_rho_delta, sum_rho_plus_P_theta
        )

        # Now loop over all species and assemble their respective y_primes
        y_prime = jnp.array([metric_eta_prime])
        for i in range(len(self.species_list)):
            species = self.species_list[i]
            piece = species.y_prime(k, lna, metric_h_prime, metric_eta_prime, y, ctx)
            # Same static layout check as in initial_conditions_one_k.
            # OK since error on static data. Gets shaken out upon tracing
            if piece.shape != (species.num_equations,):
                raise ValueError(
                    f"species '{species.name}' declares num_equations="
                    f"{species.num_equations} but its y_prime returned shape "
                    f"{piece.shape}; the perturbation vector layout would be "
                    "misaligned."
                )
            y_prime = jnp.concatenate((y_prime, piece))

        return y_prime

    def evolution_one_k(
        self, k: ArrayLike, lna: Float[Array, " n_lna"], args: EvolverArgs
    ) -> Float[Array, "n_lna n_y"]:
        """
        Evolve perturbations for single wavenumber mode.

        Integrates Einstein-Boltzmann equations from early times through
        recombination to late times using adaptive time stepping.

        Returns:
        --------
        array
            Perturbation state at each requested lna, shape

        """

        # Start and end times from tight coupling settings
        lna_start = self.get_starting_time(k, args)
        lna_end = 0.0

        # For small k's the superhorizon time can be set relatively late, but we impose a cutoff of z~20000 for all modes
        # at the very least.
        lna_start = jnp.minimum(lna_start, -10.0)

        # Initial conditions for tight coupling
        y_ini = self.initial_conditions_one_k(k, lna_start, args)

        # Settings for post-tight coupling
        term = diffrax.ODETerm(self.get_derivatives)
        solver = diffrax.Kvaerno5()

        large_k = k > self.options["k_split_PE"]
        rtol = jnp.where(
            large_k, self.options["rtol_large_k_PE"], self.options["rtol_small_k_PE"]
        )
        atol = jnp.where(
            large_k, self.options["atol_large_k_PE"], self.options["atol_small_k_PE"]
        )

        stepsize_controller = diffrax.PIDController(
            pcoeff=self.options["pcoeff_PE"],
            icoeff=self.options["icoeff_PE"],
            dcoeff=self.options["dcoeff_PE"],
            rtol=rtol,
            atol=atol,
        )
        saveat = diffrax.SaveAt(ts=lna)
        adjoint = self.adjoint()

        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=lna_start,
            t1=lna_end,
            dt0=1.0e-2,
            y0=y_ini,
            stepsize_controller=stepsize_controller,
            max_steps=self.options["max_steps_PE"],
            saveat=saveat,
            args=(k, *args),
            adjoint=adjoint,
        )

        # sol.ys is typed PyTree | None (None only without SaveAt(ts=...),
        # which this solve always passes).
        return cast(Array, sol.ys)

    def make_output_table(
        self,
        lna: Float[Array, " n_lna"],
        modes: Float[Array, "n_y n_lna n_k"],
        args: EvolverArgs,
    ) -> "PerturbationTable":
        """
        Create interpolatable perturbation table from evolution results.

        Extracts key perturbation modes and computes derived quantities.

        modes
            Perturbation evolution results
        Returns:
           Organized perturbation data for interpolation

        """
        k = self.k_axis_perturbations
        BG, params = args

        metric_eta = modes[0]

        species_perturbations = {
            s.name: s.output_perturbations(lna, modes, (BG, params))
            for s in self.species_list
        }

        # Baryon velocity derivative — backward-calculated from the Boltzmann equations.
        # Requires the Baryon and Photon objects for cs2 and the coupling R.
        # Structural requirements on the named roles (narrows the types; fails
        # loudly for an incompatible replacement): the Baryon role needs cs2
        # and the standard layout, the Photon role the layout.
        baryon = find_species(self.species_list, "Baryon", Baryon)
        photon = find_species(self.species_list, "Photon", StandardFluid)
        ctx = PerturbationContext(BG, params, self.species_list)
        delta_b = baryon.get_delta(lna, modes, ctx)
        theta_b = baryon.get_theta(lna, modes, ctx)
        theta_g = photon.get_theta(lna, modes, ctx)

        karr = k[None, :]
        a = jnp.exp(lna)[:, None]
        # Background/species quantities on the output lna grid: all follow
        # the scalar contract, so batching is an explicit vmap here.
        aH = vmap(lambda l: BG.aH(l, params))(lna)[:, None]
        cs2 = vmap(lambda l: baryon.cs2(l, ctx))(lna)[:, None]
        R = (
            4.0
            * vmap(lambda l: photon.rho(l, params))(lna)[:, None]
            / 3.0
            / vmap(lambda l: baryon.rho(l, params))(lna)[:, None]
        )
        tau_c = vmap(lambda l: BG.tau_c(l, params))(lna)[:, None]

        theta_b_prime = (
            -theta_b
            + cs2 / aH * (karr**2 * delta_b)
            + R / aH / tau_c * (theta_g - theta_b)
        )

        # Sum density/velocity/shear over all species for metric derivatives and delta_m.
        sum_rho_delta = jnp.zeros_like(modes[0])
        sum_rho_plus_P_theta = jnp.zeros_like(modes[0])
        sum_rho_plus_P_sigma = jnp.zeros_like(modes[0])
        sum_rho_delta_m = jnp.zeros_like(modes[0])
        sum_rho_m = 0.0

        for s in self.species_list:
            if s.num_equations > 0:
                rho_delta = vmap(s.rho_delta, in_axes=(0, 1, None))(lna, modes, ctx)
                sum_rho_delta += rho_delta
                sum_rho_plus_P_theta += vmap(s.rho_plus_P_theta, in_axes=(0, 1, None))(
                    lna, modes, ctx
                )
                sum_rho_plus_P_sigma += vmap(s.rho_plus_P_sigma, in_axes=(0, 1, None))(
                    lna, modes, ctx
                )

                if s.is_matter:
                    sum_rho_delta_m += rho_delta
                    # Scalar contract: batch rho over the output grid.
                    sum_rho_m += vmap(lambda l: s.rho(l, params))(lna)

        delta_m = sum_rho_delta_m / sum_rho_m[:, None]

        # Same Einstein constraints as the ODE field, batched on the
        # output (lna, k) grid.
        metric_h_prime, metric_eta_prime = _einstein_constraints(
            karr, a, aH, metric_eta, sum_rho_delta, sum_rho_plus_P_theta
        )
        metric_alpha = aH * (metric_h_prime + 6.0 * metric_eta_prime) / 2.0 / karr**2
        # alpha' from the anisotropic-stress (shear) Einstein equation.
        grav_shear = 12.0 * jnp.pi * cnst.G * a**2
        shear_term = (
            grav_shear / aH * sum_rho_plus_P_sigma / cnst.c_Mpc_over_s**2 / karr**2
        )
        metric_alpha_prime = metric_eta / aH - 2.0 * metric_alpha - shear_term

        return PerturbationTable(
            k,
            lna,
            delta_m,
            theta_b_prime,
            metric_eta,
            metric_h_prime,
            metric_eta_prime,
            metric_alpha,
            metric_alpha_prime,
            species_perturbations,
        )


class PerturbationTable(eqx.Module):
    """
    Interpolatable table of perturbation evolution.

    Stores perturbation modes as 2D arrays over wavenumber and time
    for efficient interpolation. Per-species perturbations in physically
    meaningful form are accessible via species_perturbations.

    Attributes:
    -----------
    k : array
        Wavenumber grid (units: Mpc^{-1})
    lna : array
        Logarithm of scale factor grid
    delta_m : array
        Total matter density perturbation, weighted sum over all matter species
    theta_b_prime : array
        Baryon velocity derivative (backward-calculated from Boltzmann equations)
    metric_eta : array
        Metric perturbation η
    metric_h_prime : array
        Time derivative of metric h
    metric_eta_prime : array
        Time derivative of metric η
    metric_alpha : array
        Derived metric perturbation α
    metric_alpha_prime : array
        Time derivative of metric α
    species_perturbations : dict
        Named perturbation arrays for each species, keyed by species name.
        Each value is a dict {quantity: array(Nlna, Nk)}.
        Species with no perturbations (e.g. dark energy) map to {}.
    """

    k: Float[Array, " n_k"]
    lna: Float[Array, " n_lna"]
    delta_m: Float[Array, "n_lna n_k"]
    theta_b_prime: Float[Array, "n_lna n_k"]

    metric_eta: Float[Array, "n_lna n_k"]
    metric_h_prime: Float[Array, "n_lna n_k"]
    metric_eta_prime: Float[Array, "n_lna n_k"]
    metric_alpha: Float[Array, "n_lna n_k"]
    metric_alpha_prime: Float[Array, "n_lna n_k"]

    species_perturbations: dict[str, dict[str, Float[Array, "n_lna n_k"]]]
