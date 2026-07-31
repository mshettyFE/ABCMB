"""
Cosmological perturbation evolution module.

Integrates linear perturbation equations for scalar modes across
cosmic time using background cosmology and species interactions.
"""

import os
from typing import TYPE_CHECKING

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax, vmap
from jaxtyping import Array

from . import constants as cnst
from .schema import KBatchStrategy
from .species import Baryon, Fluid, PerturbationContext, StandardFluid

if TYPE_CHECKING:
    from ._schema_types import Options

file_dir = os.path.dirname(__file__)
jax.config.update("jax_enable_x64", True)


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


class PerturbationEvolver(eqx.Module):
    """
    Linear scalar perturbation evolution solver.

    Evolves perturbations for all fluid species using Einstein-Boltzmann
    equations in synchronous gauge.

    Attributes:
    -----------
    species_list : tuple
        A list of all fluids in the cosmology
    species_dict : dict
        Maps each fluid's name to its index in species_list
        (the coupling registry).
    k_axis_perturbations : Array
        A list of wavenumbers k at which to compute perturbations
    options : dict
        A dictionary containing run options
    adjoint : type[diffrax.AbstractAdjoint]
        Adjoint mode for diffrax solves.  Default is ForwardMode.

    Methods:
    --------
    full_evolution : Evolve perturbations for multiple k modes
    evolution_one_k : Evolve perturbations for single k mode
    get_tca_on_off : Determine tight coupling approximation times
    initial_conditions_one_k : Compute initial perturbation conditions
    get_derivatives : Compute perturbation time derivatives
    make_output_table : Create interpolatable perturbation table
    """

    species_list: tuple[Fluid, ...]
    species_dict: dict[str, int]
    k_axis_perturbations: Array
    options: "Options"

    adjoint: type[diffrax.AbstractAdjoint] = eqx.field(static=True)

    def __init__(
        self,
        species_list,
        species_dict,
        k_axis_perturbations=jnp.geomspace(1.0e-4, 0.4, 600),
        options={},
        adjoint: type[diffrax.AbstractAdjoint] = diffrax.ForwardMode,
    ):
        self.species_list = species_list
        self.species_dict = species_dict
        self.k_axis_perturbations = k_axis_perturbations
        self.options = options
        self.adjoint = adjoint

    def full_evolution(self, args):
        """
        Evolve perturbations for multiple wavenumber modes.

        Integrates perturbation equations for a range of k modes,
        then interpolates results onto common time grid.

        Parameters:
        -----------
        k    : Array
            1D axis of wavenumbers k. Perturbations are computed and stored at these values.
        args : tuple
            Background cosmology and cosmological parameters (BG, params)

        Returns:
        --------
        PerturbationTable
            Interpolatable table of perturbation evolution

        Notes:
        ------
        Uses logarithmic k spacing from 10^-4 to ~0.5 Mpc^-1 with 100 points.
        Time integration runs from early times to z=1 (lna=-ln(2)).
        """
        BG, params = args
        lna = jnp.linspace(BG.lna_transfer_start, 0.0, 500)

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

        res = res.transpose(
            2, 1, 0
        )  # Transpose so the shape is (Ny, Nlna, Nk), easier for vmapping over in PT

        PT = self.make_output_table(lna, res, args)
        return PT

    def get_starting_time(self, k, args):
        """
        Determine tight coupling approximation time range.

        Finds start and end times for tight coupling between photons and baryons
        by computing when Thomson scattering becomes ineffective relative to
        Hubble and horizon crossing time scales.

        Parameters:
        -----------
        args : tuple
            Background cosmology and cosmological parameters (BG, params)

        Returns:
        --------
        tuple
            (lna_start, lna_end) for tight coupling period

        Notes:
        ------
        Uses thresholds: τc/τh < 0.0015 (start), τh/τk < 0.07 (start),
        τc/τh > 0.015 (end), τc/τk > 0.01 (end).
        """
        BG, params = args

        # 1) Starting lna
        lna_start_range = jnp.linspace(-20.0, -10.0, 10000)

        # a) τc/τh  →  f1(lna) = BG.tau_c * BG.aH
        f1 = BG.tau_c(lna_start_range, params) * BG.aH(lna_start_range, params)
        # invert f1(lna) = thr1  →  lna = interp(thr1, f1, lna_range)
        lna1 = jnp.interp(
            self.options["R_tc"], f1, lna_start_range
        )  # jnp.interp ends up being
        # faster than fast_interp through here
        # b) τh/τk  →  f2(lna) = k / BG.aH
        f2 = k / BG.aH(lna_start_range, params)
        # invert f2(lna) = thr2
        lna2 = jnp.interp(self.options["R_large"], f2, lna_start_range)

        lna_ini = jnp.minimum(lna1, lna2)

        return lna_ini

    def initial_conditions_one_k(self, k, lna_ini, args):
        """
        Compute initial conditions for perturbation evolution.

        Sets up initial values for metric and fluid perturbations at early times
        using adiabatic initial conditions.

        Parameters:
        -----------
        k : float
            Wavenumber (units: Mpc^{-1})
        lna_ini : float
            Initial logarithm of scale factor
        args : tuple
            Background cosmology and cosmological parameters (BG, params)

        Returns:
        --------
        array
            Initial perturbation state vector

        Notes:
        ------
        Uses CLASS-style initial conditions with metric perturbations h and η.
        Assumes adiabatic initial conditions with vanishing isocurvature modes.
        """
        BG, params = args
        ### CLASS Initial Conditions ###
        tau_ini = BG.tau(lna_ini)

        om = params["om"]

        metric_eta_ini = 1.0 - k**2 * tau_ini**2 / 12.0 / (
            15.0 + 4.0 * params["R_nu"]
        ) * (
            5.0
            + 4.0 * params["R_nu"]
            - (16.0 * params["R_nu"] * params["R_nu"] + 280.0 * params["R_nu"] + 325)
            / 10.0
            / (2.0 * params["R_nu"] + 15.0)
            * tau_ini
            * om
        )

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

    def get_derivatives(self, lna, y, args):
        """
        Compute time derivatives for perturbation evolution.

        Assembles the full system of Einstein-Boltzmann equations for
        metric and fluid perturbations in synchronous gauge.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        y : array
            Current perturbation state vector
        args : tuple
            Wavenumber k and background cosmology (k, BG, params)

        Returns:
        --------
        array
            Time derivatives of perturbation state
        """
        k, BG, params = args
        a = jnp.exp(lna)
        aH = BG.aH(lna, params)
        metric_eta = y[0]

        # Metric perturbation derivatives
        sum_rho_delta = 0.0
        sum_rho_plus_P_theta = 0.0

        for i in range(len(self.species_list)):
            species = self.species_list[i]
            # If species has density perturbation, add to total.
            sum_rho_delta += species.rho_delta(lna, y, params)
            # If species has velocity perturbation, add to total.
            sum_rho_plus_P_theta += species.rho_plus_P_theta(lna, y, params)

        metric_h_prime = (
            2.0
            / aH**2
            * (
                k**2 * metric_eta
                + 4.0 * jnp.pi * cnst.G * a**2 / cnst.c_Mpc_over_s**2 * sum_rho_delta
            )
        )
        metric_eta_prime = (
            4.0
            * jnp.pi
            * cnst.G
            * a**2
            / aH
            / k**2
            * sum_rho_plus_P_theta
            / cnst.c_Mpc_over_s**2
        )

        # Now loop over all species and assemble their respective y_primes
        args = PerturbationContext(BG, params, self.species_list, self.species_dict)
        y_prime = jnp.array([metric_eta_prime])
        for i in range(len(self.species_list)):
            species = self.species_list[i]
            piece = species.y_prime(k, lna, metric_h_prime, metric_eta_prime, y, args)
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

    def evolution_one_k(self, k, lna, args):
        """
        Evolve perturbations for single wavenumber mode.

        Integrates Einstein-Boltzmann equations from early times through
        recombination to late times using adaptive time stepping.

        Parameters:
        -----------
        k : float
            Wavenumber (units: Mpc^{-1})
        lna : array
            Logarithm of scale factor grid for output
        args : tuple
            Background cosmology and cosmological parameters (BG, params)

        Returns:
        --------
        diffrax.Solution
            Dense solution object for interpolation

        """

        lna_start = self.get_starting_time(
            k, args
        )  # Start and end times from tight coupling settings
        lna_end = 0.0

        # For small k's the superhorizon time can be set relatively late, but we impose a cutoff of z~20000 for all modes
        # at the very least.
        lna_start = jnp.minimum(lna_start, -10.0)

        # Initial conditions for tight coupling
        y_ini = self.initial_conditions_one_k(k, lna_start, args)

        # Settings for post-tight coupling
        term = diffrax.ODETerm(self.get_derivatives)
        solver = diffrax.Kvaerno5()

        rtol = jnp.where(
            k > self.options["k_split_PE"],
            self.options["rtol_large_k_PE"],
            self.options["rtol_small_k_PE"],
        )

        atol = jnp.where(
            k > self.options["k_split_PE"],
            self.options["atol_large_k_PE"],
            self.options["atol_small_k_PE"],
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

        return sol.ys

    def make_output_table(self, lna, modes, args):
        """
        Create interpolatable perturbation table from evolution results.

        Extracts key perturbation modes and computes derived quantities.

        Parameters:
        -----------
        lna : array
            Logarithm of scale factor grid
        modes : array
            Perturbation evolution results
        args : tuple
            Background cosmology and cosmological parameters (BG, params)

        Returns:
        --------
        PerturbationTable
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
        baryon = self.species_list[self.species_dict["Baryon"]]
        photon = self.species_list[self.species_dict["Photon"]]
        # Structural requirements on the named roles (narrows the types; fails
        # loudly at trace time for an incompatible replacement): the Baryon
        # role needs cs2 and the standard layout, the Photon role the layout.
        assert isinstance(baryon, Baryon)
        assert isinstance(photon, StandardFluid)
        delta_b = baryon.get_delta(lna, modes, params)
        theta_b = baryon.get_theta(lna, modes, params)
        theta_g = photon.get_theta(lna, modes, params)

        karr = k[None, :]
        a = jnp.exp(lna)[:, None]
        aH = BG.aH(lna, params)[:, None]
        cs2 = baryon.cs2(
            lna, PerturbationContext(BG, params, self.species_list, self.species_dict)
        )[:, None]
        R = (
            4.0
            * photon.rho(lna, params)[:, None]
            / 3.0
            / baryon.rho(lna, params)[:, None]
        )
        tau_c = BG.tau_c(lna, params)[:, None]

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
                rho_delta = vmap(s.rho_delta, in_axes=(0, 1, None))(lna, modes, params)
                sum_rho_delta += rho_delta
                sum_rho_plus_P_theta += vmap(s.rho_plus_P_theta, in_axes=(0, 1, None))(
                    lna, modes, params
                )
                sum_rho_plus_P_sigma += vmap(s.rho_plus_P_sigma, in_axes=(0, 1, None))(
                    lna, modes, params
                )

                if s.is_matter:
                    sum_rho_delta_m += rho_delta
                    sum_rho_m += s.rho(lna, params)

        delta_m = sum_rho_delta_m / sum_rho_m[:, None]

        metric_h_prime = (
            2.0
            / aH**2
            * (
                karr**2 * metric_eta
                + 4.0 * jnp.pi * cnst.G * a**2 / cnst.c_Mpc_over_s**2 * sum_rho_delta
            )
        )
        metric_eta_prime = (
            4.0
            * jnp.pi
            * cnst.G
            * a**2
            / aH
            * sum_rho_plus_P_theta
            / cnst.c_Mpc_over_s**2
            / karr**2
        )
        metric_alpha = aH * (metric_h_prime + 6.0 * metric_eta_prime) / 2.0 / karr**2
        metric_alpha_prime = (
            metric_eta / aH
            - 2.0 * metric_alpha
            - 12.0
            * jnp.pi
            * cnst.G
            * a**2
            / aH
            * sum_rho_plus_P_sigma
            / cnst.c_Mpc_over_s**2
            / karr**2
        )

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

    k: Array
    lna: Array
    delta_m: Array
    theta_b_prime: Array

    metric_eta: Array
    metric_h_prime: Array
    metric_eta_prime: Array
    metric_alpha: Array
    metric_alpha_prime: Array

    species_perturbations: dict
