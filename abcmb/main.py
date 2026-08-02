import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import config
from jax.typing import ArrayLike
from jaxtyping import Array

from . import background, model_setup, perturbations, spectrum
from .background import Background, BackgroundPreRecomb
from .hyrex import hyrex
from .inputs import derived, schema
from .linx.abundances import AbundanceModel
from .linx.background import BackgroundModel
from .linx.nuclear import NuclearRates
from .reionization import ReionizationModelFromTau, ReionizationModelFromZ
from .species import Fluid

if TYPE_CHECKING:
    # Compile-time only (generated type-checker artifact); annotations quote the names.
    from .inputs._schema_types import Options, Params
    from .perturbations import PerturbationTable
    from .recomb_interface import RecombOutput

file_dir = os.path.dirname(__file__)

config.update("jax_enable_x64", True)

# Unconditionally derived and unconditionally consumed by the background /
# perturbation stages: their presence distinguishes the output of
# add_derived_parameters from raw input params.
_DERIVED_KEY_SENTINELS = (
    "omega_r",
    "omega_Lambda",
    "omega_m",
    "om",
    "R_b",
    "R_nu",
    "H0",
)


def _check_derived(params: Mapping[str, object]) -> None:
    """
    Loud-early guard for the staged entry points: raw params fail deep in a
    trace with a bare KeyError naming a key the user never supplied (e.g.
    'omega_r'); this names the actual mistake instead. Key *presence* is
    static dict structure, so the check is jit-safe and free at runtime --
    values are deliberately not checked.
    """
    missing = [k for k in _DERIVED_KEY_SENTINELS if k not in params]
    if missing:
        raise ValueError(
            f"params is missing derived keys {missing}: pass the output of "
            "Model.add_derived_parameters(...), not raw input parameters."
        )


class Model(eqx.Module):
    """
    Model configuration and computation manager.

    Creates instances of fluid species based on user input and organizes
    them for computation. Manages the full pipeline from background
    evolution through CMB power spectrum computation.
    """

    PE: perturbations.PerturbationEvolver
    SS: spectrum.SpectrumSolver
    RecModel: hyrex.recomb_model
    options: "Options"
    raw_options: dict

    species_list: tuple[Fluid, ...]

    PArthENoPE_CLASS_table: (
        Array  # A 2D table for interpolation of the helium-4 mass fraction based
    )
    # on the user's input baryon density and Neff

    thermo_model_DNeff: (
        BackgroundModel | None
    )  # A LINX background model for BBN thermodynamics
    # A LINX abundance model used for computing the helium-4 mass fraction
    # given the user's input baryon density, Neff, neutron lifetime, and
    # nuclear reaction rates.
    abundanceModel: AbundanceModel | None
    # Solver used by diffrax. The default lives in __init__ (kwargs.pop);
    # a field default here get get overrided by the explicit constructor.
    adjoint: type[diffrax.AbstractAdjoint] = eqx.field(static=True)

    def __init__(self, user_species: "Sequence[type[Fluid]] | None" = None, **kwargs):
        """
        Initialize Model instance.

        Sets up fluid species, recombination model, and spectrum solver
        based on configuration parameters.

        **kwargs : dict
            Configuration options passed as keyword arguments.
            Any unknown keys will be preserved for custom species extensibility.
        """

        # Pull adjoint out of kwargs before resolve_options — it must NOT end up
        # inside self.options (a non-JAX pytree leaf breaks filter_jit tracing).
        adjoint = kwargs.pop("adjoint", diffrax.ForwardMode)

        # Keep the user's options exactly as supplied (keys as typed, defaults
        # absent) — save_run records these so a run file replays user intent.
        self.raw_options = dict(kwargs)

        # Fill in all user defined and missing options parameters.
        options = schema.resolve_options(kwargs)
        self.options = options

        # Populate all species
        self.species_list = model_setup.populate_species(user_species, options)

        # Initialize perturbation evolver
        k_axis_perturbations, k_axis_Pk_output, k_min, k_max_cmb = (
            model_setup.get_k_axis_perturbations(options)
        )
        self.PE = perturbations.PerturbationEvolver(
            self.species_list,
            k_axis_perturbations,
            options,
            adjoint=adjoint,
        )

        # Intialize spectrum solver
        k_axis_transfer = model_setup.get_k_axis_transfer(options, k_min, k_max_cmb)
        self.SS = spectrum.SpectrumSolver(
            options["l_min"],
            options["l_max"],
            options["lensing"],
            k_axis_transfer,
            k_axis_Pk_output,
            k_pivot=options["k_pivot"],
            scale_sw=options["scale_sw"],
            scale_isw=options["scale_isw"],
            scale_dop=options["scale_dop"],
            scale_pol=options["scale_pol"],
            lna_lensing_points=options["lna_lensing_points"],
        )

        # Initialize recombination model.
        self.RecModel = hyrex.recomb_model(adjoint=adjoint)

        # Initialize BBN model
        self.PArthENoPE_CLASS_table = jnp.asarray(
            np.loadtxt(file_dir + "/sBBN_2025_CLASS.txt")
        )
        # initialize LINX
        if self.options["bbn_type"].lower() == "linx":
            self.thermo_model_DNeff = BackgroundModel(adjoint=adjoint)
            self.abundanceModel = AbundanceModel(
                NuclearRates(nuclear_net=self.options["linx_reaction_net"]),
                adjoint=adjoint,
            )
        else:
            self.thermo_model_DNeff = None
            self.abundanceModel = None

        self.adjoint = adjoint

    # Convenience front door: eager derivation + the traceable solve. Kept
    # as __call__ because "params in, spectra out" is the overwhelmingly
    # common case and the documented entry point; staged use goes through
    # add_derived_parameters + run_derived.
    def __call__(self, params: dict | None = None) -> "Output":
        """
        Run the full pipeline: derive parameters, then compute spectra.

        Includes the *eager* derivation stage (concrete parameter checks and
        the CPU-pinned BBN solves), so do not wrap this call in ``jax.jit``
        or ``jax.vmap``. Eager autodiff is fine: ``jax.grad`` / ``jax.jacfwd``
        trace through the derivation. For staged use, :meth:`run_derived` is the jit-internal stage, applied to the
        output of :meth:`add_derived_parameters`.

        Parameters:
        -----------
        params : dict, optional
            Cosmological parameters; omitted keys resolve to the schema
            defaults (no arguments runs the fiducial cosmology).

        Returns:
        --------
        Output
            Bundle of CMB power spectra (ClTT, ClTE, ClEE) and their
            multipole grid l, matter power spectrum Pk and its k-grid,
            the Background and PerturbationTable objects, and the
            full parameter dict including derived keys.
        """
        full_params = self.add_derived_parameters({} if params is None else params)
        return self.run_derived(full_params)

    def run_derived(self, params: "Params") -> "Output":
        """
        Compute CMB spectra from *already-derived* params (the output of
        :meth:`add_derived_parameters`).

        This is the traceable stage of the pipeline and the natural
        differentiation point: gradients with respect to (derived)
        parameters flow through this method. Do not wrap it in a larger
        ``jax.jit``: the recombination and BBN companions inside are
        deliberately CPU-pinned outside the main jit context.

        Parameters:
        -----------
        params : dict
            Cosmological parameters (must already have derived keys).

        Returns:
        --------
        Output
            Bundle of CMB power spectra (ClTT, ClTE, ClEE) and their
            multipole grid l, matter power spectrum Pk and its k-grid,
            the Background and PerturbationTable objects, and the
            full parameter dict including derived keys.
        """
        _check_derived(params)

        pre_BG = self.get_BG_pre_recomb(params)
        recomb_inputs = pre_BG.make_recomb_inputs(self.RecModel, params)

        # Committed (device_put) inputs pin jit's placement
        # HyRex runs fastest on CPU.
        cpu_dev = jax.devices("cpu")[0]
        recomb_inputs_cpu = jax.device_put(recomb_inputs, cpu_dev)
        params_cpu = jax.device_put(params, cpu_dev)

        recomb_output = eqx.filter_jit(self.RecModel)((recomb_inputs_cpu, params_cpu))

        try:
            recomb_output = jax.device_put(recomb_output, jax.devices("gpu")[0])
        except Exception:
            pass

        return self._run_post_recomb(params, pre_BG, recomb_output)

    @eqx.filter_jit
    def get_BG_pre_recomb(self, params: "Params") -> "BackgroundPreRecomb":
        """
        Pre-recomb stage: tabulate conformal time (the HyRex input bundle is
        produced separately by ``pre_BG.make_recomb_inputs``).
        """
        _check_derived(params)
        # let the user know the code is compiling
        print("")
        print("              /\\  ")
        print("             /  \\   ")
        print("            / /\\ \\  ")
        print("           / /__\\ \\    ___   ___  ")
        print("          / ______ \\  | _ \\ / __\\ _  _  ")
        print("         / /      \\ \\ |  _// /   | \\/ | __  ")
        print("        / /        \\ \\| _ \\\\ \\___||\\/||| -)  ")
        print("       /_/          \\_|___/ \\___/||  |||_-) is compiling...")
        print("\\_____/      ")
        print("")
        return BackgroundPreRecomb(
            params,
            self.species_list,
            adjoint=self.adjoint,
            lna_tau_points=self.options["lna_tau_points"],
        )

    @eqx.filter_jit
    def _run_post_recomb(
        self,
        params: "Params",
        pre_BG: "BackgroundPreRecomb",
        recomb_output: "RecombOutput",
    ) -> "Output":
        """
        Post-recombination stage: full Background construction (reionization,
        optical depth, decoupling), perturbation evolution, CMB spectra.
        """

        # Compute background and linear perturbations
        PT, BG = self.get_PTBG(params, pre_BG, recomb_output)

        # Compute CMB power spectra
        Cls = self.SS.get_Cl(PT, BG, params)
        l = self.SS.ells

        # Compute linear matter power spectrum
        Pk = self.SS.Pk_lin(self.SS.k_axis_Pk_output, 0.0, PT, params)
        k = self.SS.k_axis_Pk_output

        # Package
        output = Output(Cls[0], Cls[1], Cls[2], Pk, l, k, BG, PT, params)

        return output

    @eqx.filter_jit
    def get_PTBG(
        self,
        params: "Params",
        pre_BG: "BackgroundPreRecomb",
        recomb_output: "RecombOutput",
    ) -> "tuple[PerturbationTable, Background]":
        """
        Get perturbation table and full Background.

        Constructs the post-recomb Background from ``pre_BG`` + ``recomb_output``
        and runs the perturbation evolver.

        """
        BG = self.get_BG(params, pre_BG, recomb_output)
        PT = self.PE.full_evolution((BG, params))
        return PT, BG

    def get_BG(
        self,
        params: "Params",
        pre_BG: "BackgroundPreRecomb",
        recomb_output: "RecombOutput",
    ) -> Background:
        """
        Construct the full ``Background`` from pre-recomb + HyRex output.

        The reionization model (tau-input vs z-input) follows the static
        ``input_tau_reion`` option, so the selection is plain Python at
        trace time -- only the chosen branch is ever traced.
        """
        reion_model = (
            ReionizationModelFromTau
            if self.options["input_tau_reion"]
            else ReionizationModelFromZ
        )
        return Background(
            pre_BG,
            recomb_output,
            params,
            reion_model,
            transfer_start_threshold=self.options["transfer_start_threshold"],
        )

    def add_derived_parameters(self, param_in: Mapping[str, ArrayLike]) -> "Params":
        # Resolve raw params against PARAM_SCHEMA (defaults, aliases, unknown-key
        # handling), then run the imperative cosmology derivation.
        params = schema.resolve_params(param_in)
        return derived.derive_parameters(
            params,
            self.options,
            self.species_list,
            parthenope_table=self.PArthENoPE_CLASS_table,
            linx_thermo=self.thermo_model_DNeff,
            linx_abundance=self.abundanceModel,
        )


class Output(eqx.Module):
    """
    Object containing final and intermediate results from one cosmological simulation.

    Attributes:
    -----------
    ClTT : Array
        Temperature-temperature power spectrum
    ClTE : Array
        Temperature-polarization power spectrum
    ClEE : Array
        Polarization-polarization power spectrum
    Pk : Array
        Matter power spectrum
    l : Array
        Multipoles l at which ClTT/ClTE/ClEE are output
    k : Array
        Wavenumbers k at with Pk is output
    BG  : background.Background
        Background object containing functions like Hubble, recombination history, etc
    PT : perturbations.PerturbationTable
        Perturbation table including perturbations for all fluids
    params : dict
        Complete parameter dictionary including derived parameters
    """

    # Power spectra
    ClTT: Array
    ClTE: Array
    ClEE: Array
    Pk: Array

    l: Array
    k: Array
    BG: background.Background
    PT: perturbations.PerturbationTable
    params: "Params"
