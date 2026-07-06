import os

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import config, lax
from jaxtyping import Array

from . import background, model_setup, perturbations, schema, spectrum
from .background import (
    Background,
    BackgroundPreRecomb,
    ReionizationModelFromTau,
    ReionizationModelFromZ,
)
from .hyrex import hyrex
from .linx.abundances import AbundanceModel
from .linx.background import BackgroundModel
from .linx.nuclear import NuclearRates
from .species import Fluid

file_dir = os.path.dirname(__file__)

config.update("jax_enable_x64", True)


class Model(eqx.Module):
    """
    Model configuration and computation manager.

    Creates instances of fluid species based on user input and organizes
    them for computation. Manages the full pipeline from background
    evolution through CMB power spectrum computation.

    Attributes:
    -----------
    PE : perturbations.PerturbationEvolver
        ABCMB perturbations module
    SS : spectrum.SpectrumSolver
        ABCMB spectrum module
    RecModel : hyrex.recomb_model
        HyRex recombination module
    options : dict
        A dictionary of run options (expected to be static)
    species_list : tuple
        A list of all fluids in the user cosmology
    species_dict : dict
        A dictionary containing the names of all fluids, in the same order as
        they appear in species_list.
    PArthENoPE_CLASS_table  : Array
        A 2D table for interpolation of the helium-4 mass fraction based
        on the user's input baryon density and Neff
    thermo_model_DNeff : linx.BackgroundModel
        A LINX background model for BBN thermodynamics
    abundanceModel : linx.AbundanceModel
        A LINX abundance model used for computing the helium-4 mass fraction
        given the user's input baryon density, Neff, neutron lifetime, and
        nuclear reaction rates.
    adjoint : diffrax.adjoint
        Adjoint mode for diffrax solves.  Default is ForwardMode.

    Methods:
    --------
    __call__ : Compute CMB angular power spectra
    get_PTBG : Get perturbation table and background cosmology
    get_BG : Get background cosmology
    add_derived_parameters : Compute derived parameters
    """

    PE: perturbations.PerturbationEvolver
    SS: spectrum.SpectrumSolver
    RecModel: hyrex.recomb_model
    options: dict
    options_provenance: dict

    species_list: tuple[Fluid, ...] = ()
    species_dict: dict

    PArthENoPE_CLASS_table: Array
    thermo_model_DNeff: BackgroundModel
    abundanceModel: AbundanceModel

    adjoint: "diffrax.adjoint" = eqx.field(static=True)

    ### ADDING SPECIES: add has_ parameter and add condition to append to tuple.
    # In the init, all species that are present within the model should be set to True.
    # All couplings present between species should be set to true.
    def __init__(self, user_species=None, **kwargs):
        """
        Initialize Model instance.

        Sets up fluid species, recombination model, and spectrum solver
        based on configuration parameters.

        Parameters:
        -----------
        user_species : tuple
            A tuple of user-defined fluids to be included in the cosmology
        **kwargs : dict
            Configuration options passed as keyword arguments.
            Any unknown keys will be preserved for custom species extensibility.
        """

        # Pull adjoint out of kwargs before resolve_options — it must NOT end up
        # inside self.options (a non-JAX pytree leaf breaks lax.cond / filter_jit
        # tracing).
        adjoint = kwargs.pop("adjoint", diffrax.ForwardMode)

        # Fill in all user defined and missing options parameters. resolve_options
        # also returns per-key provenance (default / user / alias / extra),
        # stored for run reproducibility and notebook introspection.
        options, self.options_provenance = schema.resolve_options(kwargs)
        self.options = options

        # Populate all species
        self.species_list, self.species_dict = model_setup.populate_species(
            user_species,
            options,
        )

        # Initialize perturbation evolver
        k_axis_perturbations, k_axis_Pk_output = model_setup.get_k_axis_perturbations(
            options
        )
        self.PE = perturbations.PerturbationEvolver(
            self.species_list,
            self.species_dict,
            k_axis_perturbations,
            options,
            adjoint=adjoint,
        )

        # Intialize spectrum solver
        k_axis_transfer = model_setup.get_k_axis_transfer(options)
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
        )

        # Initialize recombination model.
        self.RecModel = hyrex.recomb_model(adjoint=adjoint)  # DO NOT CHANGE z1 FROM 0

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

    # need this outside of the main jit context
    # since we want LINX/HyRex to run on CPU
    def __call__(self, params: dict = {}):
        """
        Runs the full pipeline from background evolution through
        perturbation integration to CMB power spectrum computation.

        Parameters:
        -----------
        params : dict
            Cosmological parameters

        Returns:
        --------
        Output
            Bundle of CMB power spectra (ClTT, ClTE, ClEE) and their
            multipole grid l, matter power spectrum Pk and its k-grid,
            the Background and PerturbationTable objects, and the
            full parameter dict including derived keys.
        """
        full_params = self.add_derived_parameters(params)
        return self.run_cosmology_abbr(full_params)

    def run_cosmology_abbr(self, params: dict):
        """
        Compute CMB angular power spectra for given parameters.

        Runs the full pipeline from background evolution through
        perturbation integration to CMB power spectrum computation.

        Parameters:
        -----------
        params : dict
            Cosmological parameters (must already have derived keys).

        Returns:
        --------
        Output
            CMB power spectra and friends.
        """

        # Cast int/bool params to float64 before entering any
        # ``eqx.filter_jit`` for custom_vjp/AD safety in
        # checkpointed_while_loop
        def _to_float(v):
            arr = jnp.asarray(v)
            if arr.dtype.kind in "iub":
                return arr.astype(jnp.float64)
            return arr

        params = jax.tree_util.tree_map(_to_float, params)

        pre_BG = self.get_BG_pre_recomb(params)

        cpu_dev = jax.devices("cpu")[0]
        recomb_inputs_cpu = jax.device_put(pre_BG.recomb_inputs, cpu_dev)
        params_cpu = jax.device_put(params, cpu_dev)

        recomb_output = eqx.filter_jit(self.RecModel, backend="cpu")(
            (recomb_inputs_cpu, params_cpu)
        )

        try:
            recomb_output = jax.device_put(recomb_output, jax.devices("gpu")[0])
        except Exception:
            pass

        # recomb_output contains array_with_padding objects whose
        # padding_size and lastnum int arrays.  The
        # checkpointed_while_loop's filter_custom_vjp inside
        # _run_post_recomb's diffrax solves trips an internal
        # _get_value_assert_unperturbed on int leaves under outer
        # AD; convert to float to avoid.
        recomb_output = jax.tree_util.tree_map(_to_float, recomb_output)

        return self._run_post_recomb(params, pre_BG, recomb_output)

    @eqx.filter_jit
    def get_BG_pre_recomb(self, params: dict):
        """
        Pre-recomb stage: tabulate conformal time and bundle H, T, nH for recombination.

        Parameters:
        -----------
        params : dict
            Cosmological parameters

        Returns:
        --------
        BackgroundPreRecomb
        """
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
            params, self.species_list, self.RecModel, adjoint=self.adjoint
        )

    @eqx.filter_jit
    def _run_post_recomb(
        self, params: dict, pre_BG: "BackgroundPreRecomb", recomb_output
    ):
        """
        Post-recombination stage: full Background construction (reionization,
        optical depth, decoupling), perturbation evolution, CMB spectra.

        Parameters:
        -----------
        params : dict
            Cosmological parameters
        pre_BG : BackgroundPreRecomb
            Output of :meth:`get_BG_pre_recomb`.
        recomb_output : tuple
            HyRex output ``(xe, lna_xe, Tm, lna_Tm)``.

        Returns:
        --------
        Output
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
    def get_PTBG(self, params: dict, pre_BG: "BackgroundPreRecomb", recomb_output):
        """
        Get perturbation table and full Background.

        Constructs the post-recomb Background from ``pre_BG`` + ``recomb_output``
        and runs the perturbation evolver.

        Parameters:
        -----------
        params : dict
            Cosmological parameters
        pre_BG : BackgroundPreRecomb
            Pre-recombination stage object.
        recomb_output : tuple
            HyRex output ``(xe, lna_xe, Tm, lna_Tm)``.

        Returns:
        --------
        tuple
            (PerturbationTable, Background)
        """
        BG = self.get_BG(params, pre_BG, recomb_output)
        PT = self.PE.full_evolution((BG, params))
        return PT, BG

    def get_BG(self, params: dict, pre_BG: "BackgroundPreRecomb", recomb_output):
        """
        Construct the full ``Background`` from pre-recomb + HyRex output.

        Selects the reionization model (z-input vs tau-input) via ``lax.cond``.
        NOT directly ``@eqx.filter_jit``-decorated; called from inside
        ``_run_post_recomb`` (which is jit-wrapped).

        Parameters:
        -----------
        params : dict
            Cosmological parameters
        pre_BG : BackgroundPreRecomb
            Pre-recombination stage object.
        recomb_output : tuple
            HyRex output ``(xe, lna_xe, Tm, lna_Tm)``.

        Returns:
        --------
        background.Background
        """

        def get_BG_z_reion(args):
            params, pre_BG, recomb_output = args
            return Background(pre_BG, recomb_output, params, ReionizationModelFromZ)

        def get_BG_tau_reion(args):
            params, pre_BG, recomb_output = args
            return Background(pre_BG, recomb_output, params, ReionizationModelFromTau)

        BG = lax.cond(
            self.options["input_tau_reion"],
            get_BG_tau_reion,
            get_BG_z_reion,
            (params, pre_BG, recomb_output),
        )

        return BG

    def param_provenance(self, param_in):
        """
        Per-parameter provenance for a raw input dict.

        Maps each parameter to a :class:`schema.Provenance` (value / source /
        origin). Derived quantities (H0, omega_m, ...) are not included — they are
        computed, not input.
        """
        return schema.resolve_params(param_in)[1]

    def add_derived_parameters(self, param_in: dict) -> dict:
        # Resolve raw params against PARAM_SCHEMA (defaults, aliases, unknown-key
        # handling), then run the imperative cosmology derivation.
        params, _ = schema.resolve_params(param_in)
        return schema.derive_parameters(
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
    ClTT : jnp.array
        Temperature-temperature power spectrum
    ClTE : jnp.array
        Temperature-polarization power spectrum
    ClEE : jnp.array
        Polarization-polarization power spectrum
    Pk : jnp.array
        Matter power spectrum
    l : jnp.array
        Multipoles l at which ClTT/ClTE/ClEE are output
    k : jnp.array
        Wavenumbers k at with Pk is output
    BG  : background.Background
        Background object containing functions like Hubble, recombination history, etc
    PT : perturbations.PerturbationTable
        Perturbation table including perturbations for all fluids
    params : dict
        Complete parameter dictionary including derived parameters
    """

    # Power spectra
    ClTT: jnp.array
    ClTE: jnp.array
    ClEE: jnp.array
    Pk: jnp.array

    l: jnp.array
    k: jnp.array
    BG: background.Background
    PT: perturbations.PerturbationTable
    params: dict
