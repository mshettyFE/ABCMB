"""
Model construction helpers.

Builds the fluid species list (``populate_species``) and the perturbation /
transfer k-axis grids from a resolved ``options`` dict. The declarative input
schema (options/params resolution, aliases, provenance) lives in
:mod:`abcmb.schema`.
"""

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from . import species

if TYPE_CHECKING:
    from ._schema_types import Options


def populate_species(
    user_species: "Sequence[type[species.Fluid]] | None", options: "Options"
) -> "tuple[tuple[species.Fluid, ...], dict[str, int]]":
    """
    Instantiate the model's fluid stack: the baseline LCDM species (when
    ``options["use_LCDM_species"]``) followed by ``user_species``, in order.

    Entries must be Fluid *classes*, not instances -- ABCMB instantiates each
    fluid itself so it can assign ``first_idx`` (the fluid's offset into the
    diffrax state vector) cumulatively from the fluids registered before it.

    Returns ``(species_list, species_dict)``, where ``species_dict`` maps
    each fluid's unique ``name`` to its index in ``species_list``. Raises at
    construction on non-Fluid entries, duplicate names, and missing or
    impostor ``Baryon``/``Photon`` roles (the baryon-photon coupling requires
    genuine subclasses; see docs/promoting_a_fluid.rst).
    """
    species_list = ()
    species_dict = {}

    lcdm_species = (
        species.DarkEnergy,
        species.ColdDarkMatter,
        species.Baryon,
        species.Photon,
        species.MasslessNeutrino,
    )

    # Baseline LCDM species (if requested), then user species
    selected = tuple(lcdm_species) if options["use_LCDM_species"] else ()
    if user_species is not None:
        selected = selected + tuple(user_species)

    # Slot 0 of the diffrax state vector is the metric perturbation eta;
    # fluid equations start at 1.
    diffrax_vector_idx = 1
    for i, s in enumerate(selected):
        # Classes, not instances: ABCMB instantiates each fluid itself so it
        # can assign first_idx consistently with the other fluids present.
        if not isinstance(s, type):
            raise TypeError(
                f"user_species entries must be Fluid classes, not instances "
                f"(got a {type(s).__name__} instance); ABCMB instantiates "
                "each fluid itself so it can assign first_idx."
            )
        if not issubclass(s, species.Fluid):
            raise TypeError(
                f"user_species entries must be species.Fluid subclasses; got "
                f"{s.__name__}, which does not inherit from Fluid "
                "(see docs/promoting_a_fluid.rst)."
            )
        instance = s(diffrax_vector_idx, options)
        if instance.name in species_dict:
            raise ValueError(
                f"duplicate species name '{instance.name}': every fluid needs a "
                "unique name -- coupling lookups (species_dict) and the "
                "perturbation output tables are keyed by it."
            )
        species_list = species_list + (instance,)
        species_dict[instance.name] = i
        diffrax_vector_idx += instance.num_equations

    # Required roles: the baryon-photon coupling (perturbations, background
    # thermodynamics, recombination) references these two fluids by name
    for required, role_cls in (
        ("Baryon", species.Baryon),
        ("Photon", species.Photon),
    ):
        if required not in species_dict:
            raise ValueError(
                f"no fluid named '{required}': the baryon-photon coupling and "
                "recombination require fluids named 'Baryon' and 'Photon' "
                "(use use_LCDM_species=True, or include them in user_species)."
            )
        found = species_list[species_dict[required]]
        if not isinstance(found, role_cls):
            raise TypeError(
                f"the fluid named '{required}' is a {type(found).__name__}, "
                f"not a species.{required} subclass: the baryon-photon "
                f"coupling requires the real {required} interface. To "
                f"customize it, subclass species.{required} "
                "(see docs/promoting_a_fluid.rst)."
            )

    return species_list, species_dict


def _check_k_step(
    step: Float[Array, ""] | float, k: Float[Array, ""] | float, knobs: str
) -> None:
    # CLASS guards its k-list loops the same way (perturbations.c,
    # smallest_allowed_variation): a step that underflows relative to k
    # would leave `k + step == k` and hang the while loop forever.
    if step <= abs(k) * 1e-14:
        raise ValueError(
            f"k-grid step underflowed (step={float(step):.3e} at "
            f"k={float(k):.3e}); the loop would never terminate. "
            f"Check the options {knobs}."
        )


def get_k_axis_perturbations(
    options: "Options",
) -> tuple[
    Float[Array, " n_k"],
    Float[Array, " n_k_pk"],
    float,
    Float[Array, ""] | float,
]:
    """
    Build the perturbation k-grid and the P(k) output k-grid.
    CLASS is a bit sparse on the details, but the idea is to dynamically change the k-grid binning
    to interpolate between the super horizon and sub horizon regimes.

    The grid is anchored to a *fiducial* cosmology (the ``*_fid`` options,
    tuned for h=0.6762) rather than the live parameters, so it can stay
    static under jit. accuracy vs CLASS
    at h=0.60 and h=0.75 on an h=0.6762-tuned grid stays at the ~0.2%
    baseline (TT/EE/Pk maxima moved by < 0.01 percentage points), so live-h
    divergence is free across the plausible Hubble range. Caveat: that test
    held the physical densities fixed, so rs_rec barely moved -- for
    cosmologies that shift the acoustic scale itself (large omega_m changes,
    early-universe physics), retune the ``*_fid`` options.

    Returns ``(k_axis, k_axis_Pk_output, k_min, k_max_cmb)``.
    """
    H0_fid = options["H0_fid"]
    tau0_fid = options["tau0_fid"]
    rs_rec_fid = options["rs_rec_fid"]
    k_rec_fid = 2.0 * jnp.pi / rs_rec_fid
    k_min = options["k_min_tau0"] / tau0_fid
    k_max = options["k_max_tau0_over_l_max"] / tau0_fid * options["l_max"]

    k = k_min
    ks = [k]
    while k < k_max:
        step = (
            # Baseline stepsize
            options["k_step_super"]
            # Interpolation between super and sub horizon scales
            + 0.5
            * (
                jnp.tanh((k - k_rec_fid) / k_rec_fid / options["k_step_transition"])
                + 1.0
            )
            * (options["k_step_sub"] - options["k_step_super"])
        ) * k_rec_fid

        # below the super-horizon plateau, increase the resolution.
        # the lowest order multipoles live here, and require finer binning
        scale2 = H0_fid**2

        step *= (k**2 / scale2 + 1.0) / (
            k**2 / scale2 + 1.0 / options["k_step_super_reduction"]
        )

        _check_k_step(step, k, "k_step_super/k_step_sub/k_step_super_reduction")
        k += step
        ks.append(k)

    # End of the CMB range; the loops below only extend the grid for lensing /
    # a user-specified k_max, which the transfer grid must not follow.
    k_max_cmb = k

    step_ext = options["k_step_extension"]

    # If lensing is needed, we need to extend max k by some amount to accurately compute high-l lensing.
    if options["lensing"]:
        k_max = k + options["k_lensing_extension"]

        while k < k_max:
            _check_k_step(step_ext, k, "k_step_extension")
            k += step_ext
            ks.append(k)

    # If the user specified a k_max above the current, we should add these as well.
    if k < options["k_max"]:
        k_max = options["k_max"]

        while k < k_max:
            _check_k_step(step_ext, k, "k_step_extension")
            k += step_ext
            ks.append(k)

    ks = np.array(ks)

    # every grid point is a full Boltzmann solve, so a grid this large is an unusually
    # expensive request, not an error.
    if ks.size > 2000:
        warnings.warn(
            f"perturbation k-grid has {ks.size} points (each one is a full "
            f"Boltzmann solve): k_max={options['k_max']} with the linear "
            f"extension step k_step_extension={options['k_step_extension']} "
            "is expensive at LSS scales, and a CLASS-style log-per-decade "
            "extension is not implemented (see get_k_axis_perturbations).",
            stacklevel=2,
        )

    k_axis_Pk_output = ks[np.where(ks <= options["k_max"])]

    return jnp.array(ks), jnp.array(k_axis_Pk_output), k_min, k_max_cmb


def get_k_axis_transfer(
    options: "Options",
    k_min: Float[Array, ""] | float,
    k_max_cmb: Float[Array, ""] | float,
) -> Float[Array, " n_k_transfer"]:
    """
    Build the transfer-integration k-grid over the CMB range
    ``[k_min, k_max_cmb]`` computed by :func:`get_k_axis_perturbations`.
    """
    k_period = 2 * jnp.pi / (options["tau0_fid"] - options["tau_rec_fid"])

    k = k_min
    ks = [k]
    while k < k_max_cmb:
        step = (
            k_period
            * options["k_transfer_linstep"]
            * k
            / (k + options["k_transfer_linstep"] / options["k_transfer_logstep"])
        )
        _check_k_step(step, k, "k_transfer_linstep/k_transfer_logstep")
        k = k + step
        ks.append(k)

    return jnp.array(ks)
