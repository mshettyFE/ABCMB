"""
Model construction helpers.

Builds the fluid species list (``populate_species``) and the perturbation /
transfer k-axis grids from a resolved ``options`` dict. The declarative input
schema (options/params resolution, aliases, provenance) lives in
:mod:`abcmb.schema`.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from . import species

if TYPE_CHECKING:
    from ._schema_types import Options


def populate_species(
    user_species: "Sequence[type[species.Fluid]] | None", options: "Options"
):
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
    # thermodynamics, recombination) references these two fluids by name.
    for required in ("Baryon", "Photon"):
        if required not in species_dict:
            raise ValueError(
                f"no fluid named '{required}': the baryon-photon coupling and "
                "recombination require fluids named 'Baryon' and 'Photon' "
                "(use use_LCDM_species=True, or include them in user_species)."
            )

    return species_list, species_dict


def get_k_axis_perturbations(options: "Options"):
    """
    Build the perturbation k-grid and the P(k) output k-grid.

    Returns ``(k_axis, k_axis_Pk_output, k_min, k_max_cmb)``.
    """
    ks = np.zeros(2000)

    H0_fid = options["H0_fid"]
    tau0_fid = options["tau0_fid"]
    rs_rec_fid = options["rs_rec_fid"]
    k_rec_fid = 2.0 * jnp.pi / rs_rec_fid

    k_min = options["k_min_tau0"] / tau0_fid
    k_max = options["k_max_tau0_over_l_max"] / tau0_fid * options["l_max"]

    k = k_min
    ks[0] = k
    i = 0
    while k < k_max:
        step = (
            options["k_step_super"]
            + 0.5
            * (
                jnp.tanh((k - k_rec_fid) / k_rec_fid / options["k_step_transition"])
                + 1.0
            )
            * (options["k_step_sub"] - options["k_step_super"])
        ) * k_rec_fid

        scale2 = H0_fid**2

        step *= (k**2 / scale2 + 1.0) / (
            k**2 / scale2 + 1.0 / options["k_step_super_reduction"]
        )

        k += step
        i += 1
        ks[i] = k

    # End of the CMB range; the loops below only extend the grid for lensing /
    # a user-specified k_max, which the transfer grid must not follow.
    k_max_cmb = k

    # If lensing is needed, we need to extend max k by some amount to accurately compute high-l lensing.
    if options["lensing"]:
        k_max = k + 0.3

        while k < k_max:
            step = 0.005

            k += step
            i += 1
            ks[i] = k

    # If the user specified a k_max above the current, we should add these as well.
    if k < options["k_max"]:
        k_max = options["k_max"]

        while k < k_max:
            step = 0.005

            k += step
            i += 1
            ks[i] = k

    ks = ks[np.where(ks > 0)]
    k_axis_Pk_output = ks[np.where(ks <= options["k_max"])]

    return jnp.array(ks), jnp.array(k_axis_Pk_output), k_min, k_max_cmb


def get_k_axis_transfer(options: "Options", k_min, k_max_cmb):
    """
    Build the transfer-integration k-grid over the CMB range
    ``[k_min, k_max_cmb]`` computed by :func:`get_k_axis_perturbations`.
    """
    ks = np.zeros(8000)

    k_period = 2 * jnp.pi / (options["tau0_fid"] - options["tau_rec_fid"])

    k = k_min
    ks[0] = k
    i = 0
    while k < k_max_cmb:
        k = k + k_period * options["k_transfer_linstep"] * k / (
            k + options["k_transfer_linstep"] / options["k_transfer_logstep"]
        )
        i += 1
        ks[i] = k

    ks = jnp.array(ks[np.where(ks > 0)])
    return ks
