"""
post-hoc validation/updating of parameter struct
"""

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.interpolate import RegularGridInterpolator
from jax.typing import ArrayLike
from jaxtyping import Array, Float

from .. import constants as cnst

if TYPE_CHECKING:
    from ..linx.abundances import AbundanceModel
    from ..linx.background import BackgroundModel
    from ..species import Fluid
    from ._schema_types import Options, Params
from ..linx import const as linxconst
from ..linx import thermo as linxThermo

# The PArthENoPE/CLASS sBBN table: rows of (omega_b, Delta_Neff, YHe).
_SBBNTable = Float[Array, "n_table 3"]


def _check_neutrino_input(params: "Params") -> None:
    """
    Enforce the ``Neff`` / ``N_nu_massless`` one-of (they are treated 1-to-1; see
    paper). A param-only invariant — no Model needed — so ``derive_parameters``
    calls it up front. Raises ``ValueError`` if both are supplied.
    """
    if params.get("N_nu_massless") is not None and params.get("Neff") is not None:
        raise ValueError(
            "Provide only one of Neff / N_nu_massless (they are treated 1-to-1)."
        )


def _require(ok: ArrayLike, message: str | Callable[[], str]) -> None:
    """Raise ``ValueError`` unless the (concrete, eager) condition holds."""
    if not bool(ok):
        raise ValueError(message() if callable(message) else message)


def _bbn_type(options: "Options") -> str:
    """
    Normalized ``bbn_type`` ('', 'table', or 'linx'); raises on anything else
    (the schema's choices check only warns -- an uninterpretable backend has no
    sensible fallback, so refuse to guess).
    """
    value = options["bbn_type"].lower()
    if value not in ("", "table", "linx"):
        raise ValueError(
            f"bbn_type={options['bbn_type']!r} is not one of '', 'table', 'linx'."
        )
    return value


def _resolve_neutrino_input(params: "Params", options: "Options") -> tuple[bool, bool]:
    """
    Check neutrino-input compatibility and apply the massless-neutrino fallback.

    Enforces the Neff / N_nu_massless one-of, rejects supplying neutrino inputs
    under LINX (which computes them), defaults an unspecified massless-neutrino
    count to ``3 - N_nu_massive``, and sets the ``T_nu_massless`` default. Returns
    ``(input_N, input_Neff)``.
    """
    _check_neutrino_input(params)  # Neff / N_nu_massless one-of (1-to-1)

    input_N = params.get("N_nu_massless") is not None
    input_Neff = params.get("Neff") is not None
    input_T_nu_massless = params.get("T_nu_massless") is not None

    bbn_type = _bbn_type(options)
    # LINX computes Neff/T_nu_massless itself, so the user must not also supply them.
    if (input_N or input_Neff or input_T_nu_massless) and bbn_type == "linx":
        raise ValueError(
            "You have specified a value for N_nu_massless and/or Neff and/or "
            "T_nu_massless, but LINX instead expects 'Delta_Neff_init', which it "
            "uses to compute Neff. See the LINX docs or "
            "https://arxiv.org/abs/2408.14538."
        )

    if not input_N and not input_Neff and bbn_type != "linx":
        params["N_nu_massless"] = 3 - params["N_nu_massive"]
        input_N = True

    # T_nu_massless (ratio to TCMB); safe to set now that inputs are validated.
    params["T_nu_massless"] = jnp.array(params.get("T_nu_massless", 0.71636856))
    return input_N, input_Neff


def _neff_from_fluid_content(params: "Params", species: "tuple[Fluid, ...]") -> None:
    """
    Case 1: the user gave the true massless-neutrino count -> infer Neff from the
    early-time fluid content, correcting for the late-time massive-neutrino
    temperature (the missing early relativistic energy is added to the massless
    sector). See the paper.
    """
    lna_early = -23.0
    rho_g = 0.0
    rho_nu = 0.0
    rho_extra = 0.0
    for s in species:
        rho = s.rho(lna_early, params)
        if s.name == "Photon":
            rho_g += rho
        elif s.is_neutrino:
            rho_nu += rho
        else:
            rho_extra += rho

    Neff_raw = (
        (rho_nu + rho_extra) / rho_g * (8.0 / 7.0) * (11.0 / 4.0) ** (4.0 / 3.0)
    )  # Uncorrected Neff using T_nu_massive today
    rho_nu_early = (
        7
        / 8
        * (params["N_nu_massless"] + params["N_nu_massive"])
        * params["T_nu_massless"] ** 4
        * rho_g
    )  # Correct using massless neutrino temp.
    params["Neff"] = (
        (rho_nu_early + rho_extra) / rho_g * (8.0 / 7.0) * (11.0 / 4.0) ** (4.0 / 3.0)
    )
    params["N_nu_massless"] = (
        params["N_nu_massless"] + params["Neff"] - Neff_raw
    )  # Add difference to massless sector.


def _helium_from_table(params: "Params", parthenope_table: _SBBNTable) -> None:
    """Interpolate ``YHe`` from the PArthENoPE/CLASS sBBN table (Neff must be set)."""
    bbn = parthenope_table
    omegab_all = bbn[:, 0]
    DNeff_all = bbn[:, 1]
    YHe_all = bbn[:, 2]

    # Hardcoded to be jit-safe; these tables don't update frequently.
    n2 = 13
    n1 = 701
    omegab = omegab_all[:n1]
    DNeff = DNeff_all[::n1]
    YHe_grid = YHe_all.reshape(n2, n1)

    # Out-of-table policy: warn, then extrapolate. Extrapolated YHe is
    # invented nuclear physics, but a hard error would break wide-prior scans;
    dneff = params["Neff"] - 3.046
    in_table = (
        omegab[0] <= params["omega_b"] <= omegab[-1] and DNeff[0] <= dneff <= DNeff[-1]
    )
    if not in_table:
        warnings.warn(
            f"(omega_b={float(params['omega_b']):.4g}, "
            f"Neff-3.046={float(dneff):.3g}) is outside the sBBN table "
            f"(omega_b in [{float(omegab[0]):.4g}, {float(omegab[-1]):.4g}], "
            f"Delta Neff in [{float(DNeff[0]):.3g}, {float(DNeff[-1]):.3g}]); "
            "YHe will be linearly extrapolated. Supply YHe directly "
            "(bbn_type='') or use bbn_type='linx' for out-of-table models.",
            stacklevel=2,
        )

    # Query at the user's omega_b and (Neff - 3.046) (3.046 assumed by the
    # table). fill_value=None linearly extrapolates outside the tabulated
    # rectangle (bitwise-matching the previous hand-rolled interpolator).
    interp = RegularGridInterpolator(
        (DNeff, omegab), YHe_grid, method="linear", bounds_error=False, fill_value=None
    )
    params["YHe"] = interp(
        jnp.stack([params["Neff"] - 3.046, params["omega_b"]])[None, :]
    )[0]


def _helium_from_linx(
    params: "Params",
    linx_thermo: "BackgroundModel",
    linx_abundance: "AbundanceModel",
) -> None:
    """
    Run LINX BBN from ``Delta_Neff_init`` (+ optional ``tau_n_fac`` /
    ``nuclear_rates_q``): sets ``Neff``, ``T_nu_massless``, and ``YHe`` in place.
    """
    params["Delta_Neff_init"] = jnp.array(params.get("Delta_Neff_init", 0.0))
    # Committing the input to CPU pins the jit placement
    dneff_cpu = jax.device_put(params["Delta_Neff_init"], jax.devices("cpu")[0])
    (
        t_vec_ref,
        a_vec_ref,
        rho_g_vec,
        rho_nu_vec,
        rho_NP_vec,
        P_NP_vec,
        Neff_vec,
    ) = eqx.filter_jit(linx_thermo)(dneff_cpu)

    # convert user input omega_b to eta_fac LINX expects
    eta_fac = params["omega_b"] * linxconst.Omegabh2_to_eta0 / linxconst.eta0

    abundances = eqx.filter_jit(linx_abundance)(
        rho_g_vec,
        rho_nu_vec,
        rho_NP_vec,
        P_NP_vec,
        t_vec=t_vec_ref,
        a_vec=a_vec_ref,
        eta_fac=eta_fac,
        tau_n_fac=jnp.asarray(params.get("tau_n_fac", 1.0)),
        nuclear_rates_q=jnp.asarray(
            params.get(
                "nuclear_rates_q",
                jnp.zeros(len(linx_abundance.nuclear_net.reactions)),
            )
        ),
    )

    try:
        params["T_nu_massless"] = jax.device_put(
            linxThermo.T_nu(rho_nu_vec[-1]) / linxThermo.T_g(rho_g_vec[-1]),
            device=jax.devices("gpu")[0],
        )
        params["Neff"] = jax.device_put(Neff_vec[-1], device=jax.devices("gpu")[0])
        YHe_BBN = jax.device_put(4 * abundances[5], device=jax.devices("gpu")[0])
    except Exception:  # no GPU
        params["T_nu_massless"] = linxThermo.T_nu(rho_nu_vec[-1]) / linxThermo.T_g(
            rho_g_vec[-1]
        )
        params["Neff"] = Neff_vec[-1]
        YHe_BBN = 4 * abundances[5]

    # CMB uses the real mass fraction
    params["YHe"] = 1.0 / (4 * cnst.mH / cnst.mHe * (1 / YHe_BBN - 1) + 1)


def _compute_helium_fraction(
    params: "Params",
    options: "Options",
    parthenope_table: _SBBNTable,
    linx_thermo: "BackgroundModel | None",
    linx_abundance: "AbundanceModel | None",
) -> bool:
    """
    Set ``params["YHe"]`` per the ``bbn_type`` backend (sBBN table, LINX, or leave
    the schema-resolved default for ""). Returns ``True`` when LINX has set
    ``Neff`` (so the caller must re-derive ``N_nu_massless``).
    """
    bbn_type = _bbn_type(options)
    if bbn_type == "table":
        # Neff must already be set (Case 1); used to interpolate YHe.
        _helium_from_table(params, parthenope_table)
        return False
    if bbn_type == "linx":
        # Reached only when no neutrino input was given, so LINX may set Neff.
        _helium_from_linx(params, linx_thermo, linx_abundance)
        return True
    # bbn_type == "": YHe is already schema-resolved (user or 0.245 default).
    return False


def _n_massless_from_neff(params: "Params", species: "tuple[Fluid, ...]") -> None:
    """
    Case 2: the user (or LINX) gave the total Neff -> subtract every
    non-massless-neutrino relativistic energy density and assign the remainder to
    massless neutrinos (adding extra neutrinos at the same temperature). Raises if
    that remainder is negative (not enough energy density).
    """
    lna_early = -23.0
    rho_g = 0.0
    rho_extra = 0.0
    for s in species:
        if s.name == "Photon":
            rho_g += s.rho(lna_early, params)
        elif s.name != "MasslessNeutrino":
            rho_extra += s.rho(lna_early, params)
    rho1nu = 7 / 8 * (4 / 11) ** (4 / 3) * rho_g

    params["N_nu_massless"] = (params["Neff"] - rho_extra / rho1nu) * (
        (4 / 11) ** (1 / 3) / params["T_nu_massless"]
    ) ** 4
    _require(
        params["N_nu_massless"] >= 0,
        lambda: (
            f"Neff={float(params['Neff']):.4g} is too small for the other "
            "relativistic species: it implies a negative massless-neutrino count "
            f"(N_nu_massless={float(params['N_nu_massless']):.4g})."
        ),
    )


def _derive_densities(params: "Params", species: "tuple[Fluid, ...]") -> None:
    """
    Derive the background densities from the fluid content: ``omega_m`` (+ ``R_b``),
    ``omega_r`` (+ ``R_nu``, at early times), the adiabatic-IC parameter ``om``, and
    ``omega_Lambda = h^2 - omega_r - omega_m``. Raises if that would be negative
    (the matter + radiation budget exceeds ``h^2``).
    """
    # matter density today
    rho_m = 0.0
    for s in species:
        if s.is_matter:
            rho_m += s.rho(0.0, params)
    params["omega_m"] = rho_m / (3 * cnst.H0_over_h**2 / 8 / jnp.pi / cnst.G)
    _require(
        params["omega_m"] > 0,
        "omega_m is not positive (no matter content?); R_b/om would divide by "
        "zero -- check omega_cdm / omega_b or the matter species.",
    )
    params["R_b"] = params["omega_b"] / params["omega_m"]  # baryon fraction

    # radiation density inferred from the very-early-time fluid energy
    a_early = jnp.exp(-23.0)
    rho_r = 0.0
    rho_nu = 0.0
    for s in species:
        rho_r += s.rho(jnp.log(a_early), params)
        if s.is_neutrino:
            rho_nu += s.rho(jnp.log(a_early), params)
    params["omega_r"] = (
        rho_r * a_early**4 / (3 * cnst.H0_over_h**2 / 8 / jnp.pi / cnst.G)
    )
    _require(
        params["omega_r"] > 0,
        "omega_r is not positive (no radiation content?); R_nu/om would divide by "
        "zero -- check TCMB0 or the photon/neutrino species.",
    )
    # Neutrino fraction of radiation (adiabatic ICs); asarray keeps the stored
    # value an Array even in the all-float edge case (identity on tracers).
    params["R_nu"] = jnp.asarray(rho_nu / rho_r)

    # Omega_m / sqrt(Omega_r) * H0, in 1/Mpc (adiabatic IC parameter)
    params["om"] = (
        params["omega_m"]
        / jnp.sqrt(params["omega_r"])
        * cnst.H0_over_h
        / cnst.c_Mpc_over_s
    )

    params["omega_Lambda"] = params["h"] ** 2 - params["omega_r"] - params["omega_m"]
    _require(
        params["omega_Lambda"] >= 0,
        lambda: (
            "omega_m + omega_r "
            f"({float(params['omega_m'] + params['omega_r']):.5g}) "
            f"exceeds h^2 ({float(params['h'] ** 2):.5g}); omega_Lambda would be "
            "negative -- check omega_cdm / omega_b / h."
        ),
    )


def derive_parameters(
    params: "Params",
    options: "Options",
    species: "tuple[Fluid, ...]",
    *,
    parthenope_table: _SBBNTable,
    linx_thermo: "BackgroundModel | None",
    linx_abundance: "AbundanceModel | None",
) -> "Params":
    """
    Imperative cosmology derivation: orchestrate the derivation of every derived
    quantity from the schema-resolved input ``params``, the model ``options``, the
    constructed ``species`` list, and the BBN backend state. Mutates and returns
    ``params``.
    """
    params["H0"] = jnp.array(params["h"] * cnst.H0_over_h)
    # tau_reion and z_reion are both schema-resolved; the physics reads whichever
    # the input_tau_reion option selects.
    # Placeholder so DE density sums to ~0 in the loops (Array, per Params).
    params["omega_Lambda"] = jnp.asarray(0.0)

    input_N, input_Neff = _resolve_neutrino_input(params, options)
    if input_N:
        _neff_from_fluid_content(params, species)
    if _compute_helium_fraction(
        params, options, parthenope_table, linx_thermo, linx_abundance
    ):
        input_Neff = True  # LINX set Neff; re-derive the massless count below
    if input_Neff:
        _n_massless_from_neff(params, species)
    _derive_densities(params, species)
    return params
