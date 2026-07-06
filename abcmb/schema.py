"""
Parameters subsystem for ABCMB: declare, resolve, derive.

Two kinds of input, split by a simple rule:
  Option: configures the computation (static, non-traceable, non-differentiable).
  Param:  a differentiable parameter of the computation.

From the CLI/config both are just KEY=VALUE, routed by name (see
``option_key_set``); the split is an internal boundary the front-end hides.

Provides default resolution, CLASS-style aliases, light type checks, per-key
provenance, human-readable listings, and the imperative ``derive_parameters``.
"""

import difflib
import warnings
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from . import constants as cnst
from .ABCMBTools import bilinear_interp
from .linx import const as linxconst
from .linx import thermo as linxThermo


class Source(StrEnum):
    """Where a resolved value came from (see :class:`Provenance`)."""

    DEFAULT = "default"  # schema default; the user did not supply this key
    USER = "user"  # supplied by the user under its canonical name
    ALIAS = "alias"  # supplied under an alias (original key in Provenance.origin)
    EXTRA = "extra"  # unrecognized passthrough (custom-species escape hatch)


class Group(StrEnum):
    """Display grouping for a schema row; organizes ``describe_schema`` output."""

    SPECIES = "species"
    INPUT = "input"
    OUTPUT = "output"
    BBN = "bbn"
    HIERARCHY = "hierarchy"
    K_GRID = "k_grid"
    TRANSFER = "transfer"
    PRIMORDIAL = "primordial"
    REIONIZATION = "reionization"
    COSMOLOGY = "cosmology"
    NEUTRINOS = "neutrinos"
    IC = "ic"
    SOLVER = "solver"
    SOURCE = "source"
    MISC = "misc"


@dataclass(frozen=True)
class Provenance:
    """
    Where one resolved value came from.

    ``value`` is the raw supplied (or default) value; ``source`` is the kind; and
    ``origin`` is the original key the user supplied when ``source`` is
    :attr:`Source.ALIAS` (otherwise ``None``).
    """

    value: Any
    source: Source
    origin: str | None = None


# The return shape of the resolve helpers: (resolved values, per-key provenance).
ResolveResult = tuple[dict[str, Any], dict[str, Provenance]]


@dataclass(frozen=True)
class Spec:
    """
    One row of a schema.

    Attributes
    ----------
    name : str
        Canonical key.
    default : Any
        Value used when the user does not supply this key.
    kind : type
        Expected Python kind (``int``/``float``/``bool``/``str``). Used for a
        light, non-fatal type check and for documentation.
    doc : str
        One-line description.
    aliases : tuple[str, ...]
        Alternative names accepted for this entry (e.g. CLASS conventions).
    group : Group
        Grouping label, used only for organizing ``describe_schema()`` output.
    choices : tuple
        Allowed values for an enum-like option (empty = unrestricted). Matched
        case-insensitively for strings; an off-list value warns with "did you mean".
    bounds : tuple
        ``(minimum, maximum)`` for a numeric value, each ``None`` if unbounded.
        An out-of-range value warns (does not raise). Inclusive.
    """

    name: str
    default: Any
    kind: type = float
    doc: str = ""
    aliases: tuple[str, ...] = ()
    group: Group = Group.MISC
    choices: tuple = ()
    bounds: tuple = (None, None)


# OPTION_SCHEMA: the source of truth for model configuration (options). Drives
# default resolution, alias handling, type checks, and ``describe_schema``.
# To add an option, add one row here.
OPTION_SCHEMA = (
    Spec(
        "use_LCDM_species",
        True,
        bool,
        "Include the default LCDM species set.",
        group=Group.SPECIES,
    ),
    Spec(
        "input_tau_reion",
        True,
        bool,
        "Reionization input mode: tau_reion (True) vs z_reion (False).",
        group=Group.INPUT,
    ),
    # Output
    Spec(
        "l_min",
        2,
        int,
        "Minimum multipole for output spectra.",
        group=Group.OUTPUT,
        bounds=(2, None),
    ),
    Spec(
        "l_max",
        2500,
        int,
        "Maximum multipole for output spectra.",
        aliases=("l_max_scalars",),
        group=Group.OUTPUT,
        bounds=(2, None),
    ),
    Spec("lensing", False, bool, "Compute lensed spectra.", group=Group.OUTPUT),
    Spec(
        "k_max",
        0.5,
        float,
        "Maximum wavenumber for the matter power spectrum (Mpc^-1).",
        group=Group.OUTPUT,
        bounds=(0.0, None),
    ),
    # BBN
    Spec(
        "bbn_type",
        "",
        str,
        "BBN backend: '' / 'Table' (sBBN table) or 'linx' (LINX).",
        group=Group.BBN,
        choices=("", "table", "linx"),
    ),
    Spec(
        "linx_reaction_net",
        "key_PRIMAT_2023",
        str,
        "LINX nuclear reaction network.",
        group=Group.BBN,
        choices=("key_PRIMAT_2023", "key_PArthENoPE"),
    ),
    # Boltzmann hierarchy cutoffs
    Spec(
        "l_max_g",
        12,
        int,
        "Photon temperature hierarchy cutoff.",
        group=Group.HIERARCHY,
        bounds=(2, None),
    ),
    Spec(
        "l_max_pol_g",
        10,
        int,
        "Photon polarization hierarchy cutoff.",
        group=Group.HIERARCHY,
        bounds=(2, None),
    ),
    Spec(
        "l_max_massless_nu",
        17,
        int,
        "Massless-neutrino hierarchy cutoff.",
        aliases=("l_max_ur",),
        group=Group.HIERARCHY,
        bounds=(2, None),
    ),
    Spec(
        "l_max_massive_nu",
        17,
        int,
        "Massive-neutrino hierarchy cutoff.",
        aliases=("l_max_ncdm",),
        group=Group.HIERARCHY,
        bounds=(2, None),
    ),
    # Perturbation k-grid resolution
    Spec(
        "k_step_sub",
        5.0e-2,
        float,
        "Perturbation k-grid step, sub-horizon.",
        group=Group.K_GRID,
        bounds=(0.0, None),
    ),
    Spec(
        "k_step_super",
        2.0e-3,
        float,
        "Perturbation k-grid step, super-horizon.",
        group=Group.K_GRID,
        bounds=(0.0, None),
    ),
    Spec(
        "k_step_transition",
        2.0e-1,
        float,
        "Perturbation k-grid transition step.",
        group=Group.K_GRID,
    ),
    Spec(
        "k_step_super_reduction",
        1.0e-1,
        float,
        "Super-horizon step reduction factor.",
        group=Group.K_GRID,
    ),
    Spec(
        "k_min_tau0",
        1.0e-1,
        float,
        "Minimum k*tau0 for the perturbation grid.",
        group=Group.K_GRID,
    ),
    Spec(
        "k_max_tau0_over_l_max",
        1.8,
        float,
        "Max k*tau0 / l_max for the perturbation grid.",
        group=Group.K_GRID,
    ),
    Spec(
        "H0_fid",
        2.255560e-04,
        float,
        "Fiducial H0 for k-grid scaling.",
        group=Group.K_GRID,
    ),
    Spec(
        "tau0_fid",
        1.418668e04,
        float,
        "Fiducial conformal time today for k-grid scaling.",
        group=Group.K_GRID,
    ),
    Spec(
        "rs_rec_fid",
        1.446279e02,
        float,
        "Fiducial sound horizon at recombination.",
        group=Group.K_GRID,
    ),
    # Transfer-integration k-grid resolution
    Spec(
        "k_transfer_linstep",
        4.5e-1,
        float,
        "Transfer-integration k-grid linear step.",
        group=Group.TRANSFER,
    ),
    Spec(
        "k_transfer_logstep",
        170.0,
        float,
        "Transfer-integration k-grid log step.",
        group=Group.TRANSFER,
    ),
    Spec(
        "tau_rec_fid",
        281.040565,
        float,
        "Fiducial conformal time at recombination.",
        group=Group.TRANSFER,
    ),
    # Primordial
    Spec(
        "k_pivot",
        0.05,
        float,
        "Primordial pivot scale (Mpc^-1).",
        group=Group.PRIMORDIAL,
    ),
    # Initial-condition start time
    Spec(
        "R_tc",
        0.0015,
        float,
        "Tight-coupling threshold for IC start time.",
        group=Group.IC,
    ),
    Spec(
        "R_large",
        0.07,
        float,
        "Large-scale threshold for IC start time.",
        group=Group.IC,
    ),
    # Perturbation-evolver (diffrax) settings
    Spec(
        "max_steps_PE",
        2048,
        int,
        "Max diffrax steps for the perturbation evolver.",
        group=Group.SOLVER,
        bounds=(1, None),
    ),
    Spec(
        "k_split_PE",
        0.01,
        float,
        "k threshold splitting solver tolerances.",
        group=Group.SOLVER,
    ),
    Spec(
        "rtol_small_k_PE",
        1.0e-5,
        float,
        "Relative tolerance, small k.",
        group=Group.SOLVER,
    ),
    Spec(
        "rtol_large_k_PE",
        1.0e-4,
        float,
        "Relative tolerance, large k.",
        group=Group.SOLVER,
    ),
    Spec(
        "atol_small_k_PE",
        1.0e-10,
        float,
        "Absolute tolerance, small k.",
        group=Group.SOLVER,
    ),
    Spec(
        "atol_large_k_PE",
        1.0e-6,
        float,
        "Absolute tolerance, large k.",
        group=Group.SOLVER,
    ),
    Spec("pcoeff_PE", 0.25, float, "PID controller P coefficient.", group=Group.SOLVER),
    Spec("icoeff_PE", 0.8, float, "PID controller I coefficient.", group=Group.SOLVER),
    Spec("dcoeff_PE", 0.0, float, "PID controller D coefficient.", group=Group.SOLVER),
    # Source-term switches for the CMB temperature transfer function
    Spec("scale_sw", 1, int, "Sachs-Wolfe term switch/scale.", group=Group.SOURCE),
    Spec(
        "scale_isw",
        1,
        int,
        "Integrated Sachs-Wolfe term switch/scale.",
        group=Group.SOURCE,
    ),
    Spec("scale_dop", 1, int, "Doppler term switch/scale.", group=Group.SOURCE),
    Spec("scale_pol", 1, int, "Polarization term switch/scale.", group=Group.SOURCE),
)


def _alias_map(schema):
    """Build a {alias: canonical_name} lookup from the schema."""
    out = {}
    for spec in schema:
        for alias in spec.aliases:
            out[alias] = spec.name
    return out


def _did_you_mean(key, names):
    """Return the closest declared name to ``key``, or None."""
    matches = difflib.get_close_matches(key, names, n=1, cutoff=0.7)
    return matches[0] if matches else None


def option_key_set() -> set[str]:
    """
    Set of all recognized option keys: canonical names plus aliases.

    Used to route flat config-file keys to options vs params (a key is an option iff
    it appears here; everything else is treated as a cosmological parameter).
    """
    return {spec.name for spec in OPTION_SCHEMA} | set(_alias_map(OPTION_SCHEMA))


def route(flat) -> tuple[dict, dict]:
    """
    Split a flat ``{key: value}`` dict into ``(options, params)`` by schema
    membership: a key is an option iff it is a known option name or alias
    (:func:`option_key_set`), otherwise a cosmological parameter. Names are globally
    unique, so a key routes correctly regardless of which config table it came from.
    """
    option_keys = option_key_set()
    options, params = {}, {}
    for key, value in flat.items():
        (options if key in option_keys else params)[key] = value
    return options, params


def _as_number(value):
    """Return ``value`` as a float if it is a real scalar (incl. 0-d array), else None."""
    if isinstance(value, (bool, int, float)):
        return float(value)
    try:
        arr = jnp.asarray(value)
        if arr.ndim == 0 and jnp.issubdtype(arr.dtype, jnp.number):
            return float(arr)
    except (TypeError, ValueError):
        pass
    return None


def _check_value(spec, value):
    """
    Light, non-fatal validation of a user-supplied value: kind, then ``choices``
    (enum-like options) and ``bounds`` (numeric range). Warns on any mismatch;
    never raises. Applied only to declared schema entries the user actually set.
    """
    # 1. kind
    if spec.kind is bool:
        ok = isinstance(value, (bool, int))  # accept 0/1
    elif spec.kind in (int, float):
        ok = _as_number(value) is not None  # Python or 0-d array numbers ok
    elif spec.kind is str:
        ok = isinstance(value, str)
    else:
        ok = True
    if not ok:
        warnings.warn(
            f"'{spec.name}' expected {spec.kind.__name__}, "
            f"got {type(value).__name__} ({value!r}).",
            stacklevel=3,
        )
        return  # remaining checks assume the kind is right

    # 2. choices (enum-like options; case-insensitive for strings)
    if spec.choices:
        allowed = {str(c).lower() for c in spec.choices}
        if isinstance(value, str) and value.lower() not in allowed:
            msg = f"'{spec.name}'={value!r} is not one of {list(spec.choices)}."
            suggestion = _did_you_mean(value, [str(c) for c in spec.choices])
            if suggestion:
                msg += f" Did you mean {suggestion!r}?"
            warnings.warn(msg, stacklevel=3)

    # 3. bounds (numeric range; inclusive, None = unbounded)
    lo, hi = spec.bounds
    if lo is not None or hi is not None:
        num = _as_number(value)
        if num is not None:
            if lo is not None and num < lo:
                warnings.warn(
                    f"'{spec.name}'={num!r} is below the minimum {lo}.", stacklevel=3
                )
            elif hi is not None and num > hi:
                warnings.warn(
                    f"'{spec.name}'={num!r} is above the maximum {hi}.", stacklevel=3
                )


def describe_schema(schema, indent="  ") -> str:
    """
    Return a human-readable, group-sectioned listing of a schema (a tuple of
    :class:`Spec`). Rows are grouped by their ``group`` label in first-seen order.
    """
    lines = []
    seen_groups = []
    by_group = {}
    for spec in schema:
        if spec.group not in by_group:
            by_group[spec.group] = []
            seen_groups.append(spec.group)
        by_group[spec.group].append(spec)
    for group in seen_groups:
        lines.append(f"{indent}[{group}]")
        for spec in by_group[group]:
            alias = f"  (aliases: {', '.join(spec.aliases)})" if spec.aliases else ""
            lines.append(
                f"{indent}  {spec.name} = {spec.default!r} "
                f"[{spec.kind.__name__}]{alias}\n"
                f"{indent}      {spec.doc}"
            )
    return "\n".join(lines)


def describe_reference() -> str:
    """
    Full, human-readable parameter/option reference for ``abcmb --list-params``.

    Lists the declared cosmological parameters and model options (grouped, with
    defaults / types / aliases / docs), plus the names of the conditional/BBN
    parameters whose values are owned by the physics logic.
    """
    cond = ", ".join(sorted(_CONDITIONAL_PARAM_KEYS))
    return "\n".join(
        [
            "COSMOLOGICAL PARAMETERS  (set via KEY=VALUE or a config file)",
            describe_schema(PARAM_SCHEMA),
            "",
            "  Also settable — defaults/values are set by the BBN and neutrino",
            f"  logic:\n      {cond}",
            "",
            "MODEL / RUN OPTIONS  (set via KEY=VALUE or a config file)",
            describe_schema(OPTION_SCHEMA),
        ]
    )


def _identity(value):
    return value


def _resolve(
    input_dict,
    schema,
    *,
    aliases,
    managed_keys=frozenset(),
    wrap=_identity,
    noun="key",
    strict=False,
) -> ResolveResult:
    """
    Resolves ``input_dict`` against ``schema`` (a tuple of :class:`Spec`): applies
    ``aliases`` (warning on use), fills declared entries from user values or
    defaults with a light type check, passes ``managed_keys`` through untouched
    (recognized but not auto-filled, for the imperative logic to consume), and
    preserves unknown keys as ``extra`` (warning, or raising if ``strict``).
    ``wrap`` transforms every stored value (e.g. ``jnp.array`` for params);
    ``noun`` labels the warnings ("option" / "parameter").

    Returns ``(resolved, provenance)`` where ``provenance[key]`` is a
    :class:`Provenance` (``value`` / ``source`` / ``origin``).
    """
    by_name = {spec.name: spec for spec in schema}
    known = set(by_name) | set(aliases) | set(managed_keys)

    resolved = {}  # canonical -> raw value
    origin = {}  # canonical -> original key as supplied
    out = {}
    provenance = {}
    for key, value in input_dict.items():
        canonical = aliases.get(key, key)
        if canonical != key:
            warnings.warn(
                f"{noun} '{key}' is an alias for '{canonical}'; "
                f"prefer the canonical name.",
                stacklevel=3,
            )
        if canonical in by_name:
            resolved[canonical] = value
            origin[canonical] = key
        elif canonical in managed_keys:
            # Recognized input handled by the imperative logic; pass through.
            out[canonical] = wrap(value)
            if canonical != key:
                provenance[canonical] = Provenance(value, Source.ALIAS, origin=key)
            else:
                provenance[canonical] = Provenance(value, Source.USER)
        else:
            msg = (
                f"unrecognized {noun} '{key}'; it is passed through unused "
                f"unless a custom species reads it."
            )
            suggestion = _did_you_mean(key, known)
            if suggestion:
                msg += f" Did you mean '{suggestion}'?"
            if strict:
                raise ValueError(msg)
            warnings.warn(msg, stacklevel=3)
            out[key] = wrap(value)
            provenance[key] = Provenance(value, Source.EXTRA)

    for spec in schema:
        if spec.name in resolved:
            value = resolved[spec.name]
            _check_value(spec, value)
            out[spec.name] = wrap(value)
            if origin[spec.name] != spec.name:
                provenance[spec.name] = Provenance(
                    value, Source.ALIAS, origin=origin[spec.name]
                )
            else:
                provenance[spec.name] = Provenance(value, Source.USER)
        else:
            out[spec.name] = wrap(spec.default)
            provenance[spec.name] = Provenance(spec.default, Source.DEFAULT)

    return out, provenance


def _check_option_consistency(options, strict=False):
    """Cross-option checks that a single ``Spec``'s ``bounds`` cannot express."""
    if options["l_max"] < options["l_min"]:
        msg = (
            f"l_max ({options['l_max']}) < l_min ({options['l_min']}); "
            "the output multipole range would be empty."
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=3)


def resolve_options(input_options, strict=False) -> ResolveResult:
    """
    Resolve user configuration against ``OPTION_SCHEMA``, returning the populated
    options and per-key provenance. See :func:`_resolve` for the semantics.
    """
    options, provenance = _resolve(
        input_options,
        OPTION_SCHEMA,
        aliases=_alias_map(OPTION_SCHEMA),
        noun="option",
        strict=strict,
    )
    _check_option_consistency(options, strict=strict)
    return options, provenance


# ---------------------------------------------------------------------------
# Only the *unconditional pure defaults* live here. Parameters whose presence
# signals user intent (Neff, N_nu_massless, T_nu_massless), or that are computed
# (H0, omega_m, R_nu, ...), are handled imperatively in ``derive_parameters`` and
# listed in ``_MANAGED_PARAM_KEYS`` below so they are recognized (not typos).
# ---------------------------------------------------------------------------
PARAM_SCHEMA = (
    Spec(
        "h",
        0.6736,
        float,
        "Dimensionless Hubble parameter H0/100.",
        group=Group.COSMOLOGY,
        bounds=(0.0, None),
    ),
    Spec(
        "omega_cdm",
        0.120,
        float,
        "Physical CDM density Omega_cdm h^2.",
        group=Group.COSMOLOGY,
        bounds=(0.0, None),
    ),
    Spec(
        "omega_b",
        0.02237,
        float,
        "Physical baryon density Omega_b h^2.",
        group=Group.COSMOLOGY,
        bounds=(0.0, None),
    ),
    Spec(
        "A_s",
        2.1e-9,
        float,
        "Scalar amplitude at the pivot scale.",
        group=Group.PRIMORDIAL,
        bounds=(0.0, None),
    ),
    Spec(
        "n_s",
        0.9649,
        float,
        "Scalar spectral index.",
        group=Group.PRIMORDIAL,
        bounds=(0.0, None),
    ),
    Spec(
        "TCMB0",
        2.34865418e-4,
        float,
        "CMB temperature today, in eV.",
        group=Group.COSMOLOGY,
        bounds=(0.0, None),
    ),
    Spec(
        "Delta_z_reion",
        0.5,
        float,
        "Reionization width in redshift.",
        aliases=("reionization_width",),
        group=Group.REIONIZATION,
        bounds=(0.0, None),
    ),
    Spec(
        "z_reion_He",
        3.5,
        float,
        "Helium reionization redshift.",
        aliases=("helium_fullreio_redshift",),
        group=Group.REIONIZATION,
        bounds=(0.0, None),
    ),
    Spec(
        "Delta_z_reion_He",
        0.5,
        float,
        "Helium reionization width.",
        aliases=("helium_fullreio_width",),
        group=Group.REIONIZATION,
        bounds=(0.0, None),
    ),
    Spec(
        "exp_reion",
        1.5,
        float,
        "Reionization exponent.",
        aliases=("reionization_exponent",),
        group=Group.REIONIZATION,
        bounds=(0.0, None),
    ),
    Spec(
        "T_nu_massive",
        0.71611,
        float,
        "Massive-neutrino temperature (ratio to TCMB).",
        aliases=("T_ncdm",),
        group=Group.NEUTRINOS,
        bounds=(0.0, None),
    ),
    Spec(
        "N_nu_massive",
        0,
        int,
        "Number of massive neutrinos.",
        aliases=("N_ncdm",),
        group=Group.NEUTRINOS,
        bounds=(0.0, None),
    ),
    Spec(
        "m_nu_massive",
        0.06,
        float,
        "Massive-neutrino mass, in eV.",
        aliases=("m_ncdm",),
        group=Group.NEUTRINOS,
        bounds=(0.0, None),
    ),
    Spec(
        "tau_reion",
        0.0544,
        float,
        "Reionization optical depth (used when input_tau_reion is True).",
        aliases=("tau_reio",),
        group=Group.REIONIZATION,
        bounds=(0.0, None),
    ),
    Spec(
        "z_reion",
        7.67,
        float,
        "Reionization redshift (used when input_tau_reion is False).",
        group=Group.REIONIZATION,
        bounds=(0.0, None),
    ),
    Spec(
        "YHe",
        0.245,
        float,
        "Primordial helium mass fraction (overwritten when bbn_type computes it).",
        group=Group.COSMOLOGY,
        bounds=(0.0, 1.0),
    ),
)

# Conditional inputs the user MAY supply but which are NOT auto-filled by
# resolve_params — their defaults/values are owned by the neutrino and BBN logic
# in derive_parameters. Two disjoint groups: the neutrino one-of
# (provide-or-derive) and the LINX BBN inputs (used only when bbn_type=='linx').
_NEUTRINO_INPUT_KEYS = ("Neff", "N_nu_massless", "T_nu_massless")
_LINX_INPUT_KEYS = ("Delta_Neff_init", "tau_n_fac", "nuclear_rates_q")
_CONDITIONAL_PARAM_KEYS = frozenset(_NEUTRINO_INPUT_KEYS + _LINX_INPUT_KEYS)
# Derived quantities — computed by derive_parameters, not user inputs.
_DERIVED_PARAM_KEYS = frozenset(
    {"H0", "omega_Lambda", "omega_m", "R_b", "omega_r", "R_nu", "om"}
)
_MANAGED_PARAM_KEYS = _CONDITIONAL_PARAM_KEYS | _DERIVED_PARAM_KEYS

# CLASS aliases for managed keys (schema-param aliases live on the Spec rows).
_MANAGED_PARAM_ALIASES = {
    "N_ur": "Neff",
}


def param_key_set() -> set[str]:
    """Set of all recognized parameter keys (schema, managed, aliases)."""
    return (
        {spec.name for spec in PARAM_SCHEMA}
        | _MANAGED_PARAM_KEYS
        | set(_alias_map(PARAM_SCHEMA))
        | set(_MANAGED_PARAM_ALIASES)
    )


def resolve_params(param_in, strict=False) -> ResolveResult:
    """
    Resolve raw cosmological parameters against ``PARAM_SCHEMA``, returning the
    populated params and per-key provenance. See :func:`_resolve` for the
    semantics.

    Values are wrapped in ``jnp.array``. Managed keys (conditional/derived; see
    ``_MANAGED_PARAM_KEYS``) are passed through untouched for the imperative logic
    in ``derive_parameters`` to consume — importantly, they are NOT auto-filled,
    so ``params.get(...) is not None`` intent checks still work.
    """
    aliases = _alias_map(PARAM_SCHEMA)
    aliases.update(_MANAGED_PARAM_ALIASES)
    return _resolve(
        param_in,
        PARAM_SCHEMA,
        aliases=aliases,
        managed_keys=_MANAGED_PARAM_KEYS,
        wrap=jnp.array,
        noun="parameter",
        strict=strict,
    )


def _check_neutrino_input(params):
    """
    Enforce the ``Neff`` / ``N_nu_massless`` one-of (they are treated 1-to-1; see
    paper). A param-only invariant — no Model needed — so ``derive_parameters``
    calls it up front. Raises ``ValueError`` if both are supplied.
    """
    if params.get("N_nu_massless") is not None and params.get("Neff") is not None:
        raise ValueError(
            "Provide only one of Neff / N_nu_massless (they are treated 1-to-1)."
        )


def _require(ok, message):
    """Raise ``ValueError(message)`` unless the (concrete, eager) condition holds."""
    if not bool(ok):
        raise ValueError(message)


def _resolve_neutrino_input(params, options):
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

    # LINX computes Neff/T_nu_massless itself, so the user must not also supply them.
    if (input_N or input_Neff or input_T_nu_massless) and options[
        "bbn_type"
    ].lower() == "linx":
        raise ValueError(
            "You have specified a value for N_nu_massless and/or Neff and/or "
            "T_nu_massless, but LINX instead expects 'Delta_Neff_init', which it "
            "uses to compute Neff. See the LINX docs or "
            "https://arxiv.org/abs/2408.14538."
        )

    if not input_N and not input_Neff and options["bbn_type"].lower() != "linx":
        params["N_nu_massless"] = 3 - params["N_nu_massive"]
        input_N = True

    # T_nu_massless (ratio to TCMB); safe to set now that inputs are validated.
    params["T_nu_massless"] = jnp.array(params.get("T_nu_massless", 0.71636856))
    return input_N, input_Neff


def _neff_from_fluid_content(params, species):
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
        elif "neutrino" in s.name.lower():
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


def _helium_from_table(params, parthenope_table):
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

    # last two args: user omega_b and (Neff - 3.046) (3.046 assumed by the table)
    params["YHe"] = bilinear_interp(
        omegab, DNeff, YHe_grid, params["omega_b"], params["Neff"] - 3.046
    )


def _helium_from_linx(params, linx_thermo, linx_abundance):
    """
    Run LINX BBN from ``Delta_Neff_init`` (+ optional ``tau_n_fac`` /
    ``nuclear_rates_q``): sets ``Neff``, ``T_nu_massless``, and ``YHe`` in place.
    """
    params["Delta_Neff_init"] = jnp.array(params.get("Delta_Neff_init", 0.0))
    (
        t_vec_ref,
        a_vec_ref,
        rho_g_vec,
        rho_nu_vec,
        rho_NP_vec,
        P_NP_vec,
        Neff_vec,
    ) = eqx.filter_jit(linx_thermo, backend="cpu")(params["Delta_Neff_init"])

    # convert user input omega_b to eta_fac LINX expects
    eta_fac = params["omega_b"] * linxconst.Omegabh2_to_eta0 / linxconst.eta0

    abundances = eqx.filter_jit(linx_abundance, backend="cpu")(
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
    params, options, parthenope_table, linx_thermo, linx_abundance
):
    """
    Set ``params["YHe"]`` per the ``bbn_type`` backend (sBBN table, LINX, or leave
    the schema-resolved default for ""). Returns ``True`` when LINX has set
    ``Neff`` (so the caller must re-derive ``N_nu_massless``).
    """
    bbn_type = options["bbn_type"].lower()
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


def _n_massless_from_neff(params, species):
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
        f"Neff={float(params['Neff']):.4g} is too small for the other relativistic "
        "species: it implies a negative massless-neutrino count "
        f"(N_nu_massless={float(params['N_nu_massless']):.4g}).",
    )


def _derive_densities(params, species):
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
        if "neutrino" in s.name.lower():
            rho_nu += s.rho(jnp.log(a_early), params)
    params["omega_r"] = (
        rho_r * a_early**4 / (3 * cnst.H0_over_h**2 / 8 / jnp.pi / cnst.G)
    )
    _require(
        params["omega_r"] > 0,
        "omega_r is not positive (no radiation content?); R_nu/om would divide by "
        "zero -- check TCMB0 or the photon/neutrino species.",
    )
    params["R_nu"] = rho_nu / rho_r  # neutrino fraction of radiation (adiabatic ICs)

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
        f"omega_m + omega_r ({float(params['omega_m'] + params['omega_r']):.5g}) "
        f"exceeds h^2 ({float(params['h'] ** 2):.5g}); omega_Lambda would be "
        "negative -- check omega_cdm / omega_b / h.",
    )


def derive_parameters(
    params,
    options,
    species,
    *,
    parthenope_table,
    linx_thermo,
    linx_abundance,
):
    """
    Imperative cosmology derivation: orchestrate the derivation of every derived
    quantity from the schema-resolved input ``params``, the model ``options``, the
    constructed ``species`` list, and the BBN backend state. Mutates and returns
    ``params``.
    """
    params["H0"] = jnp.array(params["h"] * cnst.H0_over_h)
    # tau_reion and z_reion are both schema-resolved; the physics reads whichever
    # the input_tau_reion option selects.
    params["omega_Lambda"] = 0.0  # placeholder so DE density sums to ~0 in the loops

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
