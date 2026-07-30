"""
Parameters subsystem for ABCMB: declare, resolve, derive.

Two kinds of input, split by a simple rule:
  Option: configures the computation (static, non-traceable, non-differentiable).
  Param:  a differentiable parameter of the computation.

From the CLI/config both are just KEY=VALUE, routed by name (see
``option_key_set``); the split is an internal boundary the front-end hides.

Provides default resolution, CLASS-style aliases, light type checks,
human-readable listings, and the imperative ``derive_parameters``.
"""

import difflib
import warnings
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import TYPE_CHECKING, Any, cast

import jax.numpy as jnp

if TYPE_CHECKING:
    from ._schema_types import Options, Params


class Group(StrEnum):
    """Display grouping for a schema row; organizes ``describe_schema`` output."""

    SPECIES = auto()
    INPUT = auto()
    OUTPUT = auto()
    BBN = auto()
    HIERARCHY = auto()
    K_GRID = auto()
    TRANSFER = auto()
    PRIMORDIAL = auto()
    REIONIZATION = auto()
    COSMOLOGY = auto()
    NEUTRINOS = auto()
    IC = auto()
    SOLVER = auto()
    SOURCE = auto()
    DERIVED = auto()
    MISC = auto()


class _Unset:
    """Sentinel type for :attr:`Spec.default`; see :data:`UNSET`."""

    def __repr__(self):
        return "UNSET"


# Sentinel default for a Spec that is recognized but never auto-filled: its
# absence from the resolved dict is meaningful (``params.get(...) is not None``
# signals user intent to the imperative logic in ``derive_parameters``).
UNSET = _Unset()


@dataclass(frozen=True)
class Spec:
    """
    One row of a schema.

    Attributes
    ----------
    name : str
        Canonical key.
    default : Any
        Value used when the user does not supply this key. The sentinel
        :data:`UNSET` declares a *conditional* entry: recognized (aliases, docs,
        checks all apply) but never auto-filled, so its absence still signals
        user intent downstream.
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
    derived : bool
        True for a quantity computed by ``derive_parameters``: declared here so a
        user-supplied value is recognized (not flagged as a typo), but it is not a
        model input and is excluded from the input listings. Implies
        ``default=UNSET``.
    """

    name: str
    default: Any
    kind: type = float
    doc: str = ""
    aliases: tuple[str, ...] = ()
    group: Group = Group.MISC
    choices: tuple = ()
    bounds: tuple = (None, None)
    derived: bool = False


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
            default = (
                "(no fixed default)" if spec.default is UNSET else repr(spec.default)
            )
            lines.append(
                f"{indent}  {spec.name} = {default} "
                f"[{spec.kind.__name__}]{alias}\n"
                f"{indent}      {spec.doc}"
            )
    return "\n".join(lines)


def describe_reference() -> str:
    """
    Full, human-readable parameter/option reference for ``abcmb --list-params``.

    Lists the declared cosmological parameters and model options (grouped, with
    defaults / types / aliases / docs — conditional entries show "(no fixed
    default)"), plus the names of the derived quantities computed at runtime.
    """
    inputs = tuple(spec for spec in PARAM_SCHEMA if not spec.derived)
    derived = ", ".join(spec.name for spec in PARAM_SCHEMA if spec.derived)
    return "\n".join(
        [
            "COSMOLOGICAL PARAMETERS  (set via KEY=VALUE or a config file)",
            describe_schema(inputs),
            "",
            "  Derived at runtime (computed by the model, not inputs):",
            f"      {derived}",
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
    wrap=_identity,
    noun="key",
    strict=False,
) -> dict[str, Any]:
    """
    Resolves ``input_dict`` against ``schema`` (a tuple of :class:`Spec`): applies
    ``aliases`` (warning on use), fills declared entries from user values or
    defaults with a light type check, and preserves unknown keys (warning, or
    raising if ``strict``). Entries whose default is :data:`UNSET` are recognized
    but never auto-filled, so ``.get(...) is not None`` intent checks work
    downstream. ``wrap`` transforms every stored value (e.g. ``jnp.array`` for
    params); ``noun`` labels the warnings ("option" / "parameter").
    """
    by_name = {spec.name: spec for spec in schema}
    known = set(by_name) | set(aliases)

    resolved = {}  # canonical -> raw value
    out = {}
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

    for spec in schema:
        if spec.name in resolved:
            value = resolved[spec.name]
            _check_value(spec, value)
            out[spec.name] = wrap(value)
        elif spec.default is not UNSET:
            out[spec.name] = wrap(spec.default)
        # default is UNSET and not supplied: stays absent (absence = intent).

    return out


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


def resolve_options(input_options, strict=False) -> "Options":
    """
    Resolve user configuration against ``OPTION_SCHEMA``, returning the populated
    options. See :func:`_resolve` for the semantics.
    """
    options = _resolve(
        input_options,
        OPTION_SCHEMA,
        aliases=_alias_map(OPTION_SCHEMA),
        noun="option",
        strict=strict,
    )
    _check_option_consistency(options, strict=strict)
    return cast("Options", options)


# ---------------------------------------------------------------------------
# All parameter declarations live here. Three flavors in one table:
#   * plain rows       -- unconditional pure defaults, auto-filled by _resolve;
#   * default=UNSET    -- conditional inputs whose *absence* signals user intent
#                         (values/defaults owned by ``derive_parameters``);
#   * derived=True     -- computed by ``derive_parameters``; declared so a
#                         supplied value is recognized, but not a model input.
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
    # Conditional neutrino inputs: the Neff / N_nu_massless one-of (provide one
    # or derive it); defaults are owned by derive_parameters.
    Spec(
        "Neff",
        UNSET,
        float,
        "Effective relativistic species count (one-of with N_nu_massless).",
        aliases=("N_ur",),
        group=Group.NEUTRINOS,
        bounds=(0.0, None),
    ),
    Spec(
        "N_nu_massless",
        UNSET,
        float,
        "Massless-neutrino count (one-of with Neff; default 3 - N_nu_massive).",
        group=Group.NEUTRINOS,
        bounds=(0.0, None),
    ),
    Spec(
        "T_nu_massless",
        UNSET,
        float,
        "Massless-neutrino temperature ratio to TCMB (default 0.71636856).",
        group=Group.NEUTRINOS,
        bounds=(0.0, None),
    ),
    # Conditional LINX BBN inputs (used only when bbn_type == 'linx').
    Spec(
        "Delta_Neff_init",
        UNSET,
        float,
        "LINX: initial extra relativistic energy Delta Neff (default 0).",
        group=Group.BBN,
    ),
    Spec(
        "tau_n_fac",
        UNSET,
        float,
        "LINX: neutron-lifetime scaling factor (default 1).",
        group=Group.BBN,
        bounds=(0.0, None),
    ),
    Spec(
        "nuclear_rates_q",
        UNSET,
        object,
        "LINX: per-reaction nuclear-rate perturbations (array; default zeros).",
        group=Group.BBN,
    ),
    # Derived quantities — computed by derive_parameters, not user inputs.
    Spec(
        "H0",
        UNSET,
        float,
        "Hubble rate today (h * H0_over_h).",
        group=Group.DERIVED,
        derived=True,
    ),
    Spec(
        "omega_m",
        UNSET,
        float,
        "Total matter density Omega_m h^2.",
        group=Group.DERIVED,
        derived=True,
    ),
    Spec(
        "R_b",
        UNSET,
        float,
        "Baryon fraction omega_b / omega_m.",
        group=Group.DERIVED,
        derived=True,
    ),
    Spec(
        "omega_r",
        UNSET,
        float,
        "Radiation density Omega_r h^2.",
        group=Group.DERIVED,
        derived=True,
    ),
    Spec(
        "R_nu",
        UNSET,
        float,
        "Neutrino fraction of the radiation density.",
        group=Group.DERIVED,
        derived=True,
    ),
    Spec(
        "om",
        UNSET,
        float,
        "Adiabatic-IC parameter Omega_m / sqrt(Omega_r) * H0, in 1/Mpc.",
        group=Group.DERIVED,
        derived=True,
    ),
    Spec(
        "omega_Lambda",
        UNSET,
        float,
        "Dark-energy density h^2 - omega_r - omega_m.",
        group=Group.DERIVED,
        derived=True,
    ),
)


def param_key_set() -> set[str]:
    """Set of all recognized parameter keys: canonical names plus aliases."""
    return {spec.name for spec in PARAM_SCHEMA} | set(_alias_map(PARAM_SCHEMA))


def resolve_params(param_in, strict=False) -> "Params":
    """
    Resolve raw cosmological parameters against ``PARAM_SCHEMA``, returning the
    populated params. See :func:`_resolve` for the semantics.

    Values are wrapped in ``jnp.array``. Conditional/derived entries (declared
    with ``default=UNSET``) are NOT auto-filled, so ``params.get(...) is not
    None`` intent checks in ``derive_parameters`` still work.
    """
    params = _resolve(
        param_in,
        PARAM_SCHEMA,
        aliases=_alias_map(PARAM_SCHEMA),
        wrap=jnp.array,
        noun="parameter",
        strict=strict,
    )
    return cast("Params", params)
