"""
Schema tests: option/param resolution (defaults, CLASS aliases, passthrough) and
the input-validation guards -- choices, bounds, the light kind check, the
neutrino one-of / LINX-conflict checks, and the derived-cosmology guards.
File-driven config loading, the CLI, and run-file reproducibility live in
``test_config.py``.
"""

import warnings

import pytest

from abcmb.schema import resolve_options, resolve_params


def test_resolve_options_defaults():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        options = resolve_options({})
    assert options["l_max"] == 2500
    assert options["lensing"] is False
    assert options["bbn_type"] == ""
    assert options["scale_sw"] == 1
    assert options["k_pivot"] == 0.05
    # lna-grid precision knobs (endpoints stay structural/derived)
    assert options["lna_output_points"] == 500
    assert options["lna_lensing_points"] == 500
    assert options["lna_tau_points"] == 10000
    assert options["transfer_start_threshold"] == 0.008


def test_resolve_options_aliases_and_extras():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        options = resolve_options({"l_max": 3000, "l_max_ur": 30, "foobar": 1})
    assert options["l_max"] == 3000
    # Alias actually sets the canonical spec (the old silent no-op bug).
    assert options["l_max_massless_nu"] == 30
    assert "l_max_ur" not in options
    assert options["foobar"] == 1  # unrecognized passthrough (custom species)
    assert options["lensing"] is False  # untouched default


def test_unknown_option_warns_and_strict_raises():
    with pytest.warns(UserWarning, match="unrecognized option 'l_maxx'"):
        resolve_options({"l_maxx": 1})
    with pytest.raises(ValueError, match="unrecognized option 'bogus'"):
        resolve_options({"bogus": 1}, strict=True)


def test_l_max_below_l_min_warns_and_strict_raises():
    # Cross-option consistency check (per-Spec bounds cannot see two keys at once).
    with pytest.warns(UserWarning, match=r"l_max .* < l_min"):
        resolve_options({"l_min": 100, "l_max": 50})
    with pytest.raises(ValueError, match=r"l_max .* < l_min"):
        resolve_options({"l_min": 100, "l_max": 50}, strict=True)


def test_resolve_params_defaults_aliases_extras():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params = resolve_params(
            {"omega_cdm": 0.12, "N_ur": 3.044, "T_ncdm": 0.7, "N_idr": 0.3}
        )
    assert float(params["omega_cdm"]) == 0.12
    assert float(params["T_nu_massive"]) == 0.7
    assert float(params["N_idr"]) == 0.3  # unrecognized passthrough preserved
    assert float(params["h"]) == 0.6736  # untouched default
    # CLASS alias is renamed to the canonical key; the alias name is gone.
    assert "Neff" in params and "N_ur" not in params
    assert float(params["Neff"]) == 3.044
    # Conditional keys (Neff, ...; default=UNSET) are NOT auto-filled when the
    # user omits them, so the add_derived_parameters intent checks still work.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params2 = resolve_params({})
    assert "Neff" not in params2 and "N_nu_massless" not in params2
    assert "h" in params2  # pure defaults ARE filled


def test_unknown_param_warns_and_strict_raises():
    with pytest.warns(UserWarning, match="unrecognized parameter 'omega_bb'"):
        resolve_params({"omega_bb": 0.02})
    with pytest.raises(ValueError, match="unrecognized parameter 'bogus'"):
        resolve_params({"bogus": 1}, strict=True)


def test_choices_validation():
    # Tier-1: an off-list enum value warns with a suggestion; valid values (incl.
    # a case variant) are clean.
    with pytest.warns(UserWarning, match=r"not one of.*Did you mean 'table'"):
        resolve_options({"bbn_type": "tabel"})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolve_options({"bbn_type": "Table"})  # case-insensitive
        resolve_options({"linx_reaction_net": "key_PArthENoPE"})
    assert not [x for x in caught if "not one of" in str(x.message)]


def test_bounds_validation():
    # Tier-2: out-of-range numeric values warn; in-range are clean.
    with pytest.warns(UserWarning, match="below the minimum"):
        resolve_params({"h": -0.5})
    with pytest.warns(UserWarning, match="above the maximum"):
        resolve_params({"YHe": 1.5})
    with pytest.warns(UserWarning, match="below the minimum"):
        resolve_options({"k_max": -1.0})
    # k_step_transition = 0 divides by zero in the tanh transition argument
    # (CLASS hard-rejects it); the exclusive-zero floor must catch exactly 0.
    with pytest.warns(UserWarning, match="below the minimum"):
        resolve_options({"k_step_transition": 0.0})


def test_check_value_kind():
    # The light type check warns on a clear kind mismatch and stays silent on a
    # match -- including a 0-d array for a numeric kind (the notebook jnp path that
    # _as_number exists to accept).
    import jax.numpy as jnp

    from abcmb.schema import Spec, _check_value

    num = Spec("x", 1.0, float)
    with pytest.warns(UserWarning, match=r"'x' expected float, got str"):
        _check_value(num, "oops")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a matching kind must NOT warn
        _check_value(num, 0.5)  # python float
        _check_value(num, 2)  # int is an acceptable number
        _check_value(num, jnp.asarray(0.5))  # 0-d array (jnp notebook input)

    flag = Spec("f", False, bool)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _check_value(flag, True)
        _check_value(flag, 1)  # 0/1 accepted for a bool
    with pytest.warns(UserWarning, match=r"'f' expected bool, got str"):
        _check_value(flag, "yes")

    name = Spec("s", "", str)
    with pytest.warns(UserWarning, match=r"'s' expected str, got float"):
        _check_value(name, 3.14)


def test_neutrino_one_of_invariant():
    # The 1-to-1 rule (was print + sys.exit) is a param-only helper called up front
    # by derive_parameters, so it is testable with no Model construction.
    from abcmb.derived import _check_neutrino_input

    _check_neutrino_input({"Neff": 3.044})  # one alone is fine
    _check_neutrino_input({"N_nu_massless": 2.0})  # the other alone is fine
    with pytest.raises(ValueError, match="only one of Neff"):
        _check_neutrino_input({"Neff": 3.044, "N_nu_massless": 2.0})


def test_linx_conflict_raises():
    # Supplying a neutrino input under LINX (which computes it) is rejected up front.
    from abcmb.derived import _resolve_neutrino_input

    with pytest.raises(ValueError, match="LINX"):
        _resolve_neutrino_input({"Neff": 3.044}, {"bbn_type": "linx"})


class _Fluid:
    """Lightweight mock for a species: fixed early/today energy densities.
    Skip instantiating real Fluid since it's slow. API drift should be caught
    by the accuracy check"""

    def __init__(
        self,
        name,
        is_matter=False,
        rho_early=0.0,
        rho_today=0.0,
        is_neutrino=False,
    ):
        self.name, self.is_matter = name, is_matter
        self.is_neutrino = is_neutrino
        self._early, self._today = rho_early, rho_today

    def rho(self, lna, params):
        import jax.numpy as jnp

        return jnp.asarray(self._today if lna == 0.0 else self._early)


def test_derived_density_guards():
    # The derived-cosmology guards fail fast (clear error, not a NaN spectrum).
    import jax.numpy as jnp

    from abcmb.derived import _derive_densities

    base = {"h": jnp.asarray(0.68), "omega_b": jnp.asarray(0.02), "omega_Lambda": 0.0}

    with pytest.raises(ValueError, match="omega_m is not positive"):  # no matter
        _derive_densities(dict(base), [_Fluid("Photon", rho_early=1.0, rho_today=0.0)])

    with pytest.raises(ValueError, match="omega_r is not positive"):  # no radiation
        _derive_densities(
            dict(base), [_Fluid("CDM", is_matter=True, rho_today=5.0, rho_early=0.0)]
        )

    with pytest.raises(ValueError, match="omega_Lambda would be"):  # budget > h^2
        _derive_densities(
            dict(base),
            [_Fluid("Baryon", is_matter=True, rho_today=1e30, rho_early=1.0)],
        )

    # ...and it stays SILENT with sane densities for a valid cosmology, so a
    # too-strict / always-raising guard fails here (not just the raising cases above).
    # Scale the fake rho by C so the fake densities map to realistic Omegas:
    #   omega_m = rho_m / C,   omega_r = rho_r * a_early^4 / C.
    from abcmb import constants as cnst

    C = 3 * cnst.H0_over_h**2 / 8 / jnp.pi / cnst.G
    a4 = jnp.exp(-23.0) ** 4  # the a_early^4 factor in omega_r
    valid = dict(base)
    _derive_densities(  # omega_m ~ 0.14, omega_r ~ 8e-5  ->  omega_Lambda ~ 0.32 > 0
        valid,
        [_Fluid("CDM", is_matter=True, rho_today=0.14 * C, rho_early=8e-5 * C / a4)],
    )
    assert float(valid["omega_m"]) == pytest.approx(0.14, rel=1e-3)
    assert float(valid["omega_Lambda"]) > 0  # h^2 - omega_r - omega_m, comfortably +


def test_n_massless_from_neff_guard():
    # Too-small Neff for the other relativistic species -> negative count.
    import jax.numpy as jnp

    from abcmb.derived import _n_massless_from_neff

    with pytest.raises(ValueError, match="negative massless-neutrino"):
        _n_massless_from_neff(
            {"Neff": jnp.asarray(0.1), "T_nu_massless": jnp.asarray(0.71)},
            [_Fluid("Photon", rho_early=1.0), _Fluid("ExtraRad", rho_early=1e10)],
        )

    # ...and stays SILENT with a sane count for a sufficient Neff. Only Photon and the
    # (excluded) MasslessNeutrino are present, so rho_extra = 0 and the whole Neff maps
    # to massless neutrinos: N_nu_massless ~ 3 > 0.
    valid = {"Neff": jnp.asarray(3.044), "T_nu_massless": jnp.asarray(0.71636856)}
    _n_massless_from_neff(
        valid,
        [_Fluid("Photon", rho_early=1.0), _Fluid("MasslessNeutrino", rho_early=1.0)],
    )
    assert float(valid["N_nu_massless"]) == pytest.approx(3.0, abs=0.05)


def test_bbn_type_validation():
    # The schema's choices check warns; interpretation refuses to guess: an
    # uninterpretable bbn_type raises instead of silently meaning "no BBN".
    from abcmb.derived import _bbn_type

    assert _bbn_type({"bbn_type": "Table"}) == "table"  # case-insensitive
    assert _bbn_type({"bbn_type": ""}) == ""
    with pytest.raises(ValueError, match="bbn_type='tabel'"):
        _bbn_type({"bbn_type": "tabel"})


def test_sbbn_table_out_of_range_warns():
    # Out-of-table YHe queries are extrapolated nuclear physics: warn (never
    # raise -- wide-prior scans are legitimate), in range stays silent.
    import jax.numpy as jnp
    import numpy as np

    from abcmb.derived import _helium_from_table

    # Synthetic table with the hardcoded sBBN layout (13 x 701) and a linear
    # YHe surface, so extrapolation is exactly reproducible.
    n2, n1 = 13, 701
    ob = np.linspace(0.005, 0.04, n1)
    dn = np.linspace(-3.0, 3.0, n2)
    yhe = 0.2 + 1.0 * np.tile(ob, n2) + 0.01 * np.repeat(dn, n1)
    table = jnp.asarray(np.column_stack([np.tile(ob, n2), np.repeat(dn, n1), yhe]))

    params = {"omega_b": jnp.asarray(0.022), "Neff": jnp.asarray(3.046)}
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # in range: must NOT warn
        _helium_from_table(params, table)
    assert float(params["YHe"]) == pytest.approx(0.2 + 0.022, rel=1e-12)

    params = {"omega_b": jnp.asarray(0.05), "Neff": jnp.asarray(3.046)}  # > table max
    with pytest.warns(UserWarning, match="outside the sBBN table"):
        _helium_from_table(params, table)
    assert float(params["YHe"]) == pytest.approx(0.2 + 0.05, rel=1e-10)  # linear extrap
