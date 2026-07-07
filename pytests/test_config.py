"""
Config / CLI tests: TOML routing and loading (``load_config``, ``route``), the
``abcmb`` command (``--list-params`` / ``--dump-defaults``), and the notebook-facing
config front door.

Schema option/param resolution lives in ``test_schema.py``; environment capture and
run-file reproducibility (``save_run`` / ``write_run_toml``) in ``test_provenance.py``.
"""

from abcmb import config
from abcmb.cli import _parse_key_value
from abcmb.config import load_config
from abcmb.schema import route


def test_toml_routes_by_name_not_table(tmp_path):
    cfg = tmp_path / "run.toml"
    cfg.write_text(
        "[cosmology]\n"
        "omega_cdm = 0.12\n"
        "N_idr = 0.3\n"  # unknown custom param -> routed to params
        "[output]\n"
        "l_max = 200\n"
        "l_max_ur = 15\n"  # option alias, deliberately misfiled under [output]
    )
    options, params, env = load_config(str(cfg))
    # Keys route by schema membership, not by which table they sit in.
    assert options == {"l_max": 200, "l_max_ur": 15}
    assert params == {"omega_cdm": 0.12, "N_idr": 0.3}
    assert env is None  # no [environment] table in a hand-written config


def test_read_config_explicit_mode(tmp_path):
    cfg = tmp_path / "explicit.toml"
    cfg.write_text("[params]\nomega_cdm = 0.12\n[options]\nl_max = 200\n")
    options, params, env = load_config(str(cfg))
    assert options == {"l_max": 200}
    assert params == {"omega_cdm": 0.12}
    assert env is None


def test_route_cli_assignments():
    # Positional KEY=VALUE assignments route to options vs params by name, exactly
    # like a config file (no --param/--spec distinction needed).
    flat = _parse_key_value(
        ["omega_cdm=0.12", "lensing=true", "l_max_ur=20", "N_idr=0.3"]
    )
    options, params = route(flat)
    assert options == {"lensing": True, "l_max_ur": 20}  # spec name + alias
    assert params == {"omega_cdm": 0.12, "N_idr": 0.3}  # cosmology + custom


def test_list_params_reference(capsys):
    from abcmb.cli import main

    rc = main(["--list-params"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "omega_cdm" in out  # a declared cosmological parameter
    assert "l_max" in out  # a declared spec
    assert "Neff" in out  # a conditional/BBN param is listed
    assert "[cosmology]" in out  # group heading (the 'group' field, in use)
    assert "aliases: T_ncdm" in out  # CLASS alias surfaced


def test_dump_defaults_roundtrips(tmp_path, capsys):
    # --dump-defaults prints valid TOML that load_config parses back to exactly the
    # schema defaults (routed by name), so the dumped config is faithful.
    from abcmb.cli import main
    from abcmb.schema import OPTION_SCHEMA, PARAM_SCHEMA

    rc = main(["--dump-defaults"])
    out = capsys.readouterr().out
    assert rc == 0

    path = tmp_path / "d.toml"
    path.write_text(out)
    options, params, env = load_config(str(path))
    assert options == {s.name: s.default for s in OPTION_SCHEMA}
    assert params == {s.name: s.default for s in PARAM_SCHEMA}
    assert env is None


def test_generated_artifacts_are_fresh():
    # defaults.toml and abcmb/_schema_types.py are generated from the schema and
    # committed. This verifies they're up to date (CI/check.sh only *verify*;
    # regenerate with `./check.sh fix` or `abcmb --dump-{defaults,types}`).
    from pathlib import Path

    from abcmb._codegen import dump_types
    from abcmb.config import dump_defaults

    root = Path(__file__).parents[1]
    assert dump_defaults() == (root / "defaults.toml").read_text(), (
        "defaults.toml is stale -- run ./check.sh fix"
    )
    assert dump_types() == (root / "abcmb" / "_schema_types.py").read_text(), (
        "abcmb/_schema_types.py is stale -- run ./check.sh fix"
    )


def test_public_config_api(tmp_path):
    # The notebook-facing front door lives in abcmb.config (a plain module, so
    # `import abcmb` stays jax-free until you reach for the file I/O).
    cfg = tmp_path / "cosmo.toml"
    cfg.write_text("[cosmology]\nomega_cdm = 0.12\n[output]\nl_max = 200\n")
    options, params, env = config.load_config(str(cfg))
    assert options == {"l_max": 200}
    assert params == {"omega_cdm": 0.12}
    assert env is None
    assert callable(config.save_run) and callable(config.model_from_config)


def test_import_abcmb_stays_jax_free():
    # Regression for the reason abcmb.config exists: importing the package must not
    # pull in JAX (keeps `abcmb --version`/`--help` and a bare import fast).
    import subprocess
    import sys

    code = "import abcmb, sys; assert 'jax' not in sys.modules"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
