"""
Guards for the ``abcmb`` console script.
"""

import warnings

import pytest

from abcmb import cli


def test_parse_scalar_value_types():
    # KEY=VALUE arrives as text; the CLI has to recover the intended type or
    # the schema rejects perfectly good input (bool options, int l_max).
    assert cli._parse_scalar_value("true") is True
    assert cli._parse_scalar_value("Yes") is True
    assert cli._parse_scalar_value("false") is False
    assert cli._parse_scalar_value("no") is False
    assert cli._parse_scalar_value("2500") == 2500
    assert isinstance(cli._parse_scalar_value("2500"), int)
    assert cli._parse_scalar_value("0.6762") == pytest.approx(0.6762)
    # Anything that is not a bool/int/float stays a string (e.g. bbn_type).
    assert cli._parse_scalar_value("Table") == "Table"


def test_parse_key_value_pairs_and_error():
    got = cli._parse_key_value(["l_max=300", "lensing=false", "h=0.7"])
    assert got == {"l_max": 300, "lensing": False, "h": 0.7}
    # An '=' is the whole grammar; a bare token is a user typo worth naming.
    with pytest.raises(ValueError, match="Expected KEY=VALUE"):
        cli._parse_key_value(["l_max 300"])
    assert cli._parse_key_value(None) == {}


def test_discovery_flags_exit_zero(capsys):
    # --list-params / --dump-defaults are the documented way to discover the
    # schema; they must not need JAX or a solve.
    assert cli.main(["--list-params"]) == 0
    listed = capsys.readouterr().out
    assert "omega_b" in listed and "l_max" in listed

    assert cli.main(["--dump-defaults"]) == 0
    dumped = capsys.readouterr().out
    assert "[cosmology]" in dumped or "omega_b" in dumped


@pytest.mark.slow
def test_cli_end_to_end_writes_run_artifact(tmp_path, capsys):
    # The full CLI path: config file -> Model -> solve -> npz + _run.toml.
    # Deliberately tiny (l_max=50, no lensing): this checks the plumbing
    cfg = tmp_path / "cosmo.toml"
    cfg.write_text(
        "[cosmology]\nomega_cdm = 0.12\n[output]\nl_max = 50\nk_max = 0.02\n"
    )
    out = tmp_path / "run.npz"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rc = cli.main(["--config", str(cfg), "-o", str(out)])
    assert rc == 0

    printed = capsys.readouterr().out
    assert "Computed spectra" in printed
    assert "Wrote:" in printed

    # Both artifacts land, and the run file replays as a config.
    assert out.exists()
    run_toml = tmp_path / "run_run.toml"
    assert run_toml.exists()
    text = run_toml.read_text()
    assert "[environment]" in text
    assert "[params]" in text
