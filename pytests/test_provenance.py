"""
Provenance tests: environment capture (degrading gracefully outside a git checkout),
the run-file writer (``write_run_toml``), and the shared ``save_run`` reproducibility
entry point -- including that a written run file round-trips and is itself a valid
``--config`` for replay.

Option/param resolution lives in ``test_schema.py``; config loading + the CLI in
``test_config.py``.
"""

from abcmb.inputs import config, provenance
from abcmb.inputs.config import load_config


def test_environment_degrades_outside_git_repo(tmp_path):
    # tmp_path is not a git checkout -> git info is None, not an error.
    assert provenance._git_info(cwd=str(tmp_path)) is None
    env = provenance.capture_environment()
    assert isinstance(env, provenance.Environment)
    assert env.abcmb_version
    assert env.python


def test_run_toml_omits_none_git(tmp_path):
    import tomlkit

    # An Environment with no git info (git_* default to None) -> "not a checkout".
    env = provenance.Environment(abcmb_version="0.0.0", python="3.x")
    with open(tmp_path / "r.toml", "w") as handle:
        provenance.write_run_toml(
            {"environment": env, "options": {}, "params": {}, "warnings": []}, handle
        )
    with open(tmp_path / "r.toml", encoding="utf-8") as handle:
        data = tomlkit.load(handle).unwrap()
    # No null in TOML -> git_commit/git_dirty simply omitted, and reconstruct to None.
    assert "git_commit" not in data["environment"]
    assert provenance.Environment.from_flat(data["environment"]).git_commit is None
    # A hand-edited run file with an unknown key must not be fatal.
    tolerant = provenance.Environment.from_flat({**data["environment"], "future": 1})
    assert tolerant.abcmb_version == "0.0.0"


def test_run_toml_roundtrip_and_config_compatible(tmp_path):
    import tomlkit

    run_data = {
        "environment": provenance.capture_environment(),
        "options": {"l_max": 200, "lensing": False},
        "params": {"omega_cdm": 0.12, "bbn_type_note": "keeps 'quotes' & tabs\tok"},
        "warnings": ["spec 'l_max_ur' is an alias for 'l_max_massless_nu'"],
    }
    path = tmp_path / "out_run.toml"
    with open(path, "w") as handle:
        provenance.write_run_toml(run_data, handle)

    # Valid TOML that round-trips.
    with open(path, encoding="utf-8") as handle:
        data = tomlkit.load(handle).unwrap()
    assert data["params"]["omega_cdm"] == 0.12
    assert data["options"]["lensing"] is False
    assert "l_max_ur" in data["run"]["warnings"][0]

    # The run file is ALSO a valid --config (explicit [params]/[options] mode), and
    # its [environment] is returned for drift checks.
    options, params, env = load_config(str(path))
    assert options == run_data["options"]
    assert params == run_data["params"]
    # env is an Environment reconstructed from the [environment] table.
    assert env.abcmb_version == run_data["environment"].abcmb_version


def test_save_run_artifact(tmp_path):
    # save_run is the shared (notebook + CLI) reproducibility entry point. Exercise
    # it with lightweight stand-ins for Output/Model (no solver, no compile).
    import types

    import numpy as np
    import tomlkit

    output = types.SimpleNamespace(
        l=np.array([2, 3]),
        ClTT=np.array([1.0, 2.0]),
        ClTE=np.array([0.0, 0.0]),
        ClEE=np.array([0.1, 0.2]),
        k=np.array([0.01]),
        Pk=np.array([1e3]),
    )
    model = types.SimpleNamespace(
        raw_options={"l_max": 200},
        species_list=(
            types.SimpleNamespace(name="Baryon"),
            types.SimpleNamespace(name="Photon"),
        ),
    )
    paths = config.save_run(output, str(tmp_path / "run"), model, {"omega_cdm": 0.12})
    assert paths == [str(tmp_path / "run.npz"), str(tmp_path / "run_run.toml")]

    with open(tmp_path / "run_run.toml", encoding="utf-8") as handle:
        data = tomlkit.load(handle).unwrap()
    assert data["options"] == {"l_max": 200}  # exactly what the user passed
    assert data["run"]["species"] == ["Baryon", "Photon"]  # stack recorded
    assert data["params"] == {"omega_cdm": 0.12}
    assert data["environment"]["abcmb_version"]  # a fresh stamp is present

    npz = np.load(tmp_path / "run.npz")
    assert list(npz["l"]) == [2, 3]
    assert list(npz["Pk"]) == [1e3]


def test_save_run_reads_declared_output_model_fields():
    # test_save_run_artifact uses SimpleNamespace stand-ins for Output/Model, which
    # duck-type the interface and so can't see a field rename. Pin the field names
    # save_run reads against the real (eqx.Module) classes, so a rename fails in CI
    # rather than on the first real `abcmb` run / notebook save_run.
    import dataclasses

    from abcmb.main import Model, Output

    assert {"l", "ClTT", "ClTE", "ClEE", "k", "Pk"} <= {
        f.name for f in dataclasses.fields(Output)
    }
    read_fields = {"raw_options", "species_list"}
    assert read_fields <= {f.name for f in dataclasses.fields(Model)}


def test_to_py_converts_arrays_and_passes_through_others():
    import jax.numpy as jnp
    import numpy as np

    # Scalars unwrap to Python numbers, arrays to lists -- TOML has neither
    # numpy scalars nor ndarrays.
    assert provenance._to_py(jnp.float64(2.5)) == 2.5
    assert isinstance(provenance._to_py(jnp.float64(2.5)), float)
    assert provenance._to_py(np.array([1.0, 2.0])) == [1.0, 2.0]
    assert provenance._to_py("Table") == "Table"


def test_check_drift_flags_dirty_tree_and_commit_change():
    current = provenance.capture_environment()

    # A run recorded from a dirty tree is not reproducible from git.
    dirty = provenance.Environment(
        abcmb_version=current.abcmb_version,
        git_commit=current.git_commit,
        git_dirty=True,
    )
    msgs = " ".join(provenance.check_drift(dirty))
    assert "DIRTY" in msgs

    # A different commit is drift even when everything else matches.
    moved = provenance.Environment(
        abcmb_version=current.abcmb_version,
        git_commit="0" * 40,
        git_dirty=False,
    )
    msgs = " ".join(provenance.check_drift(moved))
    assert "0" * 40 in msgs or "commit" in msgs.lower()


def test_check_drift_quiet_when_environment_matches():
    # Self-comparison is the control: apart from the current tree's own dirty
    # state, replaying the environment you are in reports nothing.
    current = provenance.capture_environment()
    msgs = [m for m in provenance.check_drift(current) if "DIRTY" not in m]
    assert msgs == [], f"unexpected drift against self: {msgs}"
