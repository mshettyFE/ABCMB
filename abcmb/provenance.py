"""
Run provenance: capture what a run used, plus enough environment to detect drift.

The run file (``<output>_run.toml``) records the *raw* user inputs (replayable:
it is itself a valid ``--config``) plus a best-effort
environment stamp. It deliberately does NOT attempt bit-for-bit reproduction:
JAX/XLA output is not bit-stable across device / jaxlib / XLA flags, and physics
runs typically happen in a dirty working tree. The goal is push-button re-run
plus drift detection — so a surprising result is never silently blamed on physics
when the code, a dependency, or the device actually changed. Per-key resolution
provenance (default/user/alias/extra) is not persisted here; it stays available
in memory via ``Model.options_provenance`` / ``Model.param_provenance``.
"""

import importlib.metadata
import os
import platform
import subprocess
from dataclasses import asdict, dataclass, fields

import jax
import numpy as np

from . import version

MANIFEST_VERSION = 1


def _git_info(cwd=None):
    """
    Best-effort git commit + dirty flag for a directory.

    Returns ``{"commit": <sha>, "dirty": <bool>}`` when run inside a git checkout,
    or ``None`` if the directory is not a repo or git is unavailable. Never raises.
    """
    cwd = cwd or os.getcwd()

    def _run(args):
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )

    try:
        head = _run(["rev-parse", "HEAD"])
        if head.returncode != 0:
            return None
        status = _run(["status", "--porcelain"])
        return {
            "commit": head.stdout.strip(),
            "dirty": bool(status.stdout.strip()),
        }
    except (OSError, subprocess.SubprocessError):
        return None


def _pkg_version(name):
    """Installed version of a package, or None if not found."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _to_py(value):
    """Convert a jnp/np scalar or array to a JSON-serializable Python object."""
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError):
        return value
    if arr.ndim == 0:
        return arr.item()
    return arr.tolist()


@dataclass(frozen=True)
class Environment:
    """
    The runtime facts captured for a run, used for drift detection.

    Every field is best-effort; ``None`` means
    "unavailable" (e.g. ``git_commit`` / ``git_dirty`` when the run directory is
    not a git checkout, or ``device`` / ``x64`` if JAX could not be queried).
    """

    abcmb_version: str | None = None  # installed ABCMB version
    git_commit: str | None = None  # HEAD sha, or None outside a checkout
    git_dirty: bool | None = None  # uncommitted changes present in the tree
    jax: str | None = None  # jax package version
    jaxlib: str | None = None  # jaxlib package version
    device: str | None = None  # jax default backend (cpu / gpu / tpu)
    x64: bool | None = None  # jax x64 (double precision) mode enabled
    python: str | None = None  # python interpreter version

    def to_flat(self):
        """Flat ``{key: value}`` for the TOML table, dropping ``None`` (no TOML null)."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_flat(cls, table):
        """Rebuild from a TOML ``[environment]`` table (missing keys become ``None``)."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in table.items() if k in names})


def capture_environment():
    """Best-effort snapshot of the runtime :class:`Environment` (never raises)."""
    try:
        device = jax.default_backend()
    except Exception:
        device = None
    try:
        x64 = bool(jax.config.jax_enable_x64)
    except Exception:
        x64 = None
    git = _git_info()  # {"commit", "dirty"} or None outside a git checkout
    return Environment(
        abcmb_version=version.__version__,
        git_commit=git["commit"] if git else None,
        git_dirty=git["dirty"] if git else None,
        jax=_pkg_version("jax"),
        jaxlib=_pkg_version("jaxlib"),
        device=device,
        x64=x64,
        python=platform.python_version(),
    )


def _toml_scalar(value):
    """Render a scalar (or list of scalars) as a TOML value."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        escaped = (
            value.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\t", "\\t")
        )
        return f'"{escaped}"'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_scalar(v) for v in value) + "]"
    raise TypeError(f"cannot serialize {type(value).__name__} to TOML")


def _write_toml_table(name, table, handle):
    """Write one flat ``[name]`` table, skipping ``None`` values (TOML has no null)."""
    handle.write(f"[{name}]\n")
    for key, value in table.items():
        value = _to_py(value)
        if value is None:
            continue
        handle.write(f"{key} = {_toml_scalar(value)}\n")
    handle.write("\n")


def write_run_toml(run_data, handle):
    """
    Write a run file: flat TOML with ``[environment]``, ``[params]``, ``[options]``,
    ``[run]``.

    Parameters
    ----------
    run_data : dict
        ``{"environment": Environment, "params": dict, "options": dict,
        "warnings": list[str]}``.
    handle : text-mode file object
    """
    handle.write("# ABCMB run file. Reproduce with: abcmb --config <this file>\n")
    handle.write("# The [params]/[options] tables below are also a valid --config.\n\n")
    _write_toml_table("environment", run_data["environment"].to_flat(), handle)
    _write_toml_table("params", run_data.get("params", {}), handle)
    _write_toml_table("options", run_data.get("options", {}), handle)
    _write_toml_table(
        "run",
        {
            "manifest_version": MANIFEST_VERSION,
            "warnings": list(run_data.get("warnings", [])),
        },
        handle,
    )


def check_drift(recorded):
    """
    Compare a recorded :class:`Environment` against the current one.

    Returns a list of human-readable drift warnings (empty if nothing moved).
    """
    current = capture_environment()
    drift = []

    if recorded.git_dirty:
        drift.append(
            "recorded run was made from a DIRTY working tree; "
            "it is not reproducible from git."
        )
    if current.git_dirty:
        drift.append("current working tree is DIRTY; replay output may not match.")
    if (
        recorded.git_commit
        and current.git_commit
        and recorded.git_commit != current.git_commit
    ):
        drift.append(
            f"git commit differs: recorded {recorded.git_commit[:10]} "
            f"vs current {current.git_commit[:10]}."
        )

    for attr, label in (
        ("abcmb_version", "ABCMB version"),
        ("jaxlib", "jaxlib"),
        ("device", "device"),
        ("x64", "x64"),
    ):
        if getattr(recorded, attr) != getattr(current, attr):
            drift.append(
                f"{label} differs: recorded {getattr(recorded, attr)!r} "
                f"vs current {getattr(current, attr)!r}."
            )
    return drift
