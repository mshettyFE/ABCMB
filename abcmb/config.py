"""
File-driven entry points for ABCMB, shared by the CLI and notebooks.

Drive a run from a TOML file (:func:`load_config`, :func:`model_from_config`),
write a run's outputs plus a reproducible run file (:func:`save_run`), or emit the
schema defaults as a starter config (:func:`dump_defaults`).

"""

import os

import numpy as np
import tomlkit

from . import provenance, schema


def load_config(path):
    """
    Load a TOML config file (or a saved ``<out>_run.toml``) into
    ``(options, params, environment)`` — the shared front door for driving a run
    from a file, from a notebook or the CLI.

    Two shapes handled by one loader:

    * **Explicit** — a file with ``[params]`` / ``[options]`` tables (as written by
      a run) uses them directly; this is how a run file replays as a ``--config``.
    * **Routing** — otherwise every non-reserved table is flattened and its keys
      routed to options vs params by schema membership (:func:`schema.route`); a key
      in the "wrong" table still routes correctly since ABCMB names are unique.

    The reserved ``[environment]`` / ``[run]`` tables never contribute inputs;
    ``[environment]`` (if present) is returned as an :class:`~abcmb.provenance.Environment`
    so a replay can be drift-checked. Returns ``(options, params, environment_or_None)``.
    """
    with open(path, encoding="utf-8") as handle:
        # unwrap() -> plain Python types (tomlkit otherwise yields wrapper objects
        # whose identity/dict comparisons differ from stdlib values downstream).
        data = tomlkit.load(handle).unwrap()

    environment = None
    if "environment" in data:
        environment = provenance.Environment.from_flat(data["environment"])

    if "params" in data or "options" in data:
        return dict(data.get("options", {})), dict(data.get("params", {})), environment

    flat = {}
    for key, value in data.items():
        if key in ("environment", "run"):
            continue
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    options, params = schema.route(flat)
    return options, params, environment


def model_from_config(path):
    """
    Build a Model from a TOML config (or a saved ``<out>_run.toml``) and return
    ``(model, params)``, ready to call as ``model(params)``.

    The file-driven counterpart to ``Model(**options)`` — the notebook mirror of the
    ``abcmb --config`` CLI path. Load ``(options, params)`` yourself with
    :func:`load_config` if you need the recorded environment for a drift check.
    """
    from .main import Model

    options, params, _environment = load_config(path)
    return Model(**options), params


def dump_defaults() -> str:
    """
    Render every option and parameter with its schema default as a human-readable
    TOML config, grouped into topical tables (by ``Spec.group``).

    Keys route to options vs params by *name* on load, so the table placement is
    cosmetic; each key is tagged ``(param)`` / ``(option)`` with its type and doc.
    Entries with no fixed default (``UNSET``: conditional inputs and derived
    quantities) are omitted. The result is a valid ``--config`` — the starting
    point printed by ``abcmb --dump-defaults``.
    """
    groups_order, group_rows = [], {}

    def add(specs, tag):
        for spec in specs:
            if spec.default is schema.UNSET:
                continue  # no value to emit; listed in the header below
            group = str(spec.group)
            if group not in group_rows:
                group_rows[group] = []
                groups_order.append(group)
            group_rows[group].append((spec, tag))

    add(schema.PARAM_SCHEMA, "param")
    add(schema.OPTION_SCHEMA, "option")

    conditional = " / ".join(
        spec.name
        for spec in schema.PARAM_SCHEMA
        if spec.default is schema.UNSET and not spec.derived
    )
    derived = " / ".join(spec.name for spec in schema.PARAM_SCHEMA if spec.derived)

    doc = tomlkit.document()
    for line in (
        "ABCMB default configuration -- every option and parameter with its schema default.",
        "",
        "Tables are topical (grouped by schema 'group'). On load (model_from_config or",
        "`abcmb --config`) keys route to options vs params by *name*, so a key's table",
        "is purely for human readability; each key is tagged (param) or (option) below.",
        "",
        "Omitted (no fixed default):",
        f"  conditional inputs (supplied only when needed): {conditional}",
        f"  derived at runtime (computed, not inputs):      {derived}",
        "NOTE: Providing derived values on the CLI will get overridden",
        "",
        "Usage:  abcmb --config defaults.toml -o out.npz",
        "        from abcmb.config import model_from_config",
        "        model, params = model_from_config('defaults.toml')",
    ):
        doc.add(tomlkit.comment(line))
    doc.add(tomlkit.nl())

    for group in groups_order:
        rows = [
            (
                spec.name,
                tomlkit.item(spec.default),
                f"({tag}, {spec.kind.__name__}) {spec.doc}",
            )
            for spec, tag in group_rows[group]
        ]
        # Pretty formatting stuff.
        # Align the inline-comment column: pad each row up to the widest
        # "key = value" prefix (tomlkit fixes the spacing around '=' at one space,
        # so the '=' itself is not hand-aligned, only the trailing '#' comments).
        left_w = max(
            len(name) + len(" = ") + len(val.as_string()) for name, val, _ in rows
        )
        tbl = tomlkit.table()
        for name, val, cmt in rows:
            pad = left_w - (len(name) + len(" = ") + len(val.as_string()))
            val.comment(cmt)
            val.trivia.comment_ws = " " * pad + "  "
            tbl[name] = val
        doc[group] = tbl

    text = tomlkit.dumps(doc)
    # Blank header lines render as "# " -- strip the trailing space per line.
    return "\n".join(line.rstrip() for line in text.splitlines()) + "\n"


def save_run(output, path, model, params, *, warnings=()):
    """
    Write the reproducible run artifact for a run of ``model`` on ``params``:
    ``<stem>.npz`` (the spectra) and ``<stem>_run.toml`` (raw inputs + environment
    stamp). Usable from a notebook or the CLI.

    The ``_run.toml`` is itself a valid ``--config`` for replay. Returns the list of
    paths written.

    Parameters
    ----------
    output : Output
        The result to save (its ``l``/``ClTT``/``ClTE``/``ClEE``/``k``/``Pk``).
    path : str
        Output path; a ``.npz`` suffix is added if missing.
    model : Model
        The model that produced ``output`` (read for ``raw_options``).
    params : dict
        The raw parameters passed to the model call.
    warnings : iterable[str], optional
        Warning messages to record in the run file.
    """
    run_data = {
        "environment": provenance.capture_environment(),
        "options": dict(model.raw_options),
        "params": dict(params),
        "warnings": list(warnings),
    }

    if not path.endswith(".npz"):
        path += ".npz"
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    run_path = path[:-4] + "_run.toml"
    with open(run_path, "w") as handle:
        provenance.write_run_toml(run_data, handle)

    np.savez(
        path,
        l=np.asarray(output.l),
        ClTT=np.asarray(output.ClTT),
        ClTE=np.asarray(output.ClTE),
        ClEE=np.asarray(output.ClEE),
        k=np.asarray(output.k),
        Pk=np.asarray(output.Pk),
    )
    return [path, run_path]
