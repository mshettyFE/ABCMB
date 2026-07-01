"""
Command-line interface for ABCMB.

Drives the full pipeline (:class:`abcmb.main.Model`) from the shell without a
notebook.  Cosmological parameters and model/config ``specs`` may be supplied
either through a config file (``--config``, JSON or YAML) or through repeatable
``--param KEY=VALUE`` / ``--spec KEY=VALUE`` flags (the latter override the
former).  Results (multipoles, power spectra) are written to disk.

Examples
--------
Run with all defaults and save to ``out.npz``::

    abcmb -o out.npz

Override a couple of parameters and enable lensing::

    abcmb -p omega_cdm=0.12 -p h=0.68 -s lensing=true -o out.npz

Drive everything from a config file, writing text output::

    abcmb --config cosmo.yaml --format txt -o run/spectra
"""

import argparse
import json
import os

from . import version


def _parse_scalar(text):
    """
    Parse a command-line ``KEY=VALUE`` value into a typed Python scalar.

    Tries, in order: boolean keyword, int, float; falls back to the raw string.

    Parameters
    ----------
    text : str
        The value portion of a ``KEY=VALUE`` token.

    Returns
    -------
    bool | int | float | str
        The value coerced to the most specific matching type.
    """
    lowered = text.strip().lower()
    if lowered in ("true", "yes"):
        return True
    if lowered in ("false", "no"):
        return False
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def _parse_key_value(pairs):
    """
    Turn a list of ``"KEY=VALUE"`` strings into a dict of typed scalars.

    Parameters
    ----------
    pairs : list[str] | None
        Tokens of the form ``KEY=VALUE`` (e.g. from ``--param``/``--spec``).

    Returns
    -------
    dict
        Mapping of key to parsed scalar value.

    Raises
    ------
    ValueError
        If a token does not contain ``=``.
    """
    out = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"Expected KEY=VALUE but got '{pair}'.")
        key, value = pair.split("=", 1)
        out[key.strip()] = _parse_scalar(value)
    return out


def _load_config(path):
    """
    Load a JSON or YAML config file into ``(specs, params)`` dicts.

    The file may either be flat (all keys under top level, interpreted as a
    combined dict split into the ``specs``/``params`` sections below) or use
    explicit top-level ``specs`` and ``params`` sections.  Explicit sections
    are recommended.

    Parameters
    ----------
    path : str
        Path to a ``.json``, ``.yaml`` or ``.yml`` file.

    Returns
    -------
    tuple[dict, dict]
        ``(specs, params)`` dictionaries (either may be empty).

    Raises
    ------
    ValueError
        If the extension is unsupported or YAML is requested but unavailable.
    """
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r") as handle:
        if ext in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover - optional dep
                raise ValueError(
                    "PyYAML is required to read YAML config files; "
                    "install it or use a JSON config instead."
                ) from exc
            data = yaml.safe_load(handle) or {}
        elif ext == ".json":
            data = json.load(handle)
        else:
            raise ValueError(
                f"Unsupported config extension '{ext}'; use .json, .yaml or .yml."
            )

    if not isinstance(data, dict):
        raise ValueError("Config file must contain a top-level mapping.")

    specs = dict(data.get("specs", {}))
    params = dict(data.get("params", {}))
    return specs, params


def _write_output(output, path, fmt, inputs):
    """
    Serialize a :class:`abcmb.main.Output` bundle to disk.

    Parameters
    ----------
    output : abcmb.main.Output
        Result of a model evaluation.
    path : str
        Output path.  For ``txt`` this is treated as a basename and files
        (``<path>_cl.txt``, ``<path>_pk.txt`` and ``<path>_inputs.json``) are
        written, since the Cl and Pk grids have different lengths.
    fmt : str
        One of ``"npz"``, ``"txt"``, ``"json"``.
    inputs : dict
        The resolved inputs that were passed to the run, i.e.
        ``{"specs": ..., "params": ...}`` after merging the config file with
        any CLI overrides.  Stored alongside the results for provenance.

    Returns
    -------
    list[str]
        Paths actually written.
    """
    import numpy as np

    inputs_json = json.dumps(inputs)

    l = np.asarray(output.l)
    ClTT = np.asarray(output.ClTT)
    ClTE = np.asarray(output.ClTE)
    ClEE = np.asarray(output.ClEE)
    k = np.asarray(output.k)
    Pk = np.asarray(output.Pk)

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    if fmt == "npz":
        if not path.endswith(".npz"):
            path += ".npz"
        # ``inputs`` is stored as a 0-d JSON string array; recover it on load
        # with ``json.loads(str(np.load(path)["inputs"]))``.
        np.savez(
            path,
            l=l,
            ClTT=ClTT,
            ClTE=ClTE,
            ClEE=ClEE,
            k=k,
            Pk=Pk,
            inputs=np.array(inputs_json),
        )
        return [path]

    if fmt == "txt":
        base = path[:-4] if path.endswith(".txt") else path
        cl_path, pk_path = base + "_cl.txt", base + "_pk.txt"
        inputs_path = base + "_inputs.json"
        np.savetxt(
            cl_path,
            np.column_stack([l, ClTT, ClTE, ClEE]),
            header="l ClTT ClTE ClEE",
        )
        np.savetxt(pk_path, np.column_stack([k, Pk]), header="k Pk")
        with open(inputs_path, "w") as handle:
            json.dump(inputs, handle, indent=2)
        return [cl_path, pk_path, inputs_path]

    if fmt == "json":
        if not path.endswith(".json"):
            path += ".json"
        with open(path, "w") as handle:
            json.dump(
                {
                    "inputs": inputs,
                    "l": l.tolist(),
                    "ClTT": ClTT.tolist(),
                    "ClTE": ClTE.tolist(),
                    "ClEE": ClEE.tolist(),
                    "k": k.tolist(),
                    "Pk": Pk.tolist(),
                },
                handle,
            )
        return [path]

    raise ValueError(f"Unknown output format '{fmt}'.")


def build_parser():
    """
    Construct the argument parser for the ``abcmb`` command.

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="abcmb",
        description="Run the ABCMB differentiable CMB Boltzmann solver.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Parameter/spec values are typed automatically: 'true'/'false' -> bool,\n"
            "integers and floats are parsed as such, everything else stays a string.\n"
            "CLI --param/--spec flags override values from --config."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"ABCMB {version.__version__}"
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="JSON/YAML config file with optional 'specs' and 'params' sections.",
    )
    parser.add_argument(
        "-p",
        "--param",
        action="append",
        metavar="KEY=VALUE",
        help="Cosmological parameter override (repeatable), e.g. -p omega_cdm=0.12.",
    )
    parser.add_argument(
        "-s",
        "--spec",
        action="append",
        metavar="KEY=VALUE",
        help="Model/config spec override (repeatable), e.g. -s lensing=true.",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        default="abcmb_out.npz",
        help="Where to write results (default: abcmb_out.npz).",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=("npz", "txt", "json"),
        default="npz",
        help="Output format (default: npz).",
    )
    return parser


def main(argv=None):
    """
    Entry point for the ``abcmb`` console script.

    Parameters
    ----------
    argv : list[str] | None
        Argument vector (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    specs, params = ({}, {})
    if args.config:
        specs, params = _load_config(args.config)

    specs.update(_parse_key_value(args.spec))
    params.update(_parse_key_value(args.param))

    # Resolved inputs (config file merged with CLI overrides), recorded with the
    # results so every run is self-documenting.
    inputs = {"specs": specs, "params": params}
    print(f"Resolved specs:  {specs}")
    print(f"Resolved params: {params}")

    # Imported here (not at module top) so that --help/--version stay fast and
    # do not pay the JAX import/compile cost.
    from .main import Model

    model = Model(**specs)
    output = model(params)

    import numpy as np

    l = np.asarray(output.l)
    ClTT = np.asarray(output.ClTT)
    print(f"Computed spectra over l = {int(l[0])}..{int(l[-1])} ({l.size} multipoles).")
    print(f"ClTT[:3] = {ClTT[:3]}")
    print(f"Pk grid: {np.asarray(output.k).size} wavenumbers.")

    written = _write_output(output, args.output, args.format, inputs)
    print("Wrote: " + ", ".join(written))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
