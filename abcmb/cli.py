"""
Command-line interface for ABCMB.

Each run writes its results plus a flat TOML run file (``<output>_run.toml``)
recording the raw inputs and a best-effort environment stamp. That file is itself
a valid ``--config``: pass it back to reproduce the run — its recorded environment
is drift-checked automatically against the current one.

Examples
--------
Run with all defaults::

    abcmb -o out.npz

Override a couple of values::

    abcmb omega_cdm=0.12 h=0.68 lensing=true -o out.npz

Drive everything from a TOML config::

    abcmb --config cosmo.toml -o run/spectra

Reproduce a previous run (drift-checked automatically)::

    abcmb --config out_run.toml -o rerun.npz
"""

import argparse
import warnings

from . import version


def _parse_scalar_value(text):
    """
    Parse a command-line ``KEY=VALUE`` value into a typed Python scalar.

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

    Raises ``ValueError`` if a token lacks ``=``.
    """
    out = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"Expected KEY=VALUE but got '{pair}'.")
        key, value = pair.split("=", 1)
        out[key.strip()] = _parse_scalar_value(value)
    return out


def build_parser():
    """Construct the argument parser for the ``abcmb`` command."""
    parser = argparse.ArgumentParser(
        prog="abcmb",
        description="Run the ABCMB differentiable CMB Boltzmann solver.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "KEY=VALUE assignments are routed to parameters vs model options by\n"
            "name (like a config file) and override values from --config. Values\n"
            "are typed automatically: 'true'/'false' -> bool, ints/floats are\n"
            "parsed as such, everything else stays a string."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"ABCMB {version.__version__}"
    )
    parser.add_argument(
        "--list-params",
        action="store_true",
        help="List all parameters and options (with defaults, types, aliases) and exit.",
    )
    parser.add_argument(
        "--dump-defaults",
        action="store_true",
        help="Print every option/parameter with its default as a TOML config, then "
        "exit (redirect to save: abcmb --dump-defaults > defaults.toml).",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="TOML config file. A previous run's <out>_run.toml also works here "
        "and is drift-checked automatically (that is the 'replay' path).",
    )
    parser.add_argument(
        "assignments",
        nargs="*",
        metavar="KEY=VALUE",
        help="Parameter or option assignments, routed by name "
        "(e.g. omega_cdm=0.12 lensing=true).",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        default="abcmb_out.npz",
        help="Where to write the .npz spectra (default: abcmb_out.npz). A "
        "<stem>_run.toml is written alongside.",
    )
    return parser


def main(argv=None):
    """
    Entry point for the ``abcmb`` console script.

    Returns the process exit code (0 on success).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Deferred until after parse_args: schema/config pull in JAX transitively,
    # so --help/--version above stay instant -- and still work in an environment
    # where JAX itself fails to import (broken jaxlib/CUDA).
    from .inputs import config, provenance, schema

    # Discovery flag: print the parameter/option reference and exit.
    if args.list_params:
        print(schema.describe_reference())
        return 0

    if args.dump_defaults:
        print(config.dump_defaults(), end="")
        return 0

    # 1. Base layer from --config, if given. If the file carries an
    #    [environment] block (i.e. it is a previous run's <out>_run.toml), it is a
    #    replay: drift-check the recorded environment against the current one.
    if args.config:
        file_options, file_params, prior_env = config.load_config(args.config)
        if prior_env is not None:
            for msg in provenance.check_drift(prior_env):
                print(f"DRIFT: {msg}")
    else:
        file_options, file_params = {}, {}

    # 2. CLI KEY=VALUE assignments layer on top (they win), routed by name just
    #    like the config file.
    cli_options, cli_params = schema.route(_parse_key_value(args.assignments))
    options = {**file_options, **cli_options}
    params = {**file_params, **cli_params}

    print(f"Resolved options:  {options}")
    print(f"Resolved params: {params}")

    # 3. Run, capturing any warnings for the run file. The Model import is
    #    deferred further still: the discovery/replay paths above never pay for
    #    the full solver stack (diffrax, equinox, LINX).
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from .main import Model

        model = Model(**options)
        # Replay guard: a run file's custom species cannot be rebuilt from a
        # config, so fail loudly if the reconstructed stack differs.
        if args.config:
            config.check_replay_species(config.recorded_species(args.config), model)
        output = model(params)
    warn_msgs = [str(w.message) for w in caught]
    for msg in warn_msgs:
        print(f"WARNING: {msg}")

    import numpy as np

    l = np.asarray(output.l)
    ClTT = np.asarray(output.ClTT)
    print(f"Computed spectra over l = {int(l[0])}..{int(l[-1])} ({l.size} multipoles).")
    print(f"ClTT[:3] = {ClTT[:3]}")
    print(f"Pk grid: {np.asarray(output.k).size} wavenumbers.")

    # 4. Write results + the reproducible run file (shared with notebook usage).
    written = config.save_run(output, args.output, model, params, warnings=warn_msgs)
    print("Wrote: " + ", ".join(written))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
