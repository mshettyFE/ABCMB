"""
Regenerate the schema-derived, committed artifacts:

* ``defaults.toml`` -- a starter config (rendered by :func:`abcmb.config.dump_defaults`).
* ``abcmb/_schema_types.py`` -- the ``Options`` / ``Params`` ``TypedDict``s the type
  checker consumes (rendered by :func:`dump_types`).

This is dev/build tooling, not part of the user-facing CLI. Run it directly::

    python -m abcmb._codegen        # or ./check.sh fix

``_schema_types.py`` is imported only under ``TYPE_CHECKING`` (never at runtime), so
this generator -- and the package as a whole -- work even if that file is missing or
stale, which is what lets it bootstrap the file it generates.
"""

from pathlib import Path

from . import config, schema


def dump_types() -> str:
    """
    Render the ``Options`` / ``Params`` ``TypedDict``s from the schema as importable
    Python source (written to ``abcmb/_schema_types.py``).

    Static type checkers need *literal* ``TypedDict`` definitions (they don't execute
    a runtime-built one), so the schema stays the single source of truth and this file
    is derived from it. ``Options`` values are Python scalars (options are static
    config); ``Params`` values are ``jax.Array`` (resolved via ``jnp.array``), and it
    is ``total=False`` because the derived keys are added in stages.
    """
    kind_name = {bool: "bool", int: "int", float: "float", str: "str"}

    lines = [
        "# GENERATED from abcmb/schema.py; do not edit by hand.",
        "# Regenerate with `./check.sh fix` (or `python -m abcmb._codegen`).",
        "from typing import TypedDict",
        "",
        "from jax import Array",
        "",
        "",
        "class Options(TypedDict):",
        '    """Resolved model options (static config); keys mirror OPTION_SCHEMA."""',
        "",
    ]
    lines += [
        f"    {spec.name}: {kind_name.get(spec.kind, 'object')}"
        for spec in schema.OPTION_SCHEMA
    ]

    lines += [
        "",
        "",
        "class Params(TypedDict, total=False):",
        '    """Resolved + derived cosmological parameters: keys mirror PARAM_SCHEMA',
        "    (inputs, conditional, derived). total=False because conditional/derived",
        '    keys are added in stages by derive_parameters."""',
        "",
    ]
    # Schema order (inputs, then conditional, then derived) -- deterministic, so
    # the generated file is stable across runs.
    lines += [f"    {spec.name}: Array" for spec in schema.PARAM_SCHEMA]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Write both committed artifacts from the current schema."""
    pkg = Path(__file__).resolve().parent
    (pkg.parent / "defaults.toml").write_text(config.dump_defaults())
    (pkg / "_schema_types.py").write_text(dump_types())
    print("Regenerated defaults.toml and abcmb/_schema_types.py")


if __name__ == "__main__":
    main()
