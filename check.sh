#!/usr/bin/env bash
#
# Run the linter, formatter check, docs build, and tests in one step (mirrors CI).
#
#   ./check.sh          lint + format-check + docs + tests (check-only, like CI)
#   ./check.sh fix      auto-apply ruff format + fixable lint, then test
#   ./check.sh test     run the tests only (skip lint/format/docs)
#   ./check.sh docs     build the docs only (warnings are errors)
#
set -euo pipefail

RUFF="uvx ruff@0.15.20"

usage() {
    cat <<'EOF'
Run the linter, formatter check, docs build, and tests (mirrors CI).

Usage: ./check.sh [command]

Commands:
  (none)   lint + format-check + docs + tests (check-only, like CI)
  fix      auto-apply ruff format + fixable lint, then run tests
  test     run the tests only (skip lint/format/docs)
  docs     build the docs only (warnings are errors), then exit
  help     show this help
EOF
}

build_docs() {
    echo ">> sphinx docs build (warnings are errors)"
    # -W mirrors the CI docs job: stale autodoc references, orphaned pages, and
    # broken cross-links fail here instead of rotting silently. -q keeps the
    # output to warnings/errors only.
    uv run --extra docs sphinx-build -W -q -b html docs docs/_build/html
}

case "${1:-}" in
    -h|--help|help)
        usage
        exit 0
        ;;
    test)
        ;;  # skip ruff/docs, just run tests below
    docs)
        build_docs
        exit 0
        ;;
    fix)
        echo ">> ruff format (applying)"
        $RUFF format .
        echo ">> ruff check --fix (applying)"
        $RUFF check --fix .
        echo ">> regenerate schema artifacts (defaults.toml, _schema_types.py)"
        uv run python -m abcmb._codegen
        ;;
    "")
        echo ">> ruff check"
        $RUFF check .
        echo ">> ruff format --check"
        $RUFF format --check .
        echo ">> pyright (type check)"
        # Gating. jax/equinox stub-noise rules are suppressed in [tool.pyright];
        # the high-signal rules (TypedDict keys, attribute access, invalid type
        # forms) catch real typos in options/params/field access.
        uv run --extra dev pyright
        build_docs
        ;;
    *)
        echo "error: unknown command '$1'" >&2
        usage >&2
        exit 2
        ;;
esac

echo ">> pytest"
uv run --extra test pytest -s -vv pytests
