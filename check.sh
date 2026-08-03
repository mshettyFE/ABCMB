#!/usr/bin/env bash
#
# Run the linter, formatter check, docs build, and tests in one step (mirrors CI).
#
#   ./check.sh          lint + format-check + docs + tests (check-only, like CI)
#   ./check.sh fix      auto-apply ruff format + fixable lint (no tests)
#   ./check.sh test     run the tests only (skip lint/format/docs)
#   ./check.sh docs     build the docs only (warnings are errors)
#   ./check.sh nb       execute the example notebooks (nbmake)
#
set -euo pipefail

RUFF="uvx ruff@0.15.20"

# `uv run --extra X` REPLACES the environment's extras, so a bare `--extra test`
# uninstalls the CUDA jaxlib on a GPU box and silently drops every later run to
# CPU. Carry the gpu extra through unless ABCMB_NO_GPU=1 (CPU-only machines and
# CI, where installing ~2 GB of CUDA wheels would be waste).
GPU_EXTRA="--extra gpu"
if [ "${ABCMB_NO_GPU:-0}" = "1" ]; then
    GPU_EXTRA=""
fi

usage() {
    cat <<'EOF'
Run the linter, formatter check, docs build, and tests (mirrors CI).

Usage: ./check.sh [command]

Commands:
  (none)   lint + format-check + docs + tests (check-only, like CI)
  fix      auto-apply ruff format + fixable lint + schema codegen, then exit
  test     run the tests only (skip lint/format/docs)
  docs     build the docs only (warnings are errors), then exit
  nb       execute the example notebooks (nbmake), then exit
  help     show this help

The test runs report coverage (pytest-cov, configured in [tool.coverage.*]).
Notebooks are a separate command because each one is a full solve; they are
also the only user-facing surface with no static checker behind them.
EOF
}

run_notebooks() {
    echo ">> nbmake (execute example notebooks)"
    # The explicit path overrides testpaths (= pytests) from pyproject.toml.
    uv run --extra test $GPU_EXTRA pytest --nbmake --nbmake-timeout=1800 example_notebooks
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
    nb)
        run_notebooks
        exit 0
        ;;
    fix)
        # Auto-apply only, then stop: the point is a fast pre-commit tidy, so
        # it does not fall through to the (slow) test run.
        echo ">> ruff format (applying)"
        $RUFF format .
        echo ">> ruff check --fix (applying)"
        $RUFF check --fix .
        echo ">> regenerate schema artifacts (defaults.toml, inputs/_schema_types.py)"
        uv run python -m abcmb.inputs._codegen
        exit 0
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
        uv run --extra dev $GPU_EXTRA pyright
        build_docs
        ;;
    *)
        echo "error: unknown command '$1'" >&2
        usage >&2
        exit 2
        ;;
esac

echo ">> pytest (with coverage)"
uv run --extra test $GPU_EXTRA pytest -s -vv --cov --cov-report=term-missing pytests
