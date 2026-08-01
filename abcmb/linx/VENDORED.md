# Vendored copy of LINX

- **Upstream**: https://github.com/cgiovanetti/LINX (docs:
  https://linx.readthedocs.io)
- **Vendor point**: first appears in ABCMB history at `3aeb9ec` (2026-02-10,
  a rename commit -- the copy itself predates it). Probable upstream base:
  `76101eb` (2026-01-28, the last pre-vendor upstream commit); the
  modification list below is measured against that base (2026-07-31).
  Upstream has 35+ commits since (through `ec2e9d2`, 2026-06-19), including
  a new `utils.py` -- absent here because it postdates the copy, not by
  local choice.

## Why vendored (not a dependency)

The PyPI name `linx` is taken by an unrelated project. Unlike `hyrex/`,
this copy is **not co-dependent on abcmb** (no abcmb imports; keeps its own
`const.py` and `background.py`): LINX's background is the z~1e9
radiation-era microphysics (e+- annihilation, QED plasma corrections,
neutrino decoupling) which ABCMB does not model, so there was no
duplication to remove. Its interface to ABCMB is thin -- (omega_b, Neff)
-> YHe -- and it runs as a CPU-pinned jit island (see docs/FAQ.rst). Note
this means ABCMB contains two physical-constants sources (abcmb.constants
and linx.const); tolerated because the interface is one scalar at ~1e-4
tolerance.

## Local modifications (vs base `76101eb`)

- **Packaging**: absolute `import linx.x` -> relative `from . import x`.
- **CPU-only**: upstream's `devices('gpu')`/`device_put` table-placement
  blocks removed; the vendored copy is CPU-pinned by construction.
- **AD**: `ForwardMode` added to diffrax imports (adjoint compatibility
  with ABCMB's forward-mode default).
- **Features removed** (present at base, cut locally): `tau_n_vary_me.py`
  (varying electron mass) and the analytic `P_QED.py` module.
- **QED numerics reworked**: the `QED_d2P_intdT2.txt` second-derivative
  table -- commented out at base with upstream's note "JAX grad obviates
  this import" -- is re-enabled here; table orientations un-flipped;
  `interpax` usage in thermo/weak_rates/abundances dropped.
- ~1,150 changed lines total across eight files.

## Syncing

Blind re-copying will break ABCMB. To port an upstream fix: diff upstream
`76101eb..<new>` for the physics change, re-apply it onto this copy by
hand, then run the accuracy suite (`pytests/accuracy_test.py`; BBN paths
via the sBBN/linx tests in `pytests/test_schema.py`). This directory is
excluded from ruff/pyright (see pyproject.toml).
