# Vendored copy of HyRex

- **Upstream**: https://github.com/TonyZhou729/HyRex (docs: https://hyrex.readthedocs.io)
- **Scientific lineage**: JAX reimplementation of the HYREC family
  (HYREC-2, https://github.com/nanoomlee/HYREC-2; Ali-Haimoud & Hirata,
  arXiv:1006.1355, 1011.3758). `tabs/*.dat` are the standard HYREC-2 tables.
- **Vendor point**: first appears in ABCMB history at `3aeb9ec` (2026-02-10,
  a rename commit -- the copy itself predates it). Upstream HEAD `0333001`
  (2026-01-06) has not moved since before vendoring, so a diff against
  upstream HEAD shows *pure local modification* (~1,030 lines across five
  files, measured 2026-07-31).

## Why vendored (not a dependency)

The PyPI name `hyrex` is taken by an unrelated project, and this copy is a
deliberate **integration fork**: upstream is a standalone recombination code
with its own internal cosmology; here it runs inside ABCMB's differentiable
graph with the duplicate background removed.

## Local modifications (vs upstream HEAD)

- **`cosmology.py` deleted**, replaced by injection: ABCMB's
  `BackgroundPreRecomb.make_recomb_inputs` computes TCMB/nH/H on HyRex's
  grid and passes them in as a `RecombInputs` bundle (defined in
  `abcmb/recomb_interface.py` -- host-side code, not part of this fork),
  making ABCMB the single background authority.
- **Constants unified**: `from abcmb import constants as cnst` replaces
  upstream's own constants (one source of physical constants across the
  fused graph).
- **`fast_interp` shared**: imports `..ABCMBTools.fast_interp` instead of a
  local interpolator.
- **Adjoint threaded**: ABCMB's diffrax adjoint choice is passed through all
  internal solves so the AD strategy is uniform.
- **Internal `jit`s removed**: ABCMB jits the whole pipeline once; nested
  jits stripped.
- **`z1` endpoint parameter removed** from `recomb_model.__init__`: the
  grid structurally ends today (lna = 0). Upstream's "integrate down to
  z1" freedom is incompatible with ABCMB's contract -- `Background`
  resamples xe/Tm onto the static `lna_axis_full` grid and clamps beyond
  its last point, so any earlier endpoint silently freezes late-time
  ionization.
- **`array_with_padding` is contained**: `get_history` sanitizes the inf
  padding and resamples every history onto the static `lna_axis_full` grid
  before returning, so the sentinel and its traced-int metadata never leave
  this package (ABCMB's `Background` consumes plain fixed-size arrays plus
  one validity-start scalar). The padding convention is now purely internal
  to hyrex.

## Syncing

Blind re-copying will break ABCMB. To port an upstream fix: diff upstream
`0333001..<new>` for the physics change, re-apply it onto this copy by hand,
then run the accuracy suite (`pytests/accuracy_test.py`). This directory is
excluded from ruff/pyright (see pyproject.toml).
