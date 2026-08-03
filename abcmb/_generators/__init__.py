"""
Offline generators for ABCMB's committed artifacts.

Nothing in here is imported at runtime. Each module regenerates something that
is checked into the repo, so the artifact has a reproducible provenance rather
than being a wall of opaque numbers, and each is pinned by a test that
re-derives the shipped values:

* :mod:`.bessel_tables` -- ``abcmb/bessel_tab/bessel_tables.npz``, the
  spherical-Bessel line-of-sight kernels (pinned by ``pytests/test_spectrum.py``).
* :mod:`.camb_stencils` -- the CAMB massive-neutrino momentum stencils carried
  in ``species/massive_neutrino.py`` (pinned by ``pytests/test_species.py``).
* :mod:`.schema_types` -- ``defaults.toml`` and ``abcmb/inputs/_schema_types.py``,
  derived from the option/param schema (pinned by ``pytests/test_config.py``).

Run one directly, e.g.::

    python -m abcmb._generators.schema_types    # or ./check.sh fix
"""
