"""
The input-parsing layer: everything between user-supplied KEY=VALUEs and the
resolved options/params the solver consumes.

* :mod:`.schema` -- declare, resolve, derive (the single source of truth).
* :mod:`.derived` -- derived-parameter computation (neutrinos, BBN, densities).
* :mod:`.config` -- TOML front door (load_config, model_from_config, save_run).
* :mod:`.provenance` -- run-file environment stamps and drift detection.
* :mod:`._schema_types` -- generated Options/Params TypedDicts (type-check only).
* :mod:`._codegen` -- regenerates the committed schema artifacts.
"""
