Promoting a notebook fluid into ABCMB
=====================================

The `fluids tutorial <https://github.com/TonyZhou729/ABCMB/blob/main/example_notebooks/ABCMB_Fluids.ipynb>`_
shows how to prototype a new species in a notebook: subclass a fluid base
class, pass it via ``user_species``, and feed its parameters through the open
``params`` dict. That workflow is deliberately frictionless — but the fluid
lives outside ABCMB's machinery. This guide walks through promoting a matured
fluid into the package proper.

What promotion buys
-------------------

While your fluid is notebook-only, its parameters are *passthrough*: they work,
but every run warns ``unrecognized parameter``, nothing validates or documents
them, and they are invisible to the static type checker. Promotion upgrades the
fluid to first-class:

* **Validation** — typos in parameter names get "did you mean" suggestions;
  values get bounds/kind checks.
* **Documentation** — the parameters appear in ``abcmb --list-params``,
  ``defaults.toml``, and the API docs; the class appears in the
  :mod:`abcmb.species` reference automatically.
* **Static checking** — the parameters join the generated ``Params`` TypedDict,
  so pyright catches key typos in code that uses them.
* **Reproducibility** — config-file and CLI runs can use the fluid (a TOML
  cannot carry a notebook class), and saved run files replay instead of
  failing the species drift check.

Step 1: move the class into the ``abcmb/species/`` package
-----------------------------------------------------------

Copy the class from the notebook into a new module under ``abcmb/species/``
(one file per species; base classes live in ``base.py``), and re-export it in
``abcmb/species/__init__.py`` (import + ``__all__`` entry). Coupled fluids
find their partners at runtime by name — ``args.find("Photon",
StandardFluid)`` — never import a sibling species module.

The base-class contracts are enforced, so an incomplete promotion fails loudly
rather than subtly:

* ``name`` and ``is_matter`` are abstract — declare both as class attributes
  (a missing one raises at instantiation; ``BackgroundFluid`` already declares
  ``is_matter = False``, so background-only fluids need only a name). ``name``
  must be unique (``populate_species`` rejects duplicates).
* ``num_equations`` has no default — declare it as a class attribute or in
  ``__init__`` (``BackgroundFluid`` already declares 0). It must match the
  size of your ``y_ini``/``y_prime`` arrays; ABCMB cross-checks this at
  compile time.
* ``y_prime`` receives the metric's contribution as a
  :class:`~abcmb.species.MetricSources` bundle rather than as raw metric
  variables. This is a **transcription table**: whichever gauge your derivation
  is written in, map its metric terms onto the three slots and ABCMB evaluates
  them correctly. (ABCMB integrates in synchronous gauge; you do not need to
  convert your equations by hand, and you must not — the slots already carry
  the right values.)

  .. list-table::
     :header-rows: 1
     :widths: 40 30

     * - Your derivation has
       - Write
     * - :math:`h'/2` (synchronous) or :math:`-3\phi'` (Newtonian)
       - ``sources.continuity``
     * - :math:`0` (synchronous) or :math:`k^2\psi` (Newtonian)
       - ``sources.euler``
     * - :math:`(h' + 6\eta')/2` (synchronous) or :math:`0` (Newtonian)
       - ``sources.shear``

  Both textbook forms collapse to the same line — this is the skeleton the two
  derivations already share, not a new formalism::

      # newtonian paper:   delta' = -(1+w)(theta - 3 phi') - 3H(cs2-w) delta
      # synchronous paper: delta' = -(1+w)(theta + h'/2)   - 3H(cs2-w) delta
      delta_prime = -(1 + w) * (theta / aH + sources.continuity) - 3 * (cs2 - w) * delta

  Because ABCMB integrates in synchronous gauge, ``sources.euler`` is
  identically zero — no Euler equation carries a metric source there. Write it
  anyway if your derivation has one: it costs nothing, it documents where the
  term belongs, and it is what makes the transcription faithful to your source.

  Initial conditions are *not* covered by this. ``y_ini`` must currently be
  written in synchronous-gauge variables (the shared series in
  :mod:`~abcmb.species.adiabatic_ics` are normalized to :math:`\eta = 1`
  superhorizon). Unlike the equations, that conversion cannot be done at the
  species level — it depends on the total stress-energy — so a fluid whose ICs
  were derived in another gauge needs care here.
* Set ``is_neutrino = True`` only if the species belongs in the neutrino
  sector for the :math:`N_{\mathrm{eff}}` / :math:`R_\nu` accounting
  (free-streaming, neutrino-like radiation). The default ``False`` is correct
  for tightly-coupled dark radiation.
* If the fluid *replaces* a default species (e.g. a dark-energy variant), it
  competes with the LCDM set under ``use_LCDM_species=False``. The names
  ``Baryon`` and ``Photon`` are structural (the baryon-photon coupling and
  recombination look them up); a replacement for either must keep the name.

Keep any notebook-era ``sys.path`` hacks out; inside the package, use relative
imports (``from . import constants as cnst``).

Step 2: declare the parameters (and options) in the schema
----------------------------------------------------------

This is the step that converts passthrough parameters into declared ones. Add
one :class:`~abcmb.inputs.schema.Spec` row per parameter in ``abcmb/inputs/schema.py``:

.. code-block:: python

   Spec(
       "N_idr",
       0.0,
       float,
       "Interacting dark radiation density, in units of one SM neutrino.",
       group=Group.NEUTRINOS,
       bounds=(0.0, None),
   ),

Fields worth knowing:

* ``default`` — used when the user omits the key. If the parameter should
  *only* exist when the user supplies it (its absence carries meaning, like
  ``Neff``), use the ``UNSET`` sentinel instead of a value.
* ``group`` — controls where the parameter appears in ``--list-params`` and
  ``defaults.toml``; add a new ``Group`` member if none fits. 
  Purely for --list-params human readability reason (parameters get flattened in the end regardless of grouping)
* ``bounds`` / ``choices`` — non-fatal validation (warns, never raises).
* ``aliases`` — accept alternative names (e.g. CLASS conventions).

**Options work the same way.** If your fluid reads precision knobs from
``options`` in ``__init__`` (a hierarchy cutoff, a grid setting), declare them
as ``Spec`` rows in ``OPTION_SCHEMA`` instead. The dividing rule:

* ``PARAM_SCHEMA`` — differentiable physics inputs that vary between calls of
  the same model (they live in the ``params`` dict, as JAX arrays).
* ``OPTION_SCHEMA`` — static configuration that shapes the computation
  (``int``/``bool``/``str``/fixed floats); changing one means a new ``Model``
  and a recompile.

Config files route keys to the right table by *name*, so users never need to
know which schema a key lives in.

Step 3: regenerate the schema artifacts
---------------------------------------

.. code-block:: bash

   ./check.sh fix

This regenerates the two committed, schema-derived artifacts:
``defaults.toml`` (your parameters appear with defaults and doc comments) and
``abcmb/inputs/_schema_types.py`` (the ``Params`` TypedDict gains your keys, which is
what activates static key checking for them). A staleness test fails CI if
this step is forgotten.

Step 4: tests
-------------

At minimum:

* Add the species to the trait assertions in ``pytests/test_species.py``
  (``is_neutrino`` flags) and, if it has interesting construction logic, an
  instantiation test. The abstract-attribute and vector-layout contracts give
  you structural checking for free at first trace.
* Run the :mod:`abcmb.species.validation` diagnostics on your fluid with
  real params. :func:`~abcmb.species.continuity_residuals` checks
  ``d(rho)/dlna = -3(rho+P)``, tying your ``rho`` and ``P`` to each other;
  for a perturbed fluid with standard adiabatic ICs,
  :func:`~abcmb.species.adiabatic_ic_residuals` checks your ``y_ini``
  against the photon's (3/4 ratio for matter, 1 for radiation) and
  :func:`~abcmb.species.ic_scaling_residuals` checks the powers of k and
  tau individually. No reference values needed: correct implementations
  sit at ~1e-14, mistakes at O(1).
* If a reference computation exists (e.g. the same model in CLASS), extend the
  accuracy test — this is the only check that validates the *physics*.

Step 5: run everything
----------------------

.. code-block:: bash

   ./check.sh

Lint, format check, pyright (your parameters are now typo-checked wherever
they are used), the docs build (your class is now in the rendered
:mod:`abcmb.species` reference — including the tutorial's live-source cells,
which display the installed code and need no editing), and the test suite.

Pitfalls
--------

* **Fields are static.** Anything stored as an ``eqx.field`` (or read from
  ``options``) triggers recompilation when changed. A quantity that varies
  across a parameter scan belongs in ``params``, not in a field.
* **Static vs traced conditionals.** Branch on *fields* (``num_equations``,
  flags) with plain Python ``if``; branch on *traced values* (anything from
  ``params`` or ``y``) with ``jnp.where``.
* **Old configs keep working.** Once the parameter is declared, config files
  that supplied it stop warning and start validating; nothing breaks — but a
  previously-tolerated out-of-bounds value will now warn.
* **Name changes are breaking.** Coupling lookups (``args.find``), saved run
  files (the species drift check), and coupled fluids all key on ``name`` —
  treat a promoted fluid's name as API (see :doc:`public_api`).
* **Options are read-only after resolution.** Never stash computed values into
  the ``options`` dict — return them explicitly instead — and annotate any new
  function taking options with ``options: "Options"`` so the type checker
  enforces this.
