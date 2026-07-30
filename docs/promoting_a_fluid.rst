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

Step 1: move the class into ``abcmb/species.py``
------------------------------------------------

Copy the class from the notebook and place it with the other concrete species.
The base-class contracts are enforced, so an incomplete promotion fails loudly
rather than subtly:

* ``name`` and ``is_matter`` are abstract — declare both as class attributes
  (a missing one raises at instantiation). ``name`` must be unique
  (``populate_species`` rejects duplicates).
* ``num_equations`` has no default — declare it as a class attribute or in
  ``__init__`` (``BackgroundFluid`` already declares 0). It must match the
  size of your ``y_ini``/``y_prime`` arrays; ABCMB cross-checks this at
  compile time.
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

Step 2: declare the parameters in ``PARAM_SCHEMA``
--------------------------------------------------

This is the step that converts passthrough parameters into declared ones. Add
one :class:`~abcmb.schema.Spec` row per parameter in ``abcmb/schema.py``:

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

Step 3: regenerate the schema artifacts
---------------------------------------

.. code-block:: bash

   ./check.sh fix

This regenerates the two committed, schema-derived artifacts:
``defaults.toml`` (your parameters appear with defaults and doc comments) and
``abcmb/_schema_types.py`` (the ``Params`` TypedDict gains your keys, which is
what activates static key checking for them). A staleness test fails CI if
this step is forgotten.

Step 5: tests
-------------

At minimum:

* Add the species to the trait assertions in ``pytests/test_species.py``
  (``is_neutrino`` flags) and, if it has interesting construction logic, an
  instantiation test. The abstract-attribute and vector-layout contracts give
  you structural checking for free at first trace.
* If a reference computation exists (e.g. the same model in CLASS), extend the
  accuracy test — this is the only check that validates the *physics*.

Step 6: run everything
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
* **Name changes are breaking.** ``species_dict`` lookups, saved run files
  (the species drift check), and coupled fluids all key on ``name`` — treat a
  promoted fluid's name as API.
