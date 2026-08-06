Promoting a notebook fluid into ABCMB
=====================================

The `fluids tutorial <https://github.com/TonyZhou729/ABCMB/blob/main/example_notebooks/ABCMB_Fluids.ipynb>`_
shows how to make a new species in a notebook. You make a subclass of a fluid
base class. You give it to ``user_species``. You put its parameters in the open
``params`` dict. That method is quick, but the fluid stays outside the ABCMB
machinery. This page shows how to move a mature fluid into the package.

What promotion gives you
------------------------

A notebook fluid has *passthrough* parameters. They operate correctly, but each
run gives an ``unrecognized parameters`` warning. ABCMB does not check them or
document them. The static type checker does not see them.

Promotion gives you these four items:

* **Validation.** A wrong parameter name gets a "did you mean" message. Values
  get bounds checks and type checks.
* **Documentation.** The parameters go into ``abcmb --list-params``, into
  ``defaults.toml``, and into the API documentation. The class goes into the
  :mod:`abcmb.species` reference.
* **Static checking.** The parameters go into the generated ``Params``
  TypedDict. Pyright then finds wrong keys in code that uses them.
* **Reproducibility.** Config files and the CLI can use the fluid, because a
  TOML file cannot hold a notebook class. Saved run files replay correctly.

Step 1: move the class into the ``abcmb/species/`` package
-----------------------------------------------------------

Copy the class from the notebook into a new module in ``abcmb/species/``. Use
one file for each species. The base classes stay in ``base.py``. Then re-export
the class in ``abcmb/species/__init__.py``. Add an import and an ``__all__``
entry.

A coupled fluid finds its partner by name at run time. Use
``args.find("Photon", StandardFluid)``. Do not import a different species
module.

ABCMB enforces the base-class contracts. An incomplete promotion therefore
fails immediately:

* ``name`` and ``is_matter`` are abstract. Declare both as class attributes. An
  absent attribute raises an error at instantiation. ``BackgroundFluid``
  declares ``is_matter = False`` already, so a background fluid needs only a
  name. Each ``name`` must be unique, because ``populate_species`` rejects a
  duplicate.
* ``num_equations`` has no default. Declare it as a class attribute, or set it
  in ``__init__``. ``BackgroundFluid`` declares 0 already. The value must agree
  with the length of your ``y_ini`` array and your ``y_prime`` array. ABCMB
  checks this at compile time.
* ``y_prime`` gets the metric contribution as a
  :class:`~abcmb.metric.MetricSources` bundle. It does not get the metric
  variables. Use this table to transcribe your equations. Your derivation can
  use either gauge. Put each metric term in the correct slot, and ABCMB
  calculates it correctly.

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

  The two textbook forms become the same line. This is the structure that the
  two derivations share::

      # newtonian paper:   delta' = -(1+w)(theta - 3 phi') - 3H(cs2-w) delta
      # synchronous paper: delta' = -(1+w)(theta + h'/2)   - 3H(cs2-w) delta
      delta_prime = -(1 + w) * (theta / aH + sources.continuity) - 3 * (cs2 - w) * delta

  .. warning::

     **Write each slot that your derivation has.** ``sources.euler`` is always
     zero in the synchronous gauge. A fluid that omits it is correct in that
     gauge, and wrong in the newtonian gauge. No synchronous test finds this
     error. The same risk applies to ``sources.shear``, which is always zero in
     the newtonian gauge.

     :func:`~abcmb.species.gauge_source_omissions` finds the omission. Run it on
     your species list. See Step 4.

  Initial conditions are different. The transformation between the gauges uses
  the *total* stress-energy, so a species cannot do it alone. Declare the gauge
  of your initial conditions, and ABCMB does the transformation:

  .. code-block:: python

     class MyFluid(StandardFluid):
         ic_gauge = GaugeName.SYNCHRONOUS   # or GaugeName.NEWTONIAN

  This declaration is mandatory for a fluid that writes its own ``y_ini``.
  ``populate_species`` raises an error without it. The attribute has a default
  value, so an omission is otherwise silent. ABCMB then reads the initial
  conditions in the wrong gauge. The error is the :math:`\alpha` shift, and
  nothing raises. Initial conditions from
  :mod:`~abcmb.species.adiabatic_ics` are synchronous.

  ``ic_gauge`` is the only gauge-dependent attribute of a fluid. It is a
  statement about ``y_ini`` only. Do not branch on it. The fluid API cannot
  tell you the gauge of the model. This is deliberate, and it is why one
  ``y_prime`` is correct in each gauge.

  :meth:`~abcmb.species.Fluid.y_ini_shift` does the transformation.
  :class:`~abcmb.species.StandardFluid` implements it for the delta, theta and
  sigma layout, and uses your own ``w``. A subclass of ``StandardFluid``
  therefore needs no work. A direct subclass of
  :class:`~abcmb.species.Fluid` must supply the method, if its ``ic_gauge`` can
  disagree with the run. ``populate_species`` raises an error at construction if
  the method is absent.

  The series in :mod:`~abcmb.species.adiabatic_ics` use the synchronous gauge,
  with the normalization :math:`\eta = 1` above the horizon. ``SYNCHRONOUS`` is
  the default for this reason.
* Set ``is_neutrino = True`` only for a species in the neutrino sector. This
  flag controls the :math:`N_{\mathrm{eff}}` and :math:`R_\nu` accounting. Use
  it for free-streaming radiation. The default ``False`` is correct for
  tightly-coupled dark radiation.
* A fluid that replaces a default species competes with the LCDM set. Set
  ``use_LCDM_species=False`` in that case. The names ``Baryon`` and ``Photon``
  are structural, because the baryon-photon coupling and the recombination code
  find them by name. A replacement must keep the same name.

Remove the notebook ``sys.path`` code. Inside the package, use relative imports
such as ``from . import constants as cnst``.

Step 2: declare the parameters and the options in the schema
------------------------------------------------------------

This step makes passthrough parameters into declared parameters. Add one
:class:`~abcmb.inputs.schema.Spec` row for each parameter in
``abcmb/inputs/schema.py``:

.. code-block:: python

   Spec(
       "N_idr",
       0.0,
       float,
       "Interacting dark radiation density, in units of one SM neutrino.",
       group=Group.NEUTRINOS,
       bounds=(0.0, None),
   ),

These fields are important:

* ``default`` gives the value when the user omits the key. Some parameters must
  exist only when the user supplies them, because the absence has a meaning.
  ``Neff`` is an example. Use the ``UNSET`` sentinel for these.
* ``group`` sets the position of the parameter in ``--list-params`` and in
  ``defaults.toml``. Add a new ``Group`` member if no member fits. The group
  controls readability only, because ABCMB flattens the parameters.
* ``bounds`` and ``choices`` give a warning for a bad value. They never raise
  an error.
* ``aliases`` accepts other names, such as the CLASS names.

Options use the same method. Your fluid can read precision values from
``options`` in ``__init__``, such as a hierarchy cutoff or a grid setting.
Declare these as ``Spec`` rows in ``OPTION_SCHEMA``. Use this rule:

* ``PARAM_SCHEMA`` holds differentiable physics inputs. These change between
  calls to the same model. They stay in the ``params`` dict, as JAX arrays.
* ``OPTION_SCHEMA`` holds static configuration. These are ``int``, ``bool``,
  ``str`` or fixed float values. A change makes a new ``Model`` and a new
  compilation.

Config files send each key to the correct table by name. A user therefore does
not need to know the schema of a key.

Step 3: regenerate the schema artifacts
---------------------------------------

.. code-block:: bash

   ./check.sh fix

This command regenerates the two schema artifacts in the repository.
``defaults.toml`` gets your parameters, with their defaults and their
descriptions. ``abcmb/inputs/_schema_types.py`` gets your keys in the ``Params``
TypedDict, which starts the static key checks. A staleness test fails in CI if
you omit this step.

Step 4: tests
-------------

Write these tests as a minimum:

* Add the species to the trait assertions in ``pytests/test_species.py``, for
  the ``is_neutrino`` flag. Add an instantiation test if the construction is
  complex. The abstract-attribute contract and the vector-layout contract give
  you structural checks at the first trace.
* Run the :mod:`abcmb.species.validation` diagnostics on your fluid, with real
  parameters. :func:`~abcmb.species.continuity_residuals` checks that
  ``d(rho)/dlna = -3(rho+P)``, and therefore connects your ``rho`` to your
  ``P``. For a perturbed fluid with standard adiabatic initial conditions,
  :func:`~abcmb.species.adiabatic_ic_residuals` compares your ``y_ini`` with
  the photon values. The ratio is 3/4 for matter and 1 for radiation.
  :func:`~abcmb.species.ic_scaling_residuals` checks each power of k and tau.
  These functions need no reference values. A correct implementation gives
  approximately 1e-14. An error gives a result of order 1.
* Run :func:`~abcmb.species.gauge_source_omissions` on your species list. It
  reports two terms that are always zero in the synchronous gauge. The terms
  are ``sources.euler``, and the ``theta/aH`` of the fluid in the continuity
  equation. The function reports a term only if your fluid does not read it.
  No synchronous test finds these omissions. An empty dict is the correct
  result.

  This function has a limit. It is a *presence* check. It differentiates your
  ``y_prime`` with respect to each slot. A measurement used five versions of one
  fluid: correct, term absent, sign wrong, value halved, and value multiplied by
  a wrong factor. The function reported the second version only. A term with a
  wrong value therefore passes.
  :func:`~abcmb.species.metric_source_dependence` shows which slots each fluid
  reads.
* :func:`~abcmb.species.adiabatic_ic_residuals` also validates a non-default
  ``ic_gauge``. It moves the initial conditions of each fluid into one gauge
  before it compares them. A wrong ``ic_gauge`` gives a residual of order 1.
  The Einstein constraints cannot find this error, because they stay correct
  when one species uses the wrong gauge.
* Run the accuracy test in each gauge. The observables must agree. This
  comparison uses each source slot.
* Extend the accuracy test if a reference calculation exists, such as the same
  model in CLASS. This test is the only check of the physics.

Step 5: run the full check
--------------------------

.. code-block:: bash

   ./check.sh

This command runs the lint check, the format check and pyright. Pyright now
finds wrong parameter names in each place that uses them. The command then
builds the documentation, which now contains your class in the
:mod:`abcmb.species` reference. The tutorial cells show the installed code, so
they need no changes. The command then runs the test suite.

Pitfalls
--------

* **Fields are static.** A value in an ``eqx.field``, or a value from
  ``options``, causes a new compilation after a change. Put a value that changes
  during a parameter scan in ``params``, not in a field.
* **Static conditionals and traced conditionals are different.** Use a Python
  ``if`` for a *field*, such as ``num_equations`` or a flag. Use ``jnp.where``
  for a *traced value*, such as a value from ``params`` or from ``y``.
* **Old config files continue to operate.** A declared parameter stops the
  warning and starts the validation. Nothing fails. An out-of-range value now
  gives a warning.
* **A name change breaks other code.** The coupling lookups (``args.find``),
  the saved run files and the coupled fluids use ``name``. Treat the name of a
  promoted fluid as API. See :doc:`public_api`.
* **Options are read-only after resolution.** Do not put calculated values into
  the ``options`` dict. Return them instead. Annotate each new function that
  takes options with ``options: "Options"``, so that the type checker enforces
  this rule.
