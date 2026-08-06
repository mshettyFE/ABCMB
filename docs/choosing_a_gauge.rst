Choosing a gauge
================

ABCMB integrates the scalar perturbations in one of two gauges. The default is
the synchronous gauge. The alternative is the conformal Newtonian gauge. Set
the gauge with the ``gauge`` option:

.. code-block:: python

   Model(gauge="synchronous")   # the default
   Model(gauge="newtonian")     # conformal Newtonian

The two gauges give the same observables. They do not give the same
perturbations.  Above the horizon, the difference is large.

This page lists each place where the choice has an effect.

A fluid is gauge agnostic in one half, and gauge dependent in the other. The
difference decides how much of this page you need:

* ``y_prime`` is gauge agnostic. The metric reaches it through three source
  slots, and ABCMB fills those slots for the gauge of the run. One
  implementation is therefore correct in each gauge, with no branch.
* ``y_ini`` is gauge **dependent**. It returns numbers, and a value of
  ``delta`` or ``theta`` means nothing without a gauge. Each fluid therefore
  declares the gauge of its own initial conditions in ``ic_gauge``. See
  `Initial conditions`_ below.

To write a fluid, read :doc:`promoting_a_fluid`. You then need the
`Initial conditions`_ section here, and nothing else.

The gauge object
----------------

A :class:`~abcmb.gauges.Gauge` holds every
gauge-dependent decision. All other code reads the inputs and the outputs of
the gauge. No other code reads the metric variables of a specific gauge.

To change the gauge, replace this one object. No other code branches on the
gauge.

Where the code uses the gauge
-----------------------------

There are six places, in four modules. No other code names a gauge.

.. list-table::
   :header-rows: 1
   :widths: 34 30 36

   * - Call site
     - Method
     - Effect
   * - ``main.Model.__init__``
     - :func:`~abcmb.gauges.resolve_gauge`, ``Gauge.check_tolerances``
     - Selects the gauge object. Gives a warning if the solver tolerances are
       too loose. See `Accuracy`_.
   * - ``model_setup.populate_species``
     - reads ``Gauge.name``
     - Makes sure that ABCMB can honour the ``ic_gauge`` of each fluid.
   * - ``perturbations.initial_conditions_one_k``
     - ``metric_y_ini``, ``ic_shift``
     - Sets slot 0 of the state vector. Transforms the initial conditions of a
       fluid that used the other gauge.
   * - ``perturbations.get_derivatives``
     - ``sources``
     - Gives the metric derivative and the three source slots. This is the ODE
       vector field.
   * - ``perturbations.make_output_table``
     - ``metric_history``, ``sources``
     - Fills ``PerturbationTable.metric``. Gives the metric source for
       ``theta_b_prime``.
   * - ``spectrum.SpectrumSolver.get_Cl``
     - ``MetricHistory.cmb_sources``
     - Gives the metric part of the CMB source functions.

The module ``spectrum.py`` does not import :mod:`abcmb.gauges`. It reads the
gauge from ``PT.gauge``. The last method above belongs to the metric history,
not to the gauge, because it reads no gauge data.

The state vector
----------------

Slot 0 holds the metric variable. The fluid equations start at slot 1. Each
gauge uses a different variable in slot 0, and closes the equations in a
different way.

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * -
     - synchronous
     - conformal Newtonian
   * - slot 0
     - :math:`\eta`
     - :math:`\phi`
   * - closed by
     - the momentum constraint for :math:`\eta'`. Then :math:`h'` follows from
       the energy constraint.
     - the momentum constraint for :math:`\phi'`. Then :math:`\psi` follows
       from the anisotropic stress. The energy constraint is redundant.
   * - ``sources.continuity``
     - :math:`h'/2`
     - :math:`-3\phi'`
   * - ``sources.euler``
     - :math:`0`
     - :math:`k^2\psi / aH`
   * - ``sources.shear``
     - :math:`(h' + 6\eta')/2`
     - :math:`0`

One slot is always zero in each gauge. This is why a fluid that omits a term is
correct in one gauge and wrong in the other. Use
:func:`~abcmb.species.gauge_source_omissions` to find the error. That function
examines fluids only. It cannot find an error in a gauge.

ABCMB integrates :math:`\phi` with the momentum constraint, as CLASS does. This
method is more stable than the energy constraint. The energy constraint is then
redundant, and gives a free test. The test in
``pytests/test_gauges_integration.py`` uses it. The constraint holds only if
the initial conditions and the gauge transformation are correct. The constraint
is an exact identity, so the test uses machine precision.

Initial conditions
------------------

This is the one gauge-dependent part of a fluid.

``y_prime`` writes *equations*. The three source slots hold each metric term,
so the equations need no gauge. ``y_ini`` writes *values*. A value of
``delta`` differs between the two gauges by :math:`3(1+w)\,aH\alpha`, and a
value of ``theta`` differs by :math:`k^2\alpha`. No fluid can therefore give
its initial conditions without a gauge.

A fluid also cannot convert them alone. The transformation uses the generator
:math:`\alpha`, and :math:`\alpha` comes from the total stress-energy of the
model. One species does not know that total. The fluid therefore states its
gauge, and ABCMB does the conversion.

The adiabatic series in :mod:`abcmb.species.adiabatic_ics` use the synchronous
gauge. They use the normalization :math:`\eta = 1` above the horizon. The
synchronous gauge is therefore the hub. ABCMB always calculates the generator
:math:`\alpha` from the synchronous constraints. Each gauge then applies the
correct sign in ``ic_shift``.

Each fluid declares the gauge of its own ``y_ini`` in ``ic_gauge``. The evolver
does the transformation. The declaration is mandatory for a fluid that writes a
``y_ini``. An omission is silent, so ``populate_species`` does not guess.

Different fluids can declare different gauges. The generator :math:`\alpha`
does not change under the shift that it makes. The density term and the
velocity term cancel. ABCMB can therefore calculate :math:`\alpha` from the
initial conditions of the fluids without a transformation.

Outputs
-------

``PerturbationTable`` records its gauge in ``PT.gauge``.

* ``PT.metric`` is a :class:`~abcmb.gauges.SynchronousMetric` or a
  :class:`~abcmb.gauges.NewtonianMetric`. The first holds
  :math:`\eta, h', \eta', \alpha, \alpha'`. The second holds
  :math:`\phi, \psi, \phi'`. The two share no field name. Code that reads the
  wrong gauge therefore gets an ``AttributeError``.
* ``PT.species_perturbations`` holds ``delta`` and ``theta`` for each fluid, in
  the gauge of the run.
* ``PT.delta_m`` and ``PT.delta_cb`` use the comoving gauge. The comoving gauge
  is gauge independent. ``Pk`` therefore does not change with the gauge. CLASS
  uses the same convention for its matter transfer functions.

.. _Accuracy:

Accuracy
--------

The default solver tolerances are correct for the synchronous gauge. They are
converged there. A tighter tolerance moves :math:`P(k)` by approximately
:math:`10^{-4}`. They are not converged in the newtonian gauge. There a tighter
tolerance moves :math:`P(k)` by approximately 1.4 per cent.

``Model`` gives a warning if the tolerances are too loose for the gauge. The
warning gives the correct values:

.. code-block:: python

   from abcmb.gauges import NewtonianGauge

   Model(gauge="newtonian",
         **NewtonianGauge.recommended_tolerances,
         max_steps_PE=NewtonianGauge.recommended_max_steps)

With these values the two gauges agree to :math:`3\times10^{-3}` in
:math:`P(k)`, and to :math:`5\times10^{-4}` in the CMB spectra. The solvers set
this limit, not the physics.

``delta_m`` has one limit. Below :math:`k \sim 10^{-3}\,\mathrm{Mpc}^{-1}` the
modes are still larger than the horizon. The conversion
:math:`\delta + 3aH\theta/k^2` then subtracts two large terms. Each term
increases as :math:`1/k^2`. Round-off therefore controls the result, although
the solution is correct. CLASS gives the same warning for this variable.

Adding a gauge
--------------

To add a gauge, do these steps:

#. Make a subclass of :class:`~abcmb.gauges.Gauge`.
#. Write the four methods.
#. Make a subclass of :class:`~abcmb.gauges.MetricHistory` for the output.
#. Add the gauge to ``GAUGES``, and to the ``choices`` of the ``gauge`` option.

Two items need a design decision first:

* ``ic_shift`` has no argument for the source gauge. The phrase "the other
  gauge" is correct only for two gauges. A third gauge must tell ``ic_shift``
  where the initial conditions come from.
* The generator :math:`\alpha` applies to the synchronous gauge and the
  newtonian gauge only. A third gauge has its own generator. Each gauge must
  then declare a shift against the hub. ABCMB can add two shifts together,
  because gauge transformations add at linear order.
  :class:`~abcmb.metric.GaugeShift` holds the two shifts separately for this
  reason.

The fluid API does not change for any of these steps.
