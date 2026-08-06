Public API and stability
========================

ABCMB is a research code. Every module is ostensibly public. You can therefore
use any part of the code, and this is often the correct thing to do for an
analysis. But the import path does not tell you if a name is a supported
interface or an implementation detail.

This page gives that information. It lists the names that stay stable, and the
names that can change without notice. A change to a stable name gets a
deprecation period. A change to any other name can happen in any release.

Supported: the ``abcmb`` namespace
----------------------------------

Each name at the top level is API. These names, their call signatures and
their behaviour do not change without a deprecation period:

.. code-block:: python

   from abcmb import Model, Output                       # the model front door
   from abcmb import load_config, model_from_config      # file-driven runs
   from abcmb import save_run, dump_defaults
   from abcmb import Fluid, StandardFluid, BackgroundFluid   # extension base classes

The package imports these names lazily. ``import abcmb`` therefore does not
import JAX. JAX arrives with the first attribute that you touch.

The file-driven functions
~~~~~~~~~~~~~~~~~~~~~~~~~

These four functions drive a run from a TOML file. Use them from a notebook,
or from your own script. The ``abcmb`` command-line tool uses the same
functions.

* ``load_config(path)`` reads a TOML file. It returns the tuple
  ``(options, params, environment)``. Each key goes to the options table or to
  the params table by *name*. You therefore do not choose the table yourself.
* ``model_from_config(path)`` builds a ``Model`` from a TOML file. It returns
  the tuple ``(model, params)``. Call ``model(params)`` next.
* ``save_run(output, path, model, params)`` writes two files. The file
  ``<stem>.npz`` holds the spectra. The file ``<stem>_run.toml`` holds your raw
  inputs and an environment stamp.
* ``dump_defaults()`` returns a TOML string. The string holds each option and
  each parameter with its default value.

A ``<stem>_run.toml`` file is itself a valid config. Give it to
``model_from_config`` or to ``--config`` to repeat the run. ABCMB then checks
the species list of the file against the species list of the new model, and
reports a difference.

The command-line tool
~~~~~~~~~~~~~~~~~~~~~

The ``abcmb`` command has the same status as the functions above:

.. code-block:: bash

   abcmb omega_cdm=0.12 h=0.68 lensing=true -o out.npz
   abcmb --config cosmo.toml -o run/spectra
   abcmb --list-params
   abcmb --dump-defaults > defaults.toml

Each ``KEY=VALUE`` argument goes to the options or to the params by name, as in
a config file. A ``KEY=VALUE`` argument overrides the same key from
``--config``.

The TOML schema has the same status, for the *names* of the options and the
parameters, and for their aliases. The default *values* can change between
releases.

Supported: the ``Model`` and ``Output`` surface
-----------------------------------------------

Build a :class:`~abcmb.main.Model` once. Then call it on a parameter dict. It
returns an :class:`~abcmb.main.Output`:

.. code-block:: python

   model = Model(lensing=True)
   out = model({"h": 0.6762, "omega_cdm": 0.1193, "omega_b": 0.0225})

A ``Model`` holds the static configuration. This includes the species list,
the k grids and the solver settings. A change to any option needs a new
``Model`` and a new compilation. The ``params`` dict holds the physics inputs,
and you can change it between calls.

The two stages
~~~~~~~~~~~~~~

The pipeline has two stages. ``model(params)`` runs both.

The first stage is ``Model.add_derived_parameters(params)``. It resolves your
input against the schema, and it adds the derived keys. Examples are
``omega_r``, ``omega_Lambda``, ``om``, ``R_nu`` and ``H0``. It also runs the
BBN calculation for ``YHe``.

This stage is **eager**. It checks concrete values, and it pins the BBN solve
to the CPU. A JAX trace cannot do either of these. The stage is therefore
separate.

The second stage is ``Model.run_derived(params)``. It takes the output of the
first stage. It solves for the background, the recombination history, the
perturbations and the spectra.

This stage is **traceable**, and it is the correct place for a derivative.
Gradients with respect to the derived parameters pass through it.

Where to put jit and grad
~~~~~~~~~~~~~~~~~~~~~~~~~

ABCMB controls its own jit boundaries. The methods ``get_BG_pre_recomb``,
``get_PTBG`` and the internal post-recombination stage each use
``eqx.filter_jit``. The recombination solver runs on the CPU, inside its own
``filter_jit``.

Follow these rules:

* Do not use ``jax.jit`` on ``model(params)``. Do not use ``jax.vmap`` on it
  either. The eager first stage cannot be traced.
* Do not use ``jax.jit`` on ``run_derived`` either. An outer jit removes the
  device placement that keeps the recombination solver on the CPU.
* Eager autodiff operates correctly. Use ``jax.grad`` or ``jax.jacfwd`` on
  ``model(params)``. The derivative passes through the derivation stage.
* For a sampler, call the two stages separately. Run
  ``add_derived_parameters`` once for each parameter set. Then call
  ``run_derived``.

``run_derived`` raises a clear error if you give it raw parameters. It checks
for the derived keys first. Without this check the failure appears much later,
as a ``KeyError`` for a key that you never supplied.

Finer entry points
~~~~~~~~~~~~~~~~~~

These methods give you the intermediate objects:

* ``Model.get_BG_pre_recomb(params)`` returns a ``BackgroundPreRecomb``. This
  object holds the conformal time table.
* ``Model.get_BG(params, pre_BG, recomb_output)`` returns the full
  ``Background``. It adds the reionization model and the optical depth.
* ``Model.get_PTBG(params, pre_BG, recomb_output)`` returns the
  ``PerturbationTable`` and the ``Background`` together. Use it to get the
  perturbations without the spectra.

The ``Output`` object
~~~~~~~~~~~~~~~~~~~~~

``Output`` holds the results:

* ``ClTT``, ``ClTE`` and ``ClEE`` are the CMB angular power spectra. ``l`` is
  their multipole grid.
* ``Pk`` is the linear matter power spectrum. ``k`` is its wavenumber grid, in
  Mpc^-1.
* ``BG``, ``PT`` and ``params`` give the intermediate quantities. ``params``
  includes the derived keys.

The field names are stable. New fields can appear.

``Output`` also gives two baryon-drag quantities. They are properties, and
ABCMB calculates each one only when you read it. Each needs its own solve, so a
run that ignores them pays nothing:

* ``output.z_d`` is the drag redshift.
* ``output.rs_d`` is the sound horizon at that redshift. This is the BAO
  standard ruler. Distance measurements use it, as ``D_V/r_d``, ``D_M/r_d``
  and ``H r_d``. A CMB and BAO analysis therefore needs it.

.. code-block:: python

   out = model(params)
   print(out.z_d, out.rs_d)     # ~1060, ~147 Mpc for Planck-like LCDM

Supported: the fluid contract
-----------------------------

Users implement the extension API. They do not call it. It is therefore the
most expensive interface to change, and the most stable one.

Choose a base class
~~~~~~~~~~~~~~~~~~~

A fluid is a subclass of one of three base classes. Use this table to choose:

.. list-table::
   :header-rows: 1
   :widths: 26 34 40

   * - Base class
     - Use it for
     - It gives you
   * - :class:`~abcmb.species.StandardFluid`
     - A species with the delta, theta, sigma layout. Most fluids use it.
     - The three stress-energy sums, the state-vector accessors, and the
       gauge transformation of the initial conditions.
   * - :class:`~abcmb.species.Fluid`
     - A species whose state is not delta, theta, sigma. An example is a
       momentum-binned distribution.
     - The attribute contract only. You write each method.
   * - :class:`~abcmb.species.BackgroundFluid`
     - A species with no perturbations, such as a cosmological constant.
     - ``num_equations = 0``, ``is_matter = False``, and an empty
       implementation of each perturbation method.

Declare these attributes
~~~~~~~~~~~~~~~~~~~~~~~~

* ``name`` is a unique string. Do not treat it as a label. The coupling
  lookups and the perturbation output tables use it as a key.
* ``num_equations`` is the number of variables that ABCMB integrates for this
  fluid. It must equal the length of the array from ``y_ini`` and the array
  from ``y_prime``. ABCMB checks this at compile time.
* ``is_matter`` is ``True`` for a species in the matter power spectrum.
* ``is_neutrino`` is optional. Set it to ``True`` only for a species in the
  neutrino sector, for the :math:`N_{\mathrm{eff}}` accounting.
* ``ic_gauge`` is optional, and mandatory for a fluid that writes its own
  ``y_ini``. It gives the gauge of those initial conditions.

ABCMB constructs each fluid with ``__init__(self, first_idx, options)``. It
gives the fluid its own offset into the state vector. Read precision values
from ``options`` here. Then read your own variables with ``self.first_idx``.

Implement these methods
~~~~~~~~~~~~~~~~~~~~~~~

The background methods apply to each fluid:

* ``rho(lna, params)`` returns the energy density, in eV cm^-3.
* ``P(lna, params)`` returns the pressure, in eV cm^-3.

Both take one scalar ``lna`` and return one scalar. The caller uses ``vmap``
for a batch. ABCMB checks this rule at model construction.

``w(lna, params)`` returns the equation of state. The base class calculates it
as ``P/rho``. Override it only for a fluid where that division loses accuracy.

A perturbed species also implements these methods:

* ``y_ini(k, tau_ini, params)`` returns the initial values, as an array of
  length ``num_equations``. Write them in the gauge that you declare in
  ``ic_gauge``.
* ``y_prime(k, lna, sources, y, args)`` returns the derivatives with respect
  to ``lna``, as an array of length ``num_equations``. The ``sources``
  argument is a :class:`~abcmb.metric.MetricSources` bundle. It holds the
  metric contribution in three slots: ``sources.continuity``,
  ``sources.euler`` and ``sources.shear``. Use these slots instead of the
  metric variables of one gauge. One implementation then operates in each
  gauge. See :doc:`promoting_a_fluid` for the transcription table.
* ``rho_delta(lna, y, args)`` returns ``rho * delta``, in eV cm^-3.
* ``rho_plus_P_theta(lna, y, args)`` returns ``(rho + P) * theta``, in
  eV cm^-3 Mpc^-1.
* ``rho_plus_P_sigma(lna, y, args)`` returns ``(rho + P) * sigma``, in
  eV cm^-3.

The last three methods give the fluid contribution to the Einstein equations.
ABCMB sums them over each species. :class:`~abcmb.species.StandardFluid`
implements them from the delta, theta and sigma slots. A direct subclass of
:class:`~abcmb.species.Fluid` must write them, because the products
``rho * delta`` and ``(rho + P) * theta`` do not exist for every state layout.

These methods are optional:

* ``output_perturbations(lna, modes, args)`` returns a dict of named arrays for
  ``PerturbationTable.species_perturbations``. The base class returns an empty
  dict.
* ``y_ini_shift(shift, params)`` returns the change to ``y_ini`` for a
  different gauge. :class:`~abcmb.species.StandardFluid` implements it.
  A direct subclass of :class:`~abcmb.species.Fluid` must write it, if its
  ``ic_gauge`` can disagree with the run.

What ``StandardFluid`` adds
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~abcmb.species.StandardFluid` assumes that slot 0 holds ``delta``,
slot 1 holds ``theta`` and slot 2 holds ``sigma``. It then gives you:

* ``get_delta``, ``get_theta`` and ``get_sigma``, which read those slots. Each
  returns zero if the fluid has fewer equations. Override one of these to
  change the layout.
* The three stress-energy sums, which use the accessors above.
* ``y_ini_shift``, which uses your own ``w``.

The gauge rule
~~~~~~~~~~~~~~

The fluid contract does not give the gauge of the model, and it will not do so.
This absence makes one ``y_prime`` correct in each gauge. ``ic_gauge`` is a
statement about the ``y_ini`` of the fluid. ABCMB supports it for that purpose
only. Do not branch on it.

What you can use inside the methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``self.first_idx`` for your own position in the state vector. Use the
fields of ``args``, which is a :class:`~abcmb.species.PerturbationContext`.
The fields are ``args.BG``, ``args.params`` and ``args.species_list``. The
lookup ``args.find(name, cls)`` returns another fluid by name, and narrows its
type. Use it for a coupled species. Do not import a different species module.
See :doc:`promoting_a_fluid` and the fluids tutorial.

Two extension points
~~~~~~~~~~~~~~~~~~~~

ABCMB has two ways to give a fluid new information without a break. Use them
when you extend the code:

* ``params`` is an open mapping. A custom species can read a key that the
  schema does not declare. Such a key gives an "unrecognized" warning.
  :doc:`promoting_a_fluid` shows how to declare it.
* You read ``PerturbationContext`` by attribute. You never unpack it. New
  fields can therefore appear. A new *method parameter* would break each
  existing fluid, but a new context field does not.

Not supported: everything else
------------------------------

A module below the top level is an implementation detail. You can import it,
and this is often correct for an analysis or for a test. But these modules can
change name, or split, or move between releases. Recent releases have done all
three.

The list includes:

* ``abcmb.main``, ``abcmb.spectrum``, ``abcmb.perturbations``,
  ``abcmb.gauges``, ``abcmb.metric``, ``abcmb.background``,
  ``abcmb.model_setup``, ``abcmb.constants`` and ``abcmb.ABCMBTools``
* the ``abcmb.inputs`` package, which holds ``schema``, ``derived``, ``config``
  and ``provenance``, and its generated ``_schema_types``
* each species module, such as ``abcmb.species.photon``. Import the classes
  from ``abcmb.species`` instead, because that path is stable.
* ``abcmb.species.validation``, and each name with an initial underscore

Two of these are worth a note. The classes in ``abcmb.gauges`` and
``abcmb.metric`` are new, and their layout can still change; the ``gauge``
*option* is stable. The diagnostics in ``abcmb.species.validation`` are for a
test suite, not for a pipeline; :doc:`promoting_a_fluid` shows how to use them.

The vendored companion codes ``abcmb.hyrex`` and ``abcmb.linx`` follow their
upstream projects. Their documentation is separate. Each package records the
ABCMB changes in its ``VENDORED.md`` file.

Type annotations
----------------

ABCMB contains a :pep:`561` ``py.typed`` marker. A type checker therefore uses
the inline annotations, for an installed copy and for a source checkout.
Pyright checks the annotations in CI.

The jaxtyping shape strings, such as ``Float[Array, "n_lna n_k"]``, document
the array axes. Nothing checks them, either statically or at run time. Treat
them as documentation.
