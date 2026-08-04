Public API and stability
========================

ABCMB is a research code, and every module is importable — nothing is hidden
behind private packages. That makes it easy to reach in and take a piece,
which is deliberate. It also means the import path alone does not tell you
whether something is a supported interface or an implementation detail.

This page is that missing signal: what will be kept working, and what may
move without notice.

Supported: the ``abcmb`` namespace
----------------------------------

Everything re-exported at the top level is API. These names, their call
signatures, and their behavior will not change incompatibly without a
deprecation cycle:

.. code-block:: python

   from abcmb import Model, Output                       # the model front door
   from abcmb import load_config, model_from_config      # file-driven runs
   from abcmb import save_run, dump_defaults
   from abcmb import Fluid, StandardFluid, BackgroundFluid   # extension base classes

The ``abcmb`` command-line tool and the TOML schema it reads (option and
parameter *names*, not their default values) are supported on the same terms.

Supported: the ``Model`` and ``Output`` surface
-----------------------------------------------

:class:`~abcmb.main.Model` is normally used through ``model(params)``, which
returns an :class:`~abcmb.main.Output`. For staged use — jit, vmap, or a
sampler — the derivation and solve stages are separate calls, because the
derivation stage validates concrete values and cannot be traced:

* ``model(params)`` — derive parameters, then solve. Not jittable; eager
  autodiff (``jax.grad``, ``jax.jacfwd``) works.
* ``Model.add_derived_parameters(params)`` — the eager derivation stage.
* ``Model.run_derived(params)`` — the traceable solve; jit/vmap this one.
* ``Model.get_BG_pre_recomb``, ``get_BG``, ``get_PTBG`` — finer entry points
  into the same pipeline.

``Output`` carries ``ClTT``, ``ClTE``, ``ClEE``, ``Pk``, ``l``, ``k``, plus
``BG``, ``PT``, and ``params`` for dropping down to intermediate quantities.
Field names are stable; new fields may be added.

It also exposes the baryon-drag observables ``output.z_d`` and ``output.rs_d``
-- the latter is the BAO standard ruler, which distance measurements are
quoted against (``D_V/r_d``, ``D_M/r_d``, ``H r_d``), so it is what a CMB+BAO
analysis needs:

.. code-block:: python

   out = model(params)
   print(out.z_d, out.rs_d)     # ~1060, ~147 Mpc for Planck-like LCDM

Supported: the fluid contract
-----------------------------

The extension API is *implemented* by users rather than called by them, so it
is the interface most expensive to change and the one held most stable. A
fluid subclasses one of the three exported base classes and provides:

* class attributes ``name``, ``num_equations``, ``is_matter``, and optionally
  ``is_neutrino``;
* ``rho`` and ``P`` (scalar in, scalar out — batching is the caller's
  ``vmap``);
* ``y_ini`` and ``y_prime`` for perturbed species, returning arrays of length
  ``num_equations``. ``y_prime`` takes the metric contribution as a
  :class:`~abcmb.species.MetricSources` bundle
  (``sources.continuity`` / ``.euler`` / ``.shear``), which is what lets one
  implementation serve every gauge — see :doc:`promoting_a_fluid`;
* optionally ``output_perturbations``.

Inside those methods, ``self.first_idx`` and the ``args``
(:class:`~abcmb.species.PerturbationContext`) fields ``args.BG``,
``args.params``, ``args.species_list``, and the lookup ``args.find(name, cls)``
are supported. See :doc:`promoting_a_fluid` and the fluids tutorial.

A fluid's ``name`` is part of the contract, not a label: coupling lookups and
the perturbation output tables are keyed on it.

Two deliberate escape hatches let ABCMB hand fluids new information without
breaking existing ones, so prefer them when extending:

* ``params`` is an open mapping — custom species may read keys the schema does
  not declare (they warn as unrecognized, and :doc:`promoting_a_fluid`
  explains how to declare them).
* ``PerturbationContext`` is attribute-accessed, never unpacked, so fields may
  be added to it. Adding a *method parameter* to the fluid contract would
  break every existing fluid; adding a context field does not.

Not supported: everything else
------------------------------

Module paths below the top level are implementation detail. Importing them is
fine — often the right thing for analysis or testing — but they may be
renamed, split, or moved between releases, and recent releases have done all
three. This includes, among others:

* ``abcmb.main``, ``abcmb.spectrum``, ``abcmb.perturbations``,
  ``abcmb.background``, ``abcmb.model_setup``, ``abcmb.constants``,
  ``abcmb.ABCMBTools``
* the ``abcmb.inputs`` package (``schema``, ``derived``, ``config``,
  ``provenance``) and its generated ``_schema_types``
* individual species modules (``abcmb.species.photon`` and friends) — import
  the classes from ``abcmb.species`` instead
* ``abcmb.species.validation``, and anything named with a leading underscore

Vendored companion codes (``abcmb.hyrex``, ``abcmb.linx``) track their
upstreams and are documented separately; ABCMB's forks of them are recorded in
each package's ``VENDORED.md``.

Type annotations
----------------

ABCMB ships a :pep:`561` ``py.typed`` marker, so type checkers honor its inline
annotations for installed copies as well as source checkouts. The annotations
are checked with pyright; jaxtyping shape strings such as
``Float[Array, "n_lna n_k"]`` document array axes but are not verified
statically or at runtime.
