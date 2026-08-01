API documentation
=================

User API
--------

Everything a typical analysis needs: build a :class:`~abcmb.main.Model` and call
it on a parameter dictionary, drive runs from TOML config files, and subclass
the fluid base classes to add new physics. All of these names are importable
directly from the package root, e.g. ``from abcmb import Model``.

.. toctree::
   :maxdepth: 2

   abcmb.main
   abcmb.inputs.config
   abcmb.species

Internals
---------

Implementation modules, documented for contributors. These are plumbing: user
code should not need to import them directly, and their interfaces may change
without notice.

.. toctree::
   :maxdepth: 1

   abcmb.background
   abcmb.perturbations
   abcmb.spectrum
   abcmb.inputs.schema
   abcmb.inputs.derived
   abcmb.model_setup
   abcmb.inputs.provenance
   abcmb.constants
   abcmb.ABCMBTools
