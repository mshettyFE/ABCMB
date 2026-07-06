<h1 align="center">
ABCMB<!-- omit from toc -->
</h1>
<h4 align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2602.15104%20-green.svg)](https://arxiv.org/abs/2602.15104)
[![Run Tests](https://github.com/TonyZhou729/ABCMB/actions/workflows/accuracy.yml/badge.svg)](https://github.com/TonyZhou729/ABCMB/actions/workflows/accuracy.yml)
<!--[![arXiv](https://img.shields.io/badge/arXiv-2408.14538%20-green.svg)](https://arxiv.org/abs/2408.14538) -->

</h4>

Autodifferentiable Boltzmann solver for the CMB (ABCMB) is a Python+JAX package for differentiable computation of the Cosmic Microwave Background.  ABCMB is **complete to linear order** in $\Lambda\rm{CDM}$ cosmology.  It computes the matter and CMB power spectra and includes effects like lensing, massive neutrinos, and a state-of-the-art treatment of the physics of recombination through the companion code [HyRex](https://github.com/TonyZhou729/HyRex).

## Installation
ABCMB is pip installable!  Just run
```
pip install ABCMB
```
We recommend always doing so in a conda environment, preferably even a clean one.

If you'd like to clone the repo instead, after cloning you can run
```
pip install .
```
from the code directory. 

Note that both methods of installing will automatically attempt to install JAX for CPU; to install for GPU, refer to the [JAX documentation](https://docs.jax.dev/en/latest/installation.html) for a quick JAX installation guide.

## Examples
We have included several pedagogical jupyter notebooks to walk you through how to get started with ABCMB in our [example_notebooks](https://github.com/TonyZhou729/ABCMB/tree/main/example_notebooks) folder.  We suggest you start with [ABCMB_basics](https://github.com/TonyZhou729/ABCMB/blob/main/example_notebooks/ABCMB_basics.ipynb) to get a sense of how to run the code.  If you'd like to add new physics to ABCMB, check out [ABCMB_Fluids](https://github.com/TonyZhou729/ABCMB/blob/main/example_notebooks/ABCMB_Fluids.ipynb).  If you'd like to run ABCMB with the Big Bang Nucleosynthesis (BBN) code [LINX](https://github.com/cgiovanetti/LINX/tree/main) to do BBN+CMB joint analyses, check out [ABCMB_with_LINX](https://github.com/TonyZhou729/ABCMB/blob/main/example_notebooks/ABCMB_with_LINX.ipynb).

## Command-line usage
ABCMB installs an `abcmb` command for running the solver from the shell without a
notebook. Configuration comes from a [TOML](https://toml.io) file and/or positional
`KEY=VALUE` assignments (which override the file). Both route each key to
parameters vs model options by name, so you never have to say which is which:

```
# run with defaults
abcmb -o out.npz

# override a couple of values
abcmb omega_cdm=0.12 h=0.68 lensing=true -o out.npz

# drive everything from a config file
abcmb --config cosmo.toml -o run/spectra
```

A config file uses TOML tables purely for readability — keys are routed to
parameters vs model `options` by name, so a key placed in the "wrong" table still
works:

```toml
[cosmology]
omega_cdm = 0.12
h         = 0.68

[output]
l_max   = 2500
lensing = true
```

CLASS-style names are accepted as aliases (e.g. `N_ur`→`Neff`, `tau_reio`→`tau_reion`,
`l_max_scalars`→`l_max`); unrecognized option names warn with a suggestion.

**Reproducibility.** Every run writes a `<output>_run.toml` recording the raw
inputs (as `[params]`/`[specs]` tables) and an environment stamp (package version,
git commit + dirty flag, jax/jaxlib versions, device). That file is itself a valid
`--config`: pass it back to reproduce the run — its recorded environment is
drift-checked automatically, so you're warned if the code or environment moved:

```
abcmb --config out_run.toml -o rerun.npz
```

## Issues
Please feel free to open an issue if something is amiss in ABCMB!

## Citation

If you use ABCMB to publish scientific research, we suggest you cite
```
@misc{abcmb,
      title={{ABCMB: A Python+JAX Package for the Cosmic Microwave Background Power Spectrum}}, 
      author={Zilu Zhou and Cara Giovanetti and Hongwan Liu},
      year={2026},
      eprint={2602.15104},
      archivePrefix={arXiv},
      primaryClass={astro-ph.CO},
      url={https://arxiv.org/abs/2602.15104}, 
}
```



