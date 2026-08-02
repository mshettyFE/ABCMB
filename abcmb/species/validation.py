"""
Numeric sanity diagnostics for fluid species.

Complements the construction-time scalar-contract probe (model_setup): these
checks need real parameter values, so they run on demand -- from the test
suite for the built-in species, or by a custom-fluid author against their own
stack (see docs/promoting_a_fluid.rst).
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp

from .base import Fluid, FluidParams, StandardFluid


def continuity_residuals(
    species_list: Sequence[Fluid],
    params: FluidParams,
    lna_grid: jnp.ndarray | None = None,
) -> dict[str, float]:
    r"""
    Maximum relative residual of the background continuity equation,

        d(rho)/d(ln a) = -3 (rho + P),

    per species, over ``lna_grid`` (default: 27 points in lna = [-13, 0]).

    Every background fluid must satisfy this identically, so it is a
    metamorphic cross-check tying a fluid's ``rho`` and ``P`` implementations
    to each other with no reference values needed"""
    if lna_grid is None:
        lna_grid = jnp.linspace(-13.0, 0.0, 27)

    residuals: dict[str, float] = {}
    for s in species_list:

        def rho_fn(lna, s=s):
            return jnp.asarray(s.rho(lna, params))

        # Eager per-point loop (not vmap): constant-returning rho/P (e.g.
        # DarkEnergy) produce unbatched outputs, and this is a diagnostic,
        # not a hot path.
        rho = jnp.asarray([rho_fn(lna) for lna in lna_grid])
        drho = jnp.asarray([jax.grad(rho_fn)(lna) for lna in lna_grid])
        pressure = jnp.asarray([s.P(lna, params) for lna in lna_grid])

        rel = jnp.max(jnp.abs(drho + 3.0 * (rho + pressure)) / rho)
        residuals[s.name] = float(rel)
    return residuals


def adiabatic_ic_residuals(
    species_list: Sequence[Fluid],
    params: FluidParams,
    k: float = 0.05,
    tau_ini: float = 0.5,
) -> dict[str, float]:
    r"""
    Residuals of the adiabatic-mode cross-relations between species'
    ``y_ini``: delta = (3/4) delta_gamma for matter, delta = delta_gamma for
    radiation (exact at all orders kept, including the om*tau corrections),
    and theta_baryon = theta_gamma (tight coupling at zeroth order).

    Returns ``{"<name>.delta": rel residual, ..., "Baryon.theta": ...}``.
    """
    photon = next((s for s in species_list if s.name == "Photon"), None)
    if photon is None:
        raise ValueError("adiabatic_ic_residuals needs a fluid named 'Photon'.")
    y_g = jnp.asarray(photon.y_ini(k, tau_ini, params))
    delta_g, theta_g = y_g[0], y_g[1]

    residuals: dict[str, float] = {}
    for s in species_list:
        if s is photon or not isinstance(s, StandardFluid) or s.num_equations < 1:
            continue
        y = jnp.asarray(s.y_ini(k, tau_ini, params))
        expected = 0.75 * delta_g if s.is_matter else delta_g
        residuals[f"{s.name}.delta"] = float(jnp.abs(y[0] / expected - 1.0))
        if s.name == "Baryon":
            residuals["Baryon.theta"] = float(jnp.abs(y[1] / theta_g - 1.0))
    return residuals


def ic_scaling_residuals(
    species_list: Sequence[Fluid],
    params: FluidParams,
    k: float = 0.05,
    tau_ini: float = 0.5,
    alpha: float = 2.0,
) -> dict[str, float]:
    r"""
    k-tau scaling degeneracy of the adiabatic ICs: they depend on k and
    tau_ini only through k*tau (deltas, shears) with one extra power of k
    in theta, and on om only through om*tau. Under

        (k, tau, om) -> (alpha*k, tau/alpha, alpha*om)

    every slot of ``y_ini`` is therefore invariant except theta, which
    scales by alpha. Catches wrong individual powers of k or tau that the
    combined forms hide. Returns max relative slot residual per species.
    """
    scaled_params = dict(params)
    scaled_params["om"] = alpha * params["om"]

    residuals: dict[str, float] = {}
    for s in species_list:
        if not isinstance(s, StandardFluid) or s.num_equations < 1:
            continue
        y = jnp.asarray(s.y_ini(k, tau_ini, params))
        y_scaled = jnp.asarray(s.y_ini(alpha * k, tau_ini / alpha, scaled_params))
        factors = jnp.ones(s.num_equations)
        if s.num_equations > 1:
            factors = factors.at[1].set(alpha)
        expected = y * factors
        # zero slots (higher moments) are exactly zero on both sides
        denom = jnp.where(expected != 0.0, jnp.abs(expected), 1.0)
        residuals[s.name] = float(jnp.max(jnp.abs(y_scaled - expected) / denom))
    return residuals
