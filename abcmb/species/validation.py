"""
Numeric sanity diagnostics for fluid species.

Complements the construction-time scalar-contract probe (model_setup): these
checks need real parameter values, so they run on demand -- from the test
suite for the built-in species, or by a custom-fluid author against their own
stack (see docs/promoting_a_fluid.rst).
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from ..metric import GaugeName, MetricSources
from .base import (
    Fluid,
    FluidParams,
    PerturbationContext,
    StandardFluid,
)

if TYPE_CHECKING:
    from ..background import Background


# Radiation-era superhorizon values of the gauge generator and the conformal
# Hubble rate, used as defaults by the IC diagnostics below. With ABCMB's
# eta -> 1 normalization the synchronous adiabatic mode has h = (k tau)^2 / 2
# and eta' = O((k tau)^2) (Ma & Bertschinger Eq. 96 with C = 1/2), so
# alpha = (h' + 6 eta')/2k^2 -> tau/2, and aH -> 1/tau.
def _radiation_era_alpha(tau_ini: float) -> float:
    return tau_ini / 2.0


def _radiation_era_aH(tau_ini: float) -> float:
    return 1.0 / tau_ini


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


def _y_ini_synchronous(
    s: Fluid,
    params: FluidParams,
    k: float,
    tau_ini: float,
    alpha: float,
    aH: float,
    lna: float,
) -> jnp.ndarray:
    """
    A fluid's ``y_ini`` expressed in synchronous-gauge variables, applying its
    own ``y_ini_shift`` when it declares that it wrote them in the other gauge.

    The shift comes from the target gauge itself rather than being rebuilt
    here, so this diagnostic cannot drift from the transformation the evolver
    actually applies. The import is deferred because :mod:`abcmb.gauges`
    imports this package.
    """
    y = jnp.asarray(s.y_ini(k, tau_ini, params))
    if s.ic_gauge == GaugeName.SYNCHRONOUS:
        return y
    from ..gauges import SynchronousGauge

    shift = SynchronousGauge().ic_shift(k, lna, aH, alpha)
    return y + jnp.asarray(s.y_ini_shift(shift, params))


def adiabatic_ic_residuals(
    species_list: Sequence[Fluid],
    params: FluidParams,
    k: float = 0.05,
    tau_ini: float = 0.5,
    alpha: float | None = None,
    aH: float | None = None,
    lna: float = -12.0,
) -> dict[str, float]:
    r"""
    Residuals of the adiabatic-mode cross-relations between species'
    ``y_ini``: delta = (3/4) delta_gamma for matter, delta = delta_gamma for
    radiation (exact at all orders kept, including the om*tau corrections),
    and theta_baryon = theta_gamma (tight coupling at zeroth order).

    **This is the check that catches a mis-declared** ``ic_gauge``. Every
    fluid's ICs are first brought to synchronous gauge according to what it
    declares, and only then compared.

    ``alpha`` and ``aH`` default to their radiation-era superhorizon values
    (``tau_ini/2`` and ``1/tau_ini``), which is where the ICs are set.
    ``lna`` only matters for a fluid whose ``w``
    varies with time. A model whose fluids all agree on ``ic_gauge`` (the usual
    case) does no transformation at all and is unaffected by any of the three.

    Returns ``{"<name>.delta": rel residual, ..., "Baryon.theta": ...}``.
    """
    if alpha is None:
        alpha = _radiation_era_alpha(tau_ini)
    if aH is None:
        aH = _radiation_era_aH(tau_ini)

    photon = next((s for s in species_list if s.name == "Photon"), None)
    if photon is None:
        raise ValueError("adiabatic_ic_residuals needs a fluid named 'Photon'.")
    y_g = _y_ini_synchronous(photon, params, k, tau_ini, alpha, aH, lna)
    delta_g, theta_g = y_g[0], y_g[1]

    residuals: dict[str, float] = {}
    for s in species_list:
        if s is photon or not isinstance(s, StandardFluid) or s.num_equations < 1:
            continue
        y = _y_ini_synchronous(s, params, k, tau_ini, alpha, aH, lna)
        expected = 0.75 * delta_g if s.is_matter else delta_g
        residuals[f"{s.name}.delta"] = float(jnp.abs(y[0] / expected - 1.0))
        if s.name == "Baryon":
            residuals["Baryon.theta"] = float(jnp.abs(y[1] / theta_g - 1.0))
    return residuals


def metric_source_dependence(
    species_list: Sequence[Fluid],
    BG: "Background",
    params: FluidParams,
    k: float = 0.05,
    lna: float = -12.0,
) -> dict[str, dict[str, bool]]:
    """
    Which of the three :class:`~.base.MetricSources` slots each fluid's
    ``y_prime`` actually reads, by differentiating it with respect to each
    slot in turn.

    Returns ``{"<name>": {"continuity": bool, "euler": bool, "shear": bool}}``.

    The perturbation equations are linear in the sources, so this is exact and
    independent of the state it is evaluated at -- a ``True`` means the term is
    present, a ``False`` means it is absent, with no thresholds involved.
    """
    ctx = PerturbationContext(BG, params, tuple(species_list))
    tau_ini = BG.tau(lna)
    y = jnp.concatenate(
        [jnp.array([1.0])]
        + [jnp.asarray(s.y_ini(k, tau_ini, params)) for s in species_list]
    )
    one = jnp.asarray(1.0)
    zero = jnp.asarray(0.0)
    base = MetricSources(continuity=one, euler=one, shear=one)

    out: dict[str, dict[str, bool]] = {}
    for s in species_list:
        if s.num_equations == 0:
            out[s.name] = dict.fromkeys(("continuity", "euler", "shear"), False)
            continue
        slots: dict[str, bool] = {}
        for slot in ("continuity", "euler", "shear"):
            tangent = MetricSources(
                **{
                    name: (one if name == slot else zero)
                    for name in ("continuity", "euler", "shear")
                }
            )
            _, dy = jax.jvp(
                lambda src, s=s: s.y_prime(k, lna, src, y, ctx), (base,), (tangent,)
            )
            slots[slot] = bool(jnp.any(dy != 0.0))
        out[s.name] = slots
    return out


def gauge_source_omissions(
    species_list: Sequence[Fluid],
    BG: "Background",
    params: FluidParams,
    k: float = 0.05,
    lna: float = -12.0,
) -> dict[str, list[str]]:
    """
    Report terms a fluid never reads that vanish identically in synchronous
    gauge -- the omissions that are correct there and silently wrong elsewhere.

    A **presence** check, not a correctness one. It differentiates ``y_prime``
    with respect to each source slot and asks whether the result depends on it
    at all, so a term that is written but wrong -- flipped sign, wrong
    coefficient -- is indistinguishable from a right one. It also inspects
    *fluids* only: a gauge that supplies a wrong value for a slot is outside
    its reach.

    Two such terms are detectable structurally, and both are checked:

    ``"sources.euler"``
        The metric's contribution to the Euler equation, zero in synchronous
        gauge. Flagged when a fluid has more than one equation and reads
        ``sources.continuity`` but not ``sources.euler``.

    ``"own velocity in the continuity equation"``
        The ``theta / aH`` that pairs with ``sources.continuity``. Synchronous
        gauge is *defined* by ``theta_c = 0`` for cold dark matter, so a
        cold-matter fluid can drop it and still be right there. Flagged when a
        fluid's first equation does not depend on any other variable of its
        own.

    Returns ``{"<name>": [missing terms]}`` for the flagged fluids only; an
    empty dict is the passing result. Fluids with no perturbations, and
    genuinely density-only fluids, are never flagged.
    """
    dependence = metric_source_dependence(species_list, BG, params, k=k, lna=lna)
    ctx = PerturbationContext(BG, params, tuple(species_list))
    tau_ini = BG.tau(lna)
    y = jnp.concatenate(
        [jnp.array([1.0])]
        + [jnp.asarray(s.y_ini(k, tau_ini, params)) for s in species_list]
    )
    one = jnp.asarray(1.0)
    base = MetricSources(continuity=one, euler=one, shear=one)

    out: dict[str, list[str]] = {}
    for s in species_list:
        if s.num_equations < 2:
            continue
        missing: list[str] = []

        if dependence[s.name]["continuity"] and not dependence[s.name]["euler"]:
            missing.append("sources.euler")

        # Does the fluid's first equation couple to any other variable it
        # owns? For every physical fluid the density equation is driven by the
        # velocity, so a flat row here means that coupling was dropped.
        jac = jax.jacfwd(lambda yy, s=s: s.y_prime(k, lna, base, yy, ctx))(y)
        own = jac[0, s.first_idx + 1 : s.first_idx + s.num_equations]
        if not bool(jnp.any(own != 0.0)):
            missing.append("own velocity in the continuity equation")

        if missing:
            out[s.name] = missing
    return out


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
