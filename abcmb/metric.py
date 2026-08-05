"""
The vocabulary shared between fluids and gauges.
"""

from enum import StrEnum, auto

import equinox as eqx
from jaxtyping import Array


class GaugeName(StrEnum):
    """
    The gauges ABCMB can integrate in.

    The concrete gauge implementations live in :mod:`abcmb.gauges`; only the
    *names* are here, so that a fluid can declare which gauge its own initial
    conditions are written in without importing (or being able to reach) the
    machinery that would tell it which gauge the run uses.
    """

    SYNCHRONOUS = auto()
    NEWTONIAN = auto()


class MetricSources(eqx.Module):
    """
    The metric's contribution to a fluid's equations, in the three slots it can
    occupy. Written once by the evolver, read by every ``y_prime``.

    Deliberately carries no gauge tag: a fluid must not be able to ask which
    gauge it is in
    """

    continuity: Array
    euler: Array
    shear: Array


class GaugeShift(eqx.Module):
    r"""
    The change of gauge a fluid must apply to its *own* initial conditions,
    evaluated at ``tau_ini`` and already signed for the direction being
    applied -- so :meth:`~abcmb.species.Fluid.y_ini_shift` is one formula and
    never asks which way it is going.

    The density entry is factored by :math:`1+w` so that the fluid supplies the
    only piece the gauge cannot know: its own equation of state.

    Attributes:
    -----------
    delta_per_one_plus_w : array
        The density shift divided by ``1 + w``; a fluid adds
        ``(1 + w) * delta_per_one_plus_w`` to its delta (units: dimensionless)
    theta : array
        The velocity-divergence shift, added as-is (units: Mpc^{-1})
    lna : array
        Log scale factor at ``tau_ini``, for fluids that need to evaluate
        their own equation of state
    """

    delta_per_one_plus_w: Array
    theta: Array
    lna: Array
