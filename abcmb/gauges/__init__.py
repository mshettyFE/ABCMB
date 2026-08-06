"""
The gauges ABCMB can integrate the scalar perturbations in.

See :doc:`choosing_a_gauge` for what the choice affects and where.
"""

from ..metric import GaugeName
from .base import CMBMetricSources, FluidTotals, Gauge, MetricHistory
from .newtonian import NewtonianGauge, NewtonianMetric
from .synchronous import SynchronousGauge, SynchronousMetric

GAUGES: dict[GaugeName, type[Gauge]] = {
    GaugeName.SYNCHRONOUS: SynchronousGauge,
    GaugeName.NEWTONIAN: NewtonianGauge,
}


def resolve_gauge(value: str) -> Gauge:
    """
    Build the :class:`Gauge` named by the ``gauge`` option.

    The schema already restricts the option to these, so raising here is a
    backstop for direct callers.
    """
    try:
        name = GaugeName(value)
    except ValueError:
        raise ValueError(
            f"gauge={value!r} is not one of "
            f"{', '.join(repr(str(g)) for g in GaugeName)}."
        ) from None
    return GAUGES[name]()


__all__ = [
    "GAUGES",
    "CMBMetricSources",
    "FluidTotals",
    "Gauge",
    "MetricHistory",
    "NewtonianGauge",
    "NewtonianMetric",
    "SynchronousGauge",
    "SynchronousMetric",
    "resolve_gauge",
]
