r"""
Synchronous gauge: the CDM-frame slicing, and ABCMB's default.
"""

from typing import ClassVar

import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Float

from ..metric import GaugeName, GaugeShift, MetricSources
from .base import (
    AllSpeciesTotals,
    CMBMetricSources,
    Gauge,
    MetricHistory,
    _grav,
    _shift_from_alpha,
)


class SynchronousMetric(MetricHistory):
    """
    The synchronous-gauge metric history on the output ``(lna, k)`` grid.

    Attributes:
    -----------
    eta : array
        Metric perturbation eta -- the evolved variable (slot 0)
    h_prime : array
        Time derivative of metric h (d/dlna). Retained for inspection; no
        internal consumer reads it.
    eta_prime : array
        Time derivative of metric eta (d/dlna)
    alpha : array
        Derived metric perturbation alpha = aH (h' + 6 eta') / 2k^2, the
        generator of the transformation to conformal Newtonian gauge
    alpha_prime : array
        Time derivative of alpha, from the anisotropic-stress Einstein
        equation (not by differentiating the energy constraint -- that would
        need h'')
    """

    eta: Float[Array, "n_lna n_k"]
    h_prime: Float[Array, "n_lna n_k"]
    eta_prime: Float[Array, "n_lna n_k"]
    alpha: Float[Array, "n_lna n_k"]
    alpha_prime: Float[Array, "n_lna n_k"]

    def _velocity_offset(self) -> tuple[Array, Array]:
        """``theta + k^2 alpha`` is the conformal-Newtonian velocity."""
        return self.alpha, self.alpha_prime

    def cmb_sources(
        self,
        k: Float[Array, " n_k"],
        aH: Float[Array, "n_lna 1"],
        aH_dot: Float[Array, "n_lna 1"],
        g: Float[Array, "n_lna 1"],
        g_prime: Float[Array, "n_lna 1"],
        expmkappa: Float[Array, "n_lna 1"],
    ) -> CMBMetricSources:
        """
        CLASS's numerically-efficient synchronous form (perturbations.c,
        ``perturbations_sources``): the naive
        ``-exp(-kappa) h'/6 + g delta_g/4`` integrated by parts so the
        integrand is not dominated by a large early-time h'.

        The Doppler offset is ``alpha``: ``theta_b + k^2 alpha`` is the baryon
        velocity in conformal Newtonian gauge, which is the combination the
        source function actually contains.
        """
        theta_offset, theta_offset_prime = self._velocity_offset()
        return CMBMetricSources(
            sw_potential=aH * self.alpha_prime,
            isw_T0=(
                g * (self.eta - aH * self.alpha_prime - 2.0 * aH * self.alpha)
                + 2.0
                * expmkappa
                * (aH * self.eta_prime - aH_dot * self.alpha - aH**2 * self.alpha_prime)
            ),
            isw_T1=(
                expmkappa
                * (aH * self.alpha_prime + 2.0 * aH * self.alpha - self.eta)
                * k
            ),
            theta_offset=theta_offset,
            theta_offset_prime=theta_offset_prime,
        )


class SynchronousGauge(Gauge):
    r"""
    Synchronous gauge: :math:`ds^2 = a^2[-d\tau^2 + (\delta_{ij} + h_{ij})dx^i dx^j]`,
    Ma & Bertschinger Eq. (21), with the residual freedom fixed by the cold
    dark matter frame (:math:`\theta_c = 0`).
    """

    name: ClassVar[GaugeName] = GaugeName.SYNCHRONOUS

    def metric_y_ini(
        self, aH: ArrayLike, eta_ini: ArrayLike, alpha_ini: ArrayLike
    ) -> Array:
        """Slot 0 is eta itself, so the adiabatic normalization passes through."""
        return jnp.asarray(eta_ini)

    def ic_shift(
        self, k: ArrayLike, lna: ArrayLike, aH: ArrayLike, alpha: ArrayLike
    ) -> GaugeShift:
        """
        Conformal Newtonian -> synchronous: MB95 Eq. 18 run backwards, so the
        generator is ``-alpha``.
        """
        return _shift_from_alpha(k, lna, aH, -jnp.asarray(alpha))

    def sources(
        self,
        k: ArrayLike,
        a: ArrayLike,
        aH: ArrayLike,
        metric_y: ArrayLike,
        totals: AllSpeciesTotals,
    ) -> tuple[Array, MetricSources]:
        """
        The energy constraint gives ``h'``, the momentum constraint ``eta'``
        (MB95 Eq. 21b, 21c).

        ``euler`` is identically zero: no Euler equation carries a metric
        source in this gauge. It is materialized as an array rather than a
        Python ``0.0`` so every field of the returned pytree is a traced leaf.
        """
        grav = _grav(a)
        eta = metric_y
        h_prime = 2.0 / aH**2 * (k**2 * eta + grav * totals.rho_delta)
        eta_prime = grav / aH / k**2 * totals.rho_plus_P_theta
        sources = MetricSources(
            continuity=h_prime / 2.0,
            euler=jnp.zeros_like(h_prime),
            shear=(h_prime + 6.0 * eta_prime) / 2.0,
        )
        return eta_prime, sources

    def metric_history(
        self,
        k: ArrayLike,
        a: ArrayLike,
        aH: ArrayLike,
        metric_y: ArrayLike,
        totals: AllSpeciesTotals,
    ) -> SynchronousMetric:
        """(eta, h', eta', alpha, alpha') on the output grid."""
        eta_prime, sources = self.sources(k, a, aH, metric_y, totals)
        # sources.shear is (h' + 6 eta')/2, so h' comes back out of it.
        h_prime = 2.0 * sources.continuity
        alpha = aH * (h_prime + 6.0 * eta_prime) / 2.0 / k**2
        # alpha' from the anisotropic-stress Einstein equation (MB95 Eq. 21d);
        # differentiating the energy constraint instead would need h''.
        shear_term = 3.0 * _grav(a) / aH * totals.rho_plus_P_sigma / k**2
        alpha_prime = metric_y / aH - 2.0 * alpha - shear_term
        return SynchronousMetric(
            eta=jnp.asarray(metric_y),
            h_prime=h_prime,
            eta_prime=eta_prime,
            alpha=alpha,
            alpha_prime=alpha_prime,
        )
