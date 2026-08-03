"""
Regenerate the CAMB massive-neutrino momentum stencil carried in
``massive_neutrino.py``, so the constants are pinned by construction rather
than taken on faith from CAMB's Fortran.

Adapted CAMB's generator (camb.info/maple/nu_integration_kernels.py),
implementing the Appendix A moment-matching construction of Howlett, Lewis,
Hall & Challinor (arXiv:1201.3654)

The basic idea: We have some integrals whose values we want to approximate with a set of nodes q_i and weights w_i.
In this case, the moments of the normalize F kernel.
- Use scipy least_squares to search in an unconstrained parameter space
- Transform the unconstrained variables to ones which are gaurentee to obey the constraints of the problem
- Return the residual: weighted sum against the reference integrals weighted by the relative size of the target
- Fixing values complicates this picture a bit (your constraints change to a sigmoid instead to stay pegged between the fixed points)

* 3-point (perturbation stencil): the unique 3-node / 3-weight match of the
  moments n = -4, -2, -1, 0, 1, 2.
* 5-point (background stencil): CAMB's original rule -- nodes 2, 4, 13
  fixed by choice, with q1, q4 and all five weights matched to the moments
  n = -4, -2, -1, 0, 1, 2, 3. (CAMB has since replaced it upstream with an
  exact+least-squares refit; ABCMB keeps the original for continuity, now
  pinned by construction rather than taken on faith.)

Not imported at runtime -- consumed by the stencil-pinning test.
"""

from math import factorial, log, pi

import numpy as np
from scipy.optimize import least_squares
from scipy.special import zeta

_KERNEL_NORM = 7.0 * pi**4 / 120.0  # int q^4 (-df0/dq) dq / 4


def _fermi_dirac_integral(p: int) -> float:
    """int_0^inf q^p f0(q) dq for integer p >= 0, f0 = 1/(e^q + 1)."""
    if p == 0:
        return log(2.0)
    return (1.0 - 2.0**-p) * factorial(p) * zeta(p + 1)


def _normalized_moment(n: int) -> float:
    """Moment of q^n against the normalized kernel (1/4) q^4 (-df0/dq)."""
    if n == -4:
        return 1.0 / (8.0 * _KERNEL_NORM)
    return ((n + 4.0) / 4.0) * _fermi_dirac_integral(n + 3) / _KERNEL_NORM


def camb_three_point_rule() -> tuple[tuple[float, ...], tuple[float, ...]]:
    """
    Solve the 3-point moment-matching system: 6 unknowns (nodes and weights,
    log-parameterized for positivity and node ordering) against the 6 moment
    conditions n = -4..2. Returns (nodes, kernel weights) in the convention
    of massive_neutrino._CAMB_Q_PERT / _CAMB_W_PERT.
    """
    exponents = (-4, -2, -1, 0, 1, 2)
    targets = np.array([_normalized_moment(n) for n in exponents])
    # Renormalization of residuals so that they have the same scale
    # targets < 1 use absolute error. targets > 1 use relative error
    scales = np.maximum(1.0, np.abs(targets))

    def unpack(v):
        # least_squares assumes that the domain of the function you are trying to optimize has no constraints
        # We therefore map the unparameterized input vector into one which enforces the constraints we want
        # - exp gives positivity (our interval is on 0 to inf)
        # - Treat v[0], v[1], v[2] as differences between the nodes, which enforces q1<q2<q3
        q1 = np.exp(v[0])
        q2 = q1 + np.exp(v[1])
        q3 = q2 + np.exp(v[2])
        # Enforce a node ordering (permuatations of nodes gives multiple equivalent minima. Fixing the ordering eliminates these)
        return np.array([q1, q2, q3]), np.exp(v[3:6])

    def residuals(v):
        q, w = unpack(v)
        matched = np.array([np.dot(w, q ** float(n)) for n in exponents])
        return (matched - targets) / scales

    # Upstream's initial guess (nodes as first value + gaps).
    guess = np.log(np.array([0.8, 2.2, 4.7, 0.08, 3.0, 2.2]))
    sol = least_squares(residuals, guess)
    if not sol.success:
        raise RuntimeError(f"3-point stencil solve failed: {sol.message}")
    nodes, norm_weights = unpack(sol.x)
    kernel_weights = norm_weights * _KERNEL_NORM
    return tuple(float(q) for q in nodes), tuple(float(w) for w in kernel_weights)


def camb_five_point_rule() -> tuple[tuple[float, ...], tuple[float, ...]]:
    """
    Solve CAMB's original 5-point background rule: nodes q2 = 2, q3 = 4,
    q5 = 13 fixed by choice (sigmoid-parameterized q1 in (0, q2) and q4 in
    (q3, q5)), all five weights free, matched to the 7 moment conditions
    n = -4..3. Returns (nodes, kernel weights) in the convention of
    massive_neutrino._CAMB_Q_BG / _CAMB_W_BG.
    """
    exponents = (-4, -2, -1, 0, 1, 2, 3)
    targets = np.array([_normalized_moment(n) for n in exponents])
    scales = np.maximum(1.0, np.abs(targets))
    FIXED_POINTS = [2.0, 4.0, 13.0]

    def unpack(v):
        # Same reparameterization trick as the 3-point rule, with sigmoids
        # for the two-sided constraints: the movable nodes are boxed between
        # the fixed ones, q1 in (0, 2) and q4 in (4, 13).
        q1 = FIXED_POINTS[0] / (1.0 + np.exp(-v[0]))
        q4 = FIXED_POINTS[1] + FIXED_POINTS[2] / (1.0 + np.exp(-v[1]))
        return np.array(
            [q1, FIXED_POINTS[0], FIXED_POINTS[1], q4, FIXED_POINTS[2]]
        ), np.exp(v[2:7])

    def residuals(v):
        q, w = unpack(v)
        matched = np.array([np.dot(w, q ** float(n)) for n in exponents])
        return (matched - targets) / scales

    # Upstream's initial guess: two node parameters, then log-weights.
    guess = np.array([-0.9, -0.6, *np.log([0.008, 0.69, 2.8, 2.05, 0.13])])
    sol = least_squares(residuals, guess)
    if not sol.success:
        raise RuntimeError(f"5-point stencil solve failed: {sol.message}")
    nodes, norm_weights = unpack(sol.x)
    kernel_weights = norm_weights * _KERNEL_NORM
    return tuple(float(q) for q in nodes), tuple(float(w) for w in kernel_weights)
