"""
Script for helper numerical tools
"""

import jax.numpy as jnp
from jax import config, lax, vmap
from jax.scipy.special import factorial
from jaxtyping import Array, Float, Int

config.update("jax_enable_x64", True)


def wigner_d_matrix(
    mu: Float[Array, " num_mu"],
    ells: Int[Array, " num_ell"],
    m: int,
    n: int,
) -> Float[Array, "num_mu num_ell"]:
    """
    Compute Wigner d-matrix elements for rotation.

    Recursively computes reduced Wigner d-matrix elements d^ell_{mn}(beta)
    for CMB lensing calculations using three-term recurrence relation.

    Parameters:
    -----------
    mu : array
        Cosine of rotation angle beta
    ells : array
        Multipole values. Contract: consecutive integers with
        ``ells[0] == m`` -- the recurrence chains *adjacent* entries, so a
        gap or shifted start yields silently wrong values.
    m : int
        First index (must be positive and >= abs(n))
    n : int
        Second index (must satisfy abs(n) <= m)

    Returns:
    --------
    array
        Wigner d-matrix elements, shape (len(mu), len(ells))

    Notes:
    ------
    Implements Kostelec & Rockmore 2003 ("FFTs on the rotation group";
    J. Fourier Anal. Appl. 14, 145 (2008)): the closed-form base case at
    l = m is their Eq. 26, seeding their Eq. 28 three-term recurrence on
    the L^2-normalized sqrt((2l+1)/2) d^l_{mn}, here kept in general
    (m, n) form -- CLASS's twelve lensing_dXX routines (lensing.c) are
    per-(m, n) specializations of the same recurrence.

    The recurrence is preferred over their closed Jacobi form (Eq. 23)
    deliberately: the lensing sums need every l up to ellmax, which the
    chain yields in O(1) each, and evaluating Eq. 23's degree-l Jacobi
    polynomial stably would itself require a recurrence. The naive
    factorial closed form is not an option at all (overflows float64 for
    l >= 86).
    """

    # The l=0 row (reached only via d00's padding, where m=n=0) would hit
    # 0/0 in B and sqrt(negative)/0 in C. Substitute a safe denominator
    ells_safe = jnp.where(ells == 0, 1, ells)
    normA = jnp.sqrt((2 * ells + 3) / (2 * ells + 1))
    normC = jnp.sqrt((2 * ells + 3) / (2 * ells_safe - 1))
    denom = jnp.sqrt(((ells + 1) ** 2 - m**2) * ((ells + 1) ** 2 - n**2))
    A = normA * (ells + 1) * (2 * ells + 1) / denom
    B = -A * m * n / ells_safe / (ells + 1)
    C = (
        -normC
        * jnp.sqrt(ells**2 - m**2)
        * jnp.sqrt(ells**2 - n**2)
        / denom
        * (ells + 1)
        / ells_safe
    )

    # Normalized d^m_{mn} (Eq. 26): the l = m seed of the recurrence.
    def base_val(mu):
        beta = jnp.arccos(mu)
        norm = jnp.sqrt((2 * m + 1) / 2) * jnp.sqrt(
            factorial(2 * m) / (factorial(m + n) * factorial(m - n))
        )
        return norm * jnp.cos(beta / 2.0) ** (m + n) * (-jnp.sin(beta / 2.0)) ** (m - n)

    def one_mu(mu):
        d_seed = base_val(mu)

        def recurrence_step(carry, coeffs):
            # One upward step of the Eq. 28 three-term recurrence: from
            # (d^l, d^{l-1}) produce d^{l+1}. The scan emits the *current*
            # d^l, so the collected outputs are the values at `ells`,
            # beginning with the l = m seed itself.
            d_l, d_lminus1 = carry
            a, b, c = coeffs
            d_lplus1 = a * mu * d_l + b * d_l + c * d_lminus1
            return (d_lplus1, d_l), d_l

        # Second carry slot starts at d^{m-1}_{mn} = 0 (d vanishes for l < m).
        (_, _), d_normalized = lax.scan(recurrence_step, (d_seed, 0.0), (A, B, C))
        # Undo the sqrt((2l+1)/2) L^2 normalization (Eq. 27).
        return d_normalized * jnp.sqrt(2.0 / (2.0 * ells + 1))

    return vmap(one_mu)(mu)


def d00(
    mu: Float[Array, " num_mu"], ells: Int[Array, " num_ell"]
) -> Float[Array, "num_mu num_ell"]:
    # Contract (also d1n/d2n/d3n/d4n): ells must be consecutive integers
    # starting exactly at 2; each wrapper's padding/slicing extends that to
    # the wigner_d_matrix requirement ells[0] == m.
    ells_patched = jnp.concatenate((jnp.array([0, 1]), ells))
    res = wigner_d_matrix(mu, ells_patched, 0, 0)
    return res[:, 2:]  # Return only the ells >= 2


def d1n(
    mu: Float[Array, " num_mu"], ells: Int[Array, " num_ell"], n: int
) -> Float[Array, "num_mu num_ell"]:
    # Wigner matrices where m=1, and abs(n)<=m.
    ells_patched = jnp.concatenate((jnp.array([1]), ells))
    res = wigner_d_matrix(mu, ells_patched, 1, n)
    return res[:, 1:]


def d2n(
    mu: Float[Array, " num_mu"], ells: Int[Array, " num_ell"], n: int
) -> Float[Array, "num_mu num_ell"]:
    # Wigner matrices where m=2, and abs(n)<=m.
    res = wigner_d_matrix(mu, ells, 2, n)
    return res


def d3n(
    mu: Float[Array, " num_mu"], ells: Int[Array, " num_ell"], n: int
) -> Float[Array, "num_mu num_ell"]:
    # Wigner matrices where m=3, and abs(n)<=m.
    ells_sliced = ells[1:]  # Compute starting at ell=3
    res = wigner_d_matrix(mu, ells_sliced, 3, n)
    res_patched = jnp.concatenate(
        (jnp.zeros((mu.size, 1)), res), axis=1
    )  # Pad zeros for ell<3.
    return res_patched


def d4n(
    mu: Float[Array, " num_mu"], ells: Int[Array, " num_ell"], n: int
) -> Float[Array, "num_mu num_ell"]:
    # Wigner matrices where m=4, and abs(n)<=m.
    ells_sliced = ells[2:]  # Compute starting at ell=4
    res = wigner_d_matrix(mu, ells_sliced, 4, n)
    res_patched = jnp.concatenate(
        (jnp.zeros((mu.size, 2)), res), axis=1
    )  # Pad zeros for ell<4.
    return res_patched


def fast_interp(
    x: Float[Array, "*batch"] | float,
    xp_min: Float[Array, ""] | float,
    xp_max: Float[Array, ""] | float,
    fp: Float[Array, " n_table"],
) -> Float[Array, "*batch"]:
    """
    Fast 1D linear interpolation for uniformly-spaced grids.

    Optimized interpolation that avoids searchsorted by exploiting
    uniform grid spacing. Significantly faster than jnp.interp for
    large arrays.

    Returns:
        Interpolated values at query points

    Notes:
    ------
    Credit: JAX issue #16182 (https://github.com/jax-ml/jax/issues/16182)
    Assumes fp is uniformly spaced between xp_min and xp_max.
    """
    # The official jnp.interp is very slow becuase it uses searchsorted.
    # Therefore, we leverage the fact that the fp is linearly increasing, evenly spaced, and has a known range
    # to make this operation much faster.
    eps = 1.0e-6
    n = fp.shape[-1]
    i = (x - xp_min) / (xp_max - xp_min) * (n - 1)  # fix bug in JAX issue
    i = jnp.clip(i, eps, n - 1.0 - eps)  # Avoid index out of range
    i_lower = jnp.floor(i).astype(jnp.int32)
    i_upper = jnp.minimum(i_lower + 1, n - 1)
    w_upper = i - i_lower
    w_lower = 1.0 - w_upper
    return w_lower * fp[i_lower] + w_upper * fp[i_upper]
