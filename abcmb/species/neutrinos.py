"""
Massless and massive neutrinos.
"""

import equinox as eqx
import jax.numpy as jnp
from jax import vmap
from jax.typing import ArrayLike
from jaxtyping import Array

from .. import constants as cnst
from .base import Fluid, FluidParams, OutputArgs, PerturbationContext, StandardFluid


class MasslessNeutrino(StandardFluid):
    """
    Represents relativistic neutrinos with multiple angular momentum modes.
    """

    name = "MasslessNeutrino"
    is_matter = False
    is_neutrino = True

    def __init__(self, first_idx, options):
        super().__init__(first_idx, options)
        self.num_equations = options["l_max_massless_nu"] + 1

    def rho(self, lna: ArrayLike, args: FluidParams) -> Array | float:
        """
        Compute neutrino density.

        Returns:
            Neutrino density (units: eV cm^{-3})
        """
        params = args

        a = jnp.exp(lna)
        rho = (
            params["N_nu_massless"]
            * 2.0
            * 7.0
            / 8.0
            * jnp.pi**2
            / 30.0
            * params["T_nu_massless"] ** 4
            * params["TCMB0"] ** 4
            / a**4
        )  # eV^4
        rho = rho / (cnst.c * cnst.hbar) ** 3  # Convert to eV cm^{-3}
        return rho

    def P(self, lna: ArrayLike, args: FluidParams) -> Array | float:
        """
        Compute neutrino pressure.

        Returns:
           Neutrino pressure (units: eV cm^{-3})
        """
        params = args
        return self.rho(lna, params) / 3.0

    def y_ini(self, k: ArrayLike, tau_ini: ArrayLike, args: FluidParams) -> Array:
        """
        Compute initial conditions for massless neutrino perturbations.
        Follows Ma & Bertschinger (1995), ApJ 455, 7 (arXiv:astro-ph/9506072).
        The adiabatic initial conditions are their Eq. (96)
        with C = 1/2, plus the next-order om*tau corrections used by CLASS
        (perturbations.c, adiabatic ICs: theta_ur, shear_ur).

         Returns:
           Initial perturbation mode values (units: 1/Mpc for theta, else dimensionless)
        """
        params = args
        R_nu = params["R_nu"]

        delta = -((k * tau_ini) ** 2) / 3.0 * (1.0 - params["om"] * tau_ini / 5.0)
        theta = (
            -k
            * (k * tau_ini) ** 3
            / 36.0
            / (4.0 * R_nu + 15.0)
            * (
                4.0 * R_nu
                + 11.0
                + 12.0
                - 3.0
                * (8.0 * R_nu**2 + 50.0 * R_nu + 275.0)
                / 20.0
                / (2.0 * R_nu + 15.0)
                * tau_ini
                * params["om"]
            )
        )
        sigma = (
            (k * tau_ini) ** 2
            / (45.0 + 12.0 * R_nu)
            * 2.0
            * (
                1.0
                + (4.0 * R_nu - 5.0)
                / 4.0
                / (2.0 * R_nu + 15.0)
                * tau_ini
                * params["om"]
            )
        )

        # Return the four non-zero ell modes, and all higher ell-modes are zero to start.
        # For the neutrinos we track Fnu_2 = 2*sigma, for better structure within the hierarchy.
        return jnp.concatenate(
            (jnp.array([delta, theta, sigma]), jnp.zeros(self.num_equations - 3))
        )

    def y_prime(
        self,
        k: ArrayLike,
        lna: ArrayLike,
        metric_h_prime: ArrayLike,
        metric_eta_prime: ArrayLike,
        y: Array,
        args: PerturbationContext,
    ) -> Array:
        """
        Compute time derivatives of massless neutrino perturbations.
        Follows Ma & Bertschinger (1995), ApJ 455, 7 (arXiv:astro-ph/9506072).
        The collisionless hierarchy is their Eq. (49), truncated at l_max with
        their Eq. (51).

        Returns:
            Time derivatives of perturbation modes (units: 1/Mpc for theta, else dimensionless)
        """
        BG, params = args.BG, args.params
        aH = BG.aH(lna, params)
        tau = BG.tau(lna)

        L = jnp.arange(self.num_equations) + self.first_idx
        F = y[L]
        delta = F[0]
        theta = F[1]
        sigma = F[2]

        # density, velocity, shear perturbations
        delta_prime = -4.0 / 3.0 / aH * theta - 2.0 / 3.0 * metric_h_prime
        theta_prime = k**2 / aH * (delta / 4.0 - sigma)
        sigma_prime = (
            4.0 / 15.0 / aH * theta
            - 3.0 / 10.0 * k / aH * F[3]
            + 2.0 / 15.0 * metric_h_prime
            + 4.0 / 5.0 * metric_eta_prime
        )
        F3_prime = 1.0 / 7.0 * k / aH * (6.0 * sigma - 4.0 * F[4])

        # Rest of the Boltzmann Hierarchy
        lmax = self.num_equations - 1
        L = jnp.arange(4, lmax)
        Fl_prime = 1.0 / (2.0 * L + 1.0) * k / aH * (L * F[L - 1] - (L + 1) * F[L + 1])
        Flmax_prime = k / aH * F[lmax - 1] - (lmax + 1) / aH / tau * F[lmax]

        return jnp.concatenate(
            (
                jnp.array([delta_prime, theta_prime, sigma_prime, F3_prime]),
                Fl_prime,
                jnp.array([Flmax_prime]),
            )
        )

    def output_perturbations(
        self, lna: ArrayLike, modes: Array, args: OutputArgs
    ) -> dict[str, Array]:
        """Output keys: ``delta``, ``theta``, ``sigma``."""
        return {
            "delta": modes[self.first_idx],
            "theta": modes[self.first_idx + 1],
            "sigma": modes[self.first_idx + 2],
        }


class MassiveNeutrino(Fluid):
    """
    Non-relativistic neutrinos with multiple angular momentum modes.

    Notes
    -----
    The sparse momentum quadrature (q_3p/w_3p, q_5p/w_5p) replaces MB95's
    dense q-grid with CAMB rules (massive_neutrinos.f90), constructed by
    moment-matching against the perturbation kernel q^4 * (-df0/dq)
    (arXiv:1201.3654, Appendix A; generator at
    camb.info/maple/nu_integration_kernels.py). The 3-point rule is CAMB's
    current one -- the unique 3-node match of the moments n = -4..2. The
    5-point rule is CAMB's *original* one, kept upstream only as a comment
    since being replaced by an exact+least-squares refit; swapping in the
    modern rule would change background rho/P at the ~1e-4 level.
    """

    num_ells_per_bin: int = eqx.field(default=0, static=True)

    # CAMB's sparse momentum quadrature (massive_neutrinos.f90): nodes q_i
    # and kernel weights w_i satisfying, with f0(q) = 1/(e^q + 1),
    #
    #   (1/4) * int dq q^2 f0(q) F(q)  ~=  sum_i w_i (1+e^{-q_i}) F(q_i)/q_i^2
    #
    # (the 1/4 is cancelled by the 4/pi^2 prefactors in the integrals below).
    # The weights fold the q^2 f0 measure into themselves, which is why every
    # integrand here carries the compensating (1 + e^{-q})/q^2 factor.
    #
    # Generated by moment-matching against the perturbation kernel
    # q^4 * (-df0/dq): Howlett, Lewis, Hall & Challinor (arXiv:1201.3654),
    # Appendix A; executable generator at
    # https://camb.info/maple/nu_integration_kernels.py (reproduces q_3p/w_3p
    # to all published digits; q_5p/w_5p is CAMB's since-replaced original).
    q_3p = jnp.array([0.913201, 3.37517, 7.79184])
    w_3p = jnp.array([0.0687359, 3.31435, 2.29911])
    q_5p = jnp.array([0.583165, 2.0, 4.0, 7.26582, 13.0])
    w_5p = jnp.array([0.0081201, 0.689407, 2.8063, 2.05156, 0.12681])

    # d(ln f0)/d(ln q) at the 3-point nodes, for the Fermi-Dirac f0: the
    # metric-source coupling of the Psi hierarchy and its initial conditions
    # (MB95 Eqs. 56 and 97) are written in terms of this quantity.
    dlnf0_dlnq_3p = -q_3p / (1.0 + jnp.exp(-q_3p))

    name = "MassiveNeutrino"
    is_matter = True
    is_neutrino = True

    def __init__(self, first_idx, options):

        super().__init__(first_idx, options)
        self.num_ells_per_bin = options["l_max_massive_nu"] + 1
        self.num_equations = 3 * self.num_ells_per_bin

    def rho(self, lna: ArrayLike, args: FluidParams) -> Array | float:
        """
        Compute massive neutrino density.

        Returns:
           Massive neutrino density (units: eV cm^{-3})
        """
        params = args

        # Ensure lna is at least 1D for broadcasting
        lna_arr = jnp.atleast_1d(lna)  # shape (N,)
        # shape (N,1):
        a = jnp.exp(lna_arr)[:, None]
        T = params["T_nu_massive"] * params["TCMB0"] / a
        x = params["m_nu_massive"] / T

        # q_5p, w_5p are shape (5,), broadcast with (N, 1)
        integrand = (
            (1.0 + jnp.exp(-self.q_5p)) / self.q_5p**2 * jnp.sqrt(self.q_5p**2 + x**2)
        )  # (N, 5)

        # Dot product along last axis with w_5p
        integral = jnp.dot(integrand, self.w_5p)  # (N,)

        rho_val = (
            params["N_nu_massive"]
            * 4.0
            * T[:, 0] ** 4
            / jnp.pi**2
            * integral
            / cnst.hbar**3
            / cnst.c**3
        )

        # Remove extra dimension if original input was scalar
        return jnp.squeeze(rho_val) if jnp.ndim(lna) == 0 else rho_val

    def P(self, lna: ArrayLike, args: FluidParams) -> Array | float:
        """
        Compute massive neutrino pressure.

        Returns:
           Massive neutrino pressure (units: eV cm^{-3})
        """
        params = args

        # Ensure lna is at least 1D for broadcasting
        lna_arr = jnp.atleast_1d(lna)  # shape (N,)
        # shape (N,1)
        a = jnp.exp(lna_arr)[:, None]
        T = params["T_nu_massive"] * params["TCMB0"] / a
        x = params["m_nu_massive"] / T

        # q_5p, w_5p are shape (5,), broadcast with (N, 1)
        integrand = (1.0 + jnp.exp(-self.q_5p)) / jnp.sqrt(
            self.q_5p**2 + x**2
        )  # (N, 5)

        # Dot product along last axis with w_5p
        integral = jnp.dot(integrand, self.w_5p)  # (N,)

        P_val = (
            params["N_nu_massive"]
            * 4.0
            / 3.0
            * T[:, 0] ** 4
            / jnp.pi**2
            * integral
            / cnst.hbar**3
            / cnst.c**3
        )

        # Remove extra dimension if original input was scalar
        return jnp.squeeze(P_val) if jnp.ndim(lna) == 0 else P_val

    def y_ini(self, k: ArrayLike, tau_ini: ArrayLike, args: FluidParams) -> Array:
        """
        Compute initial conditions for massive neutrino perturbations.
        Follows Ma & Bertschinger (1995), ApJ 455, 7 (arXiv:astro-ph/9506072).
        the initial conditions are their Eq. (97) (with
        epsilon/q -> 1 at early times).

        Returns:
            Initial perturbation mode values (units: 1/Mpc for kPsi1, else dimensionless)
        """
        params = args

        # Initial conditions for massless neutrinos first, needed here.
        R_nu = params["R_nu"]

        delta = -((k * tau_ini) ** 2) / 3.0 * (1.0 - params["om"] * tau_ini / 5.0)
        theta = (
            -k
            * (k * tau_ini) ** 3
            / 36.0
            / (4.0 * R_nu + 15.0)
            * (
                4.0 * R_nu
                + 11.0
                + 12.0
                - 3.0
                * (8.0 * R_nu**2 + 50.0 * R_nu + 275.0)
                / 20.0
                / (2.0 * R_nu + 15.0)
                * tau_ini
                * params["om"]
            )
        )
        sigma = (
            (k * tau_ini) ** 2
            / (45.0 + 12.0 * R_nu)
            * 2.0
            * (
                1.0
                + (4.0 * R_nu - 5.0)
                / 4.0
                / (2.0 * R_nu + 15.0)
                * tau_ini
                * params["om"]
            )
        )

        bins = []
        for i in range(3):
            # MB95 Eq. (97): (Psi0, kPsi1, Psi2) = -(delta/4, theta/3, sigma/2)
            # * dlnf0/dlnq.
            # ZZ : Techniclly Psi1 requires epsilon/q = 1/v, but at early times this should be 1. Should check this accuracy!
            first_three = (
                -jnp.array([delta / 4.0, theta / 3.0, sigma / 2.0])
                * self.dlnf0_dlnq_3p[i]
            )
            bins.append(
                jnp.concatenate((first_three, jnp.zeros(self.num_ells_per_bin - 3)))
            )

        return jnp.concatenate(bins)

    def y_prime(
        self,
        k: ArrayLike,
        lna: ArrayLike,
        metric_h_prime: ArrayLike,
        metric_eta_prime: ArrayLike,
        y: Array,
        args: PerturbationContext,
    ) -> Array:
        """
        Compute time derivatives of massive neutrino perturbations.
        Follows Ma & Bertschinger (1995), ApJ 455, 7 (arXiv:astro-ph/9506072).
        The per-momentum-bin Psi_l hierarchy is their Eq. (56), truncated with
        their Eq. (58);
        the momentum-integrated delta/theta/sigma perturbations
        are their Eq. (55);
        Returns:
            Time derivatives of perturbation modes (units: 1/Mpc for kPsi1, else dimensionless)
        """
        BG, params = args.BG, args.params

        a = jnp.exp(lna)
        T = params["T_nu_massive"] * params["TCMB0"] / a
        x = params["m_nu_massive"] / T
        aH = BG.aH(lna, params)
        tau = BG.tau(lna)

        # Iterate through momentum bins
        bins = []
        for i in range(3):
            q = self.q_3p[i]
            epsilon = jnp.sqrt(q**2 + x**2)
            dlnf0_dlnq = self.dlnf0_dlnq_3p[i]

            # NOTE: The entries are [Psi0, k * Psi1, Psi2, ...]. If accessing Psi1 make sure to divide out k
            L = (
                jnp.arange(self.num_ells_per_bin)
                + self.first_idx
                + i * self.num_ells_per_bin
            )
            Psi = y[L]

            Psi0_prime = -q / epsilon / aH * Psi[1] + metric_h_prime / 6.0 * dlnf0_dlnq
            kPsi1_prime = q * k**2 / 3.0 / epsilon / aH * (Psi[0] - 2.0 * Psi[2])
            Psi2_prime = (
                q * k / 5.0 / epsilon / aH * (2.0 * Psi[1] / k - 3.0 * Psi[3])
                - (metric_h_prime / 15.0 + 2.0 * metric_eta_prime / 5.0) * dlnf0_dlnq
            )

            # Intermediate hierarchy, 3<=L<lmax
            lmax = self.num_ells_per_bin - 1
            L_inter = jnp.arange(3, lmax)  # Doesn't include lmax.
            Psi_inter_prime = (
                q
                * k
                / epsilon
                / aH
                / (2 * L_inter + 1)
                * (L_inter * Psi[L_inter - 1] - (L_inter + 1) * Psi[L_inter + 1])
            )

            # lmax mode
            Psi_lmax_prime = (
                q * k / aH / epsilon * Psi[lmax - 1] - (lmax + 1) / aH / tau * Psi[lmax]
            )

            # Putting it all together
            bins.append(
                jnp.concatenate(
                    (
                        jnp.array([Psi0_prime, kPsi1_prime, Psi2_prime]),
                        Psi_inter_prime,
                        jnp.array([Psi_lmax_prime]),
                    )
                )
            )

        return jnp.concatenate(bins)

    def rho_delta(
        self, lna: ArrayLike, y: Array, args: PerturbationContext
    ) -> Array | float:
        """
        Compute massive neutrino density perturbation.

        Returns:
           Density perturbation (units: eV cm^{-3})
        """
        return self._rho_delta(lna, y, args.params)

    def _rho_delta(
        self, lna: ArrayLike, y: Array, params: FluidParams
    ) -> Array | float:
        # Params-only core, shared with output_perturbations (whose OutputArgs
        # carries no species registry to build a full context from).
        a = jnp.exp(lna)
        T = params["T_nu_massive"] * params["TCMB0"] / a  # (N,)
        x = params["m_nu_massive"] / T  # (N,)

        res = 0.0
        for i in range(3):
            q = self.q_3p[i]
            w = self.w_3p[i]
            epsilon = jnp.sqrt(q**2 + x**2)
            Psi0 = y[self.first_idx + i * self.num_ells_per_bin]

            res += w * (1.0 + jnp.exp(-q)) * epsilon / q**2 * Psi0
        return (
            params["N_nu_massive"]
            * res
            * 4.0
            / jnp.pi**2
            * T**4
            / cnst.hbar**3
            / cnst.c**3
        )

    def rho_plus_P_theta(
        self, lna: ArrayLike, y: Array, args: PerturbationContext
    ) -> Array | float:
        """
        Compute massive neutrino velocity perturbation.

        Returns:
            Velocity perturbation (units: eV cm^{-3} Mpc^{-1})
        """
        return self._rho_plus_P_theta(lna, y, args.params)

    def _rho_plus_P_theta(
        self, lna: ArrayLike, y: Array, params: FluidParams
    ) -> Array | float:
        a = jnp.exp(lna)
        T = params["T_nu_massive"] * params["TCMB0"] / a  # (N,)

        res = 0.0
        for i in range(3):
            q = self.q_3p[i]
            w = self.w_3p[i]
            kPsi1 = y[self.first_idx + 1 + i * self.num_ells_per_bin]

            res += w * (1.0 + jnp.exp(-q)) / q * kPsi1
        return (
            params["N_nu_massive"]
            * res
            * 4.0
            / jnp.pi**2
            * T**4
            / cnst.hbar**3
            / cnst.c**3
        )

    def rho_plus_P_sigma(
        self, lna: ArrayLike, y: Array, args: PerturbationContext
    ) -> Array | float:
        """
        Compute massive neutrino shear perturbation.

        Returns:
            Shear perturbation (units: eV cm^{-3})
        """
        return self._rho_plus_P_sigma(lna, y, args.params)

    def _rho_plus_P_sigma(
        self, lna: ArrayLike, y: Array, params: FluidParams
    ) -> Array | float:
        a = jnp.exp(lna)
        T = params["T_nu_massive"] * params["TCMB0"] / a  # (N,)
        x = params["m_nu_massive"] / T  # (N,)

        res = 0.0
        for i in range(3):
            q = self.q_3p[i]
            w = self.w_3p[i]
            epsilon = jnp.sqrt(q**2 + x**2)
            Psi2 = y[self.first_idx + 2 + i * self.num_ells_per_bin]

            res += w * (1.0 + jnp.exp(-q)) / epsilon * Psi2
        return (
            params["N_nu_massive"]
            * res
            * 8.0
            / 3.0
            / jnp.pi**2
            * T**4
            / cnst.hbar**3
            / cnst.c**3
        )

    def output_perturbations(
        self, lna: ArrayLike, modes: Array, args: OutputArgs
    ) -> dict[str, Array]:
        """Output keys: ``delta``, ``theta``, ``sigma`` -- momentum-integrated
        from the binned hierarchy, so directly comparable to the standard
        fluid outputs."""
        BG, params = args
        rho = vmap(self.rho, in_axes=(0, None))(lna, params)  # (Nlna,)
        rhoP = rho + vmap(self.P, in_axes=(0, None))(lna, params)

        rho_delta = vmap(self._rho_delta, in_axes=(0, 1, None))(
            lna, modes, params
        )  # (Nlna, Nk)
        rho_P_theta = vmap(self._rho_plus_P_theta, in_axes=(0, 1, None))(
            lna, modes, params
        )
        rho_P_sigma = vmap(self._rho_plus_P_sigma, in_axes=(0, 1, None))(
            lna, modes, params
        )

        return {
            "delta": rho_delta / rho[:, None],
            "theta": rho_P_theta / rhoP[:, None],
            "sigma": rho_P_sigma / rhoP[:, None],
        }
