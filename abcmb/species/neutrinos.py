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
    Massless neutrinos fluid species implementation.

    Represents relativistic neutrinos with multiple angular momentum modes.

    Methods:
    --------
    rho : Compute neutrino density (units: eV cm^{-3})
    P : Compute neutrino pressure (units: eV cm^{-3})
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

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        args : mapping
            Cosmological parameters (params)

        Returns:
        --------
        float
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

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        args : mapping
            Cosmological parameters (params)

        Returns:
        --------
        float
            Neutrino pressure (units: eV cm^{-3})
        """
        params = args
        return self.rho(lna, params) / 3.0

    def y_ini(self, k: ArrayLike, tau_ini: ArrayLike, args: FluidParams) -> Array:
        """
        Compute initial conditions for massless neutrino perturbations.

        Parameters:
        -----------
        k : float
            Wavenumber (units: Mpc^{-1})
        tau_ini : float
            Initial conformal time (units: Mpc)
        args : mapping
            Cosmological parameters (params)

        Returns:
        --------
        array
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

        Parameters:
        -----------
        k : float
            Wavenumber (units: Mpc^{-1})
        lna : float
            Logarithm of scale factor
        metric_h_prime : float
            Derivative of metric h
        metric_eta_prime : float
            Derivative of metric eta
        y : array
            Current perturbation mode values
        args : PerturbationContext
            Background cosmology, cosmological parameters, and the species
            registry for coupled fluids (use ``args.BG``, ``args.params``,
            ``args.species_list``, ``args.species_dict``)

        Returns:
        --------
        array
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
        return {
            "delta": modes[self.first_idx],
            "theta": modes[self.first_idx + 1],
            "sigma": modes[self.first_idx + 2],
        }


class MassiveNeutrino(Fluid):
    """
    Massive neutrinos fluid species implementation.

    Non-relativistic neutrinos with multiple angular momentum modes.

    Attributes:
    -----------
    num_ells_per_bin : int
        Number of multipole moments per momentum bin for massive neutrino hierarchy

    Methods:
    --------
    rho : Compute massive neutrino density (units: eV cm^{-3})
    P : Compute massive neutrino pressure (units: eV cm^{-3})
    y_ini : Compute initial perturbation conditions
    y_prime : Compute perturbation time derivatives
    rho_delta : Compute density perturbation (units: eV cm^{-3})
    rho_plus_P_theta : Compute velocity perturbation (units: eV cm^{-3} Mpc^{-1})
    rho_plus_P_sigma : Compute shear perturbation (units: eV cm^{-3})
    """

    num_ells_per_bin: int = eqx.field(default=0, static=True)

    q_3p = jnp.array([0.913201, 3.37517, 7.79184])
    w_3p = jnp.array([0.0687359, 3.31435, 2.29911])
    q_5p = jnp.array([0.583165, 2.0, 4.0, 7.26582, 13.0])
    w_5p = jnp.array([0.0081201, 0.689407, 2.8063, 2.05156, 0.12681])

    dlfdlq_3p = -q_3p / (
        1.0 + jnp.exp(-q_3p)
    )  # Log derivative of fermi-dirac w.r.t. momentum

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

        Parameters:
        -----------
        lna : float or ArrayLike
            Logarithm of scale factor
        args : mapping
            Cosmological parameters (params)

        Returns:
        --------
        float or ArrayLike
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

        Parameters:
        -----------
        lna : float or ArrayLike
            Logarithm of scale factor
        args : mapping
            Cosmological parameters (params)

        Returns:
        --------
        float or ArrayLike
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

        Parameters:
        -----------
        k : float
            Wavenumber (units: Mpc^{-1})
        tau_ini : float
            Initial conformal time (units: Mpc)
        args : mapping
            Cosmological parameters (params)

        Returns:
        --------
        array
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
            q = self.q_3p[i]
            # ZZ : Techniclly Psi1 requires epsilon/q = 1/v, but at early times this should be 1. Should check this accuracy!
            first_three = (
                jnp.array([delta / 4.0, theta / 3.0, sigma / 2.0])
                * q
                / (1.0 + jnp.exp(-q))
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

        Parameters:
        -----------
        k : float
            Wavenumber (units: Mpc^{-1})
        lna : float
            Logarithm of scale factor
        metric_h_prime : float
            Derivative of metric h
        metric_eta_prime : float
            Derivative of metric eta
        y : array
            Current perturbation mode values
        args : PerturbationContext
            Background cosmology, cosmological parameters, and the species
            registry for coupled fluids (use ``args.BG``, ``args.params``,
            ``args.species_list``, ``args.species_dict``)

        Returns:
        --------
        array
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
            dlnf0_dlnq = -q / (1 + jnp.exp(-q))

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

    def rho_delta(self, lna: ArrayLike, y: Array, args: FluidParams) -> Array | float:
        """
        Compute massive neutrino density perturbation.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        y : array
            Perturbation mode values
        args : mapping
            Cosmological parameters (params)

        Returns:
        --------
        float
            Density perturbation (units: eV cm^{-3})
        """
        params = args
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
        self, lna: ArrayLike, y: Array, args: FluidParams
    ) -> Array | float:
        """
        Compute massive neutrino velocity perturbation.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        y : array
            Perturbation mode values
        args : mapping
            Cosmological parameters (params)

        Returns:
        --------
        float
            Velocity perturbation (units: eV cm^{-3} Mpc^{-1})
        """
        params = args
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
        self, lna: ArrayLike, y: Array, args: FluidParams
    ) -> Array | float:
        """
        Compute massive neutrino shear perturbation.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        y : array
            Perturbation mode values
        args : mapping
            Cosmological parameters (params)

        Returns:
        --------
        float
            Shear perturbation (units: eV cm^{-3})
        """
        params = args
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
        BG, params = args
        rho = vmap(self.rho, in_axes=(0, None))(lna, params)  # (Nlna,)
        rhoP = rho + vmap(self.P, in_axes=(0, None))(lna, params)

        rho_delta = vmap(self.rho_delta, in_axes=(0, 1, None))(
            lna, modes, params
        )  # (Nlna, Nk)
        rho_P_theta = vmap(self.rho_plus_P_theta, in_axes=(0, 1, None))(
            lna, modes, params
        )
        rho_P_sigma = vmap(self.rho_plus_P_sigma, in_axes=(0, 1, None))(
            lna, modes, params
        )

        return {
            "delta": rho_delta / rho[:, None],
            "theta": rho_P_theta / rhoP[:, None],
            "sigma": rho_P_sigma / rhoP[:, None],
        }
