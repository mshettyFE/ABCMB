import os
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from interpax import CubicSpline
from jax import config, grad, lax, vmap
from jaxtyping import Array, Float, Int
from scipy.special import roots_legendre

from . import ABCMBTools as tools

if TYPE_CHECKING:
    from .background import Background
    from .inputs._schema_types import Options, Params
    from .perturbations import PerturbationTable

file_dir = os.path.dirname(__file__)

config.update("jax_enable_x64", True)

MINIMUM_ALLOWED_L = 2


# Tabulated spherical-Bessel kernels over (x, l): phi0 = j_l, phi1 = j_l',
# phi2 = (3 j_l'' + j_l)/2 -- the three line-of-sight source kernels. These
# are the m = 0 radial functions of Hu & White, astro-ph/9702170 Eq. (15):
# j_l^(00), j_l^(10), j_l^(20), paired in Cl_one_ell with sourceT0/T1/T2
# respectively. The E-mode kernel j_l^(22) = sqrt(3/8 (l+2)!/(l-2)!) j_l/x^2
# from the same equation is built there from phi0 rather than tabulated.
# Line-of-sight integration itself is Seljak & Zaldarriaga, astro-ph/9603033.
# Each kernel has its own x grid because each is tabulated over its own
# function's support.
# Regenerate with abcmb/_generators/bessel_tables.py (scipy-based, offline).

# Axis names: num_ell_tab is the *tabulated* ell grid
# num_x is the per-kernel sample count along its own x grid.
_bessel_tables = np.load(file_dir + "/data/bessel_tables.npz")

# Sorted sparse list of tabulated ell values. That it starts at ell=2 and
# increases strictly is checked by _generators/bessel_tables.build_tables,
# where the table is made
bessel_l_tab: Int[Array, " num_ell_tab"] = jnp.array(_bessel_tables["l"], dtype="int")

xphi0_tab: Float[Array, "num_x num_ell_tab"] = jnp.array(_bessel_tables["xphi0"])
phi0_tab: Float[Array, "num_x num_ell_tab"] = jnp.array(_bessel_tables["phi0"])
xphi1_tab: Float[Array, "num_x num_ell_tab"] = jnp.array(_bessel_tables["xphi1"])
phi1_tab: Float[Array, "num_x num_ell_tab"] = jnp.array(_bessel_tables["phi1"])
xphi2_tab: Float[Array, "num_x num_ell_tab"] = jnp.array(_bessel_tables["xphi2"])
phi2_tab: Float[Array, "num_x num_ell_tab"] = jnp.array(_bessel_tables["phi2"])


def _n_cols_through(ell: int, what: str) -> int:
    """
    Number of leading ``bessel_l_tab`` columns needed to bracket ``ell``.

    """
    i = int(jnp.searchsorted(bessel_l_tab, ell, side="left"))
    if i == bessel_l_tab.size:
        raise ValueError(
            f"{what} = {ell} exceeds the tabulated Bessel range "
            f"(2..{int(bessel_l_tab[-1])}). Lower it, or regenerate "
            f"abcmb/data/bessel_tables.npz with abcmb/_generators/bessel_tables.py."
        )

    # i stops just short of including l. +1 to gaurentee inclusion of ell
    return i + 1


# Commit the tables to the default device (the accelerator when there is one,
# otherwise a no-op)
_tab_device = jax.devices()[0]
bessel_l_tab = jax.device_put(bessel_l_tab, _tab_device)
xphi0_tab = jax.device_put(xphi0_tab, _tab_device)
phi0_tab = jax.device_put(phi0_tab, _tab_device)
xphi1_tab = jax.device_put(xphi1_tab, _tab_device)
phi1_tab = jax.device_put(phi1_tab, _tab_device)
xphi2_tab = jax.device_put(xphi2_tab, _tab_device)
phi2_tab = jax.device_put(phi2_tab, _tab_device)


# large-x asymptotic expansion of spherical bessel functions
def Q(l: Float[Array, ""] | float, x: Float[Array, " n"]) -> Float[Array, " n"]:
    return jnp.sqrt(x**2 - l**2) - l * jnp.pi / 2 + l * jnp.arcsin(l / x)


def J(l: Float[Array, ""] | float, x: Float[Array, " n"]) -> Float[Array, " n"]:
    return jnp.sqrt(2 / jnp.pi / jnp.sqrt(x**2 - l**2)) * jnp.cos(Q(l, x) - jnp.pi / 4)


def j(l: Int[Array, ""] | int, x: Float[Array, " n"]) -> Float[Array, " n"]:
    return jnp.sqrt(jnp.pi / 2 / x) * J(l + 1 / 2, x)


class SpectrumSolver(eqx.Module):
    r"""
    CMB angular power spectrum computation.

    Computes temperature and polarization angular power spectra by
    integrating transfer functions over wavenumber and time.

    Attributes:
    -----------
    evaluated_ells : Array
        Internal contiguous multipole axis, always anchored at ell=2 (a
        contract of the Wigner-d recurrences and the ``[ells - 2]`` output
        slicing); extends 'lensing_buffer' past ``l_max`` when lensing is on. Used for the
        raw-Cl spline in both the lensed and unlensed paths.
    sampled_ells : Array
        The tabulated multipoles the raw Cls are actually solved at -- a
        sparse subset of bessel_l_tab, splined onto evaluated_ells by get_Cl.
    k_axis_transfer : Array
        Wavenumber grid for transfer function integration (units: Mpc^{-1}).
        Required, no default: build with
        ``model_setup.get_k_axis_transfer``.
    options : Options
        The resolved options dictionary.

    Methods:
    --------
    Pk_lin : Compute linear matter power spectrum
    get_Cl : Compute angular power spectra for multiple :math:`\ell`
    Cl_one_ell : Compute angular power spectrum for single :math:`\ell`
    integrand_T0 : Compute SW+ISW temperature source integrand
    integrand_T1 : Compute ISW temperature source integrand
    integrand_T2 : Compute polarization temperature source integrand
    integrand_E : Compute E-mode polarization source integrand
    """

    # bounds of l for output multiple spectra
    ellmin: int
    ellmax: int

    # L grid on which the spline is evaluated at
    evaluated_ells: Int[Array, " num_lensing_ell"]
    # Knots of the cubic splines
    sampled_ells: Int[Array, " num_raw_ell"]

    k_axis_transfer: Float[Array, " n_k_transfer"]

    options: "Options" = eqx.field(default_factory=dict)

    def __init__(
        self,
        k_axis_transfer: Float[Array, " n_k_transfer"],
        options: "Options",
    ) -> None:
        """
        Initialize CMB spectrum solver.
        """

        self.options = options
        self.k_axis_transfer = k_axis_transfer

        self.ellmin = options["l_min"]
        self.ellmax = options["l_max"]

        if self.ellmin < MINIMUM_ALLOWED_L:
            raise ValueError(
                f"l_min must be >= {MINIMUM_ALLOWED_L} (the monopole and dipole are not "
                f"computed, and bessel_l_tab starts at 2); got {self.ellmin}"
            )

        if options["lensing"]:
            # Pad l support of spline to account for lensing convolution
            lensing_ellmax = self.ellmax + options["lensing_buffer"]
            self.evaluated_ells = jnp.arange(MINIMUM_ALLOWED_L, lensing_ellmax + 1)
            self.sampled_ells = bessel_l_tab[
                : _n_cols_through(
                    lensing_ellmax,
                    "l_max + lensing_buffer (lensing extends the internal ell axis)",
                )
            ]
        else:
            self.evaluated_ells = jnp.arange(MINIMUM_ALLOWED_L, self.ellmax + 1)
            self.sampled_ells = bessel_l_tab[: _n_cols_through(self.ellmax, "l_max")]

    def _primordial_spectrum(
        self, k: Float[Array, " k"], params: "Params"
    ) -> Float[Array, " k"]:
        """
        Compute primordial curvature power spectrum.

        Returns:
            Primordial power spectrum P_R(k), units Mpc^3
        """
        return (
            params["A_s"]
            * (k / self.options["k_pivot"]) ** (params["n_s"] - 1.0)
            * (2 * jnp.pi**2 / k**3)
        )

    def _Pk_from(
        self,
        delta: Float[Array, "n_lna n_k"],
        k: Float[Array, " k"],
        z: float,
        PT: "PerturbationTable",
        params: "Params",
    ) -> Float[Array, " k"]:
        """
        Linear power spectrum of a density contrast table, at (z, k).

        Returns:
            P(k, z), units Mpc^3
        """
        lna = -jnp.log(1.0 + z)

        # vmapped interpolation over Nk (columns of the 2D arrays)
        interp_over_lna = jax.vmap(
            lambda y: jnp.interp(lna, PT.lna, y),
            in_axes=1,  # loop over columns
        )

        delta_lna = interp_over_lna(delta)  # shape (Nk,)

        # now interpolate over k
        delta_k = jnp.interp(k, PT.k, delta_lna)

        return delta_k**2 * self._primordial_spectrum(k, params)

    def Pk_lin(
        self,
        k: Float[Array, " k"],
        z: float,
        PT: "PerturbationTable",
        params: "Params",
    ) -> Float[Array, " k"]:
        """
        Compute linear matter power spectrum at wavenumbers k and redshift z.

        Includes every species counted as matter (``is_matter``), massive
        neutrinos among them; see :meth:`Pk_cb` for the baryon+CDM variant.

        Returns:
            Linear matter power spectrum P(k, z), units Mpc^3
        """
        return self._Pk_from(PT.delta_m, k, z, PT, params)

    def Pk_cb(
        self,
        k: Float[Array, " k"],
        z: float,
        PT: "PerturbationTable",
        params: "Params",
    ) -> Float[Array, " k"]:
        """
        Compute linear Baryon+DarkMatter power spectrum at wavenumbers k and
        redshift z. Does not include any other massive species present --
        notably massive neutrinos, which :meth:`Pk_lin` does include.

        Returns:
            Linear Baryon+DarkMatter power spectrum P_cb(k, z), units Mpc^3
        """
        return self._Pk_from(PT.delta_cb, k, z, PT, params)

    def lensing_power_spectrum(
        self,
        k: Float[Array, " k"],
        lna: float,
        PT: "PerturbationTable",
        BG: "Background",
        params: "Params",
    ) -> Float[Array, " k"]:
        """
        Computes the lensing power spectrum at wavenumbers k and redshift z.
        Eq.(3.15) in astro-ph/0601594

        Returns:
            Lensing matter power spectrum P(k, z), dimensionless.
        """
        a = jnp.exp(lna)
        z = 1.0 / a - 1.0
        aH = BG.aH(lna, params)

        # Omega_m(a) = rho_m / rho_crit(a), and rho_crit = rho_tot because the
        # density budget is closed flat.
        Om = BG.rho_matter(lna, params) / BG.rho_tot(lna, params)

        Pk = self.Pk_lin(k, z, PT, params)  # Mpc^3

        return 9.0 / 8.0 / jnp.pi**2 * Om**2 * aH**4 * Pk / k

    def lensing_Cl(
        self,
        ells: Int[Array, " num_lensing_ell"],
        PT: "PerturbationTable",
        BG: "Background",
        params: "Params",
    ) -> Float[Array, " num_lensing_ell"]:
        """
        Angular lensing power spectrum at multipole ell.

        IMPORTANT: Assumes Limber approximation throughout, even at ell=2.

        Eq.(3.14) in astro-ph/0601594, except shifts ell -> ell+1/2
        to align with natural indexing of spherical Bessel (matches CLASS).

        Returns:
            Angular lensing matter power spectrum Cl^phiphi, dimensionless.
        """

        coeff = 8.0 * jnp.pi**2 / (ells + 0.5) ** 3

        def chi(lna):
            return BG.tau0 - BG.tau(lna)

        lna_axis = jnp.linspace(BG.lna_rec, 0.0, self.options["lna_lensing_points"])
        lna_floor = lna_axis[-2]

        def integrand_func(lna):
            # substitute lna_safe everywhere, then mask the result to 0 at the boundary.
            lna_safe = jnp.where(lna < 0.0, lna, lna_floor)
            chi_safe = chi(lna_safe)
            k = (ells + 0.5) / chi_safe
            window = (chi(BG.lna_rec) - chi_safe) / chi(BG.lna_rec) / chi_safe
            res = (
                chi_safe
                / BG.aH(lna_safe, params)
                * window**2
                * self.lensing_power_spectrum(k, lna_safe, PT, BG, params)
            )
            return jnp.where(lna < 0.0, res, 0.0)

        integrand = vmap(integrand_func)(lna_axis)
        return coeff * jnp.trapezoid(integrand, lna_axis, axis=0)

    def lensed_Cls(
        self,
        ells: Int[Array, " num_lensing_ell"],
        ClTT_unlensed: Float[Array, " num_lensing_ell"],
        ClTE_unlensed: Float[Array, " num_lensing_ell"],
        ClEE_unlensed: Float[Array, " num_lensing_ell"],
        PT: "PerturbationTable",
        BG: "Background",
        params: "Params",
    ) -> tuple[
        Float[Array, " num_lensing_ell"],
        Float[Array, " num_lensing_ell"],
        Float[Array, " num_lensing_ell"],
    ]:
        """
        Compute lensed CMB power spectra.

        All-sky correlation-function method of Challinor & Lewis,
        astro-ph/0502425 (PRD 71, 103010). The unlensed spectra are
        transformed to real-space correlation functions, the deflection
        smearing is applied there through the X_imj factors, and the result
        is transformed back by Gauss-Legendre quadrature.

        Equation numbers below are that paper's; the notation here follows
        it, and CLASS implements the same method (lensing.c)::

            Cgl, Cgl2   Eq. (35)  the d11 / d(-1)1 deflection covariances
            sigma2      unnumbered, after Eq. (35): sigma^2 = Cgl(0) - Cgl(b),
                                  hence Cgl[-1] (mu = 1) minus the rest
            X_imn       Eq. (37)  defining integral
            ksi         Eq. (38)  temperature
            X000, X220  Eqs. (39), (40)
            ksip        Eq. (54)  EE, +
            ksim        Eq. (55)  EE, -
            ksix        Eq. (56)  TE
            X022        Eq. (57)
            X121        Eq. (58)
            X132        Eq. (59)
            X242        Eq. (60)

        Returns:
            (ClTT, ClTE, ClEE) lensed power spectra
        """
        # num_mu is static (it comes from options), so scipy is fine
        # The appended mu = 1 node carries weight 0: it extends the grid to
        # the endpoint for the Wigner-d recurrences without altering the quadrature.
        num_mu = (
            self.ellmax
            + self.options["lensing_buffer"]
            + self.options["lensing_quadrature_buffer"]
        )
        mu_np, w_np = roots_legendre(num_mu)
        mu = jnp.concatenate((jnp.asarray(mu_np), jnp.array([1.0])))
        w = jnp.concatenate((jnp.asarray(w_np), jnp.array([0.0])))

        # Compute lensing Cl
        Clpp = self.lensing_Cl(ells, PT, BG, params)

        # Wigner matrices needed in general and for temperature
        # Note that for all wigner matrices, the symmetry relation is dnm = (-1)^(m-n) x dmn
        d00 = tools.d00(mu, ells)
        d11 = tools.d1n(mu, ells, 1)
        d1m1 = tools.d1n(mu, ells, -1)
        d2m2 = tools.d2n(mu, ells, -2)
        dm11 = d1m1

        # Wigner matrices needed for polarization
        d22 = tools.d2n(mu, ells, 2)
        d31 = tools.d3n(mu, ells, 1)
        d40 = tools.d4n(mu, ells, 0)
        d3m3 = tools.d3n(mu, ells, -3)
        d4m4 = tools.d4n(mu, ells, -4)
        d20 = tools.d2n(mu, ells, 0)
        d3m1 = tools.d3n(mu, ells, -1)
        d4m2 = tools.d4n(mu, ells, -2)
        d02 = d20
        dm24 = d4m2

        llp1 = ells * (ells + 1)

        def corr(Cl, kernel):
            """(1/4pi) Sum_l (2l+1) Cl K_l(mu): (num_lensing_ell,) against a
            (num_mu, num_lensing_ell) kernel, summed over ell -> (num_mu,)."""
            return jnp.sum((2.0 * ells + 1) * Cl * kernel, axis=1) / (4.0 * jnp.pi)

        # Lensing angular correlation function. sigma2 is the variance of the
        # deflection *difference* between two points separated by mu, hence
        # the value at mu = 1 (zero separation) minus the rest.
        Cgl = corr(llp1 * Clpp, d11)  # Nmu
        Cgl2 = corr(llp1 * Clpp, dm11)[:, None]
        sigma2 = (Cgl[-1] - Cgl)[:, None]

        X000 = jnp.exp(-llp1 * sigma2 / 4)
        X000_prime = -llp1 / 4.0 * X000
        X220 = (
            1.0
            / 4.0
            * jnp.sqrt((ells + 2) * (ells - 1) * ells * (ells + 1))
            * jnp.exp(-(llp1 - 2) * sigma2 / 4.0)
        )
        X022 = jnp.exp(-(llp1 - 4) * sigma2 / 4)
        X022_prime = -(llp1 - 4) / 4 * X022
        X121 = (
            -1.0
            / 2.0
            * jnp.sqrt((ells + 2) * (ells - 1))
            * jnp.exp(-(llp1 - 8.0 / 3.0) * sigma2 / 4.0)
        )
        X132 = (
            -1.0
            / 2.0
            * jnp.sqrt((ells + 3) * (ells - 2))
            * jnp.exp(-(llp1 - 20.0 / 3.0) * sigma2 / 4.0)
        )
        X242 = (
            1.0
            / 4.0
            * jnp.sqrt((ells + 4) * (ells + 3) * (ells - 2) * (ells - 3))
            * jnp.exp(-(llp1 - 10.0) * sigma2 / 4.0)
        )

        # Correlation functions
        ksi = corr(
            ClTT_unlensed,
            X000**2 * d00
            + 8.0 / llp1 * Cgl2 * X000_prime**2 * d1m1
            + Cgl2**2 * (X000_prime**2 * d00 + X220**2 * d2m2),
        )

        ksip = corr(
            ClEE_unlensed,
            X022**2 * d22
            + 2 * Cgl2 * X132 * X121 * d31
            + Cgl2**2 * (X022_prime**2 * d22 + X242 * X220 * d40),
        )

        ksim = corr(
            ClEE_unlensed,
            X022**2 * d2m2
            + Cgl2 * (X121**2 * d1m1 + X132**2 * d3m3)
            + 0.5
            * Cgl2**2
            * (2 * X022_prime**2 * d2m2 + X220**2 * d00 + X242**2 * d4m4),
        )

        ksix = corr(
            ClTE_unlensed,
            X022 * X000 * d02
            + Cgl2 * 2 * X000_prime / jnp.sqrt(llp1) * (X121 * d11 + X132 * d3m1)
            + 0.5
            * Cgl2**2
            * ((2 * X022_prime * X000_prime + X220**2) * d20 + X220 * X242 * dm24),
        )

        w = w[:, None]
        ClTT = 2 * jnp.pi * jnp.sum(ksi[:, None] * d00 * w, axis=0)
        ClTE = 2 * jnp.pi * jnp.sum(ksix[:, None] * d20 * w, axis=0)
        ClEE = (
            1.0
            / 2.0
            * 2
            * jnp.pi
            * jnp.sum((ksip[:, None] * d22 + ksim[:, None] * d2m2) * w, axis=0)
        )

        return (ClTT, ClTE, ClEE)

    def get_Cl(
        self, PT: "PerturbationTable", BG: "Background", params: "Params"
    ) -> tuple[
        Float[Array, " num_ell"],
        Float[Array, " num_ell"],
        Float[Array, " num_ell"],
    ]:
        """
        Compute angular power spectra for multiple multipoles.


        Returns:
        --------
        tuple
            (ClTT, ClTE, ClEE) angular power spectra
        """

        tt_raw, te_raw, ee_raw = vmap(self.Cl_one_ell, in_axes=(0, None, None, None))(
            self.sampled_ells, PT, BG, params
        )

        # Cubic spline for smooth Cl over user requested ells
        # The raw Cls live on the sparse tabulated ell grid; spline them up
        # onto the dense contiguous self.evaluated_ells axis.
        knots = self.sampled_ells
        tt_unlensed = CubicSpline(knots, tt_raw, check=False)(self.evaluated_ells)
        te_unlensed = CubicSpline(knots, te_raw, check=False)(self.evaluated_ells)
        ee_unlensed = CubicSpline(knots, ee_raw, check=False)(self.evaluated_ells)

        # Align the ells to index the power spectra properly and excise the lensing buffer
        out = slice(self.ellmin - MINIMUM_ALLOWED_L, self.ellmax - 1)

        def get_lensed_Cls():
            tt_lensed, te_lensed, ee_lensed = self.lensed_Cls(
                self.evaluated_ells,
                tt_unlensed,
                te_unlensed,
                ee_unlensed,
                PT,
                BG,
                params,
            )
            return (tt_lensed[out], te_lensed[out], ee_lensed[out])

        def get_unlensed_Cls():
            return (tt_unlensed[out], te_unlensed[out], ee_unlensed[out])

        return get_lensed_Cls() if self.options["lensing"] else get_unlensed_Cls()

    def Cl_one_ell(
        self,
        l: Int[Array, ""] | int,
        PT: "PerturbationTable",
        BG: "Background",
        params: "Params",
    ) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
        r"""
        Computes angular power spectrum for single multipole.

        Line-of-sight integration of Seljak & Zaldarriaga, astro-ph/9603033,
        whose Eqs. (12)-(16) are the method. Two stages here::

            Eq. (12)  the source terms -> sourceT0, sourceT1, sourceT2
                      (SW + ISW, ISW, and the polarization/anisotropic-
                      scattering Pi term), plus sourceE
            Eq. (13)  Delta_l(k) = Int dtau S(k,tau) j_l[k(tau0 - tau)]
                      -> the lax.scan over lna accumulating transferT0/T1/T2/E.
                      The single j_l of Eq. (13) is split across the m = 0
                      radial functions of Hu & White astro-ph/9702170 Eq. (15)
                      (see the module comment on the tables above)
            Eq. (9)   Cl = (4pi)^2 Int k^2 dk P_psi(k) |Delta_l(k)|^2
                      -> the closing jnp.trapezoid over k_axis

        Normalization differs from Eq. (9) as written: the integrands here are
        ``4 pi A_s (k/k_pivot)^(n_s-1) Delta^2 / k``. Both forms are the same
        quantity -- P_R(k) = A_s (k/k_pivot)^(n_s-1) 2 pi^2 / k^3 absorbs the
        k^2 measure and one factor of 4 pi -- but a term-by-term comparison
        against the paper will not match without accounting for it.

        Parameters:
        -----------
        l : int
            Multipole :math:`\ell` to evaluate. Must be one of the tabulated
            multipoles in ``bessel_l_tab`` -- the Bessel kernels exist only
            there, and the column is looked up by exact match.
        PT : perturbations.PerturbationTable
            Perturbation evolution table
        BG : background.Background
            Background cosmology module
        params : dict
            Dictionary of input and derived parameters

        Returns:
        --------
        tuple
            :math:`(C_\ell^{TT}, C_\ell^{TE}, C_\ell^{EE})` angular power spectra
        """
        # The kernel tables are keyed by column position, so translate the
        # physical multipole into one. Exact by construction: every l handed
        # here is drawn from bessel_l_tab itself (self.sampled_ells).
        idx = jnp.searchsorted(bessel_l_tab, l)
        k_axis = self.k_axis_transfer
        lna_axis = PT.lna[:-1]
        delta_lna = PT.lna[-1] - PT.lna[-2]

        ### TRANSFER FUNCTION ###
        # Background quantities, all Nlna 1D vectors
        tau0 = BG.tau0
        tau = vmap(BG.tau)(lna_axis)
        g = vmap(BG.visibility, in_axes=[0, None])(lna_axis, params)
        g_prime = vmap(grad(BG.visibility, argnums=0), in_axes=[0, None])(
            lna_axis, params
        )  # Derivative of g w.r.t. lna
        aH = vmap(BG.aH, in_axes=[0, None])(lna_axis, params)
        expmkappa = vmap(BG.expmkappa)(lna_axis)
        aH_dot = (
            vmap(BG.aH_prime, in_axes=[0, None])(lna_axis, params) * aH
        )  # Derivative of aH w.r.t. conformal time tau.

        # Keep a 1D alias of aH for the rolling-accumulator scan below.
        aH_1d = aH

        g = g[:, None]
        g_prime = g_prime[:, None]
        aH = aH[:, None]
        expmkappa = expmkappa[:, None]
        aH_dot = aH_dot[:, None]

        # Perturbations, all (Nlna, Nk) 2D vectors
        # Cubic Spline is necessary here for accuracy.
        def interp_column(col):
            return CubicSpline(jnp.log10(PT.k), col, check=False)(jnp.log10(k_axis))

        # Found that this is much much faster than RegularGridInterpolator
        photon_sp = PT.species_perturbations["Photon"]
        baryon_sp = PT.species_perturbations["Baryon"]
        delta_g = vmap(interp_column, in_axes=0, out_axes=0)(photon_sp["delta"][:-1, :])
        theta_b = vmap(interp_column, in_axes=0, out_axes=0)(baryon_sp["theta"][:-1, :])
        theta_b_prime = vmap(interp_column, in_axes=0, out_axes=0)(
            PT.theta_b_prime[:-1, :]
        )
        sigma_g = vmap(interp_column, in_axes=0, out_axes=0)(photon_sp["sigma"][:-1, :])
        Gg0 = vmap(interp_column, in_axes=0, out_axes=0)(photon_sp["G0"][:-1, :])
        Gg2 = vmap(interp_column, in_axes=0, out_axes=0)(photon_sp["G2"][:-1, :])
        # The metric history is interpolated leaf-wise, so this stays correct
        # for whichever gauge's metric struct the table carries; the metric
        # then turns its own fields into source terms.
        metric = jax.tree.map(
            lambda arr: vmap(interp_column, in_axes=0, out_axes=0)(arr[:-1, :]),
            PT.metric,
        )
        metric_src = metric.cmb_sources(k_axis, aH, aH_dot, g, g_prime, expmkappa)

        # Source terms. Identical in both gauges: what differs is entirely
        # inside metric_src (see gauges.CMBMetricSources).
        sourceT0 = (
            self.options["scale_sw"] * g * (delta_g / 4.0 + metric_src.sw_potential)
            + self.options["scale_isw"] * metric_src.isw_T0
            + self.options["scale_dop"]
            * (
                aH
                * (
                    g * (theta_b_prime / k_axis**2 + metric_src.theta_offset_prime)
                    + g_prime * (theta_b / k_axis**2 + metric_src.theta_offset)
                )
            )
        )

        sourceT1 = self.options["scale_isw"] * metric_src.isw_T1

        sourceT2 = self.options["scale_pol"] * g * (2 * sigma_g + Gg0 + Gg2) / 8.0

        sourceE = jnp.sqrt(6) * g * (2 * sigma_g + Gg0 + Gg2) / 8.0

        # Here we perform the time integral to get transfer functions from source functions.
        # previously, this block explicitly built a 2D (Nlna, Nk) tensor for each ell and summed it down to (Nk).
        # This newer version refactors into four accumulators of shape (Nk).  For each lna, we compute all four
        # (Nk), multiply by a trapezoid weight, and then add to the accumulator.  The result is identical but
        # avoids having to construct a full 2D tensor for each ell, instead just constructing the 1D (Nk) tensor
        # and accumulating down ell.  Clever "traingle term" added by hand is now handled by the trapezoid weights.

        # Pre-slice bessel-table columns so the scan body doesn't re-index
        # ..._tab[:, idx] every iteration.
        x0_min = xphi0_tab[0, idx]
        x0_max = xphi0_tab[-1, idx]
        x1_min = xphi1_tab[0, idx]
        x1_max = xphi1_tab[-1, idx]
        x2_min = xphi2_tab[0, idx]
        x2_max = xphi2_tab[-1, idx]
        col_phi0_l = phi0_tab[:, idx]
        col_phi1_l = phi1_tab[:, idx]
        col_phi2_l = phi2_tab[:, idx]
        ell_eps_factor = jnp.sqrt(3.0 / 8.0 * (l + 2) * (l + 1) * l * (l - 1))

        def phi0_local(x):
            x_safe = jnp.where(x >= x0_max, x, x0_max)
            return jnp.where(
                x < x0_min,
                0.0,
                jnp.where(
                    x >= x0_max,
                    j(l, x_safe),
                    tools.fast_interp(x, x0_min, x0_max, col_phi0_l),
                ),
            )

        def phi1_local(x):
            x_safe = jnp.where(x >= x1_max, x, x1_max)
            return jnp.where(
                x < x1_min,
                0.0,
                jnp.where(
                    x >= x1_max,
                    l / x_safe * j(l, x_safe) - j(l + 1, x_safe),
                    tools.fast_interp(x, x1_min, x1_max, col_phi1_l),
                ),
            )

        def phi2_local(x):
            x_safe = jnp.where(x >= x2_max, x, x2_max)
            return jnp.where(
                x < x2_min,
                0.0,
                jnp.where(
                    x >= x2_max,
                    (
                        (3 * l * (l - 1) - 2 * x_safe**2) * j(l, x_safe)
                        + 6 * x_safe * j(l + 1, x_safe)
                    )
                    / 2
                    / x_safe**2,
                    tools.fast_interp(x, x2_min, x2_max, col_phi2_l),
                ),
            )

        Nlna = lna_axis.shape[0]
        weights = jnp.full((Nlna,), delta_lna, dtype=sourceT0.dtype)
        weights = weights.at[0].set(0.5 * delta_lna)
        zero_k = jnp.zeros(k_axis.shape, dtype=sourceT0.dtype)

        def scan_step(carry, xs_l):
            acc_T0, acc_T1, acc_T2, acc_E = carry
            sT0_l, sT1_l, sT2_l, sE_l, aH_l, tau_l, w_l = xs_l
            chi_l = (tau0 - tau_l) * k_axis
            phi0_l = phi0_local(chi_l)
            phi1_l = phi1_local(chi_l)
            phi2_l = phi2_local(chi_l)
            eps_l = phi0_l / chi_l**2 * ell_eps_factor
            inv_aH = 1.0 / aH_l
            acc_T0 = acc_T0 + w_l * sT0_l * inv_aH * phi0_l
            acc_T1 = acc_T1 + w_l * sT1_l * inv_aH * phi1_l
            acc_T2 = acc_T2 + w_l * sT2_l * inv_aH * phi2_l
            acc_E = acc_E + w_l * sE_l * inv_aH * eps_l
            return (acc_T0, acc_T1, acc_T2, acc_E), None

        init = (zero_k, zero_k, zero_k, zero_k)
        xs = (sourceT0, sourceT1, sourceT2, sourceE, aH_1d, tau, weights)
        # jax.checkpoint on the scan body: during reverse AD, body intermediates
        # are not saved — the body is re-executed on the backward pass. Kills
        # the ~21 GiB (Nell, Nlna, Nk) integrand rematerialisation; adds ~2x on
        # this scan's compute, a small fraction of SS wall time.
        (transferT0, transferT1, transferT2, transferE), _ = lax.scan(
            jax.checkpoint(scan_step), init, xs
        )

        transferT = transferT0 + transferT1 + transferT2
        ### END OF TRANSFER FUNCTION ###

        # Now we integrate the transfer functions along the line of sight, and return.
        integrandTT = (
            4.0
            * jnp.pi
            * params["A_s"]
            * (k_axis / self.options["k_pivot"]) ** (params["n_s"] - 1.0)
            * transferT**2
            / k_axis
        )
        integrandTE = (
            4.0
            * jnp.pi
            * params["A_s"]
            * (k_axis / self.options["k_pivot"]) ** (params["n_s"] - 1.0)
            * transferT
            * transferE
            / k_axis
        )
        integrandEE = (
            4.0
            * jnp.pi
            * params["A_s"]
            * (k_axis / self.options["k_pivot"]) ** (params["n_s"] - 1.0)
            * transferE**2
            / k_axis
        )

        return (
            jnp.trapezoid(integrandTT, k_axis),
            jnp.trapezoid(integrandTE, k_axis),
            jnp.trapezoid(integrandEE, k_axis),
        )
