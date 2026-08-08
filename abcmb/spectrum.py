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


# The three m = 0 radial functions (Hu & White Eq. (15), see the table comment
# above) evaluated from the large-x asymptotic j() rather than the tables.
# Used past the tabulated support; each is valid only for x > l, since j()
# goes through sqrt(x^2 - l^2).


def phi0_asymptotic(
    l: Int[Array, ""] | int, x: Float[Array, " n"]
) -> Float[Array, " n"]:
    """j_l(x)."""
    return j(l, x)


def phi1_asymptotic(
    l: Int[Array, ""] | int, x: Float[Array, " n"]
) -> Float[Array, " n"]:
    """j_l'(x), from the recurrence j_l' = (l/x) j_l - j_{l+1}."""
    return l / x * j(l, x) - j(l + 1, x)


def phi2_asymptotic(
    l: Int[Array, ""] | int, x: Float[Array, " n"]
) -> Float[Array, " n"]:
    """(3 j_l'' + j_l)/2, with j_l'' eliminated by the same recurrence."""
    return ((3 * l * (l - 1) - 2 * x**2) * j(l, x) + 6 * x * j(l + 1, x)) / 2 / x**2


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
        whose Eqs. (12)-(16) are the method. The three stages implemented
        here, with the code they correspond to::

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

        Returns:
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

        # Perturbations, all (Nlna, Nk) 2D vectors.
        # Cubic Spline is necessary here for accuracy; found to be much much
        # faster than RegularGridInterpolator.
        def interp_column(col):
            return CubicSpline(jnp.log10(PT.k), col, check=False)(jnp.log10(k_axis))

        def to_k_grid(history):
            """(Nlna+1, Nk_pert) history -> (Nlna, Nk_transfer).

            The [:-1] drops the final lna so the result lines up with
            lna_axis, which is PT.lna[:-1].
            """
            return vmap(interp_column)(history[:-1, :])

        photon_sp = PT.species_perturbations["Photon"]
        baryon_sp = PT.species_perturbations["Baryon"]
        delta_g = to_k_grid(photon_sp["delta"])
        theta_b = to_k_grid(baryon_sp["theta"])
        theta_b_prime = to_k_grid(PT.theta_b_prime)
        sigma_g = to_k_grid(photon_sp["sigma"])
        Gg0 = to_k_grid(photon_sp["G0"])
        Gg2 = to_k_grid(photon_sp["G2"])
        # The metric history is interpolated leaf-wise, so this stays correct
        # for whichever gauge's metric struct the table carries; the metric
        # then turns its own fields into source terms.
        metric = jax.tree.map(to_k_grid, PT.metric)
        metric_src = metric.cmb_sources(
            k_axis,
            aH[:, None],
            aH_dot[:, None],
            g[:, None],
            g_prime[:, None],
            expmkappa[:, None],
        )

        # Source terms. Identical in both gauges: what differs is entirely
        # inside metric_src (see gauges.CMBMetricSources). The [:, None] lifts
        # the (Nlna,) background histories against the (Nlna, Nk) perturbations.
        g_c = g[:, None]

        # sourceT0 is Eq. (12)'s three contributions, each behind its own
        # scale_* switch: Sachs-Wolfe, integrated Sachs-Wolfe, Doppler.
        sachs_wolfe = g_c * (delta_g / 4.0 + metric_src.sw_potential)
        doppler = aH[:, None] * (
            g_c * (theta_b_prime / k_axis**2 + metric_src.theta_offset_prime)
            + g_prime[:, None] * (theta_b / k_axis**2 + metric_src.theta_offset)
        )
        sourceT0 = (
            self.options["scale_sw"] * sachs_wolfe
            + self.options["scale_isw"] * metric_src.isw_T0
            + self.options["scale_dop"] * doppler
        )

        sourceT1 = self.options["scale_isw"] * metric_src.isw_T1

        # The photon quadrupole, shared by the polarization temperature source
        # and the E-mode source up to their prefactors.
        quadrupole = (2 * sigma_g + Gg0 + Gg2) / 8.0
        sourceT2 = self.options["scale_pol"] * g_c * quadrupole
        sourceE = jnp.sqrt(6) * g_c * quadrupole

        # The time integral turning source functions into transfer functions.
        # Scanned over lna into four (Nk,) accumulators rather than building an
        # (Nlna, Nk) integrand per ell and summing it down: same result, no 2D
        # tensor per ell. The trapezoid end correction is carried by `weights`.

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

        # eval_kernel evaluates any of the three kernels in three regimes:
        #
        #   x <  x_min   0. The generator starts each column where |f| first
        #                rises through FLOOR = 1e-10, so the kernel really is
        #                negligible below it (see _generators/bessel_tables).
        #   x in range   interpolate the tabulated column.
        #   x >= x_max   the table stops at the fifth local maximum; past it
        #                the large-x asymptotic j() takes over.
        #
        # x_safe is not cosmetic. j() goes through sqrt(x^2 - l^2), which is
        # NaN below the turning point x = l, and reverse-mode AD propagates
        # NaN out of *untaken* jnp.where branches -- so the asymptotic must
        # never be evaluated off its domain, even where its result is
        # discarded. fast_interp needs no such guard: it clips its own index
        # (ABCMBTools.fast_interp), so out-of-range x returns an edge value.

        def eval_kernel(x, x_min, x_max, col, asymptotic):
            x_safe = jnp.where(x >= x_max, x, x_max)
            return jnp.where(
                x < x_min,
                0.0,
                jnp.where(
                    x >= x_max,
                    asymptotic(l, x_safe),
                    tools.fast_interp(x, x_min, x_max, col),
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
            phi0_l = eval_kernel(chi_l, x0_min, x0_max, col_phi0_l, phi0_asymptotic)
            phi1_l = eval_kernel(chi_l, x1_min, x1_max, col_phi1_l, phi1_asymptotic)
            phi2_l = eval_kernel(chi_l, x2_min, x2_max, col_phi2_l, phi2_asymptotic)
            eps_l = phi0_l / chi_l**2 * ell_eps_factor
            inv_aH = 1.0 / aH_l
            acc_T0 = acc_T0 + w_l * sT0_l * inv_aH * phi0_l
            acc_T1 = acc_T1 + w_l * sT1_l * inv_aH * phi1_l
            acc_T2 = acc_T2 + w_l * sT2_l * inv_aH * phi2_l
            acc_E = acc_E + w_l * sE_l * inv_aH * eps_l
            return (acc_T0, acc_T1, acc_T2, acc_E), None

        init = (zero_k, zero_k, zero_k, zero_k)
        xs = (sourceT0, sourceT1, sourceT2, sourceE, aH, tau, weights)
        # jax.checkpoint on the scan body: body intermediates are not saved,
        # the body is re-executed on the backward pass instead. This is inert
        # in the forward and jvp/jacfwd paths -- remat only changes how a
        # *reverse*-mode backward pass is scheduled.
        #
        # It is worth having because the saved residuals scale as
        # (Nell, Nlna, Nk), which is 675M elements = 5.4 GB each at the default
        # l_max=2500. Measured peak device memory for jax.grad of get_Cl at
        # fixed PT/BG, with vs without:
        #
        #     l_max=1000  (61, 500, 763)    0.30 GB  vs  0.38 GB
        #     l_max=2500  (99, 500, 1704)   0.35 GB  vs  0.97 GB
        #
        # -- 2.8x at the default, and growing with l_max; bigger runs OOM.
        #
        # Reaching reverse mode at all takes a non-default adjoint. Measured
        # for jax.grad over the full pipeline at l_max=100:
        #
        #     ForwardMode (default)       ValueError -- reverse-mode AD does
        #                                 not work through lax.while_loop,
        #                                 which adaptive stepping needs
        #     RecursiveCheckpointAdjoint  AssertionError inside equinox's
        #                                 checkpointed_while_loop, from the
        #                                 vendored HyRex HeII solve
        #                                 (hyrex/helium.py solve_HeII_full)
        #     BacksolveAdjoint            NotImplementedError -- incompatible
        #                                 with events, which HyRex uses
        #     DirectAdjoint               works (427 s at l_max=100; it
        #                                 unrolls rather than checkpointing)
        #
        # So end-to-end reverse AD is possible via DirectAdjoint, and reverse
        # over the spectrum stage alone always works. See the FAQ, which
        # steers users to jacfwd -- with ~10 parameters and ~1e3 outputs,
        # forward mode is usually the right tool regardless.
        (transferT0, transferT1, transferT2, transferE), _ = lax.scan(
            jax.checkpoint(scan_step), init, xs
        )

        transferT = transferT0 + transferT1 + transferT2
        ### END OF TRANSFER FUNCTION ###

        # Now we integrate the transfer functions along the line of sight, and return.
        power_law = (k_axis / self.options["k_pivot"]) ** (params["n_s"] - 1.0)

        integrandTT = 4.0 * jnp.pi * params["A_s"] * power_law * transferT**2 / k_axis
        integrandTE = (
            4.0 * jnp.pi * params["A_s"] * power_law * transferT * transferE / k_axis
        )
        integrandEE = 4.0 * jnp.pi * params["A_s"] * power_law * transferE**2 / k_axis

        return (
            jnp.trapezoid(integrandTT, k_axis),
            jnp.trapezoid(integrandTE, k_axis),
            jnp.trapezoid(integrandEE, k_axis),
        )
