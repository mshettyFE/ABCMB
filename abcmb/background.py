import os
from typing import TYPE_CHECKING, ClassVar, cast

import diffrax
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from diffrax import (
    ForwardMode,
    Kvaerno5,
    ODETerm,
    PIDController,
    SaveAt,
    Tsit5,
    diffeqsolve,
)
from jax import config, vmap
from jaxtyping import Array, Float

from . import ABCMBTools as tools
from . import constants as cnst
from .hyrex import recomb_functions
from .recomb_interface import RecombInputs
from .species import Fluid

if TYPE_CHECKING:
    from .hyrex.hyrex import recomb_model
    from .inputs._schema_types import Params
    from .reionization import ReionizationModel

file_dir = os.path.dirname(__file__)
config.update("jax_enable_x64", True)


class BackgroundPreRecomb(eqx.Module):
    """
    Pre-recombination background-cosmology object: the ionization-independent
    part of the background, computed before the recombination solver runs
    (its quantities are valid at all times).
    """

    species_list: tuple[Fluid, ...]

    # Endpoints of the conformal-time tabulation grid
    lna_tau_min: ClassVar[float] = -33.0
    lna_tau_max: ClassVar[float] = 0.0
    # Analytic/ODE seam of the tabulation
    # Guarded by pytests/test_background.py.
    lna_tau_cut: ClassVar[float] = -20.0

    lna_tau_tab: Array  # Axis for tabulating conformal time.
    tau_tab: Array  # Tabulated conformal time.
    tau0: Array  # Conformal time today

    # Solver used by Diffrax
    adjoint: type[diffrax.AbstractAdjoint] = eqx.field(static=True)

    def __init__(
        self,
        params: "Params",
        species_list: tuple[Fluid, ...],
        adjoint: type[diffrax.AbstractAdjoint] = ForwardMode,
        lna_tau_points: int = 10000,
    ) -> None:
        """
        Initialize pre-recombination background.

        Tabulates conformal time on ``lna_tau_points`` points in
        lna = [-33, 0] by default.

        Parameters:
        -----------
        params : Params
            Cosmological parameters
        species_list : tuple[Fluid, ...]
            List of fluid species for energy density calculations
        adjoint : type[diffrax.AbstractAdjoint], optional
            Adjoint class for diffrax solves (default: ForwardMode)
        lna_tau_points : int, optional
            Points of the conformal-time tabulation grid (default: 10000)
        """
        self.adjoint = adjoint
        self.species_list = species_list

        self.lna_tau_tab = jnp.linspace(
            self.lna_tau_min, self.lna_tau_max, lna_tau_points
        )
        # Static analytic/ODE split index of the grid just built (a Python
        # int, since it sets slice sizes under jit) -- computed host-side
        # from the same endpoints and count.
        n_analytic = int(
            np.sum(
                np.linspace(self.lna_tau_min, self.lna_tau_max, lna_tau_points)
                <= self.lna_tau_cut
            )
        )
        self.tau_tab = self._tabulate_conformal_time(params, n_analytic)
        self.tau0 = self.tau(0.0)

    def _tabulate_conformal_time(
        self, params: "Params", n_analytic: int
    ) -> Float[Array, " n_lna_tau"]:
        r"""
        Tabulate conformal time as function of ln(a).

        Integrates d\tau/d(ln a) = 1/aH from early times to today using
        radiation-dominated initial conditions: grid points before the
        analytic/ODE seam use the closed-form radiation-domination
        solution, the rest are saved directly from a Diffrax solve
        (``SaveAt(ts=...)``) seeded by that solution at the seam, and the
        two segments are concatenated at a static split index.
        """

        lna_cut = self.lna_tau_cut  # see the class-attribute comment

        # Analytic early-time approximation; also seeds the solve, so the
        # two segments agree exactly at the seam.
        def tau_approx(lna):
            return (
                jnp.exp(lna)
                / (cnst.H0_over_h / cnst.c_Mpc_over_s)
                / jnp.sqrt(params["omega_r"])
            )

        # Save directly at the grid points past the cut -- no dense storage,
        # no post-hoc evaluate, no out-of-bounds guard.
        sol = diffeqsolve(
            ODETerm(self._dtau_dlna),
            solver=Kvaerno5(),
            t0=lna_cut,
            t1=self.lna_tau_max,
            dt0=1e-5,
            y0=tau_approx(lna_cut),
            saveat=SaveAt(ts=self.lna_tau_tab[n_analytic:]),
            stepsize_controller=PIDController(rtol=1e-8, atol=1e-14),
            args=params,
            adjoint=self.adjoint(),
        )

        return jnp.concatenate(
            [vmap(tau_approx)(self.lna_tau_tab[:n_analytic]), sol.ys]
        )

    @eqx.filter_jit
    def make_recomb_inputs(
        self, RecModel: "recomb_model", params: "Params"
    ) -> RecombInputs:
        """
        Bundle the background quantities HyRex needs (TCMB, nH, H) onto the
        recombination model's sampling grid -- the handoff across the
        jit/device boundary (see abcmb.recomb_interface).
        """
        lna_axis = RecModel.lna_axis_full
        return RecombInputs(
            lna_grid=lna_axis,
            TCMB_arr=vmap(self.TCMB, in_axes=[0, None])(lna_axis, params),
            nH_arr=vmap(self.nH, in_axes=[0, None])(lna_axis, params),
            H_arr=vmap(self.H, in_axes=[0, None])(lna_axis, params),
        )

    def rho_tot(
        self,
        lna: Float[Array, ""] | float,
        params: "Params",
    ) -> Float[Array, ""]:
        """
         Compute total energy density.

        Returns:
            Total energy density (units: eV cm^{-3})
        """
        return jnp.sum(
            jnp.asarray([s.rho(lna, params) for s in self.species_list]), axis=0
        )

    def P_tot(
        self,
        lna: Float[Array, ""] | float,
        params: "Params",
    ) -> Float[Array, ""]:
        """
         Compute total pressure.

        Returns:
            Total pressure (units: eV cm^{-3})
        """
        return jnp.sum(
            jnp.asarray([s.P(lna, params) for s in self.species_list]), axis=0
        )

    def H(
        self,
        lna: Float[Array, ""] | float,
        params: "Params",
    ) -> Float[Array, ""]:
        """
        Compute Hubble parameter.

        Returns:
            Hubble parameter (units: s^{-1})
        """
        return jnp.sqrt(8.0 * jnp.pi * cnst.G * self.rho_tot(lna, params) / 3.0)

    def aH(
        self,
        lna: Float[Array, ""] | float,
        params: "Params",
    ) -> Float[Array, ""]:
        r"""
        Compute conformal Hubble parameter.

        Calculates conformal Hubble H = a*H = da/d\tau where \tau is conformal time.
        Uses Mpc units for perturbation calculations.
        Returns:
           Conformal Hubble parameter (units: Mpc^{-1})
        """
        return jnp.exp(lna) * self.H(lna, params) / cnst.c_Mpc_over_s

    def aH_prime(
        self,
        lna: Float[Array, ""] | float,
        params: "Params",
    ) -> Float[Array, ""]:
        """
        Compute derivative of conformal Hubble parameter.

        Uses second Friedmann equation to compute d(aH)/d(ln a).
        See Eq. (20) of Ma & Bertschinger (1995), arXiv:astro-ph/9506072.

        Returns:
           Derivative of conformal Hubble (units: Mpc^{-1})
        """
        return (
            -4.0
            * jnp.pi
            * cnst.G
            * jnp.exp(lna) ** 2
            / 3.0
            / self.aH(lna, params)
            * (self.rho_tot(lna, params) + 3.0 * self.P_tot(lna, params))
            / cnst.c_Mpc_over_s**2
        )

    def d2adtau2_over_a(
        self,
        lna: Float[Array, ""] | float,
        params: "Params",
    ) -> Float[Array, ""]:
        """
        Compute second derivative of scale factor.

        Returns:
           Second derivative of scale factor (units: Mpc^{-2})
        """
        return self.aH(lna, params) ** 2 + self.aH(lna, params) * self.aH_prime(
            lna, params
        )

    def _dtau_dlna(
        self,
        lna: Float[Array, ""] | float,
        y: Float[Array, ""],
        args: "Params",
    ) -> Float[Array, ""]:
        """
        Compute derivative of conformal time with respect to ln(a).
        """
        params = args
        return 1.0 / self.aH(lna, params)

    def tau(self, lna: Float[Array, ""] | float) -> Float[Array, ""]:
        """
        Compute conformal time.

        Interpolates from pre-tabulated conformal time history.
        Returns:
           Conformal time (units: Mpc)
        """
        return tools.fast_interp(
            lna, self.lna_tau_tab[0], self.lna_tau_tab[-1], self.tau_tab
        )

    def nH(
        self,
        lna: Float[Array, ""] | float,
        params: "Params",
    ) -> Float[Array, ""]:
        """
        Compute hydrogen number density.

        Returns:
           Hydrogen number density (units: cm^{-3})
        """
        return (
            (1 - params["YHe"])
            * 3.0
            * params["omega_b"]
            * cnst.H0_over_h**2
            / 8
            / jnp.pi
            / cnst.G
            / cnst.mH
            / jnp.exp(lna) ** 3
        )

    def TCMB(
        self,
        lna: Float[Array, ""] | float,
        params: "Params",
    ) -> Float[Array, ""]:
        """
        Compute CMB temperature.

        Returns:
           CMB temperature (units: eV)
        """
        return params["TCMB0"] / jnp.exp(lna)

    def R_ratio_lna(
        self,
        lna: Float[Array, ""] | float,
        params: "Params",
    ) -> Float[Array, ""]:
        """
        Calculates R = 3ρ_b/(4ρ_γ), the ratio of baryon to photon
        energy densities that appears in baryon drag calculations.

        Returns:
        --------
        float
            Baryon drag ratio (units: dimensionless)
        """
        rho_b = jnp.asarray(0.0)
        rho_g = jnp.asarray(0.0)

        for s in self.species_list:
            if s.name == "Photon":
                rho_g += s.rho(lna, params)
            elif s.name == "Baryon":
                rho_b += s.rho(lna, params)

        return 3.0 * rho_b / (4 * rho_g)


class Background(BackgroundPreRecomb):
    """
    Full Background cosmology module for cosmological calculations.

    Inherits all cosmology fields and methods from ``BackgroundPreRecomb``.
    Construction takes a ``BackgroundPreRecomb`` and the recombination output
    from HyRex, then applies reionization and integrates the optical depth.

    This factorization allows HyRex to always run on CPU (its faster backend).

    """

    xe_tab: Array  # Free-electron fraction on lna_xe_tab (reionization applied).
    lna_xe_tab: Array  # Static recombination grid (RecModel.lna_axis_full).
    Tm_tab: Array  # Matter temperature on lna_xe_tab (endpoint-clamped fill).
    Tm_lna_start: Array  # Scalar validity start of Tm_tab; before it, use
    # the analytic early-time approximation (see Tm()).
    kappa_func: "diffrax.Solution"  # Optical depth function
    z_reion: Array  # Redshift of hydrogen reionization (CAMB parameterization);
    # traced (root-found from tau_reion or taken from params).
    tau_reion: Array  # Optical depth to reionization; traced.
    lna_rec: Array  #  Log scale factor of recombination.
    rA_rec: Array  # Comoving angular diameter distance at recombination.
    lna_transfer_start: Array  # Time where transfer functions start integrating.
    lna_visibility_stop: Array  # Time to stop integrating T1, T2, and E sources due to small visibility functions. Only used for l<400

    def __init__(
        self,
        pre_BG: BackgroundPreRecomb,
        recomb_output: tuple[
            Float[Array, " n_rec"],  # xe on the recombination grid
            Float[Array, " n_rec"],  # the static recombination grid (lna)
            Float[Array, " n_rec"],  # Tm on the same grid
            Float[Array, ""],  # Tm validity start (scalar lna)
        ],
        params: "Params",
        ReionModel: "type[ReionizationModel]",
        transfer_start_threshold: float = 0.008,
    ) -> None:
        """
        Initialize Background cosmology module.

        Consolidates pre-recombination and recombination elements of background cosmology.

        """
        # Copy pre-recomb fields onto self.
        self.adjoint = pre_BG.adjoint
        self.species_list = pre_BG.species_list
        self.lna_tau_tab = pre_BG.lna_tau_tab
        self.tau_tab = pre_BG.tau_tab
        self.tau0 = pre_BG.tau0

        # Unpack HyRex output  and apply reionization.
        xe, self.lna_xe_tab, self.Tm_tab, self.Tm_lna_start = recomb_output

        reion_model = ReionModel(self, params)
        self.z_reion = reion_model.z_reion
        self.tau_reion = reion_model.tau_reion

        xe_reion_correction = vmap(
            lambda l: reion_model.xe_reion(l, self.z_reion, params)
        )(self.lna_xe_tab)
        self.xe_tab = xe_reion_correction + xe

        self.kappa_func = self._tabulate_optical_depth(params)

        # Find approximate maximum of visibility function.
        # TODO: What gaurentees that  a reasonable maximum is in this range?
        lna_vals = jnp.linspace(-8.0, -4.0, 1500)  # Decoupling falls in here.
        vis_vals = vmap(self.visibility, in_axes=[0, None])(lna_vals, params)
        self.lna_rec = lna_vals[jnp.argmax(vis_vals)]
        self.lna_visibility_stop = lna_vals[jnp.argmin((vis_vals - 1.0e-3) ** 2)]
        self.rA_rec = self.tau0 - self.tau(self.lna_rec)

        # Find the approximate early time when aH * tau_c crosses the
        # transfer_start_threshold option (tight coupling makes the transfer
        # sources negligible before this).
        lna_vals = jnp.linspace(-15.0, -6.0, 5000)
        aH_tau_c_vals = vmap(lambda l: self.aH(l, params) * self.tau_c(l, params))(
            lna_vals
        )
        self.lna_transfer_start = lna_vals[
            jnp.argmin((aH_tau_c_vals - transfer_start_threshold) ** 2)
        ]

    def xe(self, lna: Float[Array, ""] | float) -> Float[Array, ""]:
        """
        Compute free electron fraction.

        Interpolates from pre-tabulated recombination history with
        boundary conditions for early and late times.

        Returns:
            Free electron fraction (units: dimensionless)

        """
        return tools.fast_interp(
            lna, self.lna_xe_tab[0], self.lna_xe_tab[-1], self.xe_tab
        )

    def _Tm_early_approx(
        self,
        lna: Float[Array, ""] | float,
        params: "Params",
    ) -> Float[Array, ""]:
        """
        Compute matter temperature using post-equilibrium approximation.

        Uses approximation Tm = TCMB * (1 - H/GammaCompton) for early times
        before detailed recombination calculation begins.

        Returns:
            Matter temperature (units: eV)
        """
        TCMB = self.TCMB(lna, params)
        xe = self.xe(lna)
        return TCMB * (
            1.0
            - self.H(lna, params)
            / recomb_functions.Gamma_compton(xe, TCMB, params["YHe"])
        )

    def Tm(
        self,
        lna: Float[Array, ""] | float,
        params: "Params",
    ) -> Float[Array, ""]:
        """
        Compute matter temperature.

        Interpolates from pre-tabulated recombination history with
        early-time approximation and late-time boundary conditions.

        Returns:
            Matter temperature (units: eV)
        """
        return jnp.where(
            lna < self.Tm_lna_start,
            self._Tm_early_approx(lna, params),
            tools.fast_interp(
                lna, self.lna_xe_tab[0], self.lna_xe_tab[-1], self.Tm_tab
            ),
        )

    def tau_c(
        self,
        lna: Float[Array, ""] | float,
        params: "Params",
    ) -> Float[Array, ""]:
        r"""
        Compute Thomson scattering time.

        Calculates Thomson scattering time scale \tau_c = 1/(a × ne × \sigma T).

        Returns:
            Thomson scattering time (units: Mpc)
        """
        a = jnp.exp(lna)
        nH = self.nH(lna, params)
        ne = nH * self.xe(lna)
        return 1.0 / a / ne / cnst.thomson_xsec / cnst.c * cnst.c_Mpc_over_s

    def _tabulate_optical_depth(self, params: "Params") -> diffrax.Solution:
        r"""
        Tabulate optical depth from given scale factor to today.

        Integrates d\kappa/d(ln a) = -1/(\tau* c × aH) backwards from today
        to compute optical depth \kappa (a) = \int [a to 1] d\kappa/da' da'.

        Returns:
            Tabulated optical depth values (units: dimensionless)
        """

        def integrand(lna, y, args):
            return -1.0 / self.tau_c(lna, params) / self.aH(lna, params)

        stepsize_controller = PIDController(
            pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=1.0e-10, atol=1.0e-10
        )
        sol = diffeqsolve(
            ODETerm(integrand),
            solver=Kvaerno5(),
            stepsize_controller=stepsize_controller,
            t0=0.0,
            t1=-10.0,
            dt0=-1.0e-3,
            max_steps=2048,
            y0=0.0,
            saveat=SaveAt(dense=True),
            adjoint=self.adjoint(),
        )
        return sol

    def expmkappa(self, lna: Float[Array, ""] | float) -> Float[Array, ""]:
        """
        Compute exp(-optical depth).

        Interpolates from pre-tabulated optical depth history.
        Returns:
            exp(-(optical depth)) (units: dimensionless)
        """
        return jnp.where(lna < -10.0, 0.0, jnp.exp(-self.kappa_func.evaluate(lna)))

    def visibility(
        self,
        lna: Float[Array, ""] | float,
        params: "Params",
    ) -> Float[Array, ""]:
        r"""
        Compute visibility function.

        Calculates visibility function g(x) = -aH(x) × \kappa'(x) × exp(-\kappa(x))
        where x = ln a. Represents probability that a CMB
        photon observed today was last scattered at time x.

        Returns:
            Visibility function (units: Mpc^{-1})
        """
        return self.expmkappa(lna) / self.tau_c(lna, params)

    def find_z_at_kappad_equals_one(
        self, z: Float[Array, " n"], kappa_d: Float[Array, " n"]
    ) -> Float[Array, ""]:
        r"""
        Find redshift where baryon optical depth equals unity.

        Interpolates to find z_d such that \kappa_d(z_d) = 1, marking
        the approximate time of baryon decoupling.

        Returns:
            Decoupling redshift (units: dimensionless)
        """
        # ensure sorted ascending
        idx = jnp.argsort(z)
        z_sorted = z[idx]
        kappa_d_sorted = jnp.abs(kappa_d)[idx]

        z_d = jnp.interp(1.0, kappa_d_sorted, z_sorted)
        return z_d

    def interp_rs_at_z(
        self,
        z_bg: Float[Array, " n"],
        r_s: Float[Array, " n"],
        z_d: Float[Array, ""] | float,
    ) -> Float[Array, ""]:
        """
        Interpolate sound horizon at decoupling redshift.
        Returns:
            Sound horizon at decoupling (units: Mpc)
        """
        idx = jnp.argsort(z_bg)
        z_sorted = z_bg[idx]
        rs_sorted = r_s[idx]
        return jnp.interp(z_d, z_sorted, rs_sorted)

    def _tabulate_kappa_d(self, params: "Params") -> Float[Array, " n_lna_tau"]:
        r"""
        Tabulate baryon optical depth.

        Integrates d\kappa_d/d(ln a) = -1/(\tau_c × aH × R) backwards from today
        to compute baryon optical depth including drag effects.

        Returns:
            Tabulated baryon optical depth values (units: dimensionless)
        """

        def integrand(lna, y, args):
            return jnp.float64(
                -1.0
                / self.tau_c(lna, params)
                / self.aH(lna, params)
                / (self.R_ratio_lna(lna, params))
            )

        term = ODETerm(integrand)
        stepsize_controller = PIDController(
            pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=1.0e-3, atol=1.0e-6
        )
        adjoint = self.adjoint()

        solution = diffeqsolve(
            term,
            solver=Tsit5(),  # Kvaerno5 is just slower but gives same result
            stepsize_controller=stepsize_controller,
            t0=self.lna_tau_tab[-1],  # Initial x value (~0 in this case)
            t1=self.lna_tau_tab[0],  # Final x value (smallest x value)
            dt0=-1e-3,
            max_steps=2048,
            y0=0.0,  # Initial value tau(x=0) = 0
            saveat=SaveAt(
                ts=self.lna_tau_tab[::-1]
            ),  # Save at all points in x, reverse order since integrating backwards
            adjoint=adjoint,
        )
        result = solution.ys[::-1]
        return result

    def _tabulate_rs(self, params: "Params") -> Float[Array, " n_lna_tau"]:
        r"""
        Tabulate sound horizon evolution.

        Integrates drs/d(ln a) = cs/aH from early times to today
        where cs = 1/\sqrt{(3(1+R))} accounts for baryon loading.

        Returns:
            Tabulated sound horizon values (units: Mpc)
        """
        # initial condition assuming cs**2 = 1/3 at early times
        rs0 = 1.0 / jnp.sqrt(3) / (self.aH(self.lna_tau_tab[0], params))

        def integrand(lna, y, args):
            return (
                1.0
                / jnp.sqrt(3 * (1 + self.R_ratio_lna(lna, params)))
                / (self.aH(lna, params))
            )

        term = ODETerm(integrand)
        stepsize_controller = PIDController(
            pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=1.0e-3, atol=1.0e-6
        )
        adjoint = self.adjoint()

        solution = diffeqsolve(
            term,
            solver=Tsit5(),
            stepsize_controller=stepsize_controller,
            t0=self.lna_tau_tab[0],  # reversed direction since I know rs at early times
            t1=self.lna_tau_tab[-1],
            dt0=1e-3,
            max_steps=2048,
            y0=rs0,
            saveat=SaveAt(ts=self.lna_tau_tab),
            adjoint=adjoint,
        )
        # diffrax types .ys as PyTree | None (None when nothing is saved);
        # SaveAt(ts=...) above guarantees an Array here.
        return cast(Array, solution.ys)

    def z_d(self, params: "Params") -> Float[Array, ""]:
        """
        Compute baryon decoupling redshift.

        Finds redshift where κ_d = 1 as estimate of when baryons
        decouple from photons.

        Returns:
            Decoupling redshift (units: dimensionless)
        """
        return self.find_z_at_kappad_equals_one(
            1 / jnp.exp(self.lna_tau_tab) - 1, self._tabulate_kappa_d(params)
        )

    def rs_d(self, params: "Params") -> Float[Array, ""]:
        """
        Compute sound horizon at decoupling.

        Finds value of sound horizon at baryon decoupling redshift z_d.
        Returns:
            Sound horizon at decoupling (units: Mpc)
        """
        return self.interp_rs_at_z(
            1 / jnp.exp(self.lna_tau_tab) - 1,
            self._tabulate_rs(params),
            self.z_d(params),
        )
