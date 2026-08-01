"""
Reionization models: the correction to the free-electron fraction after
recombination (CAMB tanh parameterization), in the two input modes --
directly from ``z_reion``, or root-found from a target ``tau_reion``.

``Background.__init__`` takes one of these classes (not an instance) and
instantiates it against itself; ``Model`` selects which via the
``input_tau_reion`` option.
"""

from typing import TYPE_CHECKING, cast

import equinox as eqx
import jax.numpy as jnp
import optimistix as optx
from jax import vmap
from jaxtyping import Array, Float

from . import constants as cnst

if TYPE_CHECKING:
    from .background import Background
    from .inputs._schema_types import Params


class ReionizationModel(eqx.Module):
    """
    Object for computing the reionization correction to the free electron fraction.
    Provides the base methods

    At the moment we only support the CAMB tanh parameterization, but we need different approaches
    based on whether the use inputs the optical depth tau_reion or the reionization redshift z_reion.

    """

    z_reion: Array
    tau_reion: Array

    def xe_reion(
        self,
        lna: Float[Array, ""] | float,
        z_reion: Float[Array, ""] | float,
        params: "Params",
    ) -> Float[Array, ""]:
        """
        Scalar contract: evaluates one lna (batch with jax.vmap at the call
        site); the tanh patching follows the reionization parameters.
        """
        fHe = params["YHe"] / 4 / (1 - params["YHe"])
        z = 1 / jnp.exp(lna) - 1
        y = (1 + z) ** (params["exp_reion"])

        y_reion = (1 + z_reion) ** (params["exp_reion"])
        Delta_y_reion = (
            params["exp_reion"]
            * (1 + z_reion) ** (params["exp_reion"] - 1)
            * params["Delta_z_reion"]
        )
        tanh_arg = (y_reion - y) / Delta_y_reion
        xe_reion_H = (1 + fHe) / 2 * (1 + jnp.tanh(tanh_arg))

        # The above accounts for hydrogen and the first ionization level of helium.
        # Let's also account for the second ionization of helium:
        tanh_arg_He = (params["z_reion_He"] - z) / params["Delta_z_reion_He"]
        xe_reion_HeII = fHe / 2 * (1 + jnp.tanh(tanh_arg_He))

        return xe_reion_H + xe_reion_HeII

    def tau_reion_fn(
        self,
        z_reion: Float[Array, ""] | float,
        BG: "Background",
        params: "Params",
    ) -> Float[Array, ""]:
        lna_axis = jnp.linspace(-5.0, 0.0, 2000)

        # Scalar optical-depth integrand, vmapped over the grid (the rho/P
        # and background thermodynamics contract is scalar-in, scalar-out).
        def integrand(lna):
            # Free electron number density from reionized hydrogen only.
            ne = BG.nH(lna, params) * self.xe_reion(lna, z_reion, params)
            Gamma = jnp.exp(lna) * ne * cnst.thomson_xsec * cnst.c / cnst.c_Mpc_over_s
            return Gamma / BG.aH(lna, params)

        return jnp.trapezoid(vmap(integrand)(lna_axis), lna_axis)


class ReionizationModelFromZ(ReionizationModel):
    """
    This object is used when the user directly inputs the redshift of reionization.
    In this case the tanh correction and the optical depth can be computed directly,
    and simply returned.
    """

    def __init__(self, BG: "Background", params: "Params") -> None:
        self.z_reion = params.get("z_reion", jnp.array(7.6711))
        self.tau_reion = self.tau_reion_fn(self.z_reion, BG, params)


class ReionizationModelFromTau(ReionizationModel):
    """
    This object is used when the user inputs the optical depth and wishes to infer the redshift.
    The init finder will use an optimistix root finder to find the appropriate redshift.
    Then the appropriate tanh correction may be called and returned, as well as the inferred reionization redshift.
    """

    def __init__(self, BG: "Background", params: "Params") -> None:
        def tau_target_fn(z_reion, args):
            target = args
            return self.tau_reion_fn(z_reion, BG, params) - target

        solver = optx.Newton(rtol=1e-5, atol=1e-5)
        sol = optx.root_find(
            tau_target_fn, solver, 7.6, params.get("tau_reion", jnp.array(0.05430842))
        )
        self.z_reion = cast(Array, sol.value)
        self.tau_reion = params.get("tau_reion", jnp.array(0.05430842))
