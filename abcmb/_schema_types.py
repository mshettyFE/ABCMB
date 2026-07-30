# GENERATED from abcmb/schema.py; do not edit by hand.
# Regenerate with `./check.sh fix` (or `python -m abcmb._codegen`).
from typing import TypedDict

from jax import Array


class Options(TypedDict):
    """Resolved model options (static config); keys mirror OPTION_SCHEMA."""

    use_LCDM_species: bool
    input_tau_reion: bool
    l_min: int
    l_max: int
    lensing: bool
    k_max: float
    bbn_type: str
    linx_reaction_net: str
    l_max_g: int
    l_max_pol_g: int
    l_max_massless_nu: int
    l_max_massive_nu: int
    k_step_sub: float
    k_step_super: float
    k_step_transition: float
    k_step_super_reduction: float
    k_min_tau0: float
    k_max_tau0_over_l_max: float
    H0_fid: float
    tau0_fid: float
    rs_rec_fid: float
    k_transfer_linstep: float
    k_transfer_logstep: float
    tau_rec_fid: float
    k_pivot: float
    R_tc: float
    R_large: float
    max_steps_PE: int
    k_split_PE: float
    rtol_small_k_PE: float
    rtol_large_k_PE: float
    atol_small_k_PE: float
    atol_large_k_PE: float
    pcoeff_PE: float
    icoeff_PE: float
    dcoeff_PE: float
    scale_sw: int
    scale_isw: int
    scale_dop: int
    scale_pol: int


class Params(TypedDict, total=False):
    """Resolved + derived cosmological parameters: keys mirror PARAM_SCHEMA
    (inputs, conditional, derived). total=False because conditional/derived
    keys are added in stages by derive_parameters."""

    h: Array
    omega_cdm: Array
    omega_b: Array
    A_s: Array
    n_s: Array
    TCMB0: Array
    Delta_z_reion: Array
    z_reion_He: Array
    Delta_z_reion_He: Array
    exp_reion: Array
    T_nu_massive: Array
    N_nu_massive: Array
    m_nu_massive: Array
    tau_reion: Array
    z_reion: Array
    YHe: Array
    Neff: Array
    N_nu_massless: Array
    T_nu_massless: Array
    Delta_Neff_init: Array
    tau_n_fac: Array
    nuclear_rates_q: Array
    H0: Array
    omega_m: Array
    R_b: Array
    omega_r: Array
    R_nu: Array
    om: Array
    omega_Lambda: Array
