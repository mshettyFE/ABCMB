"""
Constants sanity oracle: rebuild the *derived combinations* abcmb.constants
feeds into the physics -- the critical-density factor and the unit-system
conversions -- from scipy's CODATA values via an independent (SI) route.

Constants stay hand-pinned in abcmb.constants on purpose: deriving them from scipy at
runtime would let a scipy upgrade silently shift every spectrum.
"""

import numpy as np
from scipy import constants as sc

import abcmb.constants as cnst

_eV = sc.electron_volt


def test_conversion_factors_match_codata():
    hbar_c_eVcm = sc.hbar / _eV * sc.c * 100.0  # eV cm
    Mpc_cm = sc.parsec * 1e6 * 100.0
    c_Mpc_s = sc.c * 100.0 / Mpc_cm

    assert abs(float(cnst.hbar * cnst.c) / hbar_c_eVcm - 1) < 1e-12
    assert abs(float(cnst.c_Mpc_over_s) / c_Mpc_s - 1) < 1e-6


def test_critical_density_combination():
    # The combination every rho() uses: 3 H0^2 / (8 pi G) at h = 1, in
    # eV cm^-3, rebuilt entirely from scipy (G, parsec, c, eV) through SI.
    H0_h = 100.0 * 1e5 / (sc.parsec * 1e6 * 100.0)  # 100 km/s/Mpc in 1/s
    rho_crit = 3.0 * H0_h**2 * sc.c**2 / (8.0 * np.pi * sc.G) / _eV / 1e6
    abcmb_val = float(3.0 * cnst.H0_over_h**2 / (8.0 * np.pi * cnst.G))
    assert abs(abcmb_val / rho_crit - 1) < 2e-6
