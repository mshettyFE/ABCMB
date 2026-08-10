import numpy as np
import pytest
from classy import Class

from abcmb import species
from abcmb.gauges import NewtonianGauge
from abcmb.main import Model

# JAX platform, x64, and debug_nans are configured in conftest.py.
np.seterr(all="raise")


@pytest.mark.parametrize(
    "n_nu_massive", [0, 1], ids=["massless_only", "one_massive_nu"]
)
def test_accuracy_checker(n_nu_massive, h=0.6762):
    ellmin = 2
    ellmax = 2500
    try:
        # Setup

        params = {
            "h": h,
            "omega_cdm": 0.1193,
            "omega_b": 0.0225,
            "A_s": 2.12424e-9,
            "n_s": 0.9709,
            "Neff": 3.044,
            "YHe": 0.245,
            "TCMB0": 2.34865418e-4,
            "N_nu_massive": n_nu_massive,
            "T_nu_massive": 0.71611,
            "m_nu_massive": 0.06,
            "tau_reion": 0.0544,
            "Delta_z_reion": 0.5,
            "z_reion_He": 3.5,
            "Delta_z_reion_He": 0.5,
            "exp_reion": 1.5,
        }

        if params["N_nu_massive"] > 0:
            user_species = (species.MassiveNeutrino,)
        else:
            user_species = None

        model = Model(
            user_species=user_species,
            l_max=ellmax,
            lensing=True,
            k_max=0.5,
            l_max_g=12,
            l_max_pol_g=10,
            l_max_ur=17,
            l_max_ncdm=17,
        )
        full_params = model.add_derived_parameters(params)

        T_nu_std = (4.0 / 11.0) ** (1.0 / 3.0)
        N_ur = (
            float(full_params["N_nu_massless"])
            * (float(full_params["T_nu_massless"]) / T_nu_std) ** 4
        )

        # CLASS
        CLASS_params = {
            "output": "mPk, tCl, pCl, lCl"
            if model.options["lensing"]
            else "mPk, tCl, pCl",
            # "temperature_contributions" : "tsw",
            "l_max_scalars": ellmax,
            "P_k_max_1/Mpc": model.options["k_max"],
            "lensing": "yes" if model.options["lensing"] else "no",
            "accurate_lensing": 1,
            "H0": full_params["h"] * 100,
            "omega_b": full_params["omega_b"],
            "omega_cdm": full_params["omega_cdm"],
            "A_s": full_params["A_s"],
            "n_s": full_params["n_s"],
            "N_ur": N_ur,
            "YHe": full_params["YHe"],
            "N_ncdm": full_params["N_nu_massive"],
            # "reio_parametrization" : "reio_none",
            "reio_parametrization": "reio_camb",
            "tau_reio": params["tau_reion"],
            "reionization_width": params["Delta_z_reion"],
            "helium_fullreio_redshift": params["z_reion_He"],
            "helium_fullreio_width": params["Delta_z_reion_He"],
            "reionization_exponent": params["exp_reion"],
            "l_max_g": model.options["l_max_g"],
            "l_max_pol_g": model.options["l_max_pol_g"],
            "l_max_ur": model.options["l_max_massless_nu"],
            "l_max_ncdm": model.options["l_max_massive_nu"],
        }

        CLASS_Model = Class()
        CLASS_Model.set(CLASS_params)
        if full_params["N_nu_massive"] > 0:
            CLASS_Model.set(
                {
                    "m_ncdm": full_params["m_nu_massive"],
                    "T_ncdm": full_params["T_nu_massive"],
                }
            )

        CLASS_Model.compute()
        if model.options["lensing"]:
            cl = CLASS_Model.lensed_cl(ellmax)
        else:
            cl = CLASS_Model.raw_cl(ellmax)
        cltt = cl["tt"][ellmin:]
        clee = cl["ee"][ellmin:]

        # ABCMB

        output = model(params)

        ABC_tt = output.ClTT
        ABC_ee = output.ClEE

        # Compare Cltt
        err_tt = abs(cltt - ABC_tt) / cltt
        print(err_tt.max())

        # Compare Clee
        err_ee = abs(clee - ABC_ee) / clee
        print(err_ee.max())

        # Compare P(k)
        ABC_Pk = output.Pk
        ABC_k = output.k
        CLA_Pk = np.vectorize(CLASS_Model.pk)(ABC_k, 0.0)
        err_Pk = abs(CLA_Pk - ABC_Pk) / CLA_Pk
        print(err_Pk.max())

        # 0.003 is ~45% headroom over the measured maxima at the default
        # solver settings (TT 0.21%, EE 0.19%, Pk 0.18%).
        assert max(err_tt) <= 0.003, f"Accuracy check failed at TT: {err_tt}"
        assert max(err_ee) <= 0.003, f"Accuracy check failed at EE: {err_ee}"
        assert max(err_Pk) <= 0.003, f"Accuracy check failed at P(k): {err_Pk}"

    except Exception as e:
        pytest.fail(f"accuracy_checks raised an exception: {e}")


def test_l_min_agrees_with_default():
    # l_min must only select which multipoles are returned -- it must not
    # shift the internal contiguous ell axis that the Wigner-d recurrences
    # and the raw-Cl spline are built on (anchored at ell=2 in
    # SpectrumSolver.__init__). Before the anchoring fix, l_min=3 with
    # lensing silently corrupted every lensed multipole, and the unlensed
    # path returned Cl(ell+1) labeled as Cl(ell).
    params = {
        "h": 0.6762,
        "omega_cdm": 0.1193,
        "omega_b": 0.0225,
        "A_s": 2.12424e-9,
        "n_s": 0.9709,
        "Neff": 3.044,
        "YHe": 0.245,
        "tau_reion": 0.0544,
    }
    outputs = {}
    for l_min in (2, 3):
        model = Model(l_min=l_min, l_max=300, lensing=True, k_max=0.2)
        outputs[l_min] = model(params)

    for name in ("ClTT", "ClTE", "ClEE"):
        a = np.asarray(getattr(outputs[2], name))[1:]  # drop ell=2
        b = np.asarray(getattr(outputs[3], name))
        rel = np.max(np.abs(a - b)) / np.max(np.abs(a))
        print(f"{name}: max scaled diff l_min=2 vs l_min=3 = {rel:.2e}")
        assert rel <= 1e-10, f"{name}: l_min shifted the computed spectra ({rel:.2e})"


def test_k_batch_strategies_agree():
    # The scan and vmap k-batching paths are different computations (vmap's
    # lockstep controller evaluates each mode at different adaptive steps than
    # a solo solve), documented to agree at solver-tolerance level. CI runs on
    # CPU where 'auto' always picks scan -- this is the only test exercising
    # the vmap (GPU-default) path. Reduced l_max/k_max keep the doubled solve
    # affordable.
    params = {
        "h": 0.6762,
        "omega_cdm": 0.1193,
        "omega_b": 0.0225,
        "A_s": 2.12424e-9,
        "n_s": 0.9709,
        "Neff": 3.044,
        "YHe": 0.245,
        "tau_reion": 0.0544,
    }
    outputs = {}
    for strategy in ("scan", "vmap"):
        model = Model(l_max=300, k_max=0.2, k_batch_strategy=strategy)
        outputs[strategy] = model(params)

    for name in ("ClTT", "ClEE", "Pk"):
        a = np.asarray(getattr(outputs["scan"], name))
        b = np.asarray(getattr(outputs["vmap"], name))
        rel = np.max(np.abs(a - b) / np.abs(a))
        print(f"{name}: max rel scan-vs-vmap = {rel:.2e}")
        assert rel <= 5e-3, f"{name}: scan/vmap paths disagree ({rel:.2e})"


def test_end_to_end_differentiability():
    # Forward-mode AD through the FULL pipeline -- eager derivation ->
    # background -> recombination -> perturbations -> spectra -- in two
    # parts:
    #
    # 1. Exact identity: unlensed Cl and Pk are exactly linear in A_s
    #    (linear theory), so the jvp tangent must equal Cl/A_s.
    # 2. Nontrivial path: d(sum ClTT)/dh runs through the eager derivation
    #    stage (omega_Lambda, H0) into everything downstream, checked
    #    against central finite differences. FD at production solver
    #    tolerances is noise-limited , so the threshold is loose: measured 3.4e-3,
    #    threshold ~6x. This catches broken/missing gradient paths (zeros,
    #    NaNs, dropped dependencies), not coefficient-level errors.
    import jax
    import jax.numpy as jnp

    base = {
        "h": 0.6762,
        "omega_cdm": 0.1193,
        "omega_b": 0.0225,
        "A_s": 2.12424e-9,
        "n_s": 0.9709,
        "Neff": 3.044,
        "YHe": 0.245,
        "tau_reion": 0.0544,
    }
    model = Model(l_max=100, k_max=0.1)
    # Resolve once, eagerly: parsing is structural and rejects tracers, so it
    # sits outside the differentiated region (Model.resolve_inputs). Both
    # traceable stages -- derive (omega_Lambda, H0, the BBN YHe) and
    # run_derived -- stay inside it, so this is still AD through the FULL
    # pipeline.
    resolved = model.resolve_inputs(base)

    def solve(p):
        return model.run_derived(model.derive(p))

    # 1. A_s linearity as an AD identity (jvp returns primals too -- one pass).
    def f_As(A):
        p = dict(resolved)
        p["A_s"] = A
        out = solve(p)
        return out.ClTT, out.Pk

    (cl, pk), (dcl, dpk) = jax.jvp(
        f_As, (jnp.asarray(base["A_s"]),), (jnp.asarray(1.0),)
    )
    r_cl = float(jnp.max(jnp.abs(dcl * base["A_s"] / cl - 1.0)))
    r_pk = float(jnp.max(jnp.abs(dpk * base["A_s"] / pk - 1.0)))
    print(f"A_s linearity identity: ClTT {r_cl:.2e}  Pk {r_pk:.2e}")
    assert r_cl < 1e-13, f"tangent chain broken somewhere: ClTT identity {r_cl:.2e}"
    assert r_pk < 1e-13, f"tangent chain broken somewhere: Pk identity {r_pk:.2e}"

    # 2. d(sum ClTT)/dh vs central FD.
    def f_h(h):
        p = dict(resolved)
        p["h"] = h
        return jnp.sum(solve(p).ClTT)

    _, ad = jax.jvp(f_h, (jnp.asarray(base["h"]),), (jnp.asarray(1.0),))
    eps = 1e-3 * base["h"]
    fd = (float(f_h(base["h"] + eps)) - float(f_h(base["h"] - eps))) / (2 * eps)
    rel = abs(float(ad) / fd - 1.0)
    print(f"d(sum ClTT)/dh: AD {float(ad):+.4e}  FD {fd:+.4e}  rel {rel:.2e}")
    assert jnp.isfinite(ad), "AD returned non-finite h-gradient"
    assert rel < 2e-2, f"AD vs FD h-gradient disagreement {rel:.2e}"


def test_gauge_independence_of_observables():
    """
    The same cosmology in both gauges must give the same observables.
    """
    opts = dict(
        l_max=1200,
        k_max=0.3,
        l_max_g=12,
        l_max_pol_g=10,
        **NewtonianGauge.recommended_tolerances,
        max_steps_PE=NewtonianGauge.recommended_max_steps,
    )
    params = {"h": 0.6762, "omega_cdm": 0.1193, "omega_b": 0.0225}

    out = {g: Model(gauge=g, **opts)(params) for g in ("synchronous", "newtonian")}
    sync, conf = out["synchronous"], out["newtonian"]

    # P(k) is reported in the comoving gauge precisely so that it is gauge
    # independent.
    k = np.asarray(sync.k)
    sub = k > 1e-3
    err_pk = np.abs(np.asarray(conf.Pk) - np.asarray(sync.Pk)) / np.asarray(sync.Pk)
    print(
        f"gauge P(k) max rel err: {err_pk[sub].max():.3e} (k>1e-3), "
        f"{err_pk.max():.3e} (all k)"
    )
    assert err_pk[sub].max() <= 0.01, (
        f"P(k) is gauge dependent: {err_pk[sub].max():.3e}"
    )

    # TT/EE relative to their own peak: TE crosses zero, so a pointwise
    # relative error there is meaningless.
    for name in ("ClTT", "ClEE", "ClTE"):
        a = np.asarray(getattr(sync, name))
        b = np.asarray(getattr(conf, name))
        err = np.abs(b - a).max() / np.abs(a).max()
        print(f"gauge {name} max err / peak: {err:.3e}")
        assert err <= 0.01, f"{name} is gauge dependent: {err:.3e}"
