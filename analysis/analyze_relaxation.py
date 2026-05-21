"""
Análisis de la ley de relajación: H̄(t) ~ exp(-t/τ_eff)
Verifica la predicción: 1/τ_eff = 1/τ_V + Γ(λ)  con  Γ(λ) = C * λ²

Bergamin & Bringa (2026) - Sección 9.5, Ecuación (28)

Uses the 1D x-marginal H̄_x to avoid the periodic-y wrap artifact.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import linregress

# ─── Load results ─────────────────────────────────────────────────────────────
def load_results(results_dir="results_full"):
    data = {}
    for f in sorted(Path(results_dir).glob("results_lam*.json")):
        with open(f) as fh:
            r = json.load(fh)
        lam = r["lambda"]
        data[lam] = r
    return data

# ─── Fit τ_eff from H-bar time series ─────────────────────────────────────────
def fit_tau(hbar_mean, hbar_std, dt=0.002, t_min=1.0):
    """
    Fit ln H̄(t) = ln H̄_0 - t/τ_eff via linear regression.
    t_min: start fitting after this time (skip initial fast mixing / wrap artifact).
    Returns (tau_eff, tau_eff_err, r_squared).
    """
    t = np.arange(len(hbar_mean)) * dt
    H = np.array(hbar_mean)

    mask = (H > 1e-6) & (t >= t_min)
    if mask.sum() < 5:
        return np.nan, np.nan, np.nan

    ln_H = np.log(H[mask])
    t_fit = t[mask]

    slope, intercept, r, p, se = linregress(t_fit, ln_H)
    if slope >= 0:
        return np.inf, np.nan, np.nan

    tau_eff = -1.0 / slope
    tau_err = se / (slope**2)
    r2 = r**2
    return tau_eff, tau_err, r2

# ─── Main analysis ────────────────────────────────────────────────────────────
def analyze(results_dir="results_full", dt=0.002, n_steps=4500,
            use_1d=True, lam_max_pert=0.05, t_min_fit=1.0):
    data = load_results(results_dir)
    if not data:
        print(f"No results in {results_dir}")
        return

    lambdas  = sorted(data.keys())
    taus     = []
    tau_errs = []
    r2s      = []

    hbar_key = "hbar1d_mean" if use_1d else "hbar_mean"
    hbar_std_key = "hbar1d_std" if use_1d else "hbar_std"
    metric_label = "H̄_x (1D marginal)" if use_1d else "H̄ (2D)"

    # Check if 1D data exists
    first = data[lambdas[0]]
    if use_1d and hbar_key not in first:
        print(f"WARNING: '{hbar_key}' not in JSON, falling back to hbar_mean")
        hbar_key = "hbar_mean"
        hbar_std_key = "hbar_std"
        metric_label = "H̄ (2D, fallback)"

    print(f"\nMetric: {metric_label}  |  t_min_fit={t_min_fit}  |  dir={results_dir}")
    print(f"\n{'λ':>8}  {'τ_eff':>10}  {'±':>8}  {'R²(fit)':>8}  {'D_KS':>8}  {'±':>6}")
    print("-" * 65)

    for lam in lambdas:
        r = data[lam]
        H = r.get(hbar_key, r["hbar_mean"])
        dH = r.get(hbar_std_key, r["hbar_std"])
        tau, tau_e, r2 = fit_tau(H, dH, dt=dt, t_min=t_min_fit)
        taus.append(tau)
        tau_errs.append(tau_e if not np.isnan(tau_e) else 0.0)
        r2s.append(r2 if not np.isnan(r2) else 0.0)
        ks_m = r.get("ks_mean", float("nan"))
        ks_s = r.get("ks_std", float("nan"))
        print(f"{lam:>8.4f}  {tau:>10.3f}  {tau_e:>8.3f}  "
              f"{r2:>8.3f}  {ks_m:>8.4f}  {ks_s:>6.4f}")

    taus     = np.array(taus)
    tau_errs = np.array(tau_errs)
    lambdas  = np.array(lambdas)
    r2s      = np.array(r2s)

    # τ_V from λ=0 run
    lam0_idx = np.where(lambdas == 0.0)[0]
    tau_V_direct = taus[lam0_idx[0]] if len(lam0_idx) else np.nan

    inv_tau = 1.0 / taus
    inv_tau_err = tau_errs / taus**2

    def model(lam2, inv_tau_V, C):
        return inv_tau_V + C * lam2

    # ── Perturbative fit (λ ≤ lam_max_pert) ──────────────────────────────────
    mask_pert = (lambdas > 0) & (lambdas <= lam_max_pert) & ~np.isinf(taus) & ~np.isnan(taus)
    popt_pert = None
    if mask_pert.sum() >= 2:
        lam2_pert = lambdas[mask_pert]**2
        inv_tau_pert = inv_tau[mask_pert]
        try:
            popt_pert, pcov_pert = curve_fit(model, lam2_pert, inv_tau_pert,
                                             p0=[1.0/tau_V_direct, 10.0])
            perr_pert = np.sqrt(np.diag(pcov_pert))
            residuals = inv_tau_pert - model(lam2_pert, *popt_pert)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((inv_tau_pert - inv_tau_pert.mean())**2)
            R2_pert = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            tau_V_pert = 1.0 / popt_pert[0]
            C_pert = popt_pert[1]

            print(f"\n── Perturbative fit (λ ≤ {lam_max_pert}) ───────────────────")
            print(f"   τ_V = {tau_V_pert:.3f} ± {perr_pert[0]/popt_pert[0]**2:.3f}  (paper: ≈3.4)")
            print(f"   C   = {C_pert:.2f} ± {perr_pert[1]:.2f}    (paper: ≈42)")
            print(f"   R²  = {R2_pert:.3f}    (paper: ≈0.85)")
            print(f"   τ_V (λ=0 direct) = {tau_V_direct:.3f}")
        except Exception as e:
            print(f"\nPerturbative fit failed: {e}")

    # ── Full fit (all λ > 0) ──────────────────────────────────────────────────
    mask_all = (lambdas > 0) & ~np.isinf(taus) & ~np.isnan(taus)
    popt_all = None
    if mask_all.sum() >= 2:
        lam2_all = lambdas[mask_all]**2
        inv_tau_all = inv_tau[mask_all]
        try:
            popt_all, pcov_all = curve_fit(model, lam2_all, inv_tau_all,
                                           p0=[1.0/tau_V_direct, 5.0])
            perr_all = np.sqrt(np.diag(pcov_all))
            residuals = inv_tau_all - model(lam2_all, *popt_all)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((inv_tau_all - inv_tau_all.mean())**2)
            R2_all = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            print(f"\n── Full fit (all λ > 0) ──────────────────────────────")
            print(f"   τ_V = {1/popt_all[0]:.3f}  (paper: ≈3.4)")
            print(f"   C   = {popt_all[1]:.2f}    (paper: ≈42)")
            print(f"   R²  = {R2_all:.3f}")
        except Exception as e:
            print(f"\nFull fit failed: {e}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Hybrid BdB/ZPF: Born rule relaxation  [{metric_label}]", fontsize=12)

    t_arr = np.arange(n_steps + 1) * dt
    colors = plt.cm.viridis(np.linspace(0, 1, len(lambdas)))

    # Panel 1: H̄_x(t) normalized
    ax = axes[0]
    for i, lam in enumerate(lambdas):
        r = data[lam]
        H  = np.array(r.get(hbar_key, r["hbar_mean"]))
        dH = np.array(r.get(hbar_std_key, r["hbar_std"]))
        H0 = H[0] if H[0] > 0 else 1.0
        t_plot = t_arr[:len(H)]
        ax.plot(t_plot, H/H0, color=colors[i], label=f"λ={lam:.3f}", alpha=0.85, lw=1.2)
        ax.fill_between(t_plot, (H-dH)/H0, (H+dH)/H0, alpha=0.12, color=colors[i])
    ax.axvline(t_min_fit, color='gray', lw=0.8, ls=':', alpha=0.6, label=f"t_fit={t_min_fit}")
    ax.set_xlabel("t (dimensionless)")
    ax.set_ylabel("H̄(t) / H̄(0)")
    ax.set_title("H-function evolution")
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(-0.05, 1.1)

    # Panel 2: ln H̄_x(t)
    ax = axes[1]
    for i, lam in enumerate(lambdas):
        r = data[lam]
        H  = np.array(r.get(hbar_key, r["hbar_mean"]))
        H0 = H[0] if H[0] > 0 else 1.0
        t_plot = t_arr[:len(H)]
        mask = (H > 1e-6) & (t_plot >= t_min_fit)
        if mask.sum() > 1:
            ax.plot(t_plot[mask], np.log(H[mask]/H0), color=colors[i],
                    alpha=0.85, label=f"λ={lam:.3f}", lw=1.2)
        if not np.isnan(taus[i]) and not np.isinf(taus[i]):
            t_line = np.linspace(t_min_fit, t_arr[:len(H)][H > 1e-6].max(), 50)
            H0_at_tmin = np.interp(t_min_fit, t_plot, H) / H0
            ax.plot(t_line, np.log(H0_at_tmin) - (t_line - t_min_fit)/taus[i],
                    '--', color=colors[i], alpha=0.4, lw=1)
    ax.set_xlabel("t (dimensionless)")
    ax.set_ylabel("ln H̄(t)/H̄(0)")
    ax.set_title("Exponential decay (cf. Fig. 2 paper)")
    ax.legend(fontsize=7, ncol=2)

    # Panel 3: 1/τ_eff vs λ²
    ax = axes[2]
    mask_valid = ~np.isinf(taus) & ~np.isnan(taus) & (lambdas >= 0)
    ax.scatter(lambdas[mask_valid]**2, inv_tau[mask_valid],
               color="crimson", zorder=5, s=40, label="Simulation (Nr=20)")
    ax.errorbar(lambdas[mask_valid]**2, inv_tau[mask_valid],
                yerr=inv_tau_err[mask_valid], fmt='none', color="crimson", capsize=3)

    lam2_plot = np.linspace(0, lambdas.max()**2, 200)

    if popt_pert is not None:
        ax.plot(lam2_plot, model(lam2_plot, *popt_pert), 'b-',
                label=f"Pert. fit: τ_V={1/popt_pert[0]:.2f}, C={popt_pert[1]:.1f}")
    if popt_all is not None:
        ax.plot(lam2_plot, model(lam2_plot, *popt_all), 'g--',
                label=f"Full fit: τ_V={1/popt_all[0]:.2f}, C={popt_all[1]:.1f}", alpha=0.7)

    # Paper prediction
    lam2_paper = np.linspace(0, lambdas.max()**2, 200)
    ax.plot(lam2_paper, 1/3.4 + 42*lam2_paper, 'k:', lw=1.5,
            label="Paper: τ_V=3.4, C=42")

    ax.axhline(1/tau_V_direct, color='salmon', ls=':', alpha=0.7,
               label=f"λ=0 direct: 1/τ={1/tau_V_direct:.3f}")
    ax.set_xlabel("λ²")
    ax.set_ylabel("1/τ_eff")
    ax.set_title("Relaxation rate vs coupling\n(cf. Fig. 3 paper)")
    ax.legend(fontsize=7)

    plt.tight_layout()
    out_fig = Path(results_dir) / "relaxation_analysis.pdf"
    plt.savefig(out_fig, dpi=150)
    print(f"\nFigure saved: {out_fig}")
    plt.savefig(str(out_fig).replace(".pdf", ".png"), dpi=120)
    plt.close()

    return data

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dir",    default="results_full")
    p.add_argument("--dt",     type=float, default=0.002)
    p.add_argument("--steps",  type=int,   default=4500)
    p.add_argument("--2d",     dest="use_2d", action="store_true",
                   help="Use 2D H̄ instead of 1D x-marginal")
    p.add_argument("--lam-max-pert", type=float, default=0.05,
                   help="Max λ for perturbative fit")
    p.add_argument("--t-min",  type=float, default=1.0,
                   help="Start time for exponential fit (skip wrap artifact)")
    args = p.parse_args()
    analyze(results_dir=args.dir, dt=args.dt, n_steps=args.steps,
            use_1d=not args.use_2d,
            lam_max_pert=args.lam_max_pert,
            t_min_fit=args.t_min)
