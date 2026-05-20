"""
analyze_AL.py  —  Analysis and figures for Paper-2 Fase 3 results.

Loads results from:
  results_AL_box/   — run_box_AL() λ sweep (ALD mode + Nelson-direct mode)
  results_AL_stat/  — run_stationary_AL() stationarity test
  results_box_om3/  — Paper-1 baseline (bohm_zpf_box.py, no ALD)

Produces:
  Fig 1: Stationarity test (born_ic vs uniform_ic, Nelson D=0.5)
  Fig 2: H̄(t) comparison at λ=0.05: Paper-1 vs ALD vs Nelson-direct
  Fig 3: τ_eff(λ) and Γ(λ)=1/τ−1/τ₀ for all modes (C-fit comparison)
  Fig 4: D_ALD(λ) vs λ² scaling, D_Nelson reference, λ_c crossover
"""

import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import linregress

# ─── Load helpers ─────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_dir(pattern, key_fn=None):
    """Load all JSON files matching glob pattern. Returns list sorted by key_fn."""
    files = sorted(Path(".").glob(pattern))
    data = [load_json(f) for f in files]
    if key_fn:
        data.sort(key=key_fn)
    return data


def fit_tau(hbar_mean, t_arr):
    """Log-linear τ_eff fit. Returns (tau_eff, inv_tau, R2)."""
    h0 = hbar_mean[0] if hbar_mean[0] > 0 else 1.0
    valid = (np.array(hbar_mean) > 0.01 * h0) & (np.array(hbar_mean) > 1e-6)
    t = np.array(t_arr)
    h = np.array(hbar_mean)
    if valid.sum() < 4:
        return float("inf"), 0.0, 0.0
    slope, intercept, r, _, _ = linregress(t[valid], np.log(h[valid]))
    if slope >= 0:
        return float("inf"), 0.0, 0.0
    tau = -1.0 / slope
    resid = np.log(h[valid]) - (slope * t[valid] + intercept)
    R2 = float(1 - np.var(resid) / np.var(np.log(h[valid])))
    return tau, 1.0 / tau, R2


def bootstrap_tau(hbar_all, t_arr, n_boot=200, rng=None):
    """Bootstrap τ_eff over hbar_all rows. Returns (tau_mean, tau_std)."""
    if rng is None:
        rng = np.random.default_rng(42)
    hbar_all = np.array(hbar_all)
    nr = hbar_all.shape[0]
    taus = []
    for _ in range(n_boot):
        idx = rng.integers(0, nr, nr)
        tau, _, _ = fit_tau(hbar_all[idx].mean(axis=0), t_arr)
        if tau < 1e8:
            taus.append(tau)
    if len(taus) < 5:
        return float("nan"), float("nan")
    return float(np.mean(taus)), float(np.std(taus))


# ─── Figure 1: Stationarity test ──────────────────────────────────────────────

def fig_stationarity(out_dir="results_AL_stat", save="fig_AL_stationarity.pdf"):
    """Plot H̄(t) for born_ic and uniform_ic (Nelson D=0.5, eigenstate φ₁₁)."""
    files = sorted(Path(out_dir).glob("stat_*.json"))
    if not files:
        print(f"No files in {out_dir}/", flush=True)
        return

    fig, ax = plt.subplots(figsize=(6, 4))

    colors = {"born_ic": "steelblue", "uniform_ic": "tomato"}
    labels = {"born_ic": r"Born IC: $\rho_0 = |\varphi_{11}|^2$  [stationarity]",
              "uniform_ic": r"Uniform IC: $\rho_0 = \mathrm{const}$  [relaxation]"}

    for fpath in files:
        r = load_json(fpath)
        ic   = r.get("ic", fpath.stem)
        t    = np.array(r["times"])
        hm   = np.array(r["hbar_mean"])
        hs   = np.array(r["hbar_std"])
        D    = r.get("D_ALD", r.get("D_zpf", 0))
        lam  = r["lambda"]
        nr   = r["n_realizations"]
        err  = hs / np.sqrt(nr)
        col  = colors.get(ic, "gray")
        lab  = labels.get(ic, ic)
        ax.fill_between(t, hm - err, hm + err, alpha=0.2, color=col)
        ax.plot(t, hm, color=col, lw=1.5, label=lab)

    ax.axhline(0, ls="--", lw=0.8, color="black")
    ax.set_xlabel(r"$t$ (natural units, $\hbar=m=1$)")
    ax.set_ylabel(r"$\bar{H}(t)$ (Valentini entropy)")
    D_ref = r.get("D_ALD", 0)
    ax.set_title(f"Nelson stochastic mechanics: $D = {D_ref:.2f}$,  "
                 r"$\psi = \varphi_{11}$,  $\lambda={:.3f}$".format(lam))
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=-0.05)
    plt.tight_layout()
    plt.savefig(save, dpi=150)
    print(f"Saved: {save}", flush=True)
    plt.close()


# ─── Figure 2: τ_eff(λ) comparison  ──────────────────────────────────────────

def fig_tau_comparison(al_dir="results_AL_box",
                       paper1_dir="results_box_om3",
                       save="fig_AL_tau_comparison.pdf"):
    """
    Three curves on the same axes:
      - Paper-1 baseline (no ALD, ZPF only)
      - ALD mode (ZPF + LL + osmotic, D_ALD)
      - Nelson direct mode (Itô noise, D = 0.5)
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    def collect(globpat, sort_key="lambda"):
        files = sorted(Path(".").glob(globpat))
        records = []
        for f in files:
            r = load_json(f)
            lam = r.get("lambda", 0)
            tau, inv_tau, R2 = fit_tau(r["hbar_mean"], r["times"])
            tau_m, tau_s = bootstrap_tau(r.get("hbar_all", [r["hbar_mean"]]),
                                         r["times"])
            records.append({"lam": lam, "tau": tau, "inv_tau": inv_tau,
                             "R2": R2, "tau_m": tau_m, "tau_s": tau_s,
                             "D_ALD": r.get("D_ALD", r.get("D_zpf", 0)),
                             "label": f.name})
        records.sort(key=lambda x: x["lam"])
        return records

    # Paper-1: results_box_om3/box_lam*.json
    p1 = collect("results_box_om3/box_lam*om3.json")
    # ALD: results_AL_box/AL_lam*.json (all — includes both modes)
    al = collect("results_AL_box/AL_lam*.json")

    def plot_series(ax, data, label, color, marker="o"):
        lams    = np.array([d["lam"] for d in data if np.isfinite(d["tau_m"])])
        taus    = np.array([d["tau_m"] for d in data if np.isfinite(d["tau_m"])])
        errs    = np.array([d["tau_s"] for d in data if np.isfinite(d["tau_m"])])
        ax.errorbar(lams, taus, yerr=errs, fmt=f"{marker}-",
                    color=color, label=label, capsize=3, lw=1.5)

    ax = axes[0]
    if p1:
        plot_series(ax, p1, "Paper-1 (no ALD)", "gray", "s")
    if al:
        plot_series(ax, al, "ALD (ZPF + LL + osmotic)", "steelblue", "o")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\tau_{\rm eff}$")
    ax.set_title(r"Relaxation time $\tau_{\rm eff}(\lambda)$")
    ax.legend()
    ax.set_xlim(left=0)

    # Right panel: Γ = 1/τ − 1/τ₀
    ax = axes[1]
    def plot_gamma(ax, data, label, color, marker="o"):
        lams0 = [d["lam"] for d in data]
        taus  = [d["tau_m"] for d in data]
        inv0 = 1.0 / taus[0] if taus and np.isfinite(taus[0]) else 0
        lams_g = np.array([d["lam"] for d in data if d["lam"] > 0 and np.isfinite(d["tau_m"])])
        gammas = np.array([1.0/d["tau_m"] - inv0 for d in data
                           if d["lam"] > 0 and np.isfinite(d["tau_m"])])
        if len(lams_g) > 2:
            ax.plot(lams_g**2, gammas, f"{marker}-", color=color, label=label, lw=1.5)
            # Fit C·λ² (perturbative region: λ ≤ 0.05)
            mask = lams_g <= 0.05
            if mask.sum() > 2:
                C = np.polyfit(lams_g[mask]**2, gammas[mask], 1)[0]
                lfit = np.linspace(0, 0.05, 50)
                ax.plot(lfit**2, C * lfit**2, "--", color=color,
                        label=f"C = {C:.1f}", lw=1)

    if p1:
        plot_gamma(ax, p1, "Paper-1", "gray", "s")
    if al:
        plot_gamma(ax, al, "ALD", "steelblue", "o")

    ax.axhline(0, ls="--", lw=0.8, color="black")
    ax.set_xlabel(r"$\lambda^2$")
    ax.set_ylabel(r"$\Gamma(\lambda) = 1/\tau - 1/\tau_0$")
    ax.set_title(r"ZPF acceleration: $\Gamma \propto C \lambda^2$ ?")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save, dpi=150)
    print(f"Saved: {save}", flush=True)
    plt.close()


# ─── Figure 3: D_ALD scaling ──────────────────────────────────────────────────

def fig_D_scaling(al_dir="results_AL_box", save="fig_AL_D_scaling.pdf"):
    """
    D_ALD(λ) from results vs λ², compared to Nelson D = 0.5.
    Shows the gap between ALD and Nelson and the critical coupling λ_c.
    """
    files = sorted(Path(al_dir).glob("AL_lam*.json"))
    lams  = []
    D_vals = []
    for f in files:
        r = load_json(f)
        D = r.get("D_ALD", 0)
        if D > 0:
            lams.append(r["lambda"])
            D_vals.append(D)

    if not lams:
        print("No D_ALD values found.", flush=True)
        return

    lams   = np.array(lams)
    D_vals = np.array(D_vals)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.scatter(lams**2, D_vals, color="steelblue", s=40, zorder=5, label=r"$D_{\rm ALD}$")

    # Fit D ∝ λ²
    if len(lams) > 2:
        slope = np.polyfit(lams**2, D_vals, 1)[0]
        l2 = np.linspace(0, max(lams)**2, 100)
        ax.plot(l2, slope * l2, "--", color="steelblue",
                label=r"$D_{\rm ALD} = C_D \lambda^2$"
                f"  ($C_D = {slope:.4f}$)")

    D_nelson = 0.5
    ax.axhline(D_nelson, ls="-.", color="tomato", lw=1.5,
               label=r"$D_{\rm Nelson} = \hbar/2m = 0.5$")

    # Critical coupling λ_c where D_ALD = D_Nelson
    if len(lams) > 2:
        lam_c = np.sqrt(D_nelson / slope) if slope > 0 else float("inf")
        if lam_c < 20:
            ax.axvline(lam_c**2, ls=":", color="orange", lw=1.5,
                       label=fr"$\lambda_c^2 = {lam_c**2:.1f}$  "
                             fr"($\lambda_c = {lam_c:.2f}$)")

    ax.set_xlabel(r"$\lambda^2$")
    ax.set_ylabel(r"$D$  (diffusion coefficient)")
    ax.set_title("ALD vs Nelson diffusion coefficient")
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(save, dpi=150)
    print(f"Saved: {save}", flush=True)
    plt.close()


# ─── Figure 4: H̄(t) for a single λ, all modes ────────────────────────────────

def fig_hbar_curves(lam=0.05, al_dir="results_AL_box", stat_dir="results_AL_stat",
                    paper1_dir="results_box_om3", save="fig_AL_hbar_curves.pdf"):
    """Compare H̄(t) at one λ for: Paper-1, ALD mode, Nelson-direct."""
    fig, ax = plt.subplots(figsize=(6, 4))

    # Paper-1
    p1_files = list(Path(paper1_dir).glob(f"box_lam{lam:.4f}*om3.json"))
    if p1_files:
        r = load_json(p1_files[0])
        t = np.array(r["times"]); hm = np.array(r["hbar_mean"])
        ax.plot(t, hm, "k-", lw=1.5, label=f"Paper-1 (no ALD), τ={fit_tau(hm,t)[0]:.1f}")

    # ALD
    al_files = list(Path(al_dir).glob(f"AL_lam{lam:.4f}*.json"))
    for f in sorted(al_files):
        r = load_json(f)
        if r.get("D_ALD", 0) > 0.4:   # Nelson-direct mode (D ≈ 0.5)
            col = "tomato"; lab = f"Nelson-direct D=0.5"
        else:
            col = "steelblue"; lab = f"ALD D={r.get('D_ALD',0):.2e}"
        t = np.array(r["times"]); hm = np.array(r["hbar_mean"])
        tau, _, _ = fit_tau(hm, t)
        ax.plot(t, hm, "-", color=col, lw=1.5, label=f"{lab}, τ={tau:.1f}")

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\bar{H}(t)$")
    ax.set_title(f"Born-rule relaxation: λ = {lam}")
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(save, dpi=150)
    print(f"Saved: {save}", flush=True)
    plt.close()


# ─── Summary table ────────────────────────────────────────────────────────────

def print_summary(al_dir="results_AL_box"):
    files = sorted(Path(al_dir).glob("AL_lam*.json"))
    if not files:
        print(f"No files in {al_dir}/")
        return

    data = [load_json(f) for f in files]
    data.sort(key=lambda r: r["lambda"])

    # Find λ=0 baseline
    baselines = [r for r in data if r["lambda"] == 0.0]
    tau0 = baselines[0]["tau_eff"] if baselines else float("inf")
    inv0 = (1.0 / tau0) if tau0 < 1e8 else 0.0

    print(f"\n{'λ':>8}  {'D_ALD':>10}  {'τ_eff':>8}  {'1/τ':>8}  "
          f"{'Δ(1/τ)':>10}  {'R²':>6}")
    print("-" * 62)
    for r in data:
        tau    = r.get("tau_eff", float("inf"))
        inv_t  = r.get("inv_tau_eff", 0.0)
        R2     = r.get("R2_fit", 0.0)
        D      = r.get("D_ALD", 0.0)
        dinv   = inv_t - inv0
        tau_s  = f"{tau:.2f}" if tau < 1e8 else "inf"
        print(f"  {r['lambda']:>6.4f}  {D:>10.2e}  {tau_s:>8}  "
              f"{inv_t:>8.4f}  {dinv:>+10.4f}  {R2:>6.3f}")

    # Fit C (perturbative λ ≤ 0.05)
    pert = [(r["lambda"], r.get("inv_tau_eff", 0.0))
            for r in data if 0 < r["lambda"] <= 0.05]
    if len(pert) > 2:
        lams_p   = np.array([x[0] for x in pert])
        gammas_p = np.array([x[1] - inv0 for x in pert])
        C = np.polyfit(lams_p**2, gammas_p, 1)[0]
        print(f"\n  C_fit (λ≤0.05, ALD) = {C:.2f}")
        print(f"  (C > 0: ZPF accelerates; C < 0: ZPF disrupts)")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Analysis for Paper-2 Fase 3 (ALD box)")
    p.add_argument("--summary",    action="store_true", help="Print summary table")
    p.add_argument("--fig-stat",   action="store_true", help="Fig 1: stationarity")
    p.add_argument("--fig-tau",    action="store_true", help="Fig 2: τ_eff comparison")
    p.add_argument("--fig-D",      action="store_true", help="Fig 3: D_ALD scaling")
    p.add_argument("--fig-hbar",   action="store_true", help="Fig 4: H̄(t) curves")
    p.add_argument("--all",        action="store_true", help="All figures + summary")
    p.add_argument("--lam",        type=float, default=0.05, help="λ for fig-hbar")
    p.add_argument("--al-dir",     default="results_AL_box")
    p.add_argument("--stat-dir",   default="results_AL_stat")
    p.add_argument("--paper1-dir", default="results_box_om3")
    args = p.parse_args()

    if args.summary or args.all:
        print_summary(args.al_dir)
    if args.fig_stat or args.all:
        fig_stationarity(args.stat_dir)
    if args.fig_tau or args.all:
        fig_tau_comparison(args.al_dir, args.paper1_dir)
    if args.fig_D or args.all:
        fig_D_scaling(args.al_dir)
    if args.fig_hbar or args.all:
        fig_hbar_curves(args.lam, args.al_dir, args.stat_dir, args.paper1_dir)

    if not any([args.summary, args.fig_stat, args.fig_tau,
                args.fig_D, args.fig_hbar, args.all]):
        print("No flag given. Use --summary or --all.")
        print_summary(args.al_dir)
