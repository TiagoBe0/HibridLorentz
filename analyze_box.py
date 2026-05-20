"""
analyze_box.py — Analysis and visualization for the 2D box Born-rule relaxation results.

Generates four figures:
  1. H̄(t) curves for all lambda (one panel per omega_max)
  2. τ_eff vs λ with ±1σ error bars
  3. 1/τ_eff vs λ² with quadratic fit (perturbative window)
  4. Comparison of omega_max=3 vs omega_max=15
"""

import json, argparse
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import linregress, ttest_ind, wilcoxon

# ─── Import matplotlib with Agg backend (no display required) ─────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ─── Bootstrap τ_eff from hbar_all ────────────────────────────────────────────

def fit_tau_from_series(t, h_mean):
    """Fit τ_eff via log-linear regression. Returns (τ_eff, R²)."""
    h0 = h_mean[0] if h_mean[0] > 0 else 1.0
    valid = (h_mean > 0.01 * h0) & (h_mean > 1e-6)
    if valid.sum() < 4:
        return float('inf'), 0.0
    slope, _, _, _, _ = linregress(t[valid], np.log(h_mean[valid]))
    if slope >= 0:
        return float('inf'), 0.0
    tau = -1.0 / slope
    resid = np.log(h_mean[valid]) - (slope * t[valid] + np.log(h_mean[valid][0]))
    R2 = float(1 - np.var(resid) / np.var(np.log(h_mean[valid])))
    return tau, R2


def bootstrap_tau(t, hall, n_boot=1000, rng_seed=0):
    """Bootstrap τ_eff distribution. Returns (mean, std, [2.5%ile, 97.5%ile])."""
    rng = np.random.default_rng(rng_seed)
    nr = hall.shape[0]
    taus = []
    for _ in range(n_boot):
        idx = rng.choice(nr, nr, replace=True)
        h_b = hall[idx].mean(axis=0)
        tau, _ = fit_tau_from_series(t, h_b)
        if np.isfinite(tau):
            taus.append(tau)
    taus = np.array(taus)
    return taus.mean(), taus.std(), np.percentile(taus, [2.5, 97.5])


# ─── Statistical comparison ───────────────────────────────────────────────────

def compare_to_baseline(r_lam, r_base, alpha=0.05):
    """
    Compare τ_eff distribution at lambda>0 vs baseline using Welch t-test.
    Also compares H̄(t) integrated area (more sensitive than τ_eff alone).

    Returns dict with:
      tau_diff, tau_diff_se, tau_p_value
      area_diff, area_diff_se, area_p_value
    """
    t = np.array(r_base["times"])
    hall_base = np.array(r_base["hbar_all"])   # (nr_base, nt)
    hall_lam  = np.array(r_lam["hbar_all"])    # (nr_lam,  nt)

    # τ_eff per realization
    taus_base = []
    for hh in hall_base:
        tau, _ = fit_tau_from_series(t, hh)
        if np.isfinite(tau) and tau < 1e6:
            taus_base.append(tau)
    taus_lam = []
    for hh in hall_lam:
        tau, _ = fit_tau_from_series(t, hh)
        if np.isfinite(tau) and tau < 1e6:
            taus_lam.append(tau)

    taus_base = np.array(taus_base)
    taus_lam  = np.array(taus_lam)

    tdiff  = taus_lam.mean() - taus_base.mean()
    tdiff_se = np.sqrt(taus_lam.var(ddof=1)/len(taus_lam) + taus_base.var(ddof=1)/len(taus_base))
    _, tp = ttest_ind(taus_lam, taus_base, equal_var=False)

    # Integrated H̄: ∫H̄(t)dt per realization (trapezoid)
    area_base = np.trapezoid(hall_base, t, axis=1)
    area_lam  = np.trapezoid(hall_lam,  t, axis=1)
    adiff = area_lam.mean() - area_base.mean()
    adiff_se = np.sqrt(area_lam.var(ddof=1)/len(area_lam) + area_base.var(ddof=1)/len(area_base))
    _, ap = ttest_ind(area_lam, area_base, equal_var=False)

    return {
        "tau_diff": tdiff, "tau_diff_se": tdiff_se, "tau_p": tp,
        "area_diff": adiff, "area_diff_se": adiff_se, "area_p": ap,
        "n_base": len(taus_base), "n_lam": len(taus_lam),
    }


def print_statistical_summary(rows):
    """Print table: lambda, Δτ±SE, p-val, Δarea±SE, p-val."""
    if not rows:
        return
    r_base = next((r for r in rows if r["lambda"] == 0.0), rows[0])
    print()
    print(f"{'λ':>7}  {'Δτ':>7}  {'SE':>6}  {'z_τ':>6}  "
          f"{'Δarea':>8}  {'SE':>6}  {'z_area':>7}  {'τ_eff':>7}")
    print("-" * 70)
    for r in rows:
        if r["lambda"] == 0.0:
            continue
        c = compare_to_baseline(r, r_base)
        z_tau  = c["tau_diff"] / c["tau_diff_se"]
        z_area = c["area_diff"] / c["area_diff_se"]
        print(f"{r['lambda']:>7.4f}  {c['tau_diff']:>+7.3f}  {c['tau_diff_se']:>6.3f}  "
              f"{z_tau:>+6.2f}  {c['area_diff']:>+8.4f}  {c['area_diff_se']:>6.4f}  "
              f"{z_area:>+7.2f}  {r['tau_boot_mean']:>7.3f}")


# ─── Load results ─────────────────────────────────────────────────────────────

def load_results(results_dir, pattern="box_lam*.json", exclude_tag=None):
    rows = []
    for path in sorted(Path(results_dir).glob(pattern)):
        if exclude_tag and exclude_tag in path.name:
            continue
        with open(path) as fh:
            d = json.load(fh)
        rows.append(d)
    return rows


def augment_with_bootstrap(rows):
    """Add bootstrap τ estimates to each row."""
    for r in rows:
        t   = np.array(r["times"])
        hall = np.array(r["hbar_all"])
        tau_m, tau_s, tau_ci = bootstrap_tau(t, hall)
        r["tau_boot_mean"] = tau_m
        r["tau_boot_std"]  = tau_s
        r["tau_boot_ci"]   = tau_ci


# ─── Plotting helpers ─────────────────────────────────────────────────────────

def plot_hbar_t(rows, ax, title=""):
    """H̄(t) for each lambda, with ±1σ band."""
    cmap = cm.viridis
    lams = [r["lambda"] for r in rows]
    norm = plt.Normalize(vmin=0, vmax=max(lams) if max(lams) > 0 else 1)
    for r in rows:
        lam = r["lambda"]
        t   = np.array(r["times"])
        hm  = np.array(r["hbar_mean"])
        hs  = np.array(r["hbar_std"])
        color = cmap(norm(lam))
        ax.plot(t, hm, color=color, label=f"λ={lam:.3f}")
        ax.fill_between(t, hm - hs, hm + hs, alpha=0.15, color=color)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\bar{H}(t)$")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    ax.set_yscale("log")


def plot_tau_vs_lam(rows, ax, label="", color="steelblue"):
    lams  = np.array([r["lambda"] for r in rows])
    taus  = np.array([r["tau_boot_mean"] for r in rows])
    stds  = np.array([r["tau_boot_std"]  for r in rows])
    ax.errorbar(lams, taus, yerr=stds, fmt="o-", color=color,
                capsize=3, label=label)
    ax.axhline(taus[0], ls="--", color=color, alpha=0.4)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\tau_\mathrm{eff}$")


def plot_inv_tau_vs_lam2(rows, ax, max_lambda=None, label="", color="steelblue"):
    lams     = np.array([r["lambda"] for r in rows])
    inv_taus = np.array([1.0 / r["tau_boot_mean"] if r["tau_boot_mean"] < 1e9 else 0
                         for r in rows])
    stds     = np.array([r["tau_boot_std"] / r["tau_boot_mean"]**2 if r["tau_boot_mean"] < 1e9 else 0
                         for r in rows])

    ax.errorbar(lams**2, inv_taus, yerr=stds, fmt="o", color=color,
                capsize=3, label=label)

    # linear fit in the perturbative window
    mask = (lams > 0) & np.isfinite(inv_taus)
    if max_lambda is not None:
        mask &= lams <= max_lambda
    if mask.sum() >= 2:
        lam2_f = lams[mask]**2
        inv_f  = inv_taus[mask]
        # intercept fixed at lambda=0 value
        inv0 = inv_taus[lams == 0][0] if np.any(lams == 0) else inv_taus[0]
        C = np.sum(lam2_f * (inv_f - inv0)) / np.sum(lam2_f**2)
        lam2_plot = np.linspace(0, lam2_f.max(), 100)
        ax.plot(lam2_plot, inv0 + C * lam2_plot, "--", color=color, alpha=0.6,
                label=f"{label} fit C={C:.1f}")
        return float(C)
    return float('nan')


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze 2D box simulation results.")
    parser.add_argument("--dir-om15", default="results_box",
                        help="Results dir for omega_max=15")
    parser.add_argument("--dir-om3",  default="results_box_om3",
                        help="Results dir for omega_max=3")
    parser.add_argument("--out",      default="box_analysis.pdf",
                        help="Output PDF filename")
    parser.add_argument("--max-lam-fit", type=float, default=0.05,
                        help="λ cutoff for perturbative fit")
    args = parser.parse_args()

    # om=15 results: files WITHOUT any _omX tag
    rows15 = load_results(args.dir_om15, "box_lam*.json", exclude_tag="_om")
    # om=3: prefer dedicated dir; fall back to _om3-tagged files in dir_om15
    om3_dir = Path(args.dir_om3)
    om3_dedicated = list(om3_dir.glob("box_lam*.json")) if om3_dir.exists() else []
    if om3_dedicated:
        # dedicated dir: all files here are om3 results, accept any tag
        rows3 = load_results(args.dir_om3, "box_lam*.json")
    else:
        rows3 = (load_results(args.dir_om15, "box_lam*_om3.json") or
                 load_results(args.dir_om3,  "box_lam*_om3.json"))

    if not rows15:
        print(f"No results in {args.dir_om15} (om=15)")
    if not rows3:
        print(f"No results in {args.dir_om3} (om=3)")

    for rows in [rows15, rows3]:
        augment_with_bootstrap(rows)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Born-rule relaxation in 2D box: ZPF coupling study", fontsize=12)

    # Panel A: H̄(t) for omega_max=15
    if rows15:
        plot_hbar_t(rows15, axes[0, 0],
                    title=r"$\bar{H}(t)$, $\omega_\mathrm{max}=15$ (double-slit calibrated)")

    # Panel B: H̄(t) for omega_max=3
    if rows3:
        plot_hbar_t(rows3, axes[0, 1],
                    title=r"$\bar{H}(t)$, $\omega_\mathrm{max}=3$ (box-matched ZPF)")

    # Panel C: τ_eff vs λ (comparison)
    ax = axes[1, 0]
    if rows15:
        plot_tau_vs_lam(rows15, ax, label=r"$\omega_\mathrm{max}=15$", color="tomato")
    if rows3:
        plot_tau_vs_lam(rows3, ax, label=r"$\omega_\mathrm{max}=3$", color="steelblue")
    ax.set_title(r"$\tau_\mathrm{eff}(\lambda)$")
    ax.legend()

    # Panel D: 1/τ vs λ²
    ax = axes[1, 1]
    C15 = float('nan')
    C3  = float('nan')
    if rows15:
        C15 = plot_inv_tau_vs_lam2(rows15, ax, max_lambda=args.max_lam_fit,
                                    label=r"$\omega_\mathrm{max}=15$", color="tomato")
    if rows3:
        C3 = plot_inv_tau_vs_lam2(rows3, ax, max_lambda=args.max_lam_fit,
                                   label=r"$\omega_\mathrm{max}=3$", color="steelblue")
    ax.set_title(r"$1/\tau_\mathrm{eff}$ vs $\lambda^2$ (Fokker-Planck prediction)")
    ax.set_xlabel(r"$\lambda^2$")
    ax.set_ylabel(r"$1/\tau_\mathrm{eff}$")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved → {args.out}")

    # Text summary
    print()
    if rows15:
        print(f"om=15 summary:")
        print(f"  {'lambda':>8}  {'tau_boot':>10}  {'±std':>8}  {'1/tau':>8}")
        for r in rows15:
            print(f"  {r['lambda']:>8.4f}  {r['tau_boot_mean']:>10.3f}  "
                  f"±{r['tau_boot_std']:>6.3f}  {1/r['tau_boot_mean']:>8.4f}")
        print(f"  C_fit (λ≤{args.max_lam_fit}) = {C15:.2f}")
        print()

    if rows3:
        print(f"om=3 summary:")
        print(f"  {'lambda':>8}  {'tau_boot':>10}  {'±std':>8}  {'1/tau':>8}")
        for r in rows3:
            print(f"  {r['lambda']:>8.4f}  {r['tau_boot_mean']:>10.3f}  "
                  f"±{r['tau_boot_std']:>6.3f}  {1/r['tau_boot_mean']:>8.4f}")
        print(f"  C_fit (λ≤{args.max_lam_fit}) = {C3:.2f}")
        print("\n  Statistical tests (Δτ and Δarea vs λ=0 baseline):")
        print_statistical_summary(rows3)


if __name__ == "__main__":
    main()
