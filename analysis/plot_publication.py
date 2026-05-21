"""
Publication-quality figures for hybrid BdB/ZPF Born-rule relaxation study.
Bergamin & Bringa (2026)

Produces four individual PDF figures:
  fig1_hbar_evolution.pdf   — H̄(t)/H̄(0) vs t
  fig2_ln_hbar.pdf          — ln[H̄(t)/H̄(0)] vs t with exponential fits
  fig3_relaxation_rate.pdf  — 1/τ_eff vs λ² (hypothesis test)
  fig4_ks_vs_lambda.pdf     — D_KS vs λ (non-monotone behaviour)
"""

import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import linregress

# ── Publication style ──────────────────────────────────────────────────────────
mpl.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "font.serif": ["STIX", "STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "axes.labelsize": 11,
    "font.size": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
    "errorbar.capsize": 3,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# Single-column width for PRL/PRX style
FIG_W = 3.375   # inches  (one column)
FIG_H = 2.8     # inches

OUT_DIR = Path("figures_pub")
OUT_DIR.mkdir(exist_ok=True)

# ── Colour palette (colorblind-friendly, ordered by λ) ────────────────────────
CMAP = mpl.colormaps["plasma"]


def lam_label(lam):
    """Format λ value with enough decimal places to avoid label collisions."""
    if lam == 0.0:
        return r"$\lambda=0$"
    s = f"{lam:.4g}"          # e.g. 0.0005, 0.001, 0.05
    return rf"$\lambda={s}$"

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_dir(results_dir):
    data = {}
    for f in sorted(Path(results_dir).glob("results_lam*.json")):
        with open(f) as fh:
            r = json.load(fh)
        data[r["lambda"]] = r
    return data


def fit_tau(hbar_mean, dt=0.002):
    t = np.arange(len(hbar_mean)) * dt
    H = np.array(hbar_mean)
    mask = H > 1e-6
    if mask.sum() < 5:
        return np.nan, np.nan
    slope, intercept, _, _, se = linregress(t[mask], np.log(H[mask]))
    if slope >= 0:
        return np.inf, np.nan
    return -1.0 / slope, se / slope**2


def extract_taus(data, dt=0.002):
    lambdas, taus, tau_errs = [], [], []
    for lam in sorted(data):
        r = data[lam]
        tau, tau_e = fit_tau(r["hbar_mean"], dt=dt)
        lambdas.append(lam)
        taus.append(tau)
        tau_errs.append(tau_e if not np.isnan(tau_e) else 0.0)
    return np.array(lambdas), np.array(taus), np.array(tau_errs)


def quadratic_model(lam2, inv_tau_V, C):
    return inv_tau_V + C * lam2


def fit_relaxation_law(lambdas, taus, tau_errs):
    inv_tau = 1.0 / taus
    inv_tau_err = tau_errs / taus**2
    mask = lambdas > 0
    lam2 = lambdas[mask]**2
    popt, pcov = curve_fit(quadratic_model, lam2, inv_tau[mask],
                           p0=[1.0 / taus[0], 10.0])
    perr = np.sqrt(np.diag(pcov))
    residuals = inv_tau[mask] - quadratic_model(lam2, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((inv_tau[mask] - inv_tau[mask].mean())**2)
    R2 = 1.0 - ss_res / ss_tot
    return popt, perr, R2, inv_tau, inv_tau_err


# ── Load both datasets ────────────────────────────────────────────────────────
d1k  = load_dir("results_full")   # Np=1000, Nr=20
d10k = load_dir("results_10k")    # Np=10000, Nr=10

DT = 0.002
N_STEPS = 300
t_arr = np.arange(N_STEPS + 1) * DT

lam_1k,  tau_1k,  terr_1k  = extract_taus(d1k,  dt=DT)
lam_10k, tau_10k, terr_10k = extract_taus(d10k, dt=DT)

popt_1k,  perr_1k,  R2_1k,  inv_1k,  ierr_1k  = fit_relaxation_law(lam_1k,  tau_1k,  terr_1k)
popt_10k, perr_10k, R2_10k, inv_10k, ierr_10k = fit_relaxation_law(lam_10k, tau_10k, terr_10k)

lambdas_10k = sorted(d10k.keys())
n_lam = len(lambdas_10k)
colors = [CMAP(i / (n_lam - 1)) for i in range(n_lam)]

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — H̄(t)/H̄(0) evolution
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

for i, lam in enumerate(lambdas_10k):
    r = d10k[lam]
    H  = np.array(r["hbar_mean"])
    dH = np.array(r["hbar_std"])
    H0 = H[0] if H[0] > 0 else 1.0
    steps = len(H)
    ax.plot(t_arr[:steps], H / H0, color=colors[i], alpha=0.85)
    ax.fill_between(t_arr[:steps],
                    (H - dH) / H0, (H + dH) / H0,
                    color=colors[i], alpha=0.12)

ax.set_xlabel(r"$t$ (dimensionless)")
ax.set_ylabel(r"$\bar{H}(t)\,/\,\bar{H}(0)$")
ax.set_xlim(0, t_arr[N_STEPS])
ax.set_ylim(0.65, 1.02)

# Colourbar indicating λ (legend redundant with colorbar)
sm = mpl.cm.ScalarMappable(cmap=CMAP,
                            norm=mpl.colors.Normalize(vmin=0, vmax=max(lambdas_10k)))
sm.set_array([])
cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.03)
cb.set_label(r"$\lambda$", labelpad=4)
cb.ax.tick_params(labelsize=9)

ax.text(0.97, 0.97, rf"$N_p=10^4,\;N_r=10$",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9))

fig.savefig(OUT_DIR / "fig1_hbar_evolution.pdf")
fig.savefig(OUT_DIR / "fig1_hbar_evolution.png")
plt.close(fig)
print("fig1_hbar_evolution.pdf  saved")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — ln[H̄(t)/H̄(0)] with exponential fits
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

for i, lam in enumerate(lambdas_10k):
    r = d10k[lam]
    H  = np.array(r["hbar_mean"])
    H0 = H[0] if H[0] > 0 else 1.0
    steps = len(H)
    mask = H > 1e-6
    t_m  = t_arr[:steps][mask]
    lnH  = np.log(H[mask] / H0)
    ax.plot(t_m, lnH, color=colors[i], alpha=0.8)

# Overlay one representative exponential envelope for τ_V (λ=0)
tau_ref = tau_10k[0]
t_env = np.linspace(0, t_arr[N_STEPS], 200)
ax.plot(t_env, -t_env / tau_ref, "k--", linewidth=1.0, alpha=0.55,
        label=rf"$-t/\tau_V$  ($\tau_V={tau_ref:.3f}$, $\lambda=0$)")

ax.set_xlabel(r"$t$ (dimensionless)")
ax.set_ylabel(r"$\ln\!\left[\bar{H}(t)/\bar{H}(0)\right]$")
ax.set_xlim(0, t_arr[N_STEPS])
ax.legend(loc="lower left", framealpha=0.9, edgecolor="0.7", fontsize=8)

sm = mpl.cm.ScalarMappable(cmap=CMAP,
                            norm=mpl.colors.Normalize(vmin=0, vmax=max(lambdas_10k)))
sm.set_array([])
cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.03)
cb.set_label(r"$\lambda$", labelpad=4)
cb.ax.tick_params(labelsize=9)

# Annotate the initial transient (ln H̄ > 0 briefly)
ax.annotate("initial transient\n" r"($\psi$ reaches slits)",
            xy=(0.155, 0.115), xytext=(0.28, 0.11),
            xycoords=("data", "data"), textcoords=("data", "data"),
            arrowprops=dict(arrowstyle="-|>", color="0.35",
                            lw=0.8, mutation_scale=9),
            fontsize=7.5, color="0.35", ha="left")

fig.savefig(OUT_DIR / "fig2_ln_hbar.pdf")
fig.savefig(OUT_DIR / "fig2_ln_hbar.png")
plt.close(fig)
print("fig2_ln_hbar.pdf         saved")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — 1/τ_eff vs λ²  (hypothesis test: Γ ∝ λ²)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

lam2_plot = np.linspace(0, max(lam_10k)**2 * 1.05, 200)

# N_p=1000
mask1 = ~np.isinf(tau_1k) & ~np.isnan(tau_1k) & (lam_1k > 0)
ax.errorbar(lam_1k[mask1]**2, inv_1k[mask1], yerr=ierr_1k[mask1],
            fmt="o", ms=4, color="#4477AA", alpha=0.7,
            label=rf"$N_p=10^3$, $N_r=20$", zorder=4)
ax.plot(lam2_plot, quadratic_model(lam2_plot, *popt_1k),
        "-", color="#4477AA", alpha=0.5, linewidth=1.0)

# N_p=10000
mask2 = ~np.isinf(tau_10k) & ~np.isnan(tau_10k) & (lam_10k > 0)
ax.errorbar(lam_10k[mask2]**2, inv_10k[mask2], yerr=ierr_10k[mask2],
            fmt="s", ms=4, color="#EE6677", alpha=0.9,
            label=rf"$N_p=10^4$, $N_r=10$", zorder=5)
ax.plot(lam2_plot, quadratic_model(lam2_plot, *popt_10k),
        "-", color="#EE6677", linewidth=1.4,
        label=(rf"$1/\tau_V + C\lambda^2$"
               rf"  ($R^2={R2_10k:.3f}$)"))

# 1/τ_V reference line (10k fit)
ax.axhline(popt_10k[0], color="#EE6677", linestyle=":", linewidth=0.9, alpha=0.7)
ax.text(0.38, popt_10k[0] * 1.003,
        rf"$1/\tau_V = {popt_10k[0]:.3f}$",
        fontsize=8, color="#EE6677", va="bottom",
        transform=ax.get_yaxis_transform())

ax.set_xlabel(r"$\lambda^2$")
ax.set_ylabel(r"$1/\tau_{\rm eff}$")
ax.legend(loc="upper left", framealpha=0.9, edgecolor="0.7")

# Fit parameters box
tbox = (rf"$N_p=10^4$:" "\n"
        rf"$\tau_V = {1/popt_10k[0]:.3f}$" "\n"
        rf"$C = {popt_10k[1]:.2f}$" "\n"
        rf"$R^2 = {R2_10k:.3f}$")
ax.text(0.97, 0.05, tbox, transform=ax.transAxes,
        ha="right", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.8", alpha=0.95))

fig.savefig(OUT_DIR / "fig3_relaxation_rate.pdf")
fig.savefig(OUT_DIR / "fig3_relaxation_rate.png")
plt.close(fig)
print("fig3_relaxation_rate.pdf saved")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — D_KS vs λ  (non-monotone behaviour)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

def ks_arrays(data):
    lams = np.array(sorted(data.keys()))
    ks_m = np.array([data[l]["ks_mean"] for l in lams])
    ks_s = np.array([data[l]["ks_std"]  for l in lams])
    return lams, ks_m, ks_s

lams1, ks1m, ks1s = ks_arrays(d1k)
lams2, ks2m, ks2s = ks_arrays(d10k)

# Reference (λ=0) for both
ref1 = ks1m[lams1 == 0.0][0]
ref2 = ks2m[lams2 == 0.0][0]

ax.axhline(ref2, color="0.65", linestyle="--", linewidth=0.9,
           label=r"$D_{\rm KS}(\lambda=0)$, $N_p=10^4$")

ax.errorbar(lams1, ks1m, yerr=ks1s, fmt="o-", ms=4,
            color="#4477AA", alpha=0.75, linewidth=1.0,
            label=rf"$N_p=10^3$, $N_r=20$")

ax.errorbar(lams2, ks2m, yerr=ks2s, fmt="s-", ms=4,
            color="#EE6677", alpha=0.9, linewidth=1.4,
            label=rf"$N_p=10^4$, $N_r=10$")

# Arrow marking the minimum
imin = np.argmin(ks2m[lams2 > 0]) + 1   # index in full array
ax.annotate(r"$\lambda^*\!\approx 0.02$",
            xy=(lams2[imin], ks2m[imin]),
            xytext=(lams2[imin] + 0.005, ks2m[imin] - 0.0015),
            arrowprops=dict(arrowstyle="-|>", color="0.3",
                            lw=0.8, mutation_scale=10),
            fontsize=8, color="0.3")

ax.set_xlabel(r"$\lambda$")
ax.set_ylabel(r"$D_{\rm KS}$")
ax.legend(loc="upper right", framealpha=0.9, edgecolor="0.7")

fig.savefig(OUT_DIR / "fig4_ks_vs_lambda.pdf")
fig.savefig(OUT_DIR / "fig4_ks_vs_lambda.png")
plt.close(fig)
print("fig4_ks_vs_lambda.pdf    saved")

print(f"\nAll figures written to {OUT_DIR.resolve()}/")
