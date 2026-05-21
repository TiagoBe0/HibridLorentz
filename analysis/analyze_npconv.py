"""
analyze_npconv.py  —  Convergence check in Np for Paper-2 Fase 3.

Loads results_AL_npconv/Np{2500,5000,10000}/AL_lam{0.0000,0.1000}_trad0.0100_om3.json
and prints tau_eff (with bootstrap CI) as a function of Np for each lambda.

If tau_eff at Np=10000 differs from Np=5000 by more than ~3 sigma, the
production sweep was undersampled and needs to be rerun at higher Np.
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import linregress


def load(p):
    with open(p) as f:
        return json.load(f)


def fit_tau(hm, t):
    h = np.asarray(hm)
    t = np.asarray(t)
    h0 = h[0] if h[0] > 0 else 1.0
    valid = (h > 0.01 * h0) & (h > 1e-6)
    if valid.sum() < 4:
        return float("inf"), 0.0
    s, i, _, _, _ = linregress(t[valid], np.log(h[valid]))
    if s >= 0:
        return float("inf"), 0.0
    tau = -1.0 / s
    resid = np.log(h[valid]) - (s * t[valid] + i)
    R2 = 1.0 - np.var(resid) / np.var(np.log(h[valid]))
    return tau, R2


def bootstrap_tau(hall, t, n_boot=400, seed=42):
    rng = np.random.default_rng(seed)
    hall = np.asarray(hall)
    nr = hall.shape[0]
    taus = []
    for _ in range(n_boot):
        idx = rng.integers(0, nr, nr)
        tau, _ = fit_tau(hall[idx].mean(axis=0), t)
        if tau < 1e8:
            taus.append(tau)
    if len(taus) < 5:
        return float("nan"), float("nan")
    return float(np.mean(taus)), float(np.std(taus))


def main(root="results_AL_npconv"):
    nps = [2500, 5000, 10000]
    lams = [0.0, 0.10]

    print(f"\n{'Np':>6}  {'λ':>6}  {'τ_pt':>8}  {'τ_boot':>10}  "
          f"{'σ(τ)':>8}  {'R²':>6}  {'H[end]':>8}  {'D_ALD':>10}  {'Nr':>4}")
    print("-" * 78)

    results = {}
    for np_val in nps:
        for lam in lams:
            path = Path(root) / f"Np{np_val}" / f"AL_lam{lam:.4f}_trad0.0100_om3.json"
            if not path.exists():
                print(f"  {np_val:>4d}  {lam:>6.3f}  [MISSING: {path}]")
                continue
            r = load(path)
            t = np.array(r["times"])
            hm = np.array(r["hbar_mean"])
            tau_pt, R2 = fit_tau(hm, t)
            tau_m, tau_s = bootstrap_tau(r["hbar_all"], t)
            D = r.get("D_ALD", 0.0)
            nr = r.get("n_realizations", 0)
            print(f"  {np_val:>4d}  {lam:>6.3f}  {tau_pt:>8.3f}  "
                  f"{tau_m:>10.3f}  {tau_s:>8.4f}  {R2:>6.3f}  "
                  f"{hm[-1]:>8.4f}  {D:>10.2e}  {nr:>4d}")
            results[(np_val, lam)] = (tau_m, tau_s, hm[-1])

    # Convergence verdict
    print("\n=== Convergence verdict ===")
    for lam in lams:
        print(f"\n  λ = {lam}:")
        prev = None
        for np_val in nps:
            cur = results.get((np_val, lam))
            if cur is None:
                continue
            tau_m, tau_s, hend = cur
            if prev is not None:
                dtau = tau_m - prev[0]
                comb_s = np.hypot(tau_s, prev[1])
                z = abs(dtau) / comb_s if comb_s > 0 else float("inf")
                tag = "OK (within 3σ)" if z < 3 else "NOT converged (>3σ)"
                print(f"    Np {prev[2]:>5d} → {np_val:>5d}:  "
                      f"Δτ = {dtau:+.3f}  ({z:.1f}σ)  {tag}")
            prev = (tau_m, tau_s, np_val)

    # Effect size: Δτ between λ=0 and λ=0.10 at each Np
    print("\n=== Effect size: τ(λ=0.10) − τ(λ=0) ===")
    print(f"  (production sweep showed +0.46 at Np=5000)")
    for np_val in nps:
        a = results.get((np_val, 0.0))
        b = results.get((np_val, 0.10))
        if a is None or b is None:
            continue
        dtau = b[0] - a[0]
        s = np.hypot(a[1], b[1])
        z = dtau / s if s > 0 else float("inf")
        print(f"  Np = {np_val:>5d}:  Δτ = {dtau:+.3f} ± {s:.3f}  ({z:.1f}σ)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results_AL_npconv")
    args = p.parse_args()
    main(args.root)
