"""
Estimate the ZPF amplitude calibration from existing lambda-sweep results.

The relaxation fit is

    1 / tau_eff = 1 / tau_V + C * lambda**2

If A_zpf is multiplied by a scale factor s, the weak-coupling prediction is
C_new ~= s**2 * C_old.  This script reports s for a requested target C.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress


def load_results(results_dir):
    rows = []
    for path in sorted(Path(results_dir).glob("results_lam*.json")):
        with open(path) as fh:
            result = json.load(fh)
        rows.append(result)
    if not rows:
        raise SystemExit(f"No result JSON files found in {results_dir}")
    return rows


def fit_tau(hbar_mean, dt):
    hbar = np.asarray(hbar_mean, dtype=float)
    t = np.arange(len(hbar)) * dt
    mask = hbar > 1e-6
    if mask.sum() < 5:
        return np.nan, np.nan

    slope, _, _, _, slope_se = linregress(t[mask], np.log(hbar[mask]))
    if slope >= 0:
        return np.inf, np.nan
    return -1.0 / slope, slope_se / slope**2


def model(lam2, inv_tau_v, c_zpf):
    return inv_tau_v + c_zpf * lam2


def fit_relaxation(rows, dt, max_lambda=None, fixed_intercept=False):
    lambdas = np.array([row["lambda"] for row in rows], dtype=float)
    taus = np.array([fit_tau(row["hbar_mean"], dt)[0] for row in rows], dtype=float)

    order = np.argsort(lambdas)
    lambdas = lambdas[order]
    taus = taus[order]

    valid = np.isfinite(taus) & (taus > 0)
    lambdas = lambdas[valid]
    taus = taus[valid]
    inv_tau = 1.0 / taus

    if not np.any(lambdas == 0):
        raise SystemExit("Need a lambda=0 result to define tau_V.")

    mask = lambdas > 0
    if max_lambda is not None:
        mask &= lambdas <= max_lambda
    if mask.sum() < 2:
        raise SystemExit("Need at least two nonzero lambda values in the fit window.")

    lam2 = lambdas[mask] ** 2
    y = inv_tau[mask]

    if fixed_intercept:
        inv_tau_v = float(inv_tau[lambdas == 0][0])
        c_zpf = float(np.sum(lam2 * (y - inv_tau_v)) / np.sum(lam2 * lam2))
    else:
        inv_tau_v, c_zpf = curve_fit(model, lam2, y, p0=[inv_tau[lambdas == 0][0], 10.0])[0]

    residuals = y - model(lam2, inv_tau_v, c_zpf)
    denom = np.sum((y - y.mean()) ** 2)
    r2 = float("nan") if denom == 0 else float(1.0 - np.sum(residuals**2) / denom)
    return float(1.0 / inv_tau_v), float(c_zpf), r2, int(mask.sum())


def main():
    parser = argparse.ArgumentParser(description="Calibrate ZPF amplitude from result JSON files.")
    parser.add_argument("--dir", default="results_10k", help="results directory to analyze")
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--target-C", type=float, default=42.0,
                        help="target slope C in 1/tau = 1/tau_V + C*lambda^2")
    parser.add_argument("--max-lambda", type=float, action="append",
                        default=None,
                        help="largest lambda to include; can be passed more than once")
    parser.add_argument("--fixed-intercept", action="store_true",
                        help="force 1/tau_V to the lambda=0 value")
    args = parser.parse_args()
    if args.max_lambda is None:
        args.max_lambda = [0.01, 0.02, 0.05]

    rows = load_results(args.dir)
    zpf_scales = sorted({row.get("zpf_scale", 1.0) for row in rows})
    print(f"Results: {args.dir}")
    print(f"Stored zpf_scale values: {', '.join(f'{s:g}' for s in zpf_scales)}")
    print(f"Target C: {args.target_C:g}")
    print()
    print(f"{'lambda max':>10}  {'Nfit':>4}  {'tau_V':>10}  {'C_fit':>10}  {'R2':>7}  {'scale':>10}")
    print("-" * 64)

    for max_lam in args.max_lambda:
        tau_v, c_fit, r2, n_fit = fit_relaxation(
            rows, args.dt, max_lambda=max_lam, fixed_intercept=args.fixed_intercept
        )
        scale = np.sqrt(args.target_C / c_fit) if c_fit > 0 else float("nan")
        print(f"{max_lam:10.4f}  {n_fit:4d}  {tau_v:10.3f}  {c_fit:10.3f}  {r2:7.3f}  {scale:10.3f}")


if __name__ == "__main__":
    main()
