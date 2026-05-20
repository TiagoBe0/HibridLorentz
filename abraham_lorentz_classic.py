"""
abraham_lorentz_classic.py  —  Fase 2, Paper-2
===============================================
Benchmark numérico de Boyer (1975): oscilador armónico clásico en ZPF
con reacción de radiación Abraham-Lorentz-Dirac (aproximación Landau-Lifshitz).

Verifica la predicción:
    ⟨x²⟩_stat = ℏ/(2mω₀) = 1/(2ω₀)   [en ℏ=m=1]

Ecuación de movimiento (Landau-Lifshitz):
    ẍ + γẋ + ω₀²x = ξ(t)
    γ = τ_rad · ω₀²           [amortiguamiento ALD via LL]
    ⟨ξ(t)ξ(t')⟩ = σ²δ(t-t')  [FDT cuántico: σ² = γω₀ = 2D·γ con D=ℏ/2m]

Método numérico: Euler-Maruyama (consistente con Paper-1)
    Δv = [-ω₀²x - γv]·dt + σ·√dt·η,   η ~ N(0,1)

NOTA sobre los modos ZPF discretos:
    Para reproducir ⟨x²⟩_Boyer con N modos, se necesita Δω ≪ γ (la resonancia
    es estrecha). Para τ_rad=0.01: γ=0.01 → requiere Δω < 0.001 → N > 5000
    modos. El límite de muchos modos colapsa en ruido blanco FDT, que es lo
    que implementamos aquí. En Fase 3 (caja de Valentini) los modos discretos
    son suficientes porque λ es mayor y la geometría es acotada.

Estructura de tests
-------------------
  single       : ω₀=1, τ_rad=0.01 — convergencia de ⟨x²⟩(t)
  omega_sweep  : ω₀ ∈ {0.5,1.0,1.5,2.0,3.0} — verifica ⟨x²⟩ ∝ 1/ω₀
  tau_sweep    : τ_rad ∈ {0.005,0.01,0.02,0.05} — verifica universalidad
  all          : los tres tests en secuencia

Uso
---
  python3 abraham_lorentz_classic.py --test single
  python3 abraham_lorentz_classic.py --test all --N_realiz 30

Referencias
-----------
Boyer (1975) Phys. Rev. D 11, 790
de la Peña & Cetto (2015) The Emerging Quantum, Cap. 4
Nelson (1966) Phys. Rev. 150, 1079
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, time, argparse

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Boyer 1975 benchmark — OA+ZPF+ALD')
parser.add_argument('--test',     default='all',
                    choices=['single', 'omega_sweep', 'tau_sweep', 'all'])
parser.add_argument('--omega0',   type=float, default=1.0)
parser.add_argument('--tau_rad',  type=float, default=0.01)
parser.add_argument('--N_realiz', type=int,   default=30)
parser.add_argument('--T_total',  type=float, default=None,
                    help='Tiempo total. Default: auto (20 × τ_relax)')
parser.add_argument('--dt',       type=float, default=0.02)
parser.add_argument('--seed',     type=int,   default=42)
parser.add_argument('--output',   default='boyer_1975_results.json')
args = parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# FÍSICA
# ─────────────────────────────────────────────────────────────────────────────

def sigma_fdt(gamma: float, omega_0: float) -> float:
    """
    Amplitud del ruido (FDT cuántico, T=0):
        σ² = 2·γ·k_BT_eff  con k_BT_eff = ℏω₀/2
        σ² = γ·ω₀   [en ℏ=m=1]
    Esto es el límite de muchos modos ZPF (continuo).
    """
    return np.sqrt(gamma * omega_0)


def run_realization(omega_0: float, gamma: float, sigma: float,
                    dt: float, T_total: float, seed: int,
                    discard_frac: float = 0.5):
    """
    Integra una realización del OA+ALD con Euler-Maruyama.

    dv = [-ω₀²x - γv]dt + σ√dt·η,  η ~ N(0,1)
    dx = v·dt

    Retorna: x2_mean, v2_mean, x_arr (fase estacionaria), t_arr
    """
    rng = np.random.default_rng(seed)
    N_t = int(T_total / dt)

    # Condición inicial fuera de equilibrio (x=0, v=0)
    x, v = 0.0, 0.0

    x2_arr = np.zeros(N_t)
    v2_arr = np.zeros(N_t)

    sqrt_dt = np.sqrt(dt)
    sigma_disc = sigma * sqrt_dt   # amplitud del ruido por paso de tiempo

    # Loop de integración Euler-Maruyama
    for n in range(N_t):
        noise = rng.standard_normal()
        a     = -omega_0**2 * x - gamma * v + sigma * noise / sqrt_dt
        x    += v * dt
        v    += a * dt
        x2_arr[n] = x * x
        v2_arr[n] = v * v

    # Descartar burn-in
    n0 = int(N_t * discard_frac)
    return (float(np.mean(x2_arr[n0:])),
            float(np.mean(v2_arr[n0:])),
            x2_arr[n0:],
            np.arange(n0, N_t) * dt)


def run_realization_vec(omega_0: float, gamma: float, sigma: float,
                        dt: float, T_total: float, seed: int,
                        discard_frac: float = 0.5):
    """
    Integra con Euler Simpléctico (semi-implícito en posición):
        v_{n+1} = v_n + [-ω₀²x_n - γv_n + σ·η_n/√dt]·dt
        x_{n+1} = x_n + v_{n+1}·dt   ← usa velocidad YA actualizada

    Estabilidad: |det(M)| = 1 - γdt < 1 para γ > 0, estable para dt < 2/ω₀.
    Euler explícito (x+= v*dt antes de actualizar v) diverge cuando dt > γ/ω₀².
    """
    rng     = np.random.default_rng(seed)
    N_t     = int(T_total / dt)
    eta     = rng.standard_normal(N_t)
    sqrt_dt = np.sqrt(dt)

    x = np.empty(N_t + 1)
    v = np.empty(N_t + 1)
    x[0], v[0] = 0.0, 0.0

    for n in range(N_t):
        a      = -omega_0**2 * x[n] - gamma * v[n] + sigma * eta[n] / sqrt_dt
        v[n+1] = v[n] + a * dt           # 1. actualizar v
        x[n+1] = x[n] + v[n+1] * dt     # 2. actualizar x con v nuevo

    x = x[1:]
    v = v[1:]

    n0 = int(N_t * discard_frac)
    return (float(np.mean(x[n0:]**2)),
            float(np.mean(v[n0:]**2)),
            x[n0:]**2,
            np.arange(n0, N_t) * dt)


def auto_T_total(gamma: float, n_relax: float = 40.0) -> float:
    """T_total = n_relax × τ_relax con burn-in del 50%."""
    return n_relax / gamma   # = n_relax × τ_relax


def run_ensemble(omega_0, tau_rad, N_realiz, T_total, dt, master_seed, label=''):
    """
    Promedia ⟨x²⟩ y ⟨v²⟩ sobre N_realiz realizaciones.
    Retorna dict con media, SEM y estadísticas.
    """
    gamma      = tau_rad * omega_0**2
    sigma      = sigma_fdt(gamma, omega_0)
    tau_relax  = 1.0 / gamma
    boyer_pred = 1.0 / (2.0 * omega_0)

    if T_total is None:
        T_total = auto_T_total(gamma)

    print(f"\n{'─'*60}")
    print(f"  {label or f'ω₀={omega_0}, τ_rad={tau_rad}'}")
    print(f"  γ = {gamma:.5f}  |  τ_relax = {tau_relax:.1f}  |  "
          f"T_total = {T_total:.0f} ({T_total*gamma:.0f}×τ_relax)")
    print(f"  σ_FDT = {sigma:.5f}  |  Predicción Boyer: ⟨x²⟩ = {boyer_pred:.4f}")
    print(f"  τ_rad·ω₀ = {tau_rad*omega_0:.3f}  [LL válido si ≪ 1]")
    print(f"  Corriendo {N_realiz} realizaciones...", flush=True)

    rng   = np.random.default_rng(master_seed)
    seeds = rng.integers(0, 2**31, size=N_realiz)

    x2_vals, v2_vals = [], []
    t_start = time.time()

    for r in range(N_realiz):
        x2, v2, _, _ = run_realization_vec(omega_0, gamma, sigma, dt, T_total, seeds[r])
        x2_vals.append(x2)
        v2_vals.append(v2)
        print(f"    r={r+1:02d}  ⟨x²⟩={x2:.4f}  ⟨v²⟩={v2:.4f}", flush=True)

    elapsed = time.time() - t_start

    x2_mean = float(np.mean(x2_vals))
    x2_sem  = float(np.std(x2_vals, ddof=1) / np.sqrt(N_realiz))
    v2_mean = float(np.mean(v2_vals))
    v2_sem  = float(np.std(v2_vals, ddof=1) / np.sqrt(N_realiz))
    error   = abs(x2_mean - boyer_pred) / boyer_pred * 100.0
    z       = (x2_mean - boyer_pred) / x2_sem if x2_sem > 0 else float('inf')

    # Equipartición: ⟨v²⟩ = ω₀²⟨x²⟩ → ratio debe ser 1
    equip_ratio = v2_mean / (omega_0**2 * x2_mean) if x2_mean > 0 else 0.0

    print(f"\n  ⟨x²⟩ = {x2_mean:.4f} ± {x2_sem:.4f}  (Boyer: {boyer_pred:.4f})")
    print(f"  ⟨v²⟩ = {v2_mean:.4f} ± {v2_sem:.4f}  "
          f"(equip pred: {boyer_pred*omega_0**2:.4f})")
    print(f"  Error: {error:.1f}%  |  z = {z:.2f}σ  |  "
          f"⟨v²⟩/(ω₀²⟨x²⟩) = {equip_ratio:.3f}")
    print(f"  Tiempo: {elapsed:.1f} s")

    return {
        'omega_0':      omega_0,
        'tau_rad':      tau_rad,
        'gamma':        gamma,
        'sigma_fdt':    sigma,
        'tau_relax':    tau_relax,
        'N_realiz':     N_realiz,
        'T_total':      T_total,
        'dt':           dt,
        'boyer_pred':   boyer_pred,
        'x2_mean':      x2_mean,
        'x2_sem':       x2_sem,
        'v2_mean':      v2_mean,
        'v2_sem':       v2_sem,
        'x2_all':       x2_vals,
        'v2_all':       v2_vals,
        'error_pct':    error,
        'z_score':      z,
        'equip_ratio':  equip_ratio,
        'elapsed_s':    elapsed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: single — convergencia ⟨x²⟩(t)
# ─────────────────────────────────────────────────────────────────────────────

def test_single(args):
    print("\n" + "="*60)
    print("TEST 1 — Convergencia temporal ⟨x²⟩(t) → 1/(2ω₀)")
    print("="*60)

    omega_0    = args.omega0
    tau_rad    = args.tau_rad
    gamma      = tau_rad * omega_0**2
    sigma      = sigma_fdt(gamma, omega_0)
    T_total    = args.T_total or auto_T_total(gamma)
    boyer_pred = 1.0 / (2.0 * omega_0)

    print(f"  ω₀={omega_0}, γ={gamma:.4f}, σ={sigma:.4f}, T={T_total:.0f}")

    # Trayectoria única para mostrar convergencia
    _, _, x2_series, t_stat = run_realization_vec(
        omega_0, gamma, sigma, args.dt, T_total, args.seed
    )
    # Media acumulativa en la fase estacionaria
    x2_run = np.cumsum(x2_series) / np.arange(1, len(x2_series) + 1)

    # Plot convergencia
    fig, axes = plt.subplots(2, 1, figsize=(9, 7))

    axes[0].plot(t_stat, x2_run, 'C0', lw=1.2,
                 label=r'$\langle x^2 \rangle(t)$ media acumulativa')
    axes[0].axhline(boyer_pred, color='C3', ls='--', lw=2.0,
                    label=f'Boyer (1975): $1/(2\\omega_0) = {boyer_pred:.3f}$')
    axes[0].axhline(float(x2_run[-1]), color='C0', ls=':', lw=1.0,
                    label=f'Final: {float(x2_run[-1]):.4f}')
    axes[0].set_xlabel('$t$ (unidades adimensionales $\\hbar=m=1$)')
    axes[0].set_ylabel(r'$\langle x^2 \rangle$')
    axes[0].set_title(
        f'OA + ZPF + ALD (LL)  —  $\\omega_0={omega_0}$, '
        f'$\\tau_{{rad}}={tau_rad}$, $\\gamma={gamma:.3f}$, $\\sigma={sigma:.3f}$'
    )
    axes[0].legend(fontsize=9)

    axes[1].plot(t_stat[:3000], x2_series[:3000], 'C0', lw=0.4, alpha=0.6)
    axes[1].axhline(boyer_pred, color='C3', ls='--', lw=1.5)
    axes[1].set_xlabel('$t$')
    axes[1].set_ylabel(r'$x^2(t)$')
    axes[1].set_title('Serie temporal de $x^2(t)$ (primeros 3000 pasos de la fase estac.)')

    plt.tight_layout()
    plt.savefig('boyer_single.pdf', dpi=150)
    plt.close()
    print(f"\n  Figura → boyer_single.pdf")

    return {
        'test':       'single',
        'omega_0':    omega_0,
        'tau_rad':    tau_rad,
        'gamma':      gamma,
        'boyer_pred': boyer_pred,
        'x2_final':   float(x2_run[-1]),
        'error_pct':  abs(float(x2_run[-1]) - boyer_pred) / boyer_pred * 100,
        't_arr':      t_stat[::200].tolist(),
        'x2_running': x2_run[::200].tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: omega_sweep — ⟨x²⟩ = 1/(2ω₀)
# ─────────────────────────────────────────────────────────────────────────────

def test_omega_sweep(args):
    print("\n" + "="*60)
    print("TEST 2 — Barrido ω₀: verifica ⟨x²⟩ = 1/(2ω₀) [Boyer]")
    print("="*60)

    omega_vals = [0.5, 1.0, 1.5, 2.0, 3.0]
    results    = []

    for omega_0 in omega_vals:
        res = run_ensemble(
            omega_0=omega_0,
            tau_rad=args.tau_rad,
            N_realiz=args.N_realiz,
            T_total=args.T_total,
            dt=args.dt,
            master_seed=args.seed + int(omega_0 * 100),
            label=f'ω₀ = {omega_0}',
        )
        results.append(res)

    # ── Plot ⟨x²⟩ vs ω₀ ──────────────────────────────────────────────────────
    omega_arr  = np.array([r['omega_0']  for r in results])
    x2_arr     = np.array([r['x2_mean'] for r in results])
    x2_sem_arr = np.array([r['x2_sem']  for r in results])
    boyer_arr  = np.array([r['boyer_pred'] for r in results])

    omega_fit = np.linspace(0.3, 3.5, 200)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel izq: ⟨x²⟩ vs ω₀
    axes[0].plot(omega_fit, 1.0 / (2.0 * omega_fit), 'C3--', lw=1.8,
                 label='Boyer: $\\hbar/(2m\\omega_0)$')
    axes[0].errorbar(omega_arr, x2_arr, yerr=x2_sem_arr,
                     fmt='oC0', ms=7, capsize=5, label='Simulación (EM + FDT)')
    axes[0].set_xlabel(r'$\omega_0$')
    axes[0].set_ylabel(r'$\langle x^2 \rangle_\mathrm{stat}$')
    axes[0].set_title(r'$\langle x^2 \rangle = \hbar/(2m\omega_0)$ — Boyer (1975)')
    axes[0].legend()

    # Panel der: error relativo
    err_arr = np.abs(x2_arr - boyer_arr) / boyer_arr * 100
    axes[1].bar(omega_arr, err_arr, width=0.2, color='C0', alpha=0.7)
    axes[1].axhline(5, color='C3', ls='--', lw=1, label='5% umbral')
    axes[1].set_xlabel(r'$\omega_0$')
    axes[1].set_ylabel('Error relativo (%)')
    axes[1].set_title('Error relativo vs. predicción Boyer')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('boyer_omega_sweep.pdf', dpi=150)
    plt.close()
    print(f"\n  Figura → boyer_omega_sweep.pdf")

    return {'test': 'omega_sweep', 'results': results}


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: tau_sweep — universalidad de ⟨x²⟩ respecto de τ_rad
# ─────────────────────────────────────────────────────────────────────────────

def test_tau_sweep(args):
    print("\n" + "="*60)
    print("TEST 3 — Barrido τ_rad: ⟨x²⟩ es universal (independiente de τ_rad)")
    print("="*60)
    print("  Esto verifica D = ℏ/(2m) — el coef. de Nelson no depende de τ_rad.")

    tau_vals = [0.005, 0.01, 0.02, 0.05, 0.10]
    omega_0  = args.omega0
    results  = []

    for tau_rad in tau_vals:
        if tau_rad * omega_0 > 0.15:
            print(f"  AVISO: τ_rad·ω₀ = {tau_rad*omega_0:.2f} — LL aprox. marginal.")
        res = run_ensemble(
            omega_0=omega_0,
            tau_rad=tau_rad,
            N_realiz=args.N_realiz,
            T_total=args.T_total,
            dt=args.dt,
            master_seed=args.seed + int(tau_rad * 10000),
            label=f'τ_rad = {tau_rad}',
        )
        results.append(res)

    # ── Plot ⟨x²⟩ vs τ_rad ───────────────────────────────────────────────────
    tau_arr    = np.array([r['tau_rad']  for r in results])
    x2_arr     = np.array([r['x2_mean'] for r in results])
    x2_sem_arr = np.array([r['x2_sem']  for r in results])
    boyer_pred = 1.0 / (2.0 * omega_0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(boyer_pred, color='C3', ls='--', lw=2.0,
               label=f'Boyer: $1/(2\\omega_0) = {boyer_pred:.3f}$')
    ax.fill_between([tau_arr[0]*0.8, tau_arr[-1]*1.2],
                    boyer_pred * 0.95, boyer_pred * 1.05,
                    color='C3', alpha=0.1, label='±5% banda')
    ax.errorbar(tau_arr, x2_arr, yerr=x2_sem_arr,
                fmt='oC0', ms=8, capsize=5, label='Simulación (EM + FDT)')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\tau_\mathrm{rad}$   ($\tau_\mathrm{rad} \cdot \omega_0 \ll 1$ para LL)')
    ax.set_ylabel(r'$\langle x^2 \rangle_\mathrm{stat}$')
    ax.set_title(f'Universalidad: $D_{{Nelson}} = \\hbar/(2m)$ no depende de $\\tau_{{rad}}$  '
                 f'($\\omega_0={omega_0}$)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('boyer_tau_sweep.pdf', dpi=150)
    plt.close()
    print(f"\n  Figura → boyer_tau_sweep.pdf")

    return {'test': 'tau_sweep', 'omega_0': omega_0, 'results': results}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("="*60)
    print("Abraham-Lorentz clásico — Benchmark Boyer (1975)")
    print(f"Test: {args.test}  |  ω₀={args.omega0}  |  τ_rad={args.tau_rad}")
    print(f"N_realiz={args.N_realiz}  |  dt={args.dt}  |  seed={args.seed}")
    print("Método: Euler-Maruyama + FDT cuántico (T=0)")
    print("        [límite de ∞ modos ZPF — equivalente al ZPF continuo]")
    print("="*60)

    # Verificación del régimen LL
    if args.tau_rad * args.omega0 > 0.1:
        print(f"ADVERTENCIA: τ_rad·ω₀ = {args.tau_rad*args.omega0:.3f} > 0.1")
        print("  La aproximación Landau-Lifshitz pierde validez.")

    all_results = {}

    if args.test in ('single', 'all'):
        all_results['single'] = test_single(args)

    if args.test in ('omega_sweep', 'all'):
        all_results['omega_sweep'] = test_omega_sweep(args)

    if args.test in ('tau_sweep', 'all'):
        all_results['tau_sweep'] = test_tau_sweep(args)

    # ── Resumen final ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("RESUMEN")
    print("="*60)
    boyer_pred = 1.0 / (2.0 * args.omega0)
    print(f"  Predicción Boyer: ⟨x²⟩ = {boyer_pred:.4f}  [ℏ=m=1, ω₀={args.omega0}]")

    if 'single' in all_results:
        r = all_results['single']
        print(f"  single:       ⟨x²⟩ = {r['x2_final']:.4f}  "
              f"(error {r['error_pct']:.1f}%)")

    if 'omega_sweep' in all_results:
        print("  omega_sweep:")
        for r in all_results['omega_sweep']['results']:
            print(f"    ω₀={r['omega_0']:.1f}: ⟨x²⟩={r['x2_mean']:.4f}±{r['x2_sem']:.4f}  "
                  f"pred={r['boyer_pred']:.4f}  err={r['error_pct']:.1f}%  z={r['z_score']:.1f}σ")

    if 'tau_sweep' in all_results:
        print(f"  tau_sweep (ω₀={all_results['tau_sweep']['omega_0']}):")
        for r in all_results['tau_sweep']['results']:
            print(f"    τ_rad={r['tau_rad']:.3f}: ⟨x²⟩={r['x2_mean']:.4f}±{r['x2_sem']:.4f}  "
                  f"err={r['error_pct']:.1f}%")

    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Resultados → {args.output}")
    print("="*60)


if __name__ == '__main__':
    main()
