"""
Co-simulation Python + LAMMPS for the hybrid De Broglie–Bohm / ZPF model.

Implements the first-order guidance equation (Eq. 11 of the paper):
    dq/dt = grad(S)/m  +  lambda * (e/m²) * A_zpf(q, t)

Integration: first-order Euler in Python.
LAMMPS role: stores particle positions, enforces 2D and periodic boundaries,
             provides the infrastructure for future GPU-accelerated scaling.

Physics: Bergamin & Bringa (2026), hybridLorentz-8.pdf
"""

import numpy as np
import sys
import json
from ctypes import c_double
from pathlib import Path
from lammps import lammps

# ─── Physical / grid parameters ──────────────────────────────────────────────
NX       = 320       # grid points per axis
DX       = 0.08      # grid spacing (dimensionless)
L        = NX * DX   # = 25.6 ; box is [-12.8, 12.8]²
DT       = 0.002     # timestep
# The paper's "300 pasos" are macro-steps of dt_macro=0.03 (= 15 micro-steps).
# Total integration time = 9 dimensionless units = 4500 micro-steps at dt=0.002.
# This is required to enter the slow exponential tail where τ_V ≈ 3.4 and the
# ZPF effect (Γ(0.05) ≈ 0.1) is distinguishable from the fast initial mixing.
N_STEPS  = 4500      # total micro-steps → t_total = 9 (matches paper Fig. 2)
HBAR     = 1.0
MASS     = 1.0

# Double-slit geometry
SLIT_Y    = 0.0      # barrier at y=0
SLIT_WIDTH = 0.5     # half-width of each slit
SLIT_SEP   = 2.0     # centre-to-centre separation
BARRIER_V  = 1e4     # hard wall height

# ZPF
N_MODES = 64
TAU_C   = 5          # phase renewal every TAU_C steps
ZPF_SCALE = 1.0      # physical baseline (A_k = ω/√N, matches paper Eq. normalization)

# Valentini H-bar coarse-graining
EPSILON_CG = 0.5     # coarse-grain cell size (dimensionless)

# ─── Grid ────────────────────────────────────────────────────────────────────
x1d = (np.arange(NX) - NX // 2) * DX
X, Y = np.meshgrid(x1d, x1d, indexing='ij')

kx1d = np.fft.fftfreq(NX, d=DX) * 2 * np.pi
KX, KY = np.meshgrid(kx1d, kx1d, indexing='ij')
K2 = KX**2 + KY**2

# ─── Double-slit potential ────────────────────────────────────────────────────
def make_double_slit_potential():
    V = np.zeros((NX, NX))
    iy = np.argmin(np.abs(x1d - SLIT_Y))
    V[:, iy] = BARRIER_V
    for center in [-SLIT_SEP / 2, SLIT_SEP / 2]:
        V[np.abs(x1d - center) < SLIT_WIDTH, iy] = 0.0
    return V

V_pot = make_double_slit_potential()

# ─── Split-operator propagator ────────────────────────────────────────────────
def propagate_psi(psi, V, dt):
    exp_V = np.exp(-1j * V * dt / 2)
    exp_K = np.exp(-1j * K2 * dt / (2 * MASS))
    psi = exp_V * psi
    psi = np.fft.ifft2(exp_K * np.fft.fft2(psi))
    return exp_V * psi

# ─── Quantum potential and phase gradient ─────────────────────────────────────
def compute_guidance_velocity(psi):
    """
    Returns (vx, vy) on the grid: Bohmian guidance velocity = grad(S)/m.
    Uses Im(psi* ∇ psi) / (|psi|² * m) instead of extracting S explicitly,
    which is numerically more stable.
    """
    psi_k = np.fft.fft2(psi)
    dpsi_dx = np.fft.ifft2(1j * KX * psi_k)
    dpsi_dy = np.fft.ifft2(1j * KY * psi_k)
    rho = np.abs(psi)**2
    rho = np.where(rho < 1e-20, 1e-20, rho)
    vx = np.imag(np.conj(psi) * dpsi_dx) / (rho * MASS)
    vy = np.imag(np.conj(psi) * dpsi_dy) / (rho * MASS)
    return np.real(vx), np.real(vy)

# ─── Bilinear interpolation ───────────────────────────────────────────────────
def interpolate_grid(field, qx, qy):
    ix_f = np.clip(qx / DX + NX / 2, 0, NX - 1 - 1e-9)
    iy_f = np.clip(qy / DX + NX / 2, 0, NX - 1 - 1e-9)
    ix0  = ix_f.astype(int)
    iy0  = iy_f.astype(int)
    ix1  = np.clip(ix0 + 1, 0, NX - 1)
    iy1  = np.clip(iy0 + 1, 0, NX - 1)
    tx = ix_f - ix0;  ty = iy_f - iy0
    return (  (1-tx)*(1-ty)*field[ix0,iy0]
            + tx    *(1-ty)*field[ix1,iy0]
            + (1-tx)*ty    *field[ix0,iy1]
            + tx    *ty    *field[ix1,iy1])

# ─── ZPF field ───────────────────────────────────────────────────────────────
class ZPFField:
    """
    Stochastic ZPF: superposition of N_modes plane waves.
    Spectral amplitude A_k ∝ omega_k (2D dispersion omega = |k|, c=1).
    Phases renewed every tau_c integration steps.
    """
    def __init__(self, n_modes=N_MODES, seed=None):
        rng = np.random.default_rng(seed)
        theta   = rng.uniform(0, 2*np.pi, n_modes)
        k_mag   = rng.uniform(0.5, 15.0, n_modes)
        self.kx = k_mag * np.cos(theta)
        self.ky = k_mag * np.sin(theta)
        self.omega  = k_mag
        self.A_k    = self.omega / np.sqrt(n_modes)  # normalized amplitude
        self.rng    = rng
        self.phases = rng.uniform(0, 2*np.pi, n_modes)

    def renew_phases(self):
        self.phases = self.rng.uniform(0, 2*np.pi, len(self.omega))

    def field_at(self, qx, qy, t):
        """
        Returns (Ax, Ay) at particle positions. Shape: (N_p,) each.
        A_zpf = sum_k A_k [cos(k·x - omega t + phi), sin(k·x - omega t + phi)]
        """
        # phi_ki: shape (n_modes, N_p)
        phi = (np.outer(self.kx, qx) + np.outer(self.ky, qy)
               - np.outer(self.omega, np.full(len(qx), t))
               + self.phases[:, None])
        Ax = np.sum(self.A_k[:, None] * np.cos(phi), axis=0)
        Ay = np.sum(self.A_k[:, None] * np.sin(phi), axis=0)
        return Ax, Ay

# ─── Valentini H-bar ──────────────────────────────────────────────────────────
def compute_hbar(qx, qy, psi, epsilon=EPSILON_CG):
    """
    H̄ = ∫ dq ρ̄ ln(ρ̄ / |ψ|²_bar)    (Eq. 17 of paper)
    Coarse-grained over cells of size epsilon.
    """
    bin_w = max(1, round(epsilon / DX))   # grid points per cell
    n_cg  = NX // bin_w                   # number of cells
    trunc = n_cg * bin_w                  # truncated grid size
    eps_a = bin_w * DX                    # actual epsilon after rounding

    edges = np.linspace(x1d[0], x1d[trunc-1] + DX, n_cg + 1)

    rho_hist, _, _ = np.histogram2d(qx, qy, bins=[edges, edges])
    rho_hist = rho_hist / (len(qx) * eps_a**2)

    psi2_cg = (np.abs(psi[:trunc, :trunc])**2
               ).reshape(n_cg, bin_w, n_cg, bin_w).mean(axis=(1, 3))
    psi2_cg = psi2_cg / (psi2_cg.sum() * eps_a**2)

    mask  = (rho_hist > 1e-15) & (psi2_cg > 1e-15)
    ratio = np.where(mask, rho_hist / psi2_cg, 1.0)
    hbar  = np.sum(np.where(mask, rho_hist * np.log(ratio) * eps_a**2, 0.0))
    return max(float(hbar), 0.0)

def compute_hbar_1d(qx, psi, epsilon=EPSILON_CG):
    """
    1D x-marginal Valentini H̄: H̄_x = ∫ dx ρ̄_x ln(ρ̄_x / |ψ̄_x|²)
    Marginalizes out the y-direction, so the periodic-y wrap artifact is absent.
    This matches the observable in the double-slit experiment (distribution in x).
    """
    bin_w = max(1, round(epsilon / DX))
    n_cg  = NX // bin_w
    trunc = n_cg * bin_w
    eps_a = bin_w * DX

    edges = np.linspace(x1d[0], x1d[trunc-1] + DX, n_cg + 1)

    rho_x, _ = np.histogram(qx, bins=edges)
    rho_x = rho_x.astype(float) / (len(qx) * eps_a)

    # x-marginal of |ψ|²: sum over y, then coarse-grain
    psi2_y = (np.abs(psi[:trunc, :])**2).sum(axis=1) * DX  # shape (trunc,)
    psi2_cg = psi2_y[:trunc].reshape(n_cg, bin_w).mean(axis=1)
    psi2_cg = psi2_cg / (psi2_cg.sum() * eps_a)

    mask  = (rho_x > 1e-15) & (psi2_cg > 1e-15)
    ratio = np.where(mask, rho_x / psi2_cg, 1.0)
    hbar  = np.sum(np.where(mask, rho_x * np.log(ratio) * eps_a, 0.0))
    return max(float(hbar), 0.0)

# ─── KS distance on detection screen ────────────────────────────────────────
def ks_1d(samples, psi2_grid):
    """1D KS vs |psi|² projected onto x-axis. Returns NaN if empty."""
    if len(samples) == 0:
        return float("nan")
    born_x   = psi2_grid.sum(axis=1) * DX
    born_cdf = np.cumsum(born_x) * DX
    born_cdf /= born_cdf[-1]
    s        = np.sort(samples)
    emp_cdf  = np.arange(1, len(s)+1) / len(s)
    return float(np.max(np.abs(emp_cdf - np.interp(s, x1d, born_cdf))))

# ─── Paper's initial condition: pre-propagate ψ to slit ──────────────────────
def make_psi_at_slit():
    """
    Pre-propagate the incoming Gaussian to the slit barrier.
    With y0=-4, k0y=15: center reaches y=0 in t_arrive = 4/15 ≈ 0.267,
    i.e. N_PREPROP = round(0.267/DT) ≈ 133 micro-steps.
    Returns ψ just after the packet has passed through the slits.
    """
    N_PREPROP = round(4.0 / (15.0 * DT))  # ≈ 133
    psi = make_initial_psi()
    for _ in range(N_PREPROP):
        psi = propagate_psi(psi, V_pot, DT)
    return psi

def sample_slit_from_psi(psi_at_slit, n, rng):
    """
    Paper's unified IC: sample x from |ψ(x,y≈0)|² restricted to slit openings.
    All particles placed at y = SLIT_Y + DX (just above barrier).
    """
    iy0 = NX // 2  # index of y=0
    iy1 = min(iy0 + 2, NX - 1)  # average a couple of rows above barrier
    psi_col = 0.5 * (psi_at_slit[:, iy0] + psi_at_slit[:, iy1])
    slit_open = (V_pot[:, iy0] == 0.0).astype(float)
    prob_x = slit_open * np.abs(psi_col)**2
    total = prob_x.sum()
    if total < 1e-30:
        prob_x = np.abs(psi_col)**2
        total = prob_x.sum()
    prob_x /= total
    ix = rng.choice(NX, size=n, p=prob_x)
    qx = x1d[ix] + rng.uniform(-DX/2, DX/2, n)
    qy = np.full(n, SLIT_Y + DX)
    return qx, qy

# ─── Initial wavefunction ─────────────────────────────────────────────────────
def make_initial_psi(y0=-4.0, sigma=1.0, k0y=15.0):
    """
    Gaussian wave packet moving toward the barrier (y direction).
    With k0y=15, sigma=1, the packet reaches the barrier (~y=0) at t ≈ 0.27.
    """
    psi  = np.exp(-(X**2 + (Y - y0)**2) / (2*sigma**2) + 1j*k0y*Y)
    norm = np.sqrt((np.abs(psi)**2).sum() * DX**2)
    return psi / norm

# ─── OUT-OF-EQUILIBRIUM initial condition (paper's "condición inicial") ───────
def sample_slit_positions(n, rng):
    """
    Sample n particles uniformly over the two slit openings at y = SLIT_Y.
    This creates ρ_0 ≠ |ψ_0|² (ψ is at y=-4, particles are at the barrier).
    H-bar_0 >> 0 → system out of Born-rule equilibrium.
    H-bar then decreases as ψ travels to meet the particles and they spread.
    ZPF coupling (λ > 0) accelerates this relaxation.
    """
    half = n // 2
    rem  = n - 2 * half
    qx = np.concatenate([
        rng.uniform(-SLIT_SEP/2 - SLIT_WIDTH, -SLIT_SEP/2 + SLIT_WIDTH, half),
        rng.uniform( SLIT_SEP/2 - SLIT_WIDTH,  SLIT_SEP/2 + SLIT_WIDTH, half + rem),
    ])
    # Place just above the barrier so guidance is well-defined from the start
    qy = np.full(n, SLIT_Y + DX)
    return qx, qy

# ─── Sample from |psi|² (for reference / test purposes) ──────────────────────
def sample_positions(psi, n, rng):
    prob  = np.abs(psi)**2 * DX**2
    prob /= prob.sum()
    idx   = rng.choice(prob.size, size=n, p=prob.ravel())
    ix, iy = np.unravel_index(idx, (NX, NX))
    qx = x1d[ix] + rng.uniform(-DX/2, DX/2, n)
    qy = x1d[iy] + rng.uniform(-DX/2, DX/2, n)
    return qx, qy

# ─── LAMMPS position scatter helper ──────────────────────────────────────────
def lmp_scatter_positions(lmp, qx, qy):
    """Push (qx, qy) arrays into LAMMPS atom positions."""
    n = len(qx)
    flat = np.zeros(n * 3, dtype=np.float64)
    flat[0::3] = qx
    flat[1::3] = qy
    cdata = (c_double * (n * 3))(*flat)
    lmp.scatter("x", 2, 3, cdata)

# ─── Zero all velocities in LAMMPS (guidance = first-order, no inertia) ──────
def lmp_zero_velocities(lmp, n):
    zeros = (c_double * (n * 3))(*np.zeros(n * 3))
    lmp.scatter("v", 2, 3, zeros)

# ─── Single-lambda co-simulation ─────────────────────────────────────────────
def run_cosimulation(lam, n_particles, n_zpf_realizations=20, seed=42,
                     input_script="in.bohm_zpf", output_dir="results",
                     zpf_scale=ZPF_SCALE, epsilon=None):
    """
    Runs the hybrid BdB/ZPF simulation for one lambda value.

    Integration: first-order Euler (matches paper's Python pipeline).
    LAMMPS: stores positions, enforces boundary conditions, provides
            infrastructure for future GPU/MPI scaling.

    Returns dict with hbar_mean, hbar_std, ks_mean, ks_std.
    """
    Path(output_dir).mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    eps_cg = epsilon if epsilon is not None else EPSILON_CG

    # Pre-propagate ψ₀ to create the Born-rule reference at the detection screen.
    # Particles will be placed at the slit (sample_slit_positions) while ψ₀ starts
    # at y=-4 — the same mismatch the paper uses to produce out-of-equilibrium IC.
    psi0 = make_initial_psi()

    # Pre-compute Born-rule reference: propagate ψ₀ to the detection screen (y≈8)
    # t_screen = (y_screen - y0)/k0y = (8+4)/15 = 0.8 → 400 steps
    N_SCREEN = round((8.0 - (-4.0)) / (15.0 * DT))  # ≈ 400 steps
    psi_screen = psi0.copy()
    for _ in range(N_SCREEN):
        psi_screen = propagate_psi(psi_screen, V_pot, DT)
    iy_screen = int(np.argmin(np.abs(x1d - 8.0)))
    psi2_born_ref_2d = np.abs(psi_screen)**2
    # 1D x-marginal at detection screen: integrate y ∈ [6,10] to get Born-rule pattern
    iy_lo = int(np.argmin(np.abs(x1d - 6.0)))
    iy_hi = int(np.argmin(np.abs(x1d - 10.0)))
    psi2_born_ref = psi2_born_ref_2d[:, iy_lo:iy_hi].sum(axis=1) * DX

    hbar_runs   = []
    hbar1d_runs = []
    ks_finals   = []

    for r in range(n_zpf_realizations):
        print(f"  lambda={lam:.4f}  realization {r+1}/{n_zpf_realizations}",
              flush=True)

        zpf = ZPFField(n_modes=N_MODES, seed=int(rng.integers(1e9)))

        # ── LAMMPS setup ─────────────────────────────────────────────────────
        lmp = lammps(cmdargs=["-screen", "none", "-log", "none"])
        lmp.command(f"variable Np equal {n_particles}")
        lmp.command(f"variable seed equal {int(rng.integers(1e9))}")
        lmp.file(input_script)

        # Out-of-equilibrium IC: particles at slit openings, ψ₀ at y=-4.
        # This is the IC described in Section 10 of the paper.
        qx, qy = sample_slit_positions(n_particles, rng)
        lmp_scatter_positions(lmp, qx, qy)
        lmp_zero_velocities(lmp, n_particles)

        # ── First-order Euler guidance loop ──────────────────────────────────
        psi_t   = psi0.copy()
        hbar_t  = []
        hbar1d_t = []

        for step in range(N_STEPS):
            # Record 2D and 1D x-marginal H̄ at start of each step
            hbar_t.append(compute_hbar(qx, qy, psi_t, epsilon=eps_cg))
            hbar1d_t.append(compute_hbar_1d(qx, psi_t, epsilon=eps_cg))

            # Bohmian guidance velocity at current positions
            vBM_x_grid, vBM_y_grid = compute_guidance_velocity(psi_t)
            vBM_x = interpolate_grid(vBM_x_grid, qx, qy)
            vBM_y = interpolate_grid(vBM_y_grid, qx, qy)

            # ZPF perturbation
            if lam > 0:
                if step % TAU_C == 0:
                    zpf.renew_phases()
                Ax, Ay = zpf.field_at(qx, qy, step * DT)
                delta_vx = lam * zpf_scale * Ax
                delta_vy = lam * zpf_scale * Ay
            else:
                delta_vx = delta_vy = 0.0

            # Euler step: q(t+dt) = q(t) + dt * [vBM + lambda*A_zpf]
            qx = qx + DT * (vBM_x + delta_vx)
            qy = qy + DT * (vBM_y + delta_vy)

            # Periodic wrap to stay in box [-L/2, L/2)
            qx = (qx + L/2) % L - L/2
            qy = (qy + L/2) % L - L/2

            # Push updated positions to LAMMPS
            lmp_scatter_positions(lmp, qx, qy)

            # Propagate psi
            psi_t = propagate_psi(psi_t, V_pot, DT)

        # Final H̄ values (after last step)
        hbar_t.append(compute_hbar(qx, qy, psi_t, epsilon=eps_cg))
        hbar1d_t.append(compute_hbar_1d(qx, psi_t, epsilon=eps_cg))

        # KS vs fixed Born-rule screen reference (paper Table 2 methodology):
        # psi2_born_ref is the 1D x-marginal of |ψ|² integrated over y∈[6,10]
        born_cdf = np.cumsum(psi2_born_ref * DX)
        born_cdf = born_cdf / born_cdf[-1]
        s = np.sort(qx)
        emp_cdf = np.arange(1, len(s)+1) / len(s)
        ks = float(np.max(np.abs(emp_cdf - np.interp(s, x1d, born_cdf))))

        hbar_runs.append(hbar_t)
        hbar1d_runs.append(hbar1d_t)
        ks_finals.append(ks)
        lmp.close()

    hbar_arr   = np.array(hbar_runs)    # shape (n_realizations, N_STEPS+1)
    hbar1d_arr = np.array(hbar1d_runs)  # shape (n_realizations, N_STEPS+1)

    results = {
        "lambda":           lam,
        "zpf_scale":        zpf_scale,
        "epsilon_cg":       eps_cg,
        "n_particles":      n_particles,
        "n_realizations":   n_zpf_realizations,
        "hbar_mean":        hbar_arr.mean(axis=0).tolist(),
        "hbar_std":         hbar_arr.std(axis=0).tolist(),
        "hbar1d_mean":      hbar1d_arr.mean(axis=0).tolist(),
        "hbar1d_std":       hbar1d_arr.std(axis=0).tolist(),
        "ks_mean":          float(np.nanmean(ks_finals)),
        "ks_std":           float(np.nanstd(ks_finals)),
    }

    out = Path(output_dir) / f"results_lam{lam:.4f}.json"
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"  -> {out}   KS={results['ks_mean']:.4f}±{results['ks_std']:.4f}")

    return results

# ─── Lambda sweep (Table 2 of paper) ─────────────────────────────────────────
LAMBDA_SWEEP = [0.0, 0.0005, 0.001, 0.002, 0.003, 0.005,
                0.007, 0.010, 0.015, 0.020, 0.050]

def run_full_sweep(n_particles=1000, n_realizations=20, output_dir="results",
                   zpf_scale=ZPF_SCALE, epsilon=None):
    all_results = []
    for lam in LAMBDA_SWEEP:
        print(f"\n=== lambda = {lam} ===")
        r = run_cosimulation(lam, n_particles,
                             n_zpf_realizations=n_realizations,
                             output_dir=output_dir,
                             zpf_scale=zpf_scale, epsilon=epsilon)
        all_results.append(r)

    ks0 = all_results[0]["ks_mean"]
    print(f"\n{'lambda':>8}  {'D_KS':>8}  {'sigma':>8}  {'vs lam=0':>10}")
    for r in all_results:
        red = (r["ks_mean"] - ks0) / ks0 * 100 if ks0 > 0 else 0
        print(f"{r['lambda']:>8.4f}  {r['ks_mean']:>8.4f}  "
              f"{r['ks_std']:>8.4f}  {red:>+9.1f}%")

    return all_results

# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Hybrid BdB/ZPF LAMMPS co-simulation")
    p.add_argument("--lam",  type=float, default=None)
    p.add_argument("--Np",   type=int,   default=1000)
    p.add_argument("--Nr",   type=int,   default=20)
    p.add_argument("--out",  type=str,   default="results")
    p.add_argument("--steps",type=int,   default=None)
    p.add_argument("--zpf-scale", type=float, default=ZPF_SCALE,
                   help="multiplicative calibration factor for A_zpf amplitude")
    p.add_argument("--epsilon", type=float, default=None,
                   help="coarse-graining cell size for H̄ (overrides EPSILON_CG)")
    args = p.parse_args()

    if args.steps:
        N_STEPS = args.steps

    if args.lam is not None:
        run_cosimulation(args.lam, args.Np, n_zpf_realizations=args.Nr,
                         output_dir=args.out, zpf_scale=args.zpf_scale,
                         epsilon=args.epsilon)
    else:
        run_full_sweep(n_particles=args.Np, n_realizations=args.Nr,
                       output_dir=args.out, zpf_scale=args.zpf_scale,
                       epsilon=args.epsilon)
