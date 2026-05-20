"""
bohm_zpf_AL_box.py  —  Hybrid BdB/ZPF simulation with Abraham-Lorentz-Dirac (ALD)
                        radiation reaction in a 2D closed square box [0,π]².

Paper-2 Fase 3: extends bohm_zpf_box.py with three ALD corrections.

═══════════════════════════════════════════════════════════════════════════════
PHYSICAL ADDITIONS (relative to bohm_zpf_box.py)
═══════════════════════════════════════════════════════════════════════════════

1. Landau-Lifshitz (LL) noise correction
   ─────────────────────────────────────
   The LL approximation to the ALD equation replaces q̈̈ with d(F_ext/m)/dt,
   giving an effective ZPF force:
       f_ZPF^eff = f_ZPF + τ_rad · ∂f_ZPF/∂t
   where ∂Ax/∂t = +Σ A_k·ω_k·sin(k·q + φ_k - ω_k·t)
         ∂Ay/∂t = -Σ A_k·ω_k·cos(k·q + φ_k - ω_k·t)
   This modifies the effective ZPF power spectrum and, via FDT, the
   diffusion coefficient.  [Fase 1 derivation: sympy_AL_derivation.py]

2. Corrected osmotic diffusion coefficient
   ────────────────────────────────────────
   Fokker-Planck analysis gives (per 2D component):
       D_ALD = λ² · Σ_k A_k² · (τ_c · Δt) / 4
   The original run_box() used A_k[0] (one mode only), underestimating D
   by a factor ~25.  run_box_AL() uses the full sum — correcting the bug.

3. ALD osmotic velocity (always enabled)
   ────────────────────────────────────
       v_osm = D_ALD · ∇ ln|ψ|² = 2 D_ALD · Re(∇ψ / ψ)
   This is the Nelson drift that drives ρ → |ψ|² in the Fokker-Planck
   equation.  In bohm_zpf_box.py it was optional (--osmotic flag, buggy D).
   Here it is always on and uses the correct D_ALD.

═══════════════════════════════════════════════════════════════════════════════
KEY RESULTS EXPECTED
═══════════════════════════════════════════════════════════════════════════════

Stationarity test (run_stationary_AL):
  - "born_ic": start from |φ₁₁|², ZPF+ALD osmotic → H̄ stays near 0 (preserved)
  - "uniform_ic": start from uniform → H̄ decreases toward 0 (Born-rule relaxation)
  - Corrected D should make osmotic term strong enough to compete with ZPF kicks

Valentini box (run_box_AL):
  - λ=0: same τ_eff ≈ 6.8 as bohm_zpf_box (pure Bohm / Valentini 2005)
  - λ>0: expect C_fit > 0 (ZPF accelerates relaxation) with correct D

Paper-2 chain:
  Fase 1: ALD + FDT → v_osm = D∇ln|ψ|² [sympy_AL_derivation.py]
  Fase 2: Boyer (1975) benchmark <x²>_stat = 1/(2ω₀) [abraham_lorentz_classic.py]
  Fase 3: Valentini box with ALD [this file]
  Fase 4: Paper-2 manuscript [hybrid_AL_paper.tex]
"""

import sys
import numpy as np
import json
from pathlib import Path

# LAMMPS import — graceful fallback for local testing
try:
    from ctypes import c_double
    from lammps import lammps
    _LAMMPS_AVAILABLE = True
except ImportError:
    _LAMMPS_AVAILABLE = False

# ─── Box / time parameters ────────────────────────────────────────────────────
L_BOX   = np.pi
HBAR    = 1.0
MASS    = 1.0
DT      = 0.002
T_FINAL = 4 * np.pi        # ≈ 12.57 (one recurrence period, Valentini 2005)
N_STEPS = round(T_FINAL / DT)   # 6283

RECORD_EVERY = max(1, round((np.pi / 4) / DT))  # record H̄ every π/4 time units

# ─── H̄ coarse-graining grid ──────────────────────────────────────────────────
N_CG_CELLS = 16                        # cells per axis  (ε = π/16)
NX_HB      = N_CG_CELLS * 10          # = 160 grid pts
DX_HB      = L_BOX / NX_HB
EPSILON_CG = np.pi / N_CG_CELLS       # ≈ 0.196

x1d_hb = (np.arange(NX_HB) + 0.5) * DX_HB
X2D_HB, Y2D_HB = np.meshgrid(x1d_hb, x1d_hb, indexing='ij')

# ─── Valentini (2005) phases — p. 257 footnote ───────────────────────────────
VALENTINI_PHASES = np.array([
    [5.1306, 2.0056, 4.1172, 3.3871],
    [6.2013, 4.6598, 1.8770, 4.3033],
    [4.0145, 6.1142, 5.4401, 1.9292],
    [3.4015, 6.2109, 6.0370, 5.9159],
])

# ─── Mode arrays (16 modes, m,n ∈ {1..4}) ────────────────────────────────────
_MIDX = np.arange(1, 5, dtype=float)
_NIDX = np.arange(1, 5, dtype=float)
_M, _N = np.meshgrid(_MIDX, _NIDX, indexing='ij')
E_MN = 0.5 * (_M**2 + _N**2)

_PHI_HB = (2 / np.pi) * (
    np.sin(np.einsum('m,i->mi', _MIDX, x1d_hb))[:, None, :, None]
    * np.sin(np.einsum('n,j->nj', _NIDX, x1d_hb))[None, :, None, :]
)

# ─── ZPF parameters ───────────────────────────────────────────────────────────
N_MODES = 32
TAU_C   = 5     # phase renewal interval (steps)

# ALD radiation reaction time  τ_rad = 2e²/(3mc³) in natural units.
# Validity of LL approximation requires τ_rad · ω_max ≪ 1.
# For box ω_max ~ 5 (E_max=16, v~4), τ_rad=0.01 gives τ_rad·ω_max=0.05 ≪ 1. ✓
TAU_RAD_DEFAULT = 0.01


# ─── Physics: analytic ψ and guidance velocity ───────────────────────────────

def _phase_coeffs(t, phases):
    """c_{mn}(t) = (1/4) exp(i(θ_mn − E_mn·t)).  Shape (4,4)."""
    return 0.25 * np.exp(1j * (phases - E_MN * t))


def vBM_at_particles(qx, qy, t, phases=VALENTINI_PHASES):
    """
    Bohmian guidance velocity + ψ quantities at particle positions.
    Analytic eigenmode sum — no grid interpolation.
    Returns (vx, vy, psi_q, dpsi_dx, dpsi_dy, rho_q), each (N_p,).
    """
    c = _phase_coeffs(t, phases)

    sin_mx = np.sin(np.outer(_MIDX, qx))
    sin_ny = np.sin(np.outer(_NIDX, qy))
    cos_mx = np.cos(np.outer(_MIDX, qx))
    cos_ny = np.cos(np.outer(_NIDX, qy))

    f = 2.0 / np.pi
    psi_q   = f * np.einsum('mn,mp,np->p', c, sin_mx, sin_ny)
    dpsi_dx = f * np.einsum('mn,m,mp,np->p', c, _MIDX, cos_mx, sin_ny)
    dpsi_dy = f * np.einsum('mn,n,mp,np->p', c, _NIDX, sin_mx, cos_ny)

    rho_q = np.maximum(np.abs(psi_q)**2, 1e-20)
    vx = np.real(np.imag(np.conj(psi_q) * dpsi_dx) / rho_q)
    vy = np.real(np.imag(np.conj(psi_q) * dpsi_dy) / rho_q)
    return vx, vy, psi_q, dpsi_dx, dpsi_dy, rho_q


def osmotic_velocity(psi_q, dpsi_dx, dpsi_dy, rho_q, D):
    """
    ALD osmotic drift v_osm = D · ∇ ln|ψ|² = 2D · Re(∇ψ / ψ).
    Drives ρ → |ψ|² in the Fokker-Planck equation.
    Returns (vox, voy), shape (N_p,).
    """
    inv_psi = np.conj(psi_q) / rho_q
    vox = 2.0 * D * np.real(inv_psi * dpsi_dx)
    voy = 2.0 * D * np.real(inv_psi * dpsi_dy)
    return vox, voy


def psi_on_hbar_grid(t, phases=VALENTINI_PHASES):
    """ψ(x,y,t) on the H̄ grid.  Returns (NX_HB, NX_HB) complex array."""
    c = _phase_coeffs(t, phases)
    return np.einsum('mn,mnij->ij', c, _PHI_HB)


def compute_hbar_box(qx, qy, psi_grid, epsilon=EPSILON_CG):
    """
    Valentini H̄ = Σ_cells ε² ρ̄_cell ln(ρ̄_cell / |ψ̄|²_cell).
    Coarse-grained over n_cg×n_cg cells on [0,π]².
    """
    n_cg  = max(4, round(np.pi / epsilon))
    eps_a = np.pi / n_cg
    bin_w = NX_HB // n_cg

    edges = np.linspace(0, np.pi, n_cg + 1)
    rho_h, _, _ = np.histogram2d(qx, qy, bins=[edges, edges])
    rho_h = rho_h / (len(qx) * eps_a**2)

    trunc   = n_cg * bin_w
    psi2_cg = (np.abs(psi_grid[:trunc, :trunc])**2
               ).reshape(n_cg, bin_w, n_cg, bin_w).mean(axis=(1, 3))
    psi2_cg /= psi2_cg.sum() * eps_a**2

    mask  = (rho_h > 1e-15) & (psi2_cg > 1e-15)
    ratio = np.where(mask, rho_h / psi2_cg, 1.0)
    hbar  = float(np.sum(np.where(mask, rho_h * np.log(ratio) * eps_a**2, 0.0)))
    return max(hbar, 0.0)


def sample_phi11_squared(n, rng):
    """
    Rejection sampling from ρ₀(x,y) = (4/π²) sin²(x) sin²(y) on [0,π]².
    Ground-state distribution, genuinely far from 16-mode |ψ|².
    Acceptance rate ≈ 61%.
    """
    qx = np.empty(n)
    qy = np.empty(n)
    filled = 0
    while filled < n:
        batch = int((n - filled) * 2.5) + 10
        xs = rng.uniform(0, np.pi, batch)
        ys = rng.uniform(0, np.pi, batch)
        accept = rng.uniform(0.0, 1.0, batch) < (np.sin(xs) * np.sin(ys))**2
        take = min(int(accept.sum()), n - filled)
        qx[filled:filled + take] = xs[accept][:take]
        qy[filled:filled + take] = ys[accept][:take]
        filled += take
    return qx, qy


def reflect_in_box(qx, qy):
    """Hard-wall reflection at x,y ∈ [0,π]."""
    qx = np.where(qx < 0,        -qx,              qx)
    qx = np.where(qx > np.pi, 2*np.pi - qx,  qx)
    qy = np.where(qy < 0,        -qy,              qy)
    qy = np.where(qy > np.pi, 2*np.pi - qy,  qy)
    return qx, qy


# ─── ZPFField with ALD extension ──────────────────────────────────────────────

class ZPFField:
    """
    Stochastic ZPF with Landau-Lifshitz time-derivative extension.

    Standard field (same as bohm_zpf_box.py):
        Ax = Σ_k A_k · cos(kx·qx + ky·qy + phases_t)
        Ay = Σ_k A_k · sin(kx·qx + ky·qy + phases_t)

    ALD time derivative (new — for LL correction term):
        ∂Ax/∂t = +Σ_k A_k·ω_k · sin(kx·qx + ky·qy + phases_t)
        ∂Ay/∂t = −Σ_k A_k·ω_k · cos(kx·qx + ky·qy + phases_t)

    These follow from ∂/∂t[cos(k·q + φ − ω·t)] = +ω·sin(k·q + φ − ω·t)
    and similarly for sin.  phases_t = φ_k − ω_k·t_renewal is precomputed
    once per TAU_C block (same piecewise-constant approximation as field_at).
    """

    def __init__(self, n_modes=N_MODES, seed=None, omega_min=0.5, omega_max=15.0):
        rng = np.random.default_rng(seed)
        theta      = rng.uniform(0, 2*np.pi, n_modes)
        k_mag      = rng.uniform(omega_min, omega_max, n_modes)
        self.kx    = k_mag * np.cos(theta)
        self.ky    = k_mag * np.sin(theta)
        self.omega = k_mag
        self.A_k   = self.omega / np.sqrt(n_modes)
        self.rng   = rng
        self._omega_min = omega_min
        self._omega_max = omega_max
        self.phases   = rng.uniform(0, 2*np.pi, n_modes)
        self.phases_t = self.phases.copy()

    def renew_phases(self, t=0.0):
        self.phases   = self.rng.uniform(0, 2*np.pi, len(self.omega))
        self.phases_t = self.phases - self.omega * t

    def field_at(self, qx, qy):
        """ZPF field at particle positions: (Ax, Ay), shape (N_p,)."""
        phi = (self.kx[:, None] * qx
               + self.ky[:, None] * qy
               + self.phases_t[:, None])
        Ax = np.dot(self.A_k, np.cos(phi))
        Ay = np.dot(self.A_k, np.sin(phi))
        return Ax, Ay

    def field_dot_at(self, qx, qy):
        """
        ∂f_ZPF/∂t at particle positions (LL correction term).
        Returns (Ax_dot, Ay_dot), shape (N_p,).
        """
        phi = (self.kx[:, None] * qx
               + self.ky[:, None] * qy
               + self.phases_t[:, None])
        Ax_dot =  np.dot(self.A_k * self.omega, np.sin(phi))
        Ay_dot = -np.dot(self.A_k * self.omega, np.cos(phi))
        return Ax_dot, Ay_dot

    def sum_Ak2(self):
        """Σ_k A_k² = Σ_k ω_k²/N (used for D_ALD computation)."""
        return float(np.sum(self.A_k**2))


# ─── LAMMPS helpers ───────────────────────────────────────────────────────────

def lmp_scatter_positions(lmp, qx, qy):
    n = len(qx)
    flat = np.zeros(n * 3, dtype=np.float64)
    flat[0::3] = qx
    flat[1::3] = qy
    lmp.scatter("x", 2, 3, (c_double * (n * 3))(*flat))


def lmp_zero_velocities(lmp, n):
    lmp.scatter("v", 2, 3, (c_double * (n * 3))(*np.zeros(n * 3)))


# ─── ALD diffusion coefficient ────────────────────────────────────────────────

def compute_D_ALD(zpf, lam):
    """
    Correct Fokker-Planck diffusion coefficient (per 2D component).

    D_ALD = λ² · Σ_k A_k² · (τ_c · Δt) / 4

    Derivation: E[Δx²] = (λ·Δt)² · E[Ax²] per step,
                E[Ax²] = 0.5 · Σ_k A_k²  (cos phases average),
                after τ_c steps: D = E[Δx²] · τ_c / (2·Δt) = lam²·ΣAk²·τ_c·Δt/4.

    Fixes the A_k[0] bug in bohm_zpf_box.run_box(), which underestimated D by ~25×.
    """
    return lam**2 * zpf.sum_Ak2() * (TAU_C * DT) / 4.0


# ─── Single-λ ALD simulation ──────────────────────────────────────────────────

def run_box_AL(lam, tau_rad=TAU_RAD_DEFAULT,
               n_particles=2000, n_realizations=10, seed=42,
               output_dir="results_AL_box", epsilon=EPSILON_CG,
               phases=None, randomize_phases=False,
               omega_max=3.0, use_lammps=True, d_override=None):
    """
    Hybrid BdB/ZPF/ALD simulation in [0,π]² for one coupling constant λ.

    Differences from run_box() in bohm_zpf_box.py:
      • D_ALD uses full Σ A_k² (not A_k[0]): factor ~25 correction
      • Osmotic velocity always enabled (not optional)
      • LL noise correction: ZPF velocity += lam·τ_rad·∂f/∂t
      • --no-lammps flag: runs without LAMMPS (local testing)

    Parameters
    ----------
    lam : float
        ZPF coupling constant (λ=0 → pure Bohm/Valentini).
    tau_rad : float
        ALD radiation reaction time τ_rad.  Default 0.01.
        Validity: τ_rad · ω_max ≪ 1.
    omega_max : float
        ZPF spectral cutoff.  Default 3.0 (matches box energy scale E_max~8).
    use_lammps : bool
        If False, skip LAMMPS initialization (local testing without HPC).
    d_override : float or None
        If given: use Nelson direct mode — Itô noise with D = d_override
        and osmotic drift v_osm = D·∇ln|ψ|² instead of ZPF kicks.
        Use d_override=0.5 to test Nelson D = ℏ/2m (ℏ=m=1).
        This separates mechanism verification from ZPF derivation.
    """
    if use_lammps and not _LAMMPS_AVAILABLE:
        print("WARNING: LAMMPS not available, falling back to --no-lammps mode.",
              flush=True)
        use_lammps = False

    Path(output_dir).mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    if phases is None:
        phases = VALENTINI_PHASES

    hbar_runs = []
    t_record  = None

    for r in range(n_realizations):
        print(f"  λ={lam:.4f}  τ_rad={tau_rad:.4f}  r={r+1}/{n_realizations}",
              flush=True)

        run_phases = phases
        if randomize_phases:
            run_phases = rng.uniform(0, 2*np.pi, (4, 4))

        zpf = ZPFField(n_modes=N_MODES, seed=int(rng.integers(1e9)),
                       omega_max=omega_max)

        # Diffusion coefficient
        if d_override is not None:
            D_ald = d_override
            print(f"    D_Nelson = {D_ald:.4f}  (override; Nelson D=ℏ/2m)", flush=True)
        else:
            D_ald = compute_D_ALD(zpf, lam) if lam > 0 else 0.0
            print(f"    D_ALD = {D_ald:.6f}", flush=True)

        # LAMMPS instance (position storage for Clementina HPC)
        lmp = None
        if use_lammps:
            lmp = lammps(cmdargs=["-screen", "none", "-log", "none"])
            lmp.command(f"variable Np equal {n_particles}")
            lmp.command(f"variable seed equal {int(rng.integers(1e9))}")
            lmp.file("in.bohm_box")

        # Non-equilibrium IC: ρ₀ = |φ₁₁|² (far from 16-mode |ψ|²)
        qx, qy = sample_phi11_squared(n_particles, rng)
        if use_lammps:
            lmp_scatter_positions(lmp, qx, qy)
            lmp_zero_velocities(lmp, n_particles)

        hbar_t = []
        t_rec  = []
        t = 0.0

        for step in range(N_STEPS):

            # Record H̄ every RECORD_EVERY steps
            if step % RECORD_EVERY == 0:
                psi_g = psi_on_hbar_grid(t, run_phases)
                hbar_t.append(compute_hbar_box(qx, qy, psi_g, epsilon))
                t_rec.append(t)

            # Bohmian guidance velocity + ψ quantities (analytic, no grid)
            vx, vy, psi_q, dpsi_dx, dpsi_dy, rho_q = vBM_at_particles(
                qx, qy, t, run_phases)

            if D_ald > 0 or lam > 0:
                if d_override is not None:
                    # ── Nelson direct: Itô noise + osmotic drift ─────────────────
                    # dX = (v_BM + v_osm) dt + √(2D) dW
                    vox, voy = osmotic_velocity(
                        psi_q, dpsi_dx, dpsi_dy, rho_q, D_ald)
                    vx += vox
                    vy += voy
                    noise_std = np.sqrt(2.0 * D_ald * DT)
                    vx += noise_std / DT * rng.standard_normal(n_particles)
                    vy += noise_std / DT * rng.standard_normal(n_particles)
                else:
                    # ── ZPF/ALD: LL-corrected kicks + osmotic ────────────────────
                    if lam > 0:
                        if step % TAU_C == 0:
                            zpf.renew_phases(t)
                        Ax, Ay = zpf.field_at(qx, qy)
                        Ax_dot, Ay_dot = zpf.field_dot_at(qx, qy)
                        vx += lam * (Ax + tau_rad * Ax_dot)
                        vy += lam * (Ay + tau_rad * Ay_dot)
                    if D_ald > 0:
                        vox, voy = osmotic_velocity(
                            psi_q, dpsi_dx, dpsi_dy, rho_q, D_ald)
                        vx += vox
                        vy += voy

            # First-order Euler + hard-wall reflection
            qx, qy = reflect_in_box(qx + DT * vx, qy + DT * vy)
            if use_lammps:
                lmp_scatter_positions(lmp, qx, qy)
            t += DT

        # Final H̄ after last step
        psi_g = psi_on_hbar_grid(t, run_phases)
        hbar_t.append(compute_hbar_box(qx, qy, psi_g, epsilon))
        t_rec.append(t)

        hbar_runs.append(hbar_t)
        if t_record is None:
            t_record = t_rec

        if use_lammps:
            lmp.close()

    # ── Aggregate ─────────────────────────────────────────────────────────────
    hbar_arr  = np.array(hbar_runs)
    hbar_mean = hbar_arr.mean(axis=0)
    hbar_std  = hbar_arr.std(axis=0)
    t_arr     = np.array(t_record)

    # τ_eff via log-linear fit: ln H̄(t) = ln H̄₀ − t/τ_eff
    h0 = hbar_mean[0] if hbar_mean[0] > 0 else 1.0
    valid   = (hbar_mean > 0.01 * h0) & (hbar_mean > 1e-6)
    tau_eff = float('inf')
    R2      = 0.0
    if valid.sum() > 3:
        t_fit  = t_arr[valid]
        ln_h   = np.log(hbar_mean[valid])
        coeffs = np.polyfit(t_fit, ln_h, 1)
        if coeffs[0] < 0:
            tau_eff = -1.0 / coeffs[0]
            resid   = ln_h - np.polyval(coeffs, t_fit)
            R2 = float(1.0 - np.var(resid) / np.var(ln_h))

    inv_tau = 1.0 / tau_eff if tau_eff < 1e9 else 0.0
    print(f"    τ_eff = {tau_eff:.3f}   1/τ = {inv_tau:.4f}   R² = {R2:.3f}",
          flush=True)

    results = {
        "lambda":           lam,
        "tau_rad":          tau_rad,
        "D_ALD":            D_ald,
        "epsilon_cg":       epsilon,
        "n_particles":      n_particles,
        "n_realizations":   n_realizations,
        "omega_max":        omega_max,
        "t_final":          float(T_FINAL),
        "times":            t_arr.tolist(),
        "hbar_mean":        hbar_mean.tolist(),
        "hbar_std":         hbar_std.tolist(),
        "hbar_all":         hbar_arr.tolist(),
        "tau_eff":          tau_eff,
        "inv_tau_eff":      inv_tau,
        "R2_fit":           R2,
    }

    tag = f"_trad{tau_rad:.4f}_om{omega_max:.0f}"
    out = Path(output_dir) / f"AL_lam{lam:.4f}{tag}.json"
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"    saved → {out}", flush=True)
    return results


# ─── Stationary eigenstate test (ALD version) ─────────────────────────────────
#
# Tests the ALD Fokker-Planck dynamics on the simplest case: ψ = φ₁₁ (ground
# state eigenstate, Bohmian velocity = 0 identically).
#
# Two initial conditions:
#   "born_ic"   : ρ₀ = |φ₁₁|²  → particles ALREADY at Born rule.
#                  ALD should PRESERVE H̄ ≈ 0 (stationarity check).
#   "uniform_ic": ρ₀ = uniform  → far from Born rule.
#                  ALD should drive H̄ → 0 (relaxation check).
#
# Expected: with correct D_ALD, osmotic drift balances ZPF diffusion exactly
# when ρ = |ψ|², so born_ic should be stable and uniform_ic should relax.

_PSI11_GRID = (2.0/np.pi) * np.sin(X2D_HB) * np.sin(Y2D_HB)


def _psi11_quantities(qx, qy):
    """ψ₁₁, ∂_xψ₁₁, ∂_yψ₁₁, |ψ₁₁|² at particle positions."""
    f    = 2.0 / np.pi
    sx   = np.sin(qx);  sy = np.sin(qy)
    cx   = np.cos(qx);  cy = np.cos(qy)
    psi  = f * sx * sy
    dpx  = f * cx * sy
    dpy  = f * sx * cy
    rho  = np.maximum(psi**2, 1e-20)
    return psi, dpx, dpy, rho


def run_stationary_AL(lam, tau_rad=TAU_RAD_DEFAULT,
                      ic="uniform_ic",
                      n_particles=1000, n_realizations=20,
                      seed=42, output_dir="results_AL_stat",
                      omega_max=3.0, t_final=8*np.pi,
                      use_lammps=False,
                      d_override=None):
    """
    ALD Born-rule relaxation test on the stationary ground eigenstate φ₁₁.

    Parameters
    ----------
    ic : str
        "born_ic"   — start from |φ₁₁|², check H̄ stays near 0 (stationarity)
        "uniform_ic"— start from uniform on [0,π]², check H̄ → 0 (relaxation)
    tau_rad : float
        ALD radiation reaction time (LL approximation).
    use_lammps : bool
        Default False for local testing; set True on Clementina.
    d_override : float or None
        If given, use this value as D instead of D_ALD from ZPF.
        Set d_override=0.5 to test with Nelson's universal D = ℏ/2m (ℏ=m=1).
        This separates mechanism verification from microscopic derivation.
    """
    if use_lammps and not _LAMMPS_AVAILABLE:
        use_lammps = False

    Path(output_dir).mkdir(exist_ok=True)
    rng     = np.random.default_rng(seed)
    n_steps = round(t_final / DT)
    rec_ev  = max(1, round((np.pi / 4) / DT))

    hbar_runs = []
    t_record  = None

    for r in range(n_realizations):
        print(f"  {ic}  λ={lam:.3f}  τ_rad={tau_rad:.4f}  r={r+1}/{n_realizations}",
              flush=True)

        zpf = ZPFField(n_modes=N_MODES, seed=int(rng.integers(1e9)),
                       omega_max=omega_max)

        if d_override is not None:
            D_ald = d_override
        else:
            D_ald = compute_D_ALD(zpf, lam) if lam > 0 else 0.0

        # Initial condition
        if ic == "born_ic":
            qx, qy = sample_phi11_squared(n_particles, rng)
        else:  # "uniform_ic"
            qx = rng.uniform(0, np.pi, n_particles)
            qy = rng.uniform(0, np.pi, n_particles)

        hbar_t = []
        t_rec  = []
        t = 0.0

        for step in range(n_steps):
            if step % rec_ev == 0:
                hbar_t.append(compute_hbar_box(qx, qy, _PSI11_GRID))
                t_rec.append(t)

            vx = np.zeros(n_particles)
            vy = np.zeros(n_particles)

            if D_ald > 0:
                if d_override is not None:
                    # ── Nelson stochastic mechanics (Itô, T=0) ──────────────────
                    # dX = v_osm dt + √(2D) dW  (no ZPF kicks; D and drift matched)
                    psi_q, dpx, dpy, rho_q = _psi11_quantities(qx, qy)
                    vox, voy = osmotic_velocity(psi_q, dpx, dpy, rho_q, D_ald)
                    v_clip = 50.0
                    vx += np.clip(vox, -v_clip, v_clip)
                    vy += np.clip(voy, -v_clip, v_clip)
                    # Itô noise: variance 2D per unit time → std = √(2D·DT) per step
                    noise_std = np.sqrt(2.0 * D_ald * DT)
                    vx += noise_std / DT * rng.standard_normal(n_particles)
                    vy += noise_std / DT * rng.standard_normal(n_particles)
                else:
                    # ── ZPF/ALD mode: LL-corrected kicks + osmotic ───────────────
                    if step % TAU_C == 0:
                        zpf.renew_phases(t)
                    Ax, Ay = zpf.field_at(qx, qy)
                    Ax_dot, Ay_dot = zpf.field_dot_at(qx, qy)
                    vx += lam * (Ax + tau_rad * Ax_dot)
                    vy += lam * (Ay + tau_rad * Ay_dot)
                    psi_q, dpx, dpy, rho_q = _psi11_quantities(qx, qy)
                    vox, voy = osmotic_velocity(psi_q, dpx, dpy, rho_q, D_ald)
                    v_clip = 50.0
                    vx += np.clip(vox, -v_clip, v_clip)
                    vy += np.clip(voy, -v_clip, v_clip)

            qx, qy = reflect_in_box(qx + DT * vx, qy + DT * vy)
            t += DT

        hbar_t.append(compute_hbar_box(qx, qy, _PSI11_GRID))
        t_rec.append(t)
        hbar_runs.append(hbar_t)
        if t_record is None:
            t_record = t_rec

    hbar_arr  = np.array(hbar_runs)
    hbar_mean = hbar_arr.mean(axis=0)
    hbar_std  = hbar_arr.std(axis=0)
    t_arr     = np.array(t_record)

    # Fit τ_eff (only if H̄ is actually decreasing)
    tau_eff = float('inf')
    R2      = 0.0
    if hbar_mean[-1] < hbar_mean[0] and hbar_mean[0] > 0:
        from scipy.stats import linregress as _lr
        h0 = hbar_mean[0]
        valid = (hbar_mean > 0.01 * h0) & (hbar_mean > 1e-6)
        if valid.sum() > 3:
            slope, _, _, _, _ = _lr(t_arr[valid], np.log(hbar_mean[valid]))
            if slope < 0:
                tau_eff = -1.0 / slope
                resid   = np.log(hbar_mean[valid]) - (
                    slope * t_arr[valid] + np.log(h0))
                R2 = float(1 - np.var(resid) / np.var(np.log(hbar_mean[valid])))

    h_init = float(hbar_mean[0])
    h_fin  = float(hbar_mean[-1])
    print(f"    H̄_init={h_init:.4f}  H̄_final={h_fin:.4f}  "
          f"τ_eff={tau_eff:.3f}  R²={R2:.3f}", flush=True)

    result = {
        "ic": ic, "lambda": lam, "tau_rad": tau_rad,
        "D_ALD": D_ald, "D_source": "override" if d_override else "ZPF",
        "omega_max": omega_max, "n_particles": n_particles,
        "n_realizations": n_realizations,
        "t_final": float(t_final),
        "H_init": h_init, "H_final": h_fin,
        "times":      t_arr.tolist(),
        "hbar_mean":  hbar_mean.tolist(),
        "hbar_std":   hbar_std.tolist(),
        "hbar_all":   hbar_arr.tolist(),
        "tau_eff": tau_eff, "R2_fit": R2,
    }
    out = (Path(output_dir)
           / f"stat_{ic}_lam{lam:.3f}_trad{tau_rad:.4f}.json")
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"    saved → {out}", flush=True)
    return result


# ─── Comparison: run_box_AL vs run_box (no ALD) for same λ ───────────────────

def run_comparison(lam, tau_rad=TAU_RAD_DEFAULT, n_particles=2000,
                   n_realizations=5, seed=42, omega_max=3.0,
                   output_dir="results_AL_box", use_lammps=True):
    """
    Run both ALD and non-ALD (no osmotic, original D) for the same λ.
    Saves both results to output_dir for direct comparison.

    The non-ALD version uses: ZPF kick only (no osmotic, no LL correction)
    The ALD version uses: ZPF + LL correction + osmotic with correct D
    """
    print(f"\n=== Comparison: λ={lam:.4f} ===", flush=True)

    # ── Non-ALD (plain ZPF kicks, no osmotic) ────────────────────────────────
    if use_lammps and not _LAMMPS_AVAILABLE:
        use_lammps = False

    Path(output_dir).mkdir(exist_ok=True)
    rng_nald = np.random.default_rng(seed)

    hbar_runs_nald = []
    t_record_nald  = None

    print("  [non-ALD: ZPF only, no osmotic]", flush=True)
    for r in range(n_realizations):
        zpf = ZPFField(n_modes=N_MODES, seed=int(rng_nald.integers(1e9)),
                       omega_max=omega_max)
        lmp = None
        if use_lammps:
            lmp = lammps(cmdargs=["-screen", "none", "-log", "none"])
            lmp.command(f"variable Np equal {n_particles}")
            lmp.command(f"variable seed equal {int(rng_nald.integers(1e9))}")
            lmp.file("in.bohm_box")

        qx, qy = sample_phi11_squared(n_particles, rng_nald)
        if use_lammps:
            lmp_scatter_positions(lmp, qx, qy)
            lmp_zero_velocities(lmp, n_particles)

        hbar_t = []; t_rec = []; t = 0.0
        for step in range(N_STEPS):
            if step % RECORD_EVERY == 0:
                psi_g = psi_on_hbar_grid(t)
                hbar_t.append(compute_hbar_box(qx, qy, psi_g))
                t_rec.append(t)

            vx, vy, _, _, _, _ = vBM_at_particles(qx, qy, t)
            if lam > 0:
                if step % TAU_C == 0:
                    zpf.renew_phases(t)
                Ax, Ay = zpf.field_at(qx, qy)
                vx += lam * Ax
                vy += lam * Ay

            qx, qy = reflect_in_box(qx + DT * vx, qy + DT * vy)
            if use_lammps:
                lmp_scatter_positions(lmp, qx, qy)
            t += DT

        psi_g = psi_on_hbar_grid(t)
        hbar_t.append(compute_hbar_box(qx, qy, psi_g))
        t_rec.append(t)
        hbar_runs_nald.append(hbar_t)
        if t_record_nald is None:
            t_record_nald = t_rec
        if use_lammps:
            lmp.close()

    hbar_nald = np.array(hbar_runs_nald).mean(axis=0)
    t_arr = np.array(t_record_nald)

    out_nald = Path(output_dir) / f"nALD_lam{lam:.4f}_om{omega_max:.0f}.json"
    with open(out_nald, "w") as fh:
        json.dump({
            "label": "non-ALD (ZPF only)", "lambda": lam,
            "hbar_mean": hbar_nald.tolist(), "times": t_arr.tolist(),
            "hbar_all": np.array(hbar_runs_nald).tolist(),
        }, fh, indent=2)
    print(f"    [non-ALD] saved → {out_nald}", flush=True)

    # ── ALD version ──────────────────────────────────────────────────────────
    result_ald = run_box_AL(lam, tau_rad=tau_rad, n_particles=n_particles,
                            n_realizations=n_realizations, seed=seed,
                            output_dir=output_dir, omega_max=omega_max,
                            use_lammps=use_lammps)

    tau_nald = _fit_tau(hbar_nald, t_arr)
    tau_ald  = result_ald["tau_eff"]
    print(f"\n  τ_eff(non-ALD) = {tau_nald:.3f}")
    print(f"  τ_eff(ALD)     = {tau_ald:.3f}")
    speedup = tau_nald / tau_ald if tau_ald > 0 and tau_nald < 1e8 else float('nan')
    print(f"  Speedup factor = {speedup:.2f}×", flush=True)

    return result_ald


def _fit_tau(hbar_mean, t_arr):
    """Quick τ_eff fit from H̄ curve (log-linear)."""
    h0 = hbar_mean[0] if hbar_mean[0] > 0 else 1.0
    valid = (hbar_mean > 0.01 * h0) & (hbar_mean > 1e-6)
    if valid.sum() < 4:
        return float('inf')
    coeffs = np.polyfit(t_arr[valid], np.log(hbar_mean[valid]), 1)
    return float(-1.0 / coeffs[0]) if coeffs[0] < 0 else float('inf')


# ─── λ sweep ─────────────────────────────────────────────────────────────────

LAMBDA_SWEEP_AL = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]


def run_box_AL_sweep(tau_rad=TAU_RAD_DEFAULT, n_particles=2000,
                     n_realizations=10, output_dir="results_AL_box",
                     lambda_list=None, omega_max=3.0, use_lammps=True,
                     d_override=None):
    """λ sweep with ALD physics."""
    lams = lambda_list if lambda_list is not None else LAMBDA_SWEEP_AL
    all_results = []

    for lam in lams:
        print(f"\n=== λ = {lam}  τ_rad = {tau_rad} ===", flush=True)
        r = run_box_AL(lam, tau_rad=tau_rad, n_particles=n_particles,
                       n_realizations=n_realizations, output_dir=output_dir,
                       omega_max=omega_max, use_lammps=use_lammps,
                       d_override=d_override)
        all_results.append(r)

    # Summary table
    r0 = all_results[0]
    print(f"\n{'λ':>8}  {'D_ALD':>10}  {'τ_eff':>8}  {'1/τ':>8}  "
          f"{'Δ(1/τ)':>10}  {'R²':>6}")
    for r in all_results:
        d_inv = r["inv_tau_eff"] - r0["inv_tau_eff"]
        print(f"{r['lambda']:>8.4f}  {r['D_ALD']:>10.6f}  {r['tau_eff']:>8.3f}  "
              f"{r['inv_tau_eff']:>8.4f}  {d_inv:>+10.4f}  {r['R2_fit']:>6.3f}")

    # Fit Γ(λ) = C·λ² for λ ≤ 0.05
    lambdas = np.array([r["lambda"] for r in all_results if r["lambda"] <= 0.05])
    inv_taus = np.array([r["inv_tau_eff"] - r0["inv_tau_eff"]
                         for r in all_results if r["lambda"] <= 0.05])
    if len(lambdas) > 2 and (lambdas > 0).sum() > 1:
        mask = lambdas > 0
        C_fit = float(np.polyfit(lambdas[mask]**2, inv_taus[mask], 1)[0])
        print(f"\n  Γ(λ) = C·λ²  fit (λ≤0.05):  C = {C_fit:.2f}")
        print(f"  (C > 0 ↔ ZPF accelerates relaxation; C < 0 ↔ ALD disrupts mixing)")

    return all_results


# ─── Quick stationarity sweep (born_ic vs uniform_ic) ────────────────────────

def run_stationarity_comparison(lam=0.05, tau_rad=TAU_RAD_DEFAULT,
                                 n_particles=500, n_realizations=10,
                                 seed=42, output_dir="results_AL_stat",
                                 omega_max=3.0, t_final=8*np.pi,
                                 d_override=None):
    """
    Run born_ic and uniform_ic tests for the same λ, print comparison.
    Use this to verify:
      born_ic:   H̄_final ≈ H̄_init (stationarity preserved)
      uniform_ic: H̄_final ≪ H̄_init (Born-rule relaxation)
    """
    print(f"\n=== Stationarity comparison: λ={lam}, τ_rad={tau_rad} ===", flush=True)
    r_born = run_stationary_AL(lam, tau_rad=tau_rad, ic="born_ic",
                               n_particles=n_particles,
                               n_realizations=n_realizations, seed=seed,
                               output_dir=output_dir, omega_max=omega_max,
                               t_final=t_final, d_override=d_override)
    r_uni  = run_stationary_AL(lam, tau_rad=tau_rad, ic="uniform_ic",
                               n_particles=n_particles,
                               n_realizations=n_realizations, seed=seed,
                               output_dir=output_dir, omega_max=omega_max,
                               t_final=t_final, d_override=d_override)

    print(f"\n  born_ic:    H̄_init={r_born['H_init']:.4f}  "
          f"H̄_final={r_born['H_final']:.4f}  "
          f"(change: {r_born['H_final'] - r_born['H_init']:+.4f})")
    print(f"  uniform_ic: H̄_init={r_uni['H_init']:.4f}  "
          f"H̄_final={r_uni['H_final']:.4f}  τ={r_uni['tau_eff']:.3f}")

    return r_born, r_uni


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Hybrid BdB/ZPF/ALD in 2D closed box (Paper-2 Fase 3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  --mode sweep      : Full λ sweep with ALD (default)
  --mode single     : Single λ run (requires --lam)
  --mode stationary : Stationarity test (born_ic vs uniform_ic)
  --mode comparison : ALD vs non-ALD comparison for single λ

Examples:
  python bohm_zpf_AL_box.py --no-lammps --mode stationary --lam 0.05
  python bohm_zpf_AL_box.py --no-lammps --mode sweep --Nr 5 --Np 500
  python bohm_zpf_AL_box.py --mode single --lam 0.05 --tau-rad 0.01
        """)
    p.add_argument("--mode",      choices=["sweep","single","stationary","comparison"],
                   default="sweep")
    p.add_argument("--lam",       type=float, default=0.05)
    p.add_argument("--tau-rad",   type=float, default=TAU_RAD_DEFAULT)
    p.add_argument("--Np",        type=int,   default=2000)
    p.add_argument("--Nr",        type=int,   default=10)
    p.add_argument("--out",       type=str,   default="results_AL_box")
    p.add_argument("--epsilon",   type=float, default=EPSILON_CG)
    p.add_argument("--omega-max", type=float, default=3.0)
    p.add_argument("--no-lammps", action="store_true",
                   help="Skip LAMMPS interface (local testing)")
    p.add_argument("--d-nelson", action="store_true",
                   help="Use Nelson D = ℏ/2m = 0.5 instead of D_ALD from ZPF. "
                        "Tests Fokker-Planck mechanism independently of ZPF derivation.")
    p.add_argument("--ic",        choices=["born_ic","uniform_ic"], default="uniform_ic",
                   help="Initial condition for --mode stationary")
    p.add_argument("--t-final",   type=float, default=8*np.pi)
    args = p.parse_args()

    use_lammps = not args.no_lammps
    d_ov = 0.5 if args.d_nelson else None  # Nelson D = ℏ/2m = 0.5 (ℏ=m=1)

    if args.mode == "sweep":
        run_box_AL_sweep(tau_rad=args.tau_rad, n_particles=args.Np,
                         n_realizations=args.Nr, output_dir=args.out,
                         omega_max=args.omega_max, use_lammps=use_lammps,
                         d_override=d_ov)

    elif args.mode == "single":
        run_box_AL(args.lam, tau_rad=args.tau_rad, n_particles=args.Np,
                   n_realizations=args.Nr, output_dir=args.out,
                   epsilon=args.epsilon, omega_max=args.omega_max,
                   use_lammps=use_lammps, d_override=d_ov)

    elif args.mode == "stationary":
        run_stationary_AL(args.lam, tau_rad=args.tau_rad, ic=args.ic,
                          n_particles=args.Np, n_realizations=args.Nr,
                          output_dir=args.out.replace("box","stat"),
                          omega_max=args.omega_max, t_final=args.t_final,
                          use_lammps=False, d_override=d_ov)

    elif args.mode == "comparison":
        run_comparison(args.lam, tau_rad=args.tau_rad, n_particles=args.Np,
                       n_realizations=args.Nr, omega_max=args.omega_max,
                       output_dir=args.out, use_lammps=use_lammps)
