"""
bohm_zpf_box.py  —  Hybrid BdB/ZPF simulation in a 2D closed square box [0,π]².

Motivation: the double-slit (hybridLorentz-8.pdf) is an open system with
t_total ≈ 0.6 units — too short to observe Born-rule relaxation. This
simulation uses a closed box (direct analog of Valentini & Westman 2005):

  - ψ = superposition of 16 energy eigenmodes (m,n ∈ {1..4}, Valentini phases)
  - ρ₀ = |φ₁₁|²  (ground-state distribution, genuinely far from |ψ|²)
  - T_final = 4π ≈ 12.57 time units (one full recurrence period)
  - Valentini H̄-function as primary metric (no KS: box has no screen)

At λ=0 this should reproduce Valentini (2005) Fig. 10: τ_eff ≈ 4.
For λ>0, ZPF coupling should accelerate relaxation: τ_eff(λ) < τ_eff(0).
The theoretical prediction is 1/τ_eff = 1/τ_V + Γ(λ), Γ ∝ λ².
"""

import numpy as np
import json
from pathlib import Path
from ctypes import c_double
from lammps import lammps

# ─── Box / time parameters ────────────────────────────────────────────────────
L_BOX   = np.pi       # box side: particle domain [0, π]
HBAR    = 1.0
MASS    = 1.0
DT      = 0.002
T_FINAL = 4 * np.pi   # ≈ 12.57 (one recurrence period, cf. Valentini 2005)
N_STEPS = round(T_FINAL / DT)  # 6283

# Record H̄ every π/4 time units (same cadence as Valentini Fig. 10)
RECORD_EVERY = max(1, round((np.pi / 4) / DT))  # ≈ 392 steps → 16 points

# ─── H̄ grid (only used for coarse-graining; NOT for guidance velocity) ───────
# NX_HB must be divisible by N_CG_CELLS for exact cell alignment.
# With N_CG_CELLS=16 → NX_HB=160 gives 10 grid pts per cell.
N_CG_CELLS = 16           # coarse-graining cells per axis (= π/ε)
NX_HB      = N_CG_CELLS * 10   # = 160; 10 grid pts per cell
DX_HB      = L_BOX / NX_HB     # = π/160 ≈ 0.01963
# Cell-centered: avoids ψ=0 nodes at walls during histogram binning
x1d_hb = (np.arange(NX_HB) + 0.5) * DX_HB   # ∈ (0, π)
X2D_HB, Y2D_HB = np.meshgrid(x1d_hb, x1d_hb, indexing='ij')

# Coarse-graining length: ε = π/16 ≈ 0.196
# (4× Valentini's π/32; appropriate for N_p ≈ 2000 instead of 160 000)
EPSILON_CG = np.pi / N_CG_CELLS

# ─── Valentini (2005) phases — footnote, p. 257 ──────────────────────────────
# θ[m-1, n-1], m,n ∈ {1,2,3,4}
VALENTINI_PHASES = np.array([
    [5.1306, 2.0056, 4.1172, 3.3871],
    [6.2013, 4.6598, 1.8770, 4.3033],
    [4.0145, 6.1142, 5.4401, 1.9292],
    [3.4015, 6.2109, 6.0370, 5.9159],
])

# ─── Precomputed mode arrays ──────────────────────────────────────────────────
_MIDX = np.arange(1, 5, dtype=float)   # m values: [1,2,3,4]
_NIDX = np.arange(1, 5, dtype=float)   # n values: [1,2,3,4]
_M, _N = np.meshgrid(_MIDX, _NIDX, indexing='ij')
E_MN = 0.5 * (_M**2 + _N**2)          # energies (4,4), ħ=m=1

# Mode functions on H̄ grid: shape (4, 4, NX_HB, NX_HB)
_PHI_HB = (2 / np.pi) * (
    np.sin(np.einsum('m,i->mi', _MIDX, x1d_hb))[:, None, :, None]  # sin(m x_i)
    * np.sin(np.einsum('n,j->nj', _NIDX, x1d_hb))[None, :, None, :]  # sin(n y_j)
)  # [m-1, n-1, i, j]

# ─── ZPF parameters ───────────────────────────────────────────────────────────
N_MODES   = 32   # 32 modes sufficient for ZPF statistical convergence (CLT)
TAU_C     = 5        # phase renewal interval (steps)
ZPF_SCALE = 1.0

# ─── Physics: analytic ψ and guidance velocity ───────────────────────────────

def _phase_coeffs(t, phases):
    """Complex coefficients c_{mn}(t) = (1/4) exp(i(θ_mn - E_mn t)). Shape (4,4)."""
    return 0.25 * np.exp(1j * (phases - E_MN * t))


def vBM_at_particles(qx, qy, t, phases=VALENTINI_PHASES):
    """
    Bohmian guidance velocity directly at particle positions (no grid interpolation).
    Analytic eigenmode sum: O(N_p × 16) vs O(NX² × 16 + N_p) grid approach.
    Returns (vx, vy), each shape (N_p,).
    """
    c = _phase_coeffs(t, phases)  # (4,4) complex

    # Trig arrays at particle positions: shape (4, N_p)
    sin_mx = np.sin(np.outer(_MIDX, qx))
    sin_ny = np.sin(np.outer(_NIDX, qy))
    cos_mx = np.cos(np.outer(_MIDX, qx))
    cos_ny = np.cos(np.outer(_NIDX, qy))

    f = 2.0 / np.pi  # normalization factor from φ_mn = (2/π) sin(mx) sin(ny)

    # ψ(q) = (2/π) Σ_{m,n} c_{mn} sin(m qx) sin(n qy)
    psi_q    = f * np.einsum('mn,mp,np->p', c, sin_mx, sin_ny)
    dpsi_dx  = f * np.einsum('mn,m,mp,np->p', c, _MIDX, cos_mx, sin_ny)
    dpsi_dy  = f * np.einsum('mn,n,mp,np->p', c, _NIDX, sin_mx, cos_ny)

    rho_q = np.maximum(np.abs(psi_q)**2, 1e-20)
    vx = np.real(np.imag(np.conj(psi_q) * dpsi_dx) / rho_q)
    vy = np.real(np.imag(np.conj(psi_q) * dpsi_dy) / rho_q)
    return vx, vy, psi_q, dpsi_dx, dpsi_dy, rho_q


def osmotic_velocity(psi_q, dpsi_dx, dpsi_dy, rho_q, D):
    """
    Osmotic drift velocity v_osm = D × ∇ln|ψ|².

    In Nelson/SED stochastic mechanics, the forward drift is
        b = v_BM + v_osm  where  v_osm = D ∇ ln|ψ|² = 2D Re(∇ψ/ψ).

    This term explicitly attracts particles toward high-|ψ|² regions and is
    the key ingredient that drives ρ → |ψ|² in the diffusion process.
    The diffusion coefficient D = λ²A_zpf_rms² τ_c / 2 (Fokker-Planck result).

    Returns (v_osm_x, v_osm_y), shape (N_p,).
    """
    # ∇ln|ψ|² = 2 Re(∇ψ / ψ)
    inv_psi = np.conj(psi_q) / rho_q      # = ψ* / |ψ|²  (safe: rho_q ≥ 1e-20)
    vox = 2.0 * D * np.real(inv_psi * dpsi_dx)
    voy = 2.0 * D * np.real(inv_psi * dpsi_dy)
    return vox, voy


def psi_on_hbar_grid(t, phases=VALENTINI_PHASES):
    """ψ(x,y,t) on the H̄ grid. Returns (NX_HB, NX_HB) complex array."""
    c = _phase_coeffs(t, phases)
    return np.einsum('mn,mnij->ij', c, _PHI_HB)


# ─── Valentini H̄ for the [0,π]² box ─────────────────────────────────────────

def compute_hbar_box(qx, qy, psi_grid, epsilon=EPSILON_CG):
    """
    H̄ = Σ_cells ε² ρ̄_cell ln(ρ̄_cell / |ψ̄|²_cell)

    Coarse-grained over n_cg×n_cg cells covering [0,π]².
    ρ̄ from particle histogram (n_cg bins per axis, spanning full [0,π]).
    |ψ̄|² from grid average (bin_w = NX_HB//n_cg grid pts per cell).
    With NX_HB = n_cg * bin_w (exact division), shapes match perfectly.
    """
    n_cg  = max(4, round(np.pi / epsilon))
    eps_a = np.pi / n_cg                    # actual cell side
    bin_w = NX_HB // n_cg                   # grid points per cell

    # Histogram: n_cg bins spanning [0, π] per axis → shape (n_cg, n_cg)
    edges = np.linspace(0, np.pi, n_cg + 1)
    rho_h, _, _ = np.histogram2d(qx, qy, bins=[edges, edges])
    rho_h = rho_h / (len(qx) * eps_a**2)

    # Coarse-grain |ψ|²: use first n_cg*bin_w grid rows/cols → shape (n_cg, n_cg)
    trunc   = n_cg * bin_w
    psi2_cg = (np.abs(psi_grid[:trunc, :trunc])**2
               ).reshape(n_cg, bin_w, n_cg, bin_w).mean(axis=(1, 3))
    psi2_cg /= psi2_cg.sum() * eps_a**2

    mask  = (rho_h > 1e-15) & (psi2_cg > 1e-15)
    ratio = np.where(mask, rho_h / psi2_cg, 1.0)
    hbar  = float(np.sum(np.where(mask, rho_h * np.log(ratio) * eps_a**2, 0.0)))
    return max(hbar, 0.0)


# ─── Initial condition: ρ₀ = |φ₁₁|² ─────────────────────────────────────────

def sample_phi11_squared(n, rng):
    """
    Rejection sampling from ρ₀(x,y) = (4/π²) sin²(x) sin²(y) on [0,π]².
    This is the ground-state distribution, far from the 16-mode |ψ|².
    Acceptance rate ≈ (π/4)² ≈ 61%.
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


# ─── Wall reflection ─────────────────────────────────────────────────────────

def reflect_in_box(qx, qy):
    """
    Hard-wall reflection at x,y ∈ [0,π].
    Rarely triggered: v_BM diverges near walls, pushing particles inward.
    Safety net for large-λ ZPF kicks.
    """
    qx = np.where(qx < 0,        -qx,             qx)
    qx = np.where(qx > np.pi, 2*np.pi - qx, qx)
    qy = np.where(qy < 0,        -qy,             qy)
    qy = np.where(qy > np.pi, 2*np.pi - qy, qy)
    return qx, qy


# ─── ZPF field ────────────────────────────────────────────────────────────────

class ZPFField:
    """
    Stochastic ZPF: N_modes plane waves with Lorentz-invariant spectrum.
    Amplitudes A_k ∝ ω (2D: ω = |k|), phases renewed every TAU_C steps.

    omega_min / omega_max set the spectral cutoff.  Default [0.5, 15] matches
    the double-slit code.  For the box (E_max = 16, v_max ≈ 5.7) use
    omega_max ≈ 6 to suppress high-k modes that disrupt near-node dynamics.

    Optimization: phases_t = phases - omega*t is precomputed once per TAU_C
    block (renewal step), avoiding redundant omega*t per call.
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
        self.phases_t = self.phases.copy()   # precomputed: phases - omega*t_renewal

    def renew_phases(self, t=0.0):
        self.phases   = self.rng.uniform(0, 2*np.pi, len(self.omega))
        self.phases_t = self.phases - self.omega * t

    def field_at(self, qx, qy):
        """
        Evaluate A_zpf at particle positions using precomputed phases_t.
        phi[k,p] = kx[k]*qx[p] + ky[k]*qy[p] + phases_t[k]
        """
        phi = (self.kx[:, None] * qx
               + self.ky[:, None] * qy
               + self.phases_t[:, None])       # broadcasting: no np.outer, no np.full
        cos_phi = np.cos(phi)
        Ax = np.dot(self.A_k, cos_phi)         # (N_p,) via BLAS dot — faster than sum
        Ay = np.dot(self.A_k, np.sin(phi))
        return Ax, Ay


# ─── LAMMPS helpers ───────────────────────────────────────────────────────────

def lmp_scatter_positions(lmp, qx, qy):
    n = len(qx)
    flat = np.zeros(n * 3, dtype=np.float64)
    flat[0::3] = qx
    flat[1::3] = qy
    lmp.scatter("x", 2, 3, (c_double * (n * 3))(*flat))

def lmp_zero_velocities(lmp, n):
    lmp.scatter("v", 2, 3, (c_double * (n * 3))(*np.zeros(n * 3)))


# ─── Single-λ simulation ──────────────────────────────────────────────────────

def run_box(lam, n_particles=2000, n_realizations=10, seed=42,
            output_dir="results_box", zpf_scale=ZPF_SCALE,
            epsilon=EPSILON_CG, phases=None, randomize_phases=False,
            omega_max=15.0, osmotic=False):
    """
    Hybrid BdB/ZPF simulation in [0,π]² for one coupling constant λ.

    Parameters
    ----------
    lam : float
        ZPF coupling (λ=0 reproduces pure Bohm / Valentini 2005).
    n_particles : int
        Ensemble size. ≥ 2000 recommended for reliable H̄.
    n_realizations : int
        Number of independent ZPF realizations (different random seeds).
    randomize_phases : bool
        If True, draw random ψ phases for each realization (broader ensemble).
        If False, use the deterministic Valentini phases for all realizations.
    """
    Path(output_dir).mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    if phases is None:
        phases = VALENTINI_PHASES

    hbar_runs = []
    t_record  = None

    for r in range(n_realizations):
        print(f"  λ={lam:.4f}  realization {r+1}/{n_realizations}", flush=True)

        run_phases = phases
        if randomize_phases:
            run_phases = rng.uniform(0, 2*np.pi, (4, 4))

        zpf = ZPFField(n_modes=N_MODES, seed=int(rng.integers(1e9)), omega_max=omega_max)

        # LAMMPS: box [0,π]², fixed walls, same role as double-slit code
        lmp = lammps(cmdargs=["-screen", "none", "-log", "none"])
        lmp.command(f"variable Np equal {n_particles}")
        lmp.command(f"variable seed equal {int(rng.integers(1e9))}")
        lmp.file("in.bohm_box")

        # Non-equilibrium IC: ρ₀ = |φ₁₁|², genuinely far from multi-mode |ψ|²
        qx, qy = sample_phi11_squared(n_particles, rng)
        lmp_scatter_positions(lmp, qx, qy)
        lmp_zero_velocities(lmp, n_particles)

        hbar_t = []
        t_rec  = []
        t = 0.0

        for step in range(N_STEPS):

            # Record H̄ every RECORD_EVERY steps (≈ π/4 time units)
            if step % RECORD_EVERY == 0:
                psi_g = psi_on_hbar_grid(t, run_phases)
                hbar_t.append(compute_hbar_box(qx, qy, psi_g, epsilon))
                t_rec.append(t)

            # Bohmian guidance velocity at particle positions (analytic, no grid)
            vx, vy, psi_q, dpsi_dx, dpsi_dy, rho_q = vBM_at_particles(
                qx, qy, t, run_phases)

            # ZPF perturbation
            if lam > 0:
                if step % TAU_C == 0:
                    zpf.renew_phases(t)      # precomputes phases - omega*t
                Ax, Ay = zpf.field_at(qx, qy)
                vx = vx + lam * zpf_scale * Ax
                vy = vy + lam * zpf_scale * Ay

                # SED-correct osmotic drift: D × ∇ln|ψ|²
                # D = (λ A_zpf_rms)² τ_c / 2  (Fokker-Planck diffusion coefficient)
                if osmotic:
                    A_rms  = zpf.A_k[0]      # representative A_k (≈ ω_mean/√N)
                    D_zpf  = (lam * zpf_scale * A_rms)**2 * (TAU_C * DT) / 2
                    vox, voy = osmotic_velocity(psi_q, dpsi_dx, dpsi_dy, rho_q, D_zpf)
                    vx = vx + vox
                    vy = vy + voy

            # First-order Euler + hard-wall reflection
            qx, qy = reflect_in_box(qx + DT * vx, qy + DT * vy)
            lmp_scatter_positions(lmp, qx, qy)
            t += DT

        # Final H̄ after last step
        psi_g = psi_on_hbar_grid(t, run_phases)
        hbar_t.append(compute_hbar_box(qx, qy, psi_g, epsilon))
        t_rec.append(t)

        hbar_runs.append(hbar_t)
        if t_record is None:
            t_record = t_rec
        lmp.close()

    # ── Aggregate results ─────────────────────────────────────────────────────
    hbar_arr  = np.array(hbar_runs)     # (n_realizations, n_timepoints)
    hbar_mean = hbar_arr.mean(axis=0)
    hbar_std  = hbar_arr.std(axis=0)
    t_arr     = np.array(t_record)

    # Fit τ_eff via log-linear regression: ln H̄(t) = ln H̄₀ - t/τ_eff
    # Use only points where H̄ > 1% of initial (avoid noise floor)
    h0 = hbar_mean[0] if hbar_mean[0] > 0 else 1.0
    valid = (hbar_mean > 0.01 * h0) & (hbar_mean > 1e-6)
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
    print(f"    τ_eff = {tau_eff:.3f}   1/τ = {inv_tau:.4f}   R² = {R2:.3f}")

    results = {
        "lambda":           lam,
        "epsilon_cg":       epsilon,
        "n_particles":      n_particles,
        "n_realizations":   n_realizations,
        "omega_max":        omega_max,
        "zpf_scale":        zpf_scale,
        "t_final":          float(T_FINAL),
        "times":            t_arr.tolist(),
        "hbar_mean":        hbar_mean.tolist(),
        "hbar_std":         hbar_std.tolist(),
        "hbar_all":         hbar_arr.tolist(),   # full matrix for bootstrap
        "tau_eff":          tau_eff,
        "inv_tau_eff":      inv_tau,
        "R2_fit":           R2,
    }

    tag = f"_om{omega_max:.0f}" if omega_max != 15.0 else ""
    out = Path(output_dir) / f"box_lam{lam:.4f}{tag}.json"
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"    saved → {out}")
    return results


# ─── Stationary eigenstate Born-rule relaxation test ─────────────────────────
#
# In a STATIONARY eigenstate ψ = φ_{11} the Bohmian velocity is identically
# zero (real wave function, no phase gradient).  Without ZPF particles freeze
# forever; H̄ = const.  This isolates the ZPF contribution:
#
#   Test A (ZPF kicks only)   : H̄ slowly INCREASES (isotropic diffusion)
#   Test B (ZPF + osmotic)    : H̄ DECREASES → 0 (Fokker-Planck Born-rule)
#
# The osmotic velocity v_osm = D × ∇ ln|ψ|² = D × (2 cot qx, 2 cot qy)
# attracts particles toward high-|ψ|² regions; together with the ZPF noise it
# implements the Nelson stochastic dynamics whose stationary state is |ψ|².
# D = (λ A_zpf_rms)² τ_c / 2  is the Fokker-Planck diffusion coefficient.

def _psi11_quantities(qx, qy):
    """ψ₁₁, ∂_xψ₁₁, ∂_yψ₁₁, |ψ₁₁|² at particle positions."""
    f    = 2.0 / np.pi
    sx   = np.sin(qx);  sy   = np.sin(qy)
    cx   = np.cos(qx);  cy   = np.cos(qy)
    psi  = f * sx * sy
    dpx  = f * cx * sy
    dpy  = f * sx * cy
    rho  = np.maximum(psi**2, 1e-20)
    return psi, dpx, dpy, rho


def _osmotic_v11(qx, qy, D):
    """Osmotic velocity for ψ = φ₁₁: v_osm = D × 2 (cot qx, cot qy)."""
    sin_x = np.where(np.abs(np.sin(qx)) < 1e-9, 1e-9*np.sign(np.sin(qx)+1e-30), np.sin(qx))
    sin_y = np.where(np.abs(np.sin(qy)) < 1e-9, 1e-9*np.sign(np.sin(qy)+1e-30), np.sin(qy))
    return 2*D*np.cos(qx)/sin_x, 2*D*np.cos(qy)/sin_y


# Pre-compute ψ₁₁ on the H̄ grid (same grid as the multi-mode simulation)
_PSI11_GRID = (2.0/np.pi) * np.sin(X2D_HB) * np.sin(Y2D_HB)


def run_stationary_eigenstate(lam, mode="zpf_osmotic",
                               n_particles=1000, n_realizations=20,
                               seed=42, output_dir="results_stationary",
                               omega_max=3.0, t_final=8*np.pi):
    """
    Born-rule relaxation test in the stationary ground eigenstate φ_{11}.

    mode options
    ───────────
    "frozen"      : λ=0, no ZPF, no osmotic → H̄ = const (control)
    "zpf_only"    : λ>0, ZPF kicks only     → H̄ increases (wrong sign)
    "zpf_osmotic" : λ>0, ZPF + osmotic      → H̄ decreases (Born rule ✓)

    IC: uniform distribution on [0,π]².
    Initial H̄ ≈ 2 ln 2 ≈ 1.39 (vs Born IC |φ₁₁|² which gives H̄=0).
    """
    from pathlib import Path as _Path
    _Path(output_dir).mkdir(exist_ok=True)

    rng     = np.random.default_rng(seed)
    n_steps = round(t_final / DT)
    record_every = max(1, round((np.pi / 4) / DT))

    hbar_runs = []
    t_record  = None

    for r in range(n_realizations):
        print(f"  {mode}  λ={lam:.3f}  r={r+1}/{n_realizations}", flush=True)

        zpf = ZPFField(n_modes=N_MODES, seed=int(rng.integers(1e9)), omega_max=omega_max)

        # ZPF diffusion coefficient (Fokker-Planck)
        A_rms_sq = float(np.mean(zpf.A_k**2))   # = mean(omega²/N) ≈ <omega²>/N × N = <omega²>
        # Correction: E[Ax²] = sum_k A_k² × 0.5 per-component
        A_zpf_rms_sq = 0.5 * float(np.sum(zpf.A_k**2))
        D_zpf = (lam * A_zpf_rms_sq**0.5)**2 * (TAU_C * DT) / 2  if lam > 0 else 0.0

        # IC: uniform on [0, π]²
        qx = rng.uniform(0, np.pi, n_particles)
        qy = rng.uniform(0, np.pi, n_particles)

        hbar_t = []
        t_rec  = []
        t = 0.0

        for step in range(n_steps):
            if step % record_every == 0:
                hbar_t.append(compute_hbar_box(qx, qy, _PSI11_GRID))
                t_rec.append(t)

            vx = np.zeros(n_particles)
            vy = np.zeros(n_particles)

            if lam > 0 and mode in ("zpf_only", "zpf_osmotic"):
                if step % TAU_C == 0:
                    zpf.renew_phases(t)
                Ax, Ay = zpf.field_at(qx, qy)
                vx += lam * Ax
                vy += lam * Ay

            if lam > 0 and mode == "zpf_osmotic":
                vox, voy = _osmotic_v11(qx, qy, D_zpf)
                # clamp osmotic velocity to avoid singularity near walls
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

    # Fit τ (only for zpf_osmotic mode where decay is expected)
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
                resid = np.log(hbar_mean[valid]) - (slope*t_arr[valid] + np.log(h0))
                R2 = float(1 - np.var(resid) / np.var(np.log(hbar_mean[valid])))

    print(f"    mode={mode}  τ_eff={tau_eff:.3f}  R²={R2:.3f}")

    result = {
        "mode": mode, "lambda": lam, "omega_max": omega_max,
        "n_particles": n_particles, "n_realizations": n_realizations,
        "t_final": float(t_final), "D_zpf": D_zpf,
        "times": t_arr.tolist(),
        "hbar_mean": hbar_mean.tolist(),
        "hbar_std":  hbar_std.tolist(),
        "hbar_all":  hbar_arr.tolist(),
        "tau_eff": tau_eff, "R2_fit": R2,
    }
    out = _Path(output_dir) / f"stat_{mode}_lam{lam:.3f}.json"
    with open(out, "w") as fh:
        import json as _json
        _json.dump(result, fh, indent=2)
    print(f"    saved → {out}")
    return result


# ─── λ sweep ─────────────────────────────────────────────────────────────────

LAMBDA_SWEEP_BOX = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20]


def run_box_sweep(n_particles=2000, n_realizations=10, output_dir="results_box",
                  lambda_list=None):
    lams = lambda_list if lambda_list is not None else LAMBDA_SWEEP_BOX
    all_results = []
    for lam in lams:
        print(f"\n=== λ = {lam} ===")
        r = run_box(lam, n_particles=n_particles,
                    n_realizations=n_realizations,
                    output_dir=output_dir)
        all_results.append(r)

    # Summary table
    lam0 = all_results[0]
    print(f"\n{'λ':>8}  {'τ_eff':>8}  {'1/τ':>8}  {'Δ(1/τ)':>10}  {'R²':>6}")
    for r in all_results:
        d_inv = r["inv_tau_eff"] - lam0["inv_tau_eff"]
        print(f"{r['lambda']:>8.4f}  {r['tau_eff']:>8.3f}  "
              f"{r['inv_tau_eff']:>8.4f}  {d_inv:>+10.4f}  {r['R2_fit']:>6.3f}")

    return all_results


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Hybrid BdB/ZPF in 2D closed box (Valentini system)")
    p.add_argument("--lam",     type=float, default=None,
                   help="Single λ value (omit for full sweep)")
    p.add_argument("--Np",      type=int,   default=2000,
                   help="Particles per realization")
    p.add_argument("--Nr",      type=int,   default=10,
                   help="ZPF realizations")
    p.add_argument("--out",     type=str,   default="results_box")
    p.add_argument("--epsilon", type=float, default=EPSILON_CG,
                   help="Coarse-graining cell size ε for H̄")
    p.add_argument("--rand-phases", action="store_true",
                   help="Randomize ψ phases per realization")
    p.add_argument("--omega-max", type=float, default=15.0,
                   help="ZPF spectral cutoff ω_max (box: ~6, default: 15)")
    args = p.parse_args()

    if args.lam is not None:
        run_box(args.lam, n_particles=args.Np, n_realizations=args.Nr,
                output_dir=args.out, epsilon=args.epsilon,
                randomize_phases=args.rand_phases, omega_max=args.omega_max)
    else:
        run_box_sweep(n_particles=args.Np, n_realizations=args.Nr,
                      output_dir=args.out)
