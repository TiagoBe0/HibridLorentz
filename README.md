# HibridLorentz

Hybrid Bohm–de Broglie / Zero-Point-Field simulations with Abraham–Lorentz–Dirac
radiation reaction, in a 2D closed box.  Project supports two manuscripts:

- **Paper 1** (legacy): pure BdB + ZPF, no ALD. `bohm_zpf_box.py` pathway.
- **Paper 2** (active): adds ALD via Landau–Lifshitz approximation, corrected
  osmotic diffusion, and always-on Nelson drift. `bohm_zpf_AL_box.py`.

## Repository layout

```
src/
  paper2/   bohm_zpf_AL_box.py            ← Paper-2 Fase 3 driver (ALD)
            abraham_lorentz_classic.py    ← Paper-2 Fase 2 Boyer benchmark
            sympy_AL_derivation.py        ← Paper-2 Fase 1 analytic derivation
            in.bohm_box                   ← LAMMPS init (used for position storage only)
  paper1/   bohm_zpf_box.py               ← Paper-1 driver (no ALD)
            bohm_zpf_lammps.py            ← Paper-1 LAMMPS variant
            calibrate_zpf.py
            in.bohm_zpf

analysis/   analyze_AL.py                 ← Paper-2 main analyzer
            analyze_npconv.py             ← Np convergence check
            analyze_box.py                ← Paper-1 analyzer
            analyze_relaxation.py
            plot_publication.py           ← Paper-1 figures

slurm/      slurm_AL_sweep.sh             ← Np=5000 λ sweep (original prod)
            slurm_AL_sweep_Np10k.sh       ← Np=10000 λ sweep (post-audit prod)
            slurm_AL_npconv.sh            ← Np convergence array

papers/     paper2/hybrid_AL_paper.tex
            paper1/hybridLorentz-8.pdf

refs/       boyer1975.pdf, valentini2005.pdf, boyer_1975_results.json

docs/       CONTEXTO.md, contexto.txt, README_results.md

figures/    paper2/  fig_AL_*.pdf            ← Paper-2 figures (D scaling, tau, Npconv)
            paper1/  fig{1..4}_*.{pdf,png}   ← Paper-1 publication figures
            boyer/   boyer_*sweep.pdf, …      ← Fase 2 benchmark figures

results/    paper2_fase3/
              prod_Np5000/                  ← original production sweep
              prod_Np10000/                 ← canonical production sweep (post-audit)
              npconv/Np{2500,5000,10000}/   ← Np convergence
              AL_box/, AL_box_nelson/, AL_stat/  ← staging / Nelson-D tests
            paper1/
              box/, box_om3/, run_10k/

logs/       SLURM stdout/stderr (gitignored)
```

Heavy production output dirs (`prod_Np5000/`, `prod_Np10000/`, `npconv/`) are
gitignored — they are regenerable from the slurm scripts.

## Running

On Clementina:
```bash
sbatch slurm/slurm_AL_sweep_Np10k.sh         # full λ sweep at Np=10000
sbatch slurm/slurm_AL_npconv.sh              # Np convergence check
```

Local smoke test (`--no-lammps` falls back to pure NumPy):
```bash
python src/paper2/bohm_zpf_AL_box.py --no-lammps --mode single --lam 0.10 --Np 500 --Nr 2 --out /tmp/test
```

Analyze:
```bash
python analysis/analyze_AL.py     --al-dir results/paper2_fase3/prod_Np10000
python analysis/analyze_npconv.py --root  results/paper2_fase3/npconv
```

## Key methodological note (May 2026 audit)

`compute_hbar_box` uses Miller–Madow bias correction `(K_eff−1)/(2 N_p)` on
the empirical KL divergence.  Without it, `τ_eff` carries an O(1/N_p) tail
offset that masquerades as physics; see `docs/CONTEXTO.md` and the npconv
analysis for the diagnostic.  Production results in `prod_Np10000/` use the
corrected estimator and are the canonical numbers for the Paper-2 manuscript.
