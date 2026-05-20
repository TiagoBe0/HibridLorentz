#!/bin/bash
#SBATCH --job-name=AL_sweep
#SBATCH --output=logs/AL_sweep_%A_%a.out
#SBATCH --error=logs/AL_sweep_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --partition=cpunode
#SBATCH --array=0-9%4

# Paper-2 Fase 3: ALD sweep on Clementina
# Array: 10 lambda values, max 4 concurrent jobs
# Usage: sbatch slurm_AL_sweep.sh
#        sbatch slurm_AL_sweep.sh --export=MODE=nelson  (Nelson D=0.5)

export http_proxy=http://172.28.3.3:3128
export https_proxy=$http_proxy
export MPLBACKEND=Agg

# Lambda values (same as LAMBDA_SWEEP_AL in bohm_zpf_AL_box.py)
LAMBDAS=(0.0 0.005 0.01 0.02 0.03 0.05 0.07 0.10 0.15 0.20)
LAM=${LAMBDAS[$SLURM_ARRAY_TASK_ID]}

# Default production parameters
NP=${NP:-5000}
NR=${NR:-50}
TAU_RAD=${TAU_RAD:-0.01}
OMEGA_MAX=${OMEGA_MAX:-3.0}
MODE=${MODE:-ald}   # "ald" or "nelson"

echo "=== SLURM Job: AL_sweep ==="
echo "  Array task: $SLURM_ARRAY_TASK_ID"
echo "  Lambda:     $LAM"
echo "  Np / Nr:    $NP / $NR"
echo "  tau_rad:    $TAU_RAD"
echo "  omega_max:  $OMEGA_MAX"
echo "  mode:       $MODE"
echo "  Start:      $(date)"
echo ""

mkdir -p logs results_AL_box_prod results_AL_box_nelson

if [ "$MODE" = "nelson" ]; then
    python3 bohm_zpf_AL_box.py \
        --mode single \
        --lam "$LAM" \
        --Np "$NP" \
        --Nr "$NR" \
        --tau-rad "$TAU_RAD" \
        --omega-max "$OMEGA_MAX" \
        --out results_AL_box_nelson \
        --d-nelson
else
    python3 bohm_zpf_AL_box.py \
        --mode single \
        --lam "$LAM" \
        --Np "$NP" \
        --Nr "$NR" \
        --tau-rad "$TAU_RAD" \
        --omega-max "$OMEGA_MAX" \
        --out results_AL_box_prod
fi

echo ""
echo "  End: $(date)"
echo "  Exit code: $?"
