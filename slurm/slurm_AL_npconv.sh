#!/bin/bash
#SBATCH --account=pad_70
#SBATCH --job-name=AL_npconv
#SBATCH --partition=cpunode
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/AL_npconv_%A_%a.out
#SBATCH --error=logs/AL_npconv_%A_%a.err
#SBATCH --array=0-5%6

# Paper-2 Fase 3 — Np convergence sanity check.
# 6 tasks = {2500, 5000, 10000} × {λ=0, λ=0.10}.  Nr=50 fixed.
#
# Driver:  src/paper2/bohm_zpf_AL_box.py
# Output:  results/paper2_fase3/npconv/Np{2500,5000,10000}/

module purge

export http_proxy=http://172.28.3.3:3128
export https_proxy=$http_proxy
export MPLBACKEND=Agg
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source ~/miniconda3/etc/profile.d/conda.sh
conda activate distortion_env

set -e

cd /home/sbergamin/lammps-22Jul2025/build-clementina/HibridLorentz

NPS=(2500 2500 5000 5000 10000 10000)
LAMS=(0.0  0.10 0.0  0.10 0.0   0.10)

NP=${NPS[$SLURM_ARRAY_TASK_ID]}
LAM=${LAMS[$SLURM_ARRAY_TASK_ID]}

NR=${NR:-50}
TAU_RAD=${TAU_RAD:-0.01}
OMEGA_MAX=${OMEGA_MAX:-3.0}
OUTDIR="results/paper2_fase3/npconv/Np${NP}"

echo "=== SLURM Job: AL_npconv ==="
echo "  Array task: $SLURM_ARRAY_TASK_ID"
echo "  Np:         $NP"
echo "  Lambda:     $LAM"
echo "  Nr:         $NR"
echo "  tau_rad:    $TAU_RAD"
echo "  omega_max:  $OMEGA_MAX"
echo "  outdir:     $OUTDIR"
echo "  Start:      $(date)"
echo ""

mkdir -p logs "$OUTDIR"

cd src/paper2
python bohm_zpf_AL_box.py \
    --mode single \
    --lam "$LAM" \
    --Np "$NP" \
    --Nr "$NR" \
    --tau-rad "$TAU_RAD" \
    --omega-max "$OMEGA_MAX" \
    --out "../../$OUTDIR"

echo ""
echo "  End: $(date)"
echo "  Exit code: $?"
