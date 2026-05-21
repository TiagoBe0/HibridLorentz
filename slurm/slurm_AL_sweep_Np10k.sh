#!/bin/bash
#SBATCH --account=pad_70
#SBATCH --job-name=AL_sweep_Np10k
#SBATCH --partition=cpunode
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/AL_sweep_Np10k_%A_%a.out
#SBATCH --error=logs/AL_sweep_Np10k_%A_%a.err
#SBATCH --array=0-9%5

# Paper-2 Fase 3: ALD sweep at Np=10000 (post-Miller-Madow audit).
#
# Driver:  src/paper2/bohm_zpf_AL_box.py
#          (Miller-Madow bias correction + v_clip + per-realization D_ALD)
# Output:  results/paper2_fase3/prod_Np10000/

module purge

export http_proxy=http://172.28.3.3:3128
export https_proxy=$http_proxy
export MPLBACKEND=Agg
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source ~/miniconda3/etc/profile.d/conda.sh
conda activate distortion_env

set -e

cd /home/sbergamin/lammps-22Jul2025/build-clementina/HibridLorentz

LAMBDAS=(0.0 0.005 0.01 0.02 0.03 0.05 0.07 0.10 0.15 0.20)
LAM=${LAMBDAS[$SLURM_ARRAY_TASK_ID]}

NP=${NP:-10000}
NR=${NR:-50}
TAU_RAD=${TAU_RAD:-0.01}
OMEGA_MAX=${OMEGA_MAX:-3.0}
OUTDIR=${OUTDIR:-results/paper2_fase3/prod_Np10000}

echo "=== SLURM Job: AL_sweep_Np10k ==="
echo "  Array task: $SLURM_ARRAY_TASK_ID"
echo "  Lambda:     $LAM"
echo "  Np / Nr:    $NP / $NR"
echo "  tau_rad:    $TAU_RAD"
echo "  omega_max:  $OMEGA_MAX"
echo "  outdir:     $OUTDIR"
echo "  Start:      $(date)"
echo ""

mkdir -p logs "$OUTDIR"

# Driver lives under src/paper2/ — use it as a module path so its imports
# (in.bohm_box sibling, etc.) resolve relative to its own directory.
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
