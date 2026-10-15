#!/bin/bash
# HVA parameter sweep on 4x3, N=(5,4), starting from the ED ground state at the lowest U
# (--ref=eigenstate, so u_idx=8 => U=2.0 and u_idx=60 => U=15; the CSV's U column records the value),
# with the target on a periodic or an open lattice:
#   TARGET_BC=periodic : momentum-sector ED target; the HVA is open on the odd axis (the boundary
#                        mismatch tested against Cade et al. 2020).
#   TARGET_BC=open     : open-boundary ground states computed in real space (--target_bc=open),
#                        open HVA, i.e. Cade et al.'s setting.
#
# One sbatch per layer count, each writing its own CSV so concurrent jobs never append to the same file.
#
# Usage:  TARGET_BC=<periodic|open> U_IDX=<8|60|...> [USE_GPU=1] bash submit_hva_sweep_4x3_5_4_eigref_bc.sh [layers...]
#         (default layers: 2 5 10 15 25 35 45 55 65; USE_GPU=1 gives one GPU per job and a _gpu suffix)
# Output: experimenting/ed/benchmarks/hva_parameter_sweep_N_5_4_4x3_u<U_IDX>_eigref[_obc]/P<layers>[_gpu].csv

if [ $# -gt 0 ]; then LAYERS=("$@"); else LAYERS=(2 5 10 15 25 35 45 55 65); fi
for P in "${LAYERS[@]}"; do
    [[ "$P" =~ ^[0-9]+$ ]] || { echo "error: layer count '$P' is not an integer" >&2; exit 1; }
done
TARGET_BC=${TARGET_BC:?set TARGET_BC=periodic or TARGET_BC=open}
U_IDX=${U_IDX:?set U_IDX (8 => U=2, 60 => U=15 with --ref=eigenstate)}
case "$TARGET_BC" in periodic) BC_SUFFIX="" ;; open) BC_SUFFIX=_obc ;; *) echo "error: TARGET_BC must be periodic or open" >&2; exit 1 ;; esac

FOLDER="N=(5, 4)_4x3"
TIE=full
RUNS=1
INIT_SAMPLES=10
MAXITERS=1000
CPUS=40
MEM=64g
PARTITION=kim
TIME=7-00:00:00

GPU_SUFFIX=""
RESOURCES="#SBATCH --cpus-per-task=$CPUS"
USE_GPU_ARG=""
if [ "${USE_GPU:-0}" = "1" ]; then
    CPUS=4
    GPU_SUFFIX=_gpu
    RESOURCES="#SBATCH --cpus-per-task=$CPUS
#SBATCH --gres=gpu:1
#SBATCH --exclude=kim-compute-01"
    USE_GPU_ARG="--use_gpu"
fi

ED_DIR=/home/jek354/research/ML-signproblem/experimenting/ed
JOB_DIR=/home/jek354/research/ML-signproblem/jobs
OUT_DIR=$ED_DIR/benchmarks/hva_parameter_sweep_N_5_4_4x3_u${U_IDX}_eigref${BC_SUFFIX}

mkdir -p "$OUT_DIR"

for P in "${LAYERS[@]}"; do
    NAME=hva_4x3_54_u${U_IDX}_eigref${BC_SUFFIX}_P${P}${GPU_SUFFIX}
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$NAME
#SBATCH --output=$JOB_DIR/${NAME}_%j.out
#SBATCH --error=$JOB_DIR/${NAME}_%j.err
#SBATCH --partition=$PARTITION
$RESOURCES
#SBATCH --mem=$MEM
#SBATCH --time=$TIME

cd $ED_DIR
julia -t $CPUS --project=.. run_hva_parameter_sweep.jl "$FOLDER" $U_IDX \\
    --ref=eigenstate --target_bc=$TARGET_BC --tie=$TIE --layers=$P --loss=overlap \\
    --runs=$RUNS --initialization_samples=$INIT_SAMPLES --maxiters=$MAXITERS $USE_GPU_ARG \\
    --out=$OUT_DIR/P${P}${GPU_SUFFIX}.csv
EOF
done
