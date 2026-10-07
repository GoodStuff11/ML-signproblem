#!/bin/bash
# HVA parameter sweep on 4x3, N=(5,4), target U=15 (u_idx=61), starting from the
# U=0 Slater determinant in the target's momentum sector (--ref=slater prepends a
# U=0 row, which shifts u_idx so that 61 => U=15.0; the CSV's U column records
# the actual value).
#
# One sbatch per layer count, each writing its own CSV so concurrent jobs never
# append to the same file. Concatenate afterwards (same header in every file).
#
# Usage:  bash submit_hva_sweep_4x3_5_4_u61_slaterref.sh [layers...]
#         (default layers: 25 35 45 55 65; one A100 per job, --use_gpu)
#         USE_GPU=0 bash submit_hva_sweep_4x3_5_4_u61_slaterref.sh [layers...]
#         (CPU only; job names and CSVs drop the _gpu suffix)

if [ $# -gt 0 ]; then LAYERS=("$@"); else LAYERS=(25 35 45 55 65); fi
for P in "${LAYERS[@]}"; do
    [[ "$P" =~ ^[0-9]+$ ]] || { echo "error: layer count '$P' is not an integer (use USE_GPU=0/1, not a flag)" >&2; exit 1; }
done

FOLDER="N=(5, 4)_4x3"
U_IDX=60
TIE=full
RUNS=1
INIT_SAMPLES=10
MAXITERS=2000
CPUS=40
MEM=200g
PARTITION=kim
TIME=7-00:00:00

GPU_SUFFIX=""
RESOURCES="#SBATCH --cpus-per-task=$CPUS"
USE_GPU_ARG=""
if [ "${USE_GPU:-1}" = "1" ]; then
    CPUS=4
    GPU_SUFFIX=_gpu
    RESOURCES="#SBATCH --cpus-per-task=$CPUS
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1"
    USE_GPU_ARG="--use_gpu"
fi

ED_DIR=/home/jek354/research/ML-signproblem/experimenting/ed
JOB_DIR=/home/jek354/research/ML-signproblem/jobs
OUT_DIR=$ED_DIR/benchmarks/hva_parameter_sweep_N_5_4_4x3_u${U_IDX}_eigref

mkdir -p "$OUT_DIR"

for P in "${LAYERS[@]}"; do
    NAME=hva_4x3_54_u${U_IDX}_eigref_P${P}${GPU_SUFFIX}
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
    --ref=eigenstate --tie=$TIE --layers=$P --loss=overlap \\
    --runs=$RUNS --initialization_samples=$INIT_SAMPLES --maxiters=$MAXITERS $USE_GPU_ARG \\
    --out=$OUT_DIR/P${P}${GPU_SUFFIX}.csv
EOF
done
