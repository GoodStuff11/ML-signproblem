#!/bin/bash
# Rerun of the hva_parameter_sweep_N_5_4_4x3_u60_eigref experiment on every
# (N, lattice) folder in the ED data with a lattice smaller than 4x3:
# target U=15 (u_idx=60 with --ref=eigenstate, identical U grid in every folder),
# starting from the ED ground state at the lowest U, tie=full, overlap loss.
#
# One sbatch per (folder, layer count), each writing its own CSV to
#   benchmarks/hva_parameter_sweep_N_<nu>_<nd>_<LxW>_u60_eigref/P<layers>.csv
# These systems are small, so they run CPU-only.
#
# Usage:  bash submit_hva_sweep_small_u60_eigref.sh [layers...]
#         (default layers: 2 5 10 15 25 35 45 55 65)

if [ $# -gt 0 ]; then LAYERS=("$@"); else LAYERS=(2 5 10 15 25 35 45 55 65); fi
for P in "${LAYERS[@]}"; do
    [[ "$P" =~ ^[0-9]+$ ]] || { echo "error: layer count '$P' is not an integer" >&2; exit 1; }
done

FOLDERS=(
    "N=(2, 2)_3x2"
    "N=(3, 2)_3x2"
    "N=(3, 3)_3x2"
    "N=(3, 3)_4x2"
    "N=(3, 2)_3x3"
    "N=(3, 3)_3x3"
    "N=(4, 3)_3x3"
    "N=(4, 4)_3x3"
    "N=(5, 4)_3x3"
)
U_IDX=60
TIE=full
RUNS=1
INIT_SAMPLES=10
MAXITERS=2000
CPUS=4
MEM=16g
PARTITION=kim
TIME=3-00:00:00

ED_DIR=/home/jek354/research/ML-signproblem/experimenting/ed
JOB_DIR=/home/jek354/research/ML-signproblem/jobs

for FOLDER in "${FOLDERS[@]}"; do
    # "N=(5, 4)_3x3" -> NU=5 ND=4 LAT=3x3
    [[ "$FOLDER" =~ ^N=\(([0-9]+),\ ([0-9]+)\)_([0-9]+x[0-9]+)$ ]] ||
        { echo "error: cannot parse folder '$FOLDER'" >&2; exit 1; }
    NU=${BASH_REMATCH[1]}; ND=${BASH_REMATCH[2]}; LAT=${BASH_REMATCH[3]}
    OUT_DIR=$ED_DIR/benchmarks/hva_parameter_sweep_N_${NU}_${ND}_${LAT}_u${U_IDX}_eigref
    mkdir -p "$OUT_DIR"

    for P in "${LAYERS[@]}"; do
        NAME=hva_${LAT}_${NU}${ND}_u${U_IDX}_eigref_P${P}
        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$NAME
#SBATCH --output=$JOB_DIR/${NAME}_%j.out
#SBATCH --error=$JOB_DIR/${NAME}_%j.err
#SBATCH --partition=$PARTITION
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --time=$TIME

cd $ED_DIR
julia -t $CPUS --project=.. run_hva_parameter_sweep.jl "$FOLDER" $U_IDX \\
    --ref=eigenstate --tie=$TIE --layers=$P --loss=overlap \\
    --runs=$RUNS --initialization_samples=$INIT_SAMPLES --maxiters=$MAXITERS \\
    --out=$OUT_DIR/P${P}.csv
EOF
    done
done
