#!/bin/bash
# Rerun of every periodic-target HVA parameter sweep on a lattice with an odd axis (3x2, 3x3, 4x3),
# now with a periodic HVA. Before 2026-11-06, enumerate_ferm_excitations_HVA rejected use_pbc=true on
# an odd axis > 2, so run_hva_parameter_sweep.jl silently fell back to an OPEN-bond HVA on these
# periodic (torus) targets. The HVA now gives each odd periodic axis a third hopping group holding the
# wrap bond (L-1, 0), with its own coefficient under tie=full (see TamFermion.enumerate_ferm_excitations_HVA
# and testing/test_hva_odd_pbc.jl). 4x2 is not rerun: both axes are even, so its HVA was already periodic.
#
# Sweeps (each over the layer counts below, one sbatch per (sweep, layer count)):
#   tag             folder           u_idx  --ref       target U   resources        maxiters
#   u60_eigref      N=(2, 2)_3x2     60     eigenstate  15         CPU, 4 cores     2000
#   u60_eigref      N=(3, 2)_3x2     60     eigenstate  15         CPU, 4 cores     2000
#   u60_eigref      N=(3, 3)_3x2     60     eigenstate  15         CPU, 4 cores     2000
#   u60_eigref      N=(3, 2)_3x3     60     eigenstate  15         CPU, 4 cores     2000
#   u60_eigref      N=(3, 3)_3x3     60     eigenstate  15         CPU, 4 cores     2000
#   u60_eigref      N=(4, 3)_3x3     60     eigenstate  15         CPU, 4 cores     2000
#   u60_eigref      N=(4, 4)_3x3     60     eigenstate  15         CPU, 4 cores     2000
#   u60_eigref      N=(5, 4)_3x3     60     eigenstate  15         CPU, 4 cores     2000
#   u8_eigref       N=(3, 3)_3x3     8      eigenstate  2          CPU, 4 cores     2000
#   u9_slater       N=(3, 3)_3x3     9      slater      2          CPU, 4 cores     2000
#   u60_eigref      N=(5, 4)_4x3     60     eigenstate  15         1 GPU, 4 cores   1000
#   u8_eigref       N=(5, 4)_4x3     8      eigenstate  2          1 GPU, 4 cores   1000
#   u9_slater       N=(5, 4)_4x3     9      slater      2          1 GPU, 4 cores   1000
#   u61_slaterref   N=(5, 4)_4x3     61     slater      15         1 GPU, 4 cores   1000
# Settings are those of the original periodic sweeps; 4x3 uses maxiters 1000, the value of its
# open-boundary (_obc) counterparts. All: --target_bc=periodic (default), no --hva_pbc (defaults to
# true), --tie=full, --loss=overlap, --runs=1, --initialization_samples=10, default --seed.
# --ref=eigenstate starts from the ED ground state at the lowest U (0.25); --ref=slater from the U=0
# Slater determinant (u_idx shifted by one: 9 => U=2, 61 => U=15).
#
# Code: run from THIS checkout's experimenting/ed (the hva-odd-pbc worktree when submitted from there),
# so the worktree must stay on disk until every job has started.
# Output: $OUT_ROOT/hva_parameter_sweep_N_<nu>_<nd>_<LxW>_<tag>_pbchva/P<layers>[_gpu].csv, where
# OUT_ROOT defaults to the main checkout's benchmarks folder (next to the old open-bond results,
# which keep their original folder names without _pbchva). A CSV that already exists is skipped.
# Slurm stdout/stderr: /home/jek354/research/ML-signproblem/jobs/<job name>_<jobid>.{out,err}.
#
# Usage:  bash jobs/submit_hva_sweep_pbc_odd.sh [layers...]
#         (default layers: 2 5 10 15 25 35 45 55 65)
#         DRY_RUN=1 bash jobs/submit_hva_sweep_pbc_odd.sh   # print the jobs, submit nothing
#         ONLY=<regex> bash jobs/submit_hva_sweep_pbc_odd.sh # only sweeps whose "<folder> <tag>" matches

if [ $# -gt 0 ]; then LAYERS=("$@"); else LAYERS=(2 5 10 15 25 35 45 55 65); fi
for P in "${LAYERS[@]}"; do
    [[ "$P" =~ ^[0-9]+$ ]] || { echo "error: layer count '$P' is not an integer" >&2; exit 1; }
done

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ED_DIR=$(cd "$SCRIPT_DIR/../experimenting/ed" && pwd)
OUT_ROOT=${OUT_ROOT:-/home/jek354/research/ML-signproblem/experimenting/ed/benchmarks}
JOB_DIR=/home/jek354/research/ML-signproblem/jobs
PARTITION=kim
TIME=7-00:00:00

# folder|u_idx|ref|tag|gpu
SWEEPS=(
    "N=(2, 2)_3x2|60|eigenstate|u60_eigref|0"
    "N=(3, 2)_3x2|60|eigenstate|u60_eigref|0"
    "N=(3, 3)_3x2|60|eigenstate|u60_eigref|0"
    "N=(3, 2)_3x3|60|eigenstate|u60_eigref|0"
    "N=(3, 3)_3x3|60|eigenstate|u60_eigref|0"
    "N=(4, 3)_3x3|60|eigenstate|u60_eigref|0"
    "N=(4, 4)_3x3|60|eigenstate|u60_eigref|0"
    "N=(5, 4)_3x3|60|eigenstate|u60_eigref|0"
    "N=(3, 3)_3x3|8|eigenstate|u8_eigref|0"
    "N=(3, 3)_3x3|9|slater|u9_slater|0"
    "N=(5, 4)_4x3|60|eigenstate|u60_eigref|1"
    "N=(5, 4)_4x3|8|eigenstate|u8_eigref|1"
    "N=(5, 4)_4x3|9|slater|u9_slater|1"
    "N=(5, 4)_4x3|61|slater|u61_slaterref|1"
)

n_sub=0; n_skip=0
for S in "${SWEEPS[@]}"; do
    IFS='|' read -r FOLDER U_IDX REF TAG GPU <<< "$S"
    if [ -n "$ONLY" ] && ! [[ "$FOLDER $TAG" =~ $ONLY ]]; then continue; fi
    [[ "$FOLDER" =~ ^N=\(([0-9]+),\ ([0-9]+)\)_([0-9]+x[0-9]+)$ ]] ||
        { echo "error: cannot parse folder '$FOLDER'" >&2; exit 1; }
    NU=${BASH_REMATCH[1]}; ND=${BASH_REMATCH[2]}; LAT=${BASH_REMATCH[3]}
    OUT_DIR=$OUT_ROOT/hva_parameter_sweep_N_${NU}_${ND}_${LAT}_${TAG}_pbchva

    if [ "$GPU" = "1" ]; then
        CPUS=4; MEM=64g; MAXITERS=1000; SUFFIX=_gpu; GPU_ARG="--use_gpu"
        RESOURCES="#SBATCH --gres=gpu:1
#SBATCH --exclude=kim-compute-01"
    else
        CPUS=4; MEM=16g; MAXITERS=2000; SUFFIX=""; GPU_ARG=""; RESOURCES=""
    fi

    for P in "${LAYERS[@]}"; do
        OUT=$OUT_DIR/P${P}${SUFFIX}.csv
        if [ -e "$OUT" ]; then echo "skip (exists): $OUT"; n_skip=$((n_skip+1)); continue; fi
        NAME=hva_${LAT}_${NU}${ND}_${TAG}_pbchva_P${P}${SUFFIX}
        if [ -n "$DRY_RUN" ]; then echo "would submit $NAME -> $OUT"; n_sub=$((n_sub+1)); continue; fi
        mkdir -p "$OUT_DIR"
        sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=$NAME
#SBATCH --output=$JOB_DIR/${NAME}_%j.out
#SBATCH --error=$JOB_DIR/${NAME}_%j.err
#SBATCH --partition=$PARTITION
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --time=$TIME
$RESOURCES

cd $ED_DIR
julia -t $CPUS --project=.. run_hva_parameter_sweep.jl "$FOLDER" $U_IDX \\
    --ref=$REF --tie=full --layers=$P --loss=overlap \\
    --runs=1 --initialization_samples=10 --maxiters=$MAXITERS $GPU_ARG \\
    --out=$OUT
EOF
        n_sub=$((n_sub+1))
    done
done
echo "submitted (or would submit): $n_sub, skipped: $n_skip"
