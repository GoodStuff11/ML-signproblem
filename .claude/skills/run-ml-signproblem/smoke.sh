#!/usr/bin/env bash
# Smoke-test driver for ML-signproblem: computes a small Exact Diagonalization
# (ED) dataset from scratch via Lanczos + momentum sectors, then loads it back
# through the library's own loader and checks basic physical invariants
# (normalized eigenvectors, expected shapes).
#
# Usage:
#   .claude/skills/run-ml-signproblem/smoke.sh [output_dir]
#
# output_dir defaults to a fresh mktemp -d directory. Safe to run repeatedly;
# never writes into the repo's own data/ tree.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ED_DIR="$UNIT_ROOT/experimenting/ed"

OUT_DIR="${1:-$(mktemp -d)}"
mkdir -p "$OUT_DIR"

# Preflight: this host enforces a hard per-user memory cgroup (commonly 4GiB on
# shared academic login nodes), separate from and much smaller than the
# machine-wide total `free -h` reports. VSCode/editor extension processes
# under the same account often eat most of it, and the Julia run below only
# needs on the order of ~1GB peak RSS but has been observed to get SIGKILL'd
# (silently, or printed as "Killed") when headroom is too thin. See Gotchas
# in SKILL.md if this check reports low headroom or the run gets killed.
CG_MAX_FILE="/sys/fs/cgroup/user.slice/user-$(id -u).slice/memory.max"
CG_CUR_FILE="/sys/fs/cgroup/user.slice/user-$(id -u).slice/memory.current"
if [ -r "$CG_MAX_FILE" ] && [ -r "$CG_CUR_FILE" ]; then
    CG_MAX=$(cat "$CG_MAX_FILE")
    CG_CUR=$(cat "$CG_CUR_FILE")
    if [ "$CG_MAX" != "max" ]; then
        HEADROOM_MB=$(( (CG_MAX - CG_CUR) / 1024 / 1024 ))
        echo "== Preflight: per-user cgroup memory headroom ~${HEADROOM_MB}MiB (cap $((CG_MAX / 1024 / 1024))MiB) =="
        if [ "$HEADROOM_MB" -lt 500 ]; then
            echo "   WARNING: headroom is low; the Julia run below may get OOM-killed. See Gotchas in SKILL.md." >&2
        fi
    fi
fi

echo "== Step 1/2: computing ED ground states for a 3x2 lattice, N=(2,2) Hubbard model =="
echo "   (Lanczos + momentum-sector projection, 61 U values x 6 sectors; ~60-90s incl. Julia startup)"
cd "$ED_DIR"
julia --project=.. run_ed_lanczos_momentum.jl 3 2 2 2 "$OUT_DIR"

DATA_FILE="$OUT_DIR/N=(2, 2)_3x2/meta_data_and_E.jld2"
if [ ! -f "$DATA_FILE" ]; then
    echo "FAILED: expected output file not found: $DATA_FILE" >&2
    exit 1
fi

echo "== Step 2/2: loading the computed data back through load_jld2_ED_data and verifying it =="
julia --project=.. -e "
using HDF5, LinearAlgebra, SparseArrays, Lattices
include(\"ed_objects.jl\"); include(\"utility_functions.jl\"); include(\"trotter.jl\"); include(\"ed_functions.jl\")
using .Trotter

U_values, target_vecs, indexer, precomp, N, spin_conserved, use_symmetry, sign_conv = load_jld2_ED_data(\"$DATA_FILE\"; verbose=false)

@assert length(U_values) == 61 \"expected 61 U values, got \$(length(U_values))\"
@assert size(target_vecs) == (61, 39) \"expected target_vecs size (61, 39), got \$(size(target_vecs))\"
@assert all(isapprox.(norm.(eachrow(target_vecs)), 1.0; atol=1e-8)) \"eigenvectors are not normalized\"

println(\"OK: loaded \", length(U_values), \" ground states, Hilbert space dim=\", size(target_vecs, 2), \", N=\", N)
"

echo "SMOKE TEST PASSED (output left in: $OUT_DIR)"
