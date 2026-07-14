#!/bin/bash
# submit_aimi_12.sh
# Resubmit the last 12 jobs on partition aimi using node aimi-cpu-01.
# This script should be run from the directory: /home/jek354/research/ML-signproblem/experimenting/ed

FOLDERS=(
    "N=(4, 5)_3x3_2"
    "N=(4, 5)_3x3_3"
    "N=(4, 5)_4x4"
    "N=(5, 5)_4x3"
    "N=(5, 5)_4x4"
    "N=(6, 6)_4x3"
)
LOSSES=("overlap" "energy")
JOBS_DIR="/home/jek354/research/ML-signproblem/jobs"

for folder in "${FOLDERS[@]}"; do
    for loss in "${LOSSES[@]}"; do
        job_name="trotter_${folder}_${loss}"
        # Sanitize folder name for filenames: replace space with _, = with _, (, ) with empty, , with _
        safe_name=$(echo "$folder" | sed 's/ /_/g' | sed 's/=/_/g' | sed 's/(//g' | sed 's/)//g' | sed 's/,/_/g')
        out_log="${JOBS_DIR}/trotter_${safe_name}_${loss}.out"
        err_log="${JOBS_DIR}/trotter_${safe_name}_${loss}.err"
        
        cmd_str="julia --project=.. run_trotter_scan_optimization.jl \"data/${folder}\" 60 2 --maxiters=300 --loss=${loss}"
        
        echo "Submitting job for folder='${folder}', loss='${loss}' on aimi (aimi-cpu-01)..."
        sbatch --mem=60G \
               --cpus-per-task=60 \
               --time=7-00:00:00 \
               --partition=aimi \
               -w aimi-cpu-01 \
               --job-name="${job_name}" \
               --output="${out_log}" \
               --error="${err_log}" \
               --wrap="${cmd_str}"
        sleep 0.5
    done
done
