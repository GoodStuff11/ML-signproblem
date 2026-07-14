#!/bin/bash
# submit_optimization_sweep_pure_mild.sh
# Loop through 8 experiment configurations (4 U values x nn init x 2 losses) using the pure_mild_mid_narrow model.

U_VALUES=(25 39 49 53)
INITS=("nn")
LOSSES=("overlap" "energy")
MODEL_PATH="trained_neural_networks/trained_neural_network_pure_mild_mid_narrow.jld2"

for u in "${U_VALUES[@]}"; do
    for init in "${INITS[@]}"; do
        for loss in "${LOSSES[@]}"; do
            JOB_NAME="opt_pure_mild_u${u}_${loss}"
            OUT_FILE="/home/jek354/research/ML-signproblem/jobs/opt_pure_mild_u${u}_${loss}_%j.out"
            ERR_FILE="/home/jek354/research/ML-signproblem/jobs/opt_pure_mild_u${u}_${loss}_%j.err"
            
            CMD="cd /home/jek354/research/ML-signproblem/experimenting/ed/ && julia --project=.. run_optimization_experiments.jl $MODEL_PATH --u-idx=$u --init=$init --loss=$loss"
            
            echo "Submitting: $JOB_NAME"
            sbatch -J "$JOB_NAME" \
                   -o "$OUT_FILE" \
                   -e "$ERR_FILE" \
                   -N 1 \
                   -n 4 \
                   --mem=20G \
                   -t 1-00:00:00 \
                   --partition=kim \
                   --wrap="$CMD"
                   
            # Brief sleep to prevent scheduler contention
            sleep 1
        done
    done
done

echo "All 8 jobs submitted successfully!"
