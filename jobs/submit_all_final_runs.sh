#!/bin/bash
# submit_all_final_runs.sh
# Loop through all 24 configurations (4 U values x 2 losses x 3 initializations)
# and submit them as GPU jobs on partition kim.

U_VALUES=(25 39 49 53)
LOSSES=("overlap" "energy")

for u in "${U_VALUES[@]}"; do
    for loss in "${LOSSES[@]}"; do
        # 1. Random multi-start
        JOB_NAME="opt_random_u${u}_${loss}"
        OUT_FILE="/home/jek354/research/ML-signproblem/jobs/opt_random_u${u}_${loss}_%j.out"
        ERR_FILE="/home/jek354/research/ML-signproblem/jobs/opt_random_u${u}_${loss}_%j.err"
        CMD="cd /home/jek354/research/ML-signproblem/experimenting/ed/ && julia --project=.. run_optimization_experiments.jl trained_neural_networks/trained_neural_network_pure_mild_mid_narrow.jld2 --u-idx=$u --init=random --loss=$loss --use-gpu=true"
        
        echo "Submitting: $JOB_NAME"
        sbatch -J "$JOB_NAME" \
               -o "$OUT_FILE" \
               -e "$ERR_FILE" \
               -N 1 \
               --gres=gpu:1 \
               --exclude=kim-compute-01 \
               --mem=20G \
               -t 3-00:00:00 \
               --partition=kim \
               --wrap="$CMD"
        sleep 1

        # 2. Neural Network: 2x2_only_one_minus_loss_power_10
        JOB_NAME="opt_nn_2x2_u${u}_${loss}"
        OUT_FILE="/home/jek354/research/ML-signproblem/jobs/opt_nn_2x2_u${u}_${loss}_%j.out"
        ERR_FILE="/home/jek354/research/ML-signproblem/jobs/opt_nn_2x2_u${u}_${loss}_%j.err"
        CMD="cd /home/jek354/research/ML-signproblem/experimenting/ed/ && julia --project=.. run_optimization_experiments.jl trained_neural_networks/trained_neural_network_2x2_only_one_minus_loss_power_10.jld2 --u-idx=$u --init=nn --loss=$loss --use-gpu=true"
        
        echo "Submitting: $JOB_NAME"
        sbatch -J "$JOB_NAME" \
               -o "$OUT_FILE" \
               -e "$ERR_FILE" \
               -N 1 \
               --gres=gpu:1 \
               --exclude=kim-compute-01 \
               --mem=20G \
               -t 3-00:00:00 \
               --partition=kim \
               --wrap="$CMD"
        sleep 1

        # 3. Neural Network: pure_mild_mid_narrow
        JOB_NAME="opt_nn_pure_mild_u${u}_${loss}"
        OUT_FILE="/home/jek354/research/ML-signproblem/jobs/opt_nn_pure_mild_u${u}_${loss}_%j.out"
        ERR_FILE="/home/jek354/research/ML-signproblem/jobs/opt_nn_pure_mild_u${u}_${loss}_%j.err"
        CMD="cd /home/jek354/research/ML-signproblem/experimenting/ed/ && julia --project=.. run_optimization_experiments.jl trained_neural_networks/trained_neural_network_pure_mild_mid_narrow.jld2 --u-idx=$u --init=nn --loss=$loss --use-gpu=true"
        
        echo "Submitting: $JOB_NAME"
        sbatch -J "$JOB_NAME" \
               -o "$OUT_FILE" \
               -e "$ERR_FILE" \
               -N 1 \
               --gres=gpu:1 \
               --exclude=kim-compute-01 \
               --mem=20G \
               -t 3-00:00:00 \
               --partition=kim \
               --wrap="$CMD"
        sleep 1
    done
done

echo "All 24 jobs submitted successfully!"
