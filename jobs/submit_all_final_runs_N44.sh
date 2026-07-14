#!/bin/bash
# submit_all_final_runs_N44.sh
# Loop through all 24 configurations (4 U values x 2 losses x 3 initializations)
# and submit them as CPU-only jobs with 20 threads each.

U_VALUES=(25 39 49 53)
LOSSES=("overlap" "energy")
FOLDER="nn_test_data/N=(4, 4)_3x3"

for u in "${U_VALUES[@]}"; do
    for loss in "${LOSSES[@]}"; do
        # 1. Random multi-start
        JOB_NAME="opt_n44_random_u${u}_${loss}"
        OUT_FILE="/home/jek354/research/ML-signproblem/jobs/opt_n44_random_u${u}_${loss}_%j.out"
        ERR_FILE="/home/jek354/research/ML-signproblem/jobs/opt_n44_random_u${u}_${loss}_%j.err"
        CMD="cd /home/jek354/research/ML-signproblem/experimenting/ed/ && julia --project=.. --threads=auto run_optimization_experiments.jl trained_neural_networks/trained_neural_network_pure_mild_mid_narrow.jld2 --u-idx=$u --init=random --loss=$loss --use-gpu=false --folder='$FOLDER'"
        
        echo "Submitting: $JOB_NAME"
        sbatch -J "$JOB_NAME" \
               -o "$OUT_FILE" \
               -e "$ERR_FILE" \
               -N 1 \
               --cpus-per-task=20 \
               --mem=20G \
               -t 3-00:00:00 \
               --partition=kim \
               --wrap="$CMD"
        sleep 0.1

        # 2. Neural Network: 2x2_only_one_minus_loss_power_10
        JOB_NAME="opt_n44_nn_2x2_u${u}_${loss}"
        OUT_FILE="/home/jek354/research/ML-signproblem/jobs/opt_n44_nn_2x2_u${u}_${loss}_%j.out"
        ERR_FILE="/home/jek354/research/ML-signproblem/jobs/opt_n44_nn_2x2_u${u}_${loss}_%j.err"
        CMD="cd /home/jek354/research/ML-signproblem/experimenting/ed/ && julia --project=.. --threads=auto run_optimization_experiments.jl trained_neural_networks/trained_neural_network_2x2_only_one_minus_loss_power_10.jld2 --u-idx=$u --init=nn --loss=$loss --use-gpu=false --folder='$FOLDER'"
        
        echo "Submitting: $JOB_NAME"
        sbatch -J "$JOB_NAME" \
               -o "$OUT_FILE" \
               -e "$ERR_FILE" \
               -N 1 \
               --cpus-per-task=20 \
               --mem=20G \
               -t 3-00:00:00 \
               --partition=kim \
               --wrap="$CMD"
        sleep 0.1

        # 3. Neural Network: pure_mild_mid_narrow
        JOB_NAME="opt_n44_nn_pure_mild_u${u}_${loss}"
        OUT_FILE="/home/jek354/research/ML-signproblem/jobs/opt_n44_nn_pure_mild_u${u}_${loss}_%j.out"
        ERR_FILE="/home/jek354/research/ML-signproblem/jobs/opt_n44_nn_pure_mild_u${u}_${loss}_%j.err"
        CMD="cd /home/jek354/research/ML-signproblem/experimenting/ed/ && julia --project=.. --threads=auto run_optimization_experiments.jl trained_neural_networks/trained_neural_network_pure_mild_mid_narrow.jld2 --u-idx=$u --init=nn --loss=$loss --use-gpu=false --folder=$FOLDER"
        
        echo "Submitting: $JOB_NAME"
        sbatch -J "$JOB_NAME" \
               -o "$OUT_FILE" \
               -e "$ERR_FILE" \
               -N 1 \
               --cpus-per-task=20 \
               --mem=20G \
               -t 3-00:00:00 \
               --partition=kim \
               --wrap="$CMD"
        sleep 0.1
    done
done

echo "All 24 N=(4, 4)_3x3 jobs submitted successfully!"
