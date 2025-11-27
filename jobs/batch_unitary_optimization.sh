#!/bin/bash
for level1 in 58; do
    for level2 in 58; do # 8 46 58
        for ((i=1; i<=8; i++)); do
            sbatch ./unitary_optimization.sub "$level1" "$level2" "$(( i * 5 - 4 ))" "$(( i * 5 ))"
        done
    done
done