#!/bin/bash

for ((i=1; i<=9; i++)); do
    sbatch ./unitary_optimization.sh true "$(( i * 5 - 4 ))" "$(( i * 5 ))"
    sbatch ./unitary_optimization.sh false "$(( i * 5 - 4 ))" "$(( i * 5 ))"
done

