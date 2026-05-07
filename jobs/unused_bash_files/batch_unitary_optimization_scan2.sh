#!/bin/bash
for ((i=26; i<=61; i++)); do
    sbatch ./unitary_optimization_scan.sh "$i"
done

