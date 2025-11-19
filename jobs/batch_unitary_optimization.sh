for level1 in 8 46 58; do
    for level2 in 8 46 58; do
        for ((i=1; i<=8; i++)); do
            sh ./unitary_optimization.sub "$level1" "$level2" "$(( i * 5 - 4 ))" "$(( i * 5 ))"
        done
    done
done