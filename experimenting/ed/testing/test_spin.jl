include("testing/benchmark_operator_construction.jl")
indexer = make_indexer(3, 2, 3, 3)
sz1 = length(indexer.inv_comb_dict)
println("Size of indexer: ", sz1)
t_dict = create_randomized_nth_order_operator(1, indexer; conserve_spin=true)
for k in keys(t_dict)
    c, a = k[1:length(k)÷2], k[length(k)÷2+1:end]
    sum_c = sum((s[2] * 2 - 3) for s in c)
    sum_a = sum((s[2] * 2 - 3) for s in a)
    if sum_c != sum_a
        println("FAILED: ", k)
    end
end
rows, cols, signs, ops_list = build_n_body_structure(t_dict, indexer)
println("Success!")
