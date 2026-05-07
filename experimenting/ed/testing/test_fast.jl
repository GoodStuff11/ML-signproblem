using Lattices
using LinearAlgebra
using Combinatorics
using SparseArrays
using Random

include("ed_objects.jl")
include("ed_functions.jl")

function make_indexer(Lx, Ly, N_up, N_down)
    lattice = Square((Lx, Ly), Periodic())
    hs = HubbardSubspace(N_up, N_down, lattice)
    CombinationIndexer(hs)
end

function test_large_fast(Lx, Ly, N_up, N_down, order=1, subset_keys=4)
    println("\n──────────────────────────────────────────────────")
    println("Testing N=($N_up,$N_down)_$(Lx)x$(Ly) (Order $order, with $subset_keys keys)")
    
    t_indexer = @elapsed indexer = make_indexer(Lx, Ly, N_up, N_down)
    dim = length(indexer.inv_comb_dict)
    println("Indexer created in $(round(t_indexer, digits=2))s. Dim: $dim")

    t_opt = create_randomized_nth_order_operator(order, indexer; conserve_spin=true, omit_H_conj=false)
    keys_arr = collect(keys(t_opt))
    
    subset = randperm(length(keys_arr))[1:min(subset_keys, length(keys_arr))]
    t_dict_small = Dict(keys_arr[i] => t_opt[keys_arr[i]] for i in subset)
    
    println("Operator dict subset created. Keys: $(length(t_dict_small))")

    t_build = @elapsed rows, cols, signs, ops_list = build_n_body_structure(t_dict_small, indexer; skip_lower_triangular=false)
    println("Structure built natively in $(round(t_build, digits=2))s. Nonzeros: $(length(rows))")
    
    if dim <= 5000000 # Under 5 million check local Hermiticity array
        param_index_map = build_param_index_map(ops_list, sort!(collect(keys(t_dict_small))))
        t_keys_sorted = sort!(collect(keys(t_dict_small)))
        vals = [signs[i] * t_dict_small[t_keys_sorted[param_index_map[i]]] for i in 1:length(signs)]
        H = sparse(rows, cols, vals, dim, dim)
        
        println("Sparse matrix generated. Density: $(length(rows))")
        diff_norm = norm(H - H')
        println("Hermiticity diff norm: $diff_norm")
        if diff_norm < 1e-10
            println("✓ Matrix is Hermitian!")
        else
            println("✗ Matrix is NOT Hermitian!")
        end
    else
        println("✓ Passed Large Scale System evaluation safely without OoM!")
    end
end

test_large_fast(4, 4, 4, 4, 1, 4)
test_large_fast(4, 4, 5, 5, 1, 4)
