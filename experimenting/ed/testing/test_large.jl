using Lattices
using LinearAlgebra
using Combinatorics
using SparseArrays
using Random

include("ed_objects.jl")
include("ed_functions.jl")

# Minimal mock for `HubbardSubspace` just for getting `indexer` since `run_ed_lanczos.jl` has complex deps
function make_indexer(Lx, Ly, N_up, N_down)
    lattice = Square((Lx, Ly), Periodic())
    hs = HubbardSubspace(N_up, N_down, lattice)
    CombinationIndexer(hs)
end

function test_large(Lx, Ly, N_up, N_down, order=1)
    println("\n──────────────────────────────────────────────────")
    println("Testing N=($N_up,$N_down)_$(Lx)x$(Ly) (Order $order)")
    
    t_indexer = @elapsed indexer = make_indexer(Lx, Ly, N_up, N_down)
    dim = length(indexer.inv_comb_dict)
    println("Indexer created in $(round(t_indexer, digits=2))s. Dim: $dim")

    t_dict = @elapsed t_opt = create_randomized_nth_order_operator(order, indexer; conserve_spin=true, omit_H_conj=false)
    println("Operator dict created in $(round(t_dict, digits=2))s. Keys: $(length(t_opt))")

    t_build = @elapsed rows, cols, signs, ops_list = build_n_body_structure(t_opt, indexer)
    println("Structure built in $(round(t_build, digits=2))s. Nonzeros: $(length(rows))")

    t_sparse = @elapsed begin
        param_index_map = build_param_index_map(ops_list, sort!(collect(keys(t_opt))))
        # t_keys might not match sort! exactly if we just use keys, but we map properly
        t_keys_sorted = sort!(collect(keys(t_opt)))
        vals = [signs[i] * t_opt[t_keys_sorted[param_index_map[i]]] for i in 1:length(signs)]
        H = sparse(rows, cols, vals, dim, dim)
    end
    
    println("Sparse matrix built in $(round(t_sparse, digits=2))s. Matrix density: $(length(rows)) elements.")
    if dim <= 50000
        diff_norm = norm(H - H')
        println("Hermiticity diff norm: $diff_norm")
        if diff_norm < 1e-10
            println("✓ Matrix is Hermitian!")
        else
            println("✗ Matrix is NOT Hermitian!")
        end
    else
        println("Is Hermitian? $(ishermitian(H))")
    end
end

test_large(4, 4, 3, 3, 1)
test_large(4, 4, 4, 4, 1)
test_large(4, 4, 5, 5, 1)
