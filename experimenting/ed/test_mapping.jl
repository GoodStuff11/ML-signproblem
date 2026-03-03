using Lattices
using LinearAlgebra
using Combinatorics
using SparseArrays
using Plots
using Zygote
using Optimization, OptimizationOptimisers
using JSON
using OptimizationOptimJL
using JLD2
using ExponentialUtilities

include("ed_objects.jl")
include("ed_functions.jl")
include("ed_optimization.jl")

println("Setting up small system (2x2)...")
lattice_small = Square((2, 2), Periodic())
subspace_small = HubbardSubspace(2, 2, lattice_small)

println("Setting up large system (2x4)...")
lattice_large = Square((2, 4), Periodic())
subspace_large = HubbardSubspace(2, 2, lattice_large)

println("Generating dummy t_vals_small...")
Lx_small, Ly_small = size(lattice_small)
indexer_small = CombinationIndexer(subspace_small)
t_dict_small = create_randomized_nth_order_operator(1, indexer_small; omit_H_conj=true, conserve_spin=true)
t_keys_small = collect(keys(t_dict_small))
sym_small = find_symmetry_groups(t_keys_small, Lx_small, Ly_small; trans_x=true, trans_y=true, spin_symmetry=true, hermitian=true)

# Assign random values to small system parameter groups
t_vals_small = rand(length(sym_small[1]))

println("Testing wrapper function...")
my_cache = Dict{Symbol,Any}()
t_vals_large, index_mapping, t_keys_large, sym_large = map_symmetry_groups(t_vals_small, subspace_small, subspace_large;
    order=1, antihermitian=false, spin_symmetry=true, trans_x=true, trans_y=true, conserve_spin=true, omit_H_conj=true, cache=my_cache)

println("t_vals_large length: ", length(t_vals_large))
println("Number of non-zero entries in mapped t_vals: ", sum(t_vals_large .!= 0.0))
println("Index mapping non-zero entries: ", sum(index_mapping .> 0))

println(t_vals_small)
println(t_vals_large)

println("\nTesting mapped_unitary_loss...")
dim_large = length(my_cache[:indexer_large].inv_comb_dict)
state1 = normalize!(randn(ComplexF64, dim_large))
state2 = normalize!(randn(ComplexF64, dim_large))

# First call (generates sparse dict structure)
loss1 = mapped_unitary_loss(t_vals_small, state1, state2, subspace_small, subspace_large;
    order=1, antihermitian=false, spin_symmetry=true, trans_x=true, trans_y=true, conserve_spin=true, omit_H_conj=true, cache=my_cache)

# Second call (utilizes cache)
loss2 = mapped_unitary_loss(t_vals_small, state1, state2, subspace_small, subspace_large;
    order=1, antihermitian=false, spin_symmetry=true, trans_x=true, trans_y=true, conserve_spin=true, omit_H_conj=true, cache=my_cache)

println("Loss 1: ", loss1)
println("Loss 2 (Cached): ", loss2)

