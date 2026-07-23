using JLD2, HDF5, LinearAlgebra
include("TamFermion.jl")
include("utility_functions.jl")
include("indexer.jl")
include("trotter.jl")
include("trotter_optimization.jl")

using .Trotter
using .TrotterOptimization

folder = "/home/jek354/research/data/new_data/data_new_sign/N=(4, 3)_3x3"
data = JLD2.load(joinpath(folder, "trotter_N=9_ref_slater_antihermitian_u_60.jld2"))["dict"]
A_base = data["coefficients"]

println("Loading HDF5 data...")
ref_file = joinpath(folder, "HubbardED_Slater_3x3_(4,3)_t_1_m_2.h5")
f = h5open(ref_file, "r")
state1 = read(f, "ground_state")
state2 = read(f, "interaction_sweep/states/U_15.0")
close(f)

println("Loading indexer...")
idx_data = JLD2.load(joinpath(folder, "indexer_N=(4, 3)_3x3.jld2"))
indexer = idx_data["indexer"]

basis_ints = Trotter.get_basis_sector(indexer, (3,3), 9)
gates = Trotter.enumerate_ferm_excitations(2, (3,3); conserve_mom=true, conserve_sz=true, include_diagonal=true)

stored_num_exp = length(A_base) ÷ length(gates)

println("Original A_base max abs: ", maximum(abs.(A_base)))

A_pruned = copy(A_base)
A_pruned[abs.(A_pruned) .< 1.0] .= 0.0

psi_full = TrotterOptimization.apply_unitary(A_base, gates, state1, basis_ints, 9, stored_num_exp; antihermitian=true)
overlap_full = state2' * psi_full
loss_full = 1 - abs2(overlap_full)
println("Loss (Full): ", loss_full)

psi_pruned = TrotterOptimization.apply_unitary(A_pruned, gates, state1, basis_ints, 9, stored_num_exp; antihermitian=true)
overlap_pruned = state2' * psi_pruned
loss_pruned = 1 - abs2(overlap_pruned)
println("Loss (Pruned): ", loss_pruned)

println("psi_full ≈ psi_pruned? ", isapprox(psi_full, psi_pruned))
println("state1 ≈ psi_full? ", isapprox(state1, psi_full))
println("Loss state1: ", 1 - abs2(state2' * state1))
