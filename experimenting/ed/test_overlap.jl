using JLD2, LinearAlgebra, SparseArrays
include("ed_objects.jl")
include("ed_functions.jl")
using Lattices
include("trotter.jl")
include("trotter_optimization.jl")
using .Trotter
using .TrotterOptimization

folder = "/home/jek354/research/data/new_data/data/N=(4, 4)_3x3_2"
U_values, target_vecs, indexer, _, N, _, use_symmetry, sign_convention = load_ED_data(folder; verbose=false, use_slater_reference=true, sign_convention=:spin_first)
shared_data = load(joinpath(folder, "trotter_N=9_shared.jld2"))["dict"]
iter_data = load(joinpath(folder, "trotter_N=9_u_10.jld2"))["dict"]
A_base = iter_data["coefficients"]
gates = shared_data["gates"]
state1 = target_vecs[1, :]
state2 = target_vecs[10, :]

dim_parsed = [3,3]
N_sites = 9
basis_ints = Trotter.get_basis_sector(indexer, dim_parsed, N_sites)
psi = TrotterOptimization.apply_unitary(A_base, gates, state1, basis_ints, N_sites, 1; antihermitian=false)

println("True Loss: ", 1 - abs2(state2' * psi))
println("Baseline: ", 1 - abs2(state2' * state1))
