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
state1 = target_vecs[1, :]
state2 = target_vecs[10, :]
println("Baseline: ", 1 - abs2(state2' * state1))
