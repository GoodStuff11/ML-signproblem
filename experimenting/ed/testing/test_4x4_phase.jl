#=
test_4x4_phase.jl
=#

using Lattices
using LinearAlgebra
using SparseArrays
using HDF5

include(joinpath(@__DIR__, "..", "data_path.jl"))
include(joinpath(@__DIR__, "..", "utility_functions.jl"))
using .UtilityFunctions
include(joinpath(@__DIR__, "..", "trotter.jl"))
using .Trotter
include(joinpath(@__DIR__, "..", "ed_objects.jl"))
include(joinpath(@__DIR__, "..", "ed_functions.jl"))

folder = data_folder("N=(6, 6)_4x4")
U_values, state_vecs, indexer, _, N_elec, _, _, sign_convention =
    load_ED_data(folder; verbose=false, sign_convention=:spin_first, use_slater_reference=true)

u_idx = 33 # U=8.0
state1 = state_vecs[1, :] # Slater
state2 = state_vecs[u_idx, :] # Ground state at U=8.0

println("state1 length: ", length(state1))
println("state2 length: ", length(state2))
println("state1 norm: ", norm(state1))
println("state2 norm: ", norm(state2))

ref_stripped, phase_ref = strip_global_phase(state1)
target_stripped, phase_tgt = strip_global_phase(state2)

println("phase_ref: ", phase_ref)
println("phase_tgt: ", phase_tgt)

max_imag_ref = maximum(abs, imag.(state1 .* conj(phase_ref)))
max_imag_tgt = maximum(abs, imag.(state2 .* conj(phase_tgt)))
println("max imag in ref: ", max_imag_ref)
println("max imag in target: ", max_imag_tgt)

println("norm of stripped ref: ", norm(ref_stripped))
println("norm of stripped target: ", norm(target_stripped))
