#=
test_grow_loss_3x3.jl

Check what happens to the loss when growing from 1 to 2 exponentials on 3x3
=#

using Lattices
using LinearAlgebra
using SparseArrays
using JLD2
using HDF5

include(joinpath(@__DIR__, "..", "data_path.jl"))
include(joinpath(@__DIR__, "..", "utility_functions.jl"))
using .UtilityFunctions
include(joinpath(@__DIR__, "..", "trotter.jl"))
using .Trotter
include(joinpath(@__DIR__, "..", "ed_objects.jl"))
include(joinpath(@__DIR__, "..", "ed_functions.jl"))

folder = data_folder("N=(3, 3)_3x3")
old_file = joinpath(folder, "trotter_N=9_ref_slater_antihermitian_u_33.jld2")
old_data = JLD2.load(old_file)["dict"]
old_coeffs = old_data["coefficients"]
println("Old coeffs length: ", length(old_coeffs))
println("Old loss in file: ", get(old_data, "loss_metrics", get(old_data, "loss", "none")))

U_values, state_vecs, indexer, _, N_elec, _, _, sign_convention =
    load_ED_data(folder; verbose=false, sign_convention=:spin_first, use_slater_reference=true)

Lvec = parse_lattice_dimension(folder)
N_sites = prod(Lvec)
basis_sector = Trotter.get_basis_sector(indexer, Lvec, N_sites)
gates = Trotter.enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true, include_diagonal=false)
num_gates = length(gates)
println("Num gates: ", num_gates)

u_idx = 33
state1 = state_vecs[1, :] # Slater reference
state2 = state_vecs[u_idx, :] # Target ground state at U_idx=33

tau_terms = Trotter.fgateToTauSector(gates, N_sites, basis_sector; antihermitian=true)

# CPU ComplexF64:
loss_1_c64 = adjoint_loss(old_coeffs, gates, tau_terms, state1, state2, basis_sector, N_sites;
    num_exponentials=1, antihermitian=true, use_gpu=false, datatype=ComplexF64)
println("CPU ComplexF64 (num_exp=1, old_coeffs) loss: ", loss_1_c64)

grown_coeffs = grow_coefficients(old_coeffs, 1, 2, num_gates)
loss_2_c64 = adjoint_loss(grown_coeffs, gates, tau_terms, state1, state2, basis_sector, N_sites;
    num_exponentials=2, antihermitian=true, use_gpu=false, datatype=ComplexF64)
println("CPU ComplexF64 (num_exp=2, grown_coeffs) loss: ", loss_2_c64)

# Phase stripped Float32:
ref_stripped, _ = strip_global_phase(state1)
target_stripped, _ = strip_global_phase(state2)
loss_1_f32 = adjoint_loss(old_coeffs, gates, tau_terms, ref_stripped, target_stripped, basis_sector, N_sites;
    num_exponentials=1, antihermitian=true, use_gpu=false, datatype=Float32)
println("CPU Float32 (num_exp=1, old_coeffs) loss: ", loss_1_f32)

loss_2_f32 = adjoint_loss(grown_coeffs, gates, tau_terms, ref_stripped, target_stripped, basis_sector, N_sites;
    num_exponentials=2, antihermitian=true, use_gpu=false, datatype=Float32)
println("CPU Float32 (num_exp=2, grown_coeffs) loss: ", loss_2_f32)

# Unstripped Float32:
loss_1_f32_raw = adjoint_loss(old_coeffs, gates, tau_terms, state1, state2, basis_sector, N_sites;
    num_exponentials=1, antihermitian=true, use_gpu=false, datatype=Float32)
println("CPU Float32 raw unstripped (num_exp=1, old_coeffs) loss: ", loss_1_f32_raw)
