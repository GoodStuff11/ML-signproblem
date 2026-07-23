using Lattices
using HDF5
using JLD2, LinearAlgebra, SparseArrays

include("trotter.jl")
using .Trotter
include("utility_functions.jl")
include("ed_objects.jl")
include("ed_functions.jl")
include("data_path.jl")




use_ref = false
antihermitian = true
println("use_ref: $use_ref antihermitian: $antihermitian")
for folder in readdir(get_data_root())
    dim_parsed = parse_lattice_dimension(folder)
    N_ud = parse_electron_count(folder)

    N_sites = prod(dim_parsed)

    if N_sites > 10 # speed up program.
        continue
    end
    ref = use_ref ? "ref_slater_" : ""
    ah = antihermitian ? "antihermitian_" : ""

    i = 25
    full_folder_name = data_folder(folder)
    local shared_data, target_vecs, indexer, iter_data

    try
        _, target_vecs, indexer, _, _, _, _, _ = load_ED_data(full_folder_name; verbose=false, use_slater_reference=use_ref, sign_convention=:spin_first)
        shared_data = load(joinpath(full_folder_name, "trotter_N=$(N_sites)_$(ref)$(ah)shared.jld2"))["dict"]
        println("folder: $folder")

        iter_data = load(joinpath(full_folder_name, "trotter_N=$(N_sites)_$(ref)$(ah)u_$i.jld2"))["dict"]
    catch e
        continue
    end
    A_base = iter_data["coefficients"]
    gates = shared_data["gates"]
    # gates = Trotter.enumerate_ferm_excitations(2, dim_parsed; conserve_mom=true, conserve_sz=true, include_diagonal=!antihermitian)
    state1 = target_vecs[1, :]
    state2 = target_vecs[i, :]
    baseline = 1 - abs2(state1' * state2)
    println("baseline: |<ref|target>|^2 = $baseline")

    basis_ints = Trotter.get_basis_sector(indexer, dim_parsed, N_sites)
    loss = Trotter.adjoint_loss(A_base .* 0, gates, nothing, state1, state2, basis_ints, N_sites; num_exponentials=length(A_base) ÷ length(gates), antihermitian=antihermitian)
    println("Optimized loss: $loss")
    println()
end