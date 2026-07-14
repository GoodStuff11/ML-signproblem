#=
run_ed_lanczos_momentum.jl

Run exact diagonalization using Lanczos with momentum sector projections.

Usage:
  julia --project=.. run_ed_lanczos_momentum.jl [Lx] [Ly] [N_up] [N_down] [save_dir]

Arguments:
  Lx (optional): Lattice width (default: 2)
  Ly (optional): Lattice height (default: 2)
  N_up (optional): Number of spin-up electrons (default: 2)
  N_down (optional): Number of spin-down electrons (default: 2)
  save_dir (optional): Subdirectory where the output is saved, relative to the script directory (default: "data")

Examples:
  julia --project=.. run_ed_lanczos_momentum.jl 4 3 4 4 nn_test_data
  julia --project=.. run_ed_lanczos_momentum.jl 2 2 2 2 data
=#

using Lattices
using LinearAlgebra
using Combinatorics
using SparseArrays
using Plots
import Graphs
using LaTeXStrings
using Statistics
using Random
using JSON
using JLD2
using KrylovKit
using Zygote
using ExponentialUtilities
using CUDA
using Dates

include("utility_functions.jl")
include("ed_objects.jl")
include("ed_functions.jl")
include("ed_optimization.jl")
include("logging.jl")


"""
    parse_arguments(args::Vector{String})

Parse command line arguments for running Lanczos momentum.
Expected arguments:
1. Lx (Int): Lattice X dimension. Default: 2
2. Ly (Int): Lattice Y dimension. Default: 2
3. N_up (Int): Number of spin-up electrons. Default: 2
4. N_down (Int): Number of spin-down electrons. Default: 2
5. save_dir (String): Directory to save the output files. Default: "data"
"""
function parse_arguments(args::Vector{String})
    Lx = 2
    Ly = 2
    N_up = 2
    N_down = 2
    save_dir = "data"

    if length(args) >= 1
        Lx = parse(Int, args[1])
    end
    if length(args) >= 2
        Ly = parse(Int, args[2])
    end
    if length(args) >= 3
        N_up = parse(Int, args[3])
    end
    if length(args) >= 4
        N_down = parse(Int, args[4])
    end
    if length(args) >= 5
        save_dir = args[5]
    end

    return Lx, Ly, N_up, N_down, save_dir
end


function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "run_ed_lanczos_momentum")
    with_logging(log_path) do

    Lx, Ly, N_up_val, N_down_val, save_dir = parse_arguments(ARGS)

    U_values = [0.00001; LinRange(2.1, 9, 20)]
    U_values = sort([U_values; 10.0 .^ LinRange(-3, 2, 40)])

    sign_convention = :spin_first

    lattice_dimension = (Lx, Ly)
    spin_polarized = true

    if spin_polarized
        N_up = N_up_val
        N_down = N_down_val
        N = (N_up, N_down)
    else
        N = 6
    end
    file_name = joinpath(@__DIR__, save_dir, "N=$(N)_" * join(lattice_dimension, "x"))
    bc = "periodic"
    lattice = Square(lattice_dimension, if bc == "periodic"
        Periodic()
    else
        Open()
    end)

    hopping_model = HubbardModel(1.0, 0.0, 0.0, false)
    interaction_model = HubbardModel(0.0, 1.0, 0.0, false)

    n_eigs = collect(lattice_dimension)
    all_eig_indices = collect(Iterators.product([1:n for n in lattice_dimension]...))
    n_sectors = length(all_eig_indices)

    # Pre-allocate result arrays so parallel writes go to distinct indices (thread-safe)
    all_full_eig_vecs = Vector{Any}(undef, n_sectors)
    all_E = Vector{Any}(undef, n_sectors)
    all_indexers = Vector{Any}(undef, n_sectors)

    @safe_threads for k in 1:n_sectors
        eig_indices = collect(all_eig_indices[k])
        k_tuple = Tuple(eig_indices)

        if spin_polarized
            subspace = HubbardSubspace(N_up, N_down, lattice; k=k_tuple)
        else
            subspace = HubbardSubspace(N, lattice; k=k_tuple)
        end

        dim = get_subspace_dimension(subspace)
        if dim == 0
            all_full_eig_vecs[k] = []
            all_E[k] = []
            all_indexers[k] = nothing
            continue
        end

        new_hopping, indexer = create_Hubbard(hopping_model, subspace; get_indexer=true, momentum_basis=true, sign_convention=sign_convention)
        new_interaction = create_Hubbard(interaction_model, subspace; indexer=indexer, momentum_basis=true, sign_convention=sign_convention)

        ops = []
        eig_values = []
        if !spin_polarized
            particle_n = subspace.N
            push!(ops, Hermitian(create_operator(subspace, :Sx; momentum_basis=true, sign_convention=sign_convention)))
            push!(eig_values, -particle_n÷2:1:particle_n÷2)
            push!(ops, Hermitian(create_operator(subspace, :S2; momentum_basis=true, sign_convention=sign_convention)))
            push!(eig_values, [s * (s + 1) for s in (particle_n%2)/2:1:particle_n/2])
        else
            push!(ops, Hermitian(create_operator(subspace, :S2; momentum_basis=true, sign_convention=sign_convention)))
            particle_n = subspace.N_up + subspace.N_down
            push!(eig_values, [s * (s + 1) for s in (particle_n%2)/2:1:particle_n/2])
        end
        println("k=$k")

        all_eig_vecs = zeros(ComplexF64, length(U_values), dim)
        E_values = Vector{Float64}(undef, length(U_values))

        targets = Float64[]
        should_project = false

        for (i, U) in enumerate(U_values)
            new_h = new_hopping + new_interaction * U
            E, H_vecs = eigsolve(new_h, rand(ComplexF64, size(new_h)[1]), 5, :SR, ishermitian=true)

            vec_idx = nothing
            if i == 1
                vec_idx = 1

                if length(E) >= 2 && abs(E[1] - E[2]) < 1e-10
                    println("Degeneracy detected at U=$U. Fixing gauge.")
                    should_project = true

                    for (op, allowed_vals) in zip(ops, eig_values)
                        target = allowed_vals[1]
                        push!(targets, target)

                        target_idx = findfirst(x -> abs(x - target) < 1e-9, allowed_vals)
                        if target_idx === nothing
                            target_idx = 1
                        end

                        H_vecs[1] = project_hermitian(op, H_vecs[1], target_idx, collect(allowed_vals))
                        println(H_vecs[1]' * op * H_vecs[1])
                    end
                end

                println("overlap: $U $(real(H_vecs[1]' * ops[1] * real(H_vecs[1])))")
            else
                # Adiabatic tracking: sequential dependency, cannot parallelize over U
                prev_vec = all_eig_vecs[i-1, :]
                for ki in eachindex(H_vecs)
                    if abs(H_vecs[ki]' * prev_vec) > 0.9
                        vec_idx = ki
                        break
                    end
                end
            end

            if vec_idx === nothing
                vec_idx = 1
            end

            if should_project && i > 1
                for (j, op) in enumerate(ops)
                    target = targets[j]
                    allowed = collect(eig_values[j])
                    target_idx = findfirst(x -> abs(x - target) < 1e-9, allowed)
                    if target_idx !== nothing
                        H_vecs[vec_idx] = project_hermitian(op, H_vecs[vec_idx], target_idx, allowed)
                        println(H_vecs[vec_idx]' * op * H_vecs[vec_idx])
                    end
                end
            end

            E_values[i] = E[vec_idx]
            all_eig_vecs[i, :] .= H_vecs[vec_idx]

            if i >= 2
                overlap = abs(all_eig_vecs[i, :]' * all_eig_vecs[i-1, :])
                if overlap < 0.8
                    println(E)
                    eig_value = real(all_eig_vecs[i, :]' * ops[1] * all_eig_vecs[i, :])
                    if !isapprox(eig_value, round(eig_value), atol=1e-9)
                        println("overlap: $U $overlap $(eig_value) $vec_idx")
                        error("error is bad: $overlap")
                    end
                end
                println("overlap: $U $overlap $(real(all_eig_vecs[i, :]' * ops[1] * all_eig_vecs[i, :])) $vec_idx")
            end
        end

        # Write results to pre-allocated slots — each k writes to a distinct index
        all_full_eig_vecs[k] = all_eig_vecs
        all_E[k] = E_values
        all_indexers[k] = indexer
        # println("k: $k, indexer: $(indexer.inv_comb_dict[1])")
        # println("k: $k, all_indexers: $(all_indexers[max(k-1, 1)].inv_comb_dict[1])")
    end
    for k in eachindex(all_indexers)
        if all_indexers[k] !== nothing
            println("k: $k, indexer: $(all_indexers[k].inv_comb_dict[1])")
        end
    end
    selected_index = find_best_energy_sector(all_E, U_values)

    # --- Precompute operator structures for optimization ---
    println("Precomputing n_body_structure for optimization...")
    main_indexer = all_indexers[selected_index]
    precomputed_structures = Dict()
    if main_indexer !== nothing
        precomputed_structures = precompute_n_body_structures(main_indexer, 2; spin_conserved=!isa(N, Number), momentum_basis=true, sign_convention=sign_convention)
    end

    meta_data = Dict(
        "electron count" => N,
        "sites" => join(lattice_dimension, "x"),
        "bc" => bc,
        "basis" => "adiabatic",
        "solver" => "Lanczos",
        "U_values" => U_values,
        "maxiters" => 200,
        "optimizer" => "BFGS"
    )

    # dict = Dict(
    #     "meta_data" => meta_data,
    #     "E" => [all_E[selected_index]],
    #     "all_full_eig_vecs" => [all_full_eig_vecs[selected_index]],
    #     "indexer" => [all_indexers[selected_index]],
    #     "all_eig_indices" => [all_eig_indices[selected_index]],
    #     "precomputed_structures" => precomputed_structures
    # )
    dict = Dict(
        "meta_data" => meta_data,
        "E" => all_E,
        "all_full_eig_vecs" => all_full_eig_vecs,
        "indexer" => all_indexers,
        "all_eig_indices" => all_eig_indices,
        "precomputed_structures" => precomputed_structures
    )

    save_energy_with_metadata(file_name, dict)
    println("saved to: $file_name")

    return 0
    end # with_logging
end