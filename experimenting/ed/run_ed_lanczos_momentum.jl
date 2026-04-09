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


include("ed_objects.jl")
include("ed_functions.jl")
include("ed_optimization.jl")
include("utility_functions.jl")


function (@main)(ARGS)
    U_values = [0.00001; LinRange(2.1, 9, 20)]
    U_values = sort([U_values; 10.0 .^ LinRange(-3, 2, 40)])

    lattice_dimension = (4, 4)
    spin_polarized = true

    if spin_polarized
        N_up = 5
        N_down = 5
        N = (N_up, N_down)
    else
        N = 6
    end
    file_name = joinpath(@__DIR__, "data", "N=$(N)_" * join(lattice_dimension, "x"))
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
    all_E             = Vector{Any}(undef, n_sectors)
    all_indexers      = Vector{Any}(undef, n_sectors)

    Threads.@threads for k in 1:n_sectors
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
            all_E[k]             = []
            all_indexers[k]      = nothing
            continue
        end

        new_hopping, indexer = create_Hubbard(hopping_model, subspace; get_indexer=true, momentum_basis=true)
        new_interaction = create_Hubbard(interaction_model, subspace; indexer=indexer, momentum_basis=true)

        ops = []
        eig_values = []
        if !spin_polarized
            particle_n = subspace.N
            push!(ops, Hermitian(create_operator(subspace, :Sx; momentum_basis=true)))
            push!(eig_values, -particle_n÷2:1:particle_n÷2)
            push!(ops, Hermitian(create_operator(subspace, :S2; momentum_basis=true)))
            push!(eig_values, [s * (s + 1) for s in (particle_n%2)/2:1:particle_n/2])
        else
            push!(ops, Hermitian(create_operator(subspace, :S2; momentum_basis=true)))
            particle_n = subspace.N_up + subspace.N_down
            push!(eig_values, [s * (s + 1) for s in (particle_n%2)/2:1:particle_n/2])
        end
        println("k=$k")

        all_eig_vecs = zeros(ComplexF64, length(U_values), dim)
        E_values     = Vector{Float64}(undef, length(U_values))

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

            E_values[i]          = E[vec_idx]
            all_eig_vecs[i, :]  .= H_vecs[vec_idx]

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
        all_E[k]             = E_values
        all_indexers[k]      = indexer
    end

    selected_index = 1
    minimum_energy = Inf
    for i in eachindex(all_E)
        if !isempty(all_E[i]) && all_E[i][30] < minimum_energy
            minimum_energy = all_E[i][30]
            selected_index = i
        end
    end

    # --- Precompute operator structures for optimization ---
    println("Precomputing n_body_structure for optimization...")
    main_indexer = all_indexers[selected_index]
    precomputed_structures = Dict()
    if main_indexer !== nothing
        precomputed_structures = precompute_n_body_structures(main_indexer, 2; spin_conserved=!isa(N, Number), momentum_basis=true)
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

    dict = Dict(
        "meta_data" => meta_data,
        "E" => [all_E[selected_index]],
        "all_full_eig_vecs" => [all_full_eig_vecs[selected_index]],
        "indexer" => [all_indexers[selected_index]],
        "all_eig_indices" => [all_eig_indices[selected_index]],
        "precomputed_structures" => precomputed_structures
    )

    save_energy_with_metadata(file_name, dict)

    return 0
end