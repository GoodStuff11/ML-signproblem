
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
using ExponentialUtilities


include("ed_objects.jl")
include("ed_functions.jl")
include("ed_optimization.jl")
include("utility_functions.jl")




function (@main)(ARGS)
    U_values = [0.00001; LinRange(2.1, 9, 20)]
    U_values = sort([U_values; 10.0 .^ LinRange(-3, 2, 40)])

    lattice_dimension = (3, 4)
    spin_polarized = true

    if spin_polarized
        N_up = 6
        N_down = 6
        N = (N_up, N_down)
    else
        N = 8
    end
    file_name = "/home/jek354/research/data/N=$(N)_" * join(lattice_dimension, "x")

    bc = "periodic"
    lattice = Square(lattice_dimension, if bc == "periodic"
        Periodic()
    else
        Open()
    end)

    hopping_model = HubbardModel(1.0, 0.0, 0.0, false)
    interaction_model = HubbardModel(0.0, 1.0, 0.0, false)

    if spin_polarized
        subspace = HubbardSubspace(N_up, N_down, lattice)
    else
        subspace = HubbardSubspace(N, lattice)
    end
    # 

    H_hopping, indexer = create_Hubbard(hopping_model, subspace; get_indexer=true)
    H_interaction = create_Hubbard(interaction_model, subspace; indexer=indexer)

    mapping = []
    s_mapping = []
    for kind in 1:2
        op = create_operator(subspace, :T, kind=kind)
        r, _, v = findnz(op)
        push!(mapping, r)
        push!(s_mapping, v)
    end


    # create symmetry projected states
    n_eigs = collect(lattice_dimension)
    all_eig_indices = collect(Iterators.product([1:n for n in lattice_dimension]...))

    all_full_eig_vecs = []
    all_E = []


    for (k, eig_indices) in enumerate(all_eig_indices)
        push!(all_E, [])

        eig_indices = collect(eig_indices)
        checked_indices, representative_indices, associated_representative, magnitude = find_representatives(size(H_hopping)[1], eig_indices, n_eigs, mapping, s_mapping)
        symmetrize(h) = construct_hamiltonian(
            findnz(h)...;
            checked_indices, representative_indices, associated_representative,
            magnitude, n_eigs, eig_indices
        )
        new_hopping = symmetrize(H_hopping)
        new_interaction = symmetrize(H_interaction)

        ops = []
        eig_values = []
        if !spin_polarized
            particle_n = subspace.N
            push!(ops, Hermitian(symmetrize(create_operator(subspace, :Sx))))
            push!(eig_values, -particle_n÷2:1:particle_n÷2)
            push!(ops, Hermitian(symmetrize(create_operator(subspace, :S2))))
            push!(eig_values, [s * (s + 1) for s in (particle_n%2)/2:1:particle_n/2])
        else
            push!(ops, Hermitian(symmetrize(create_operator(subspace, :S2))))
            particle_n = subspace.N_up + subspace.N_down
            push!(eig_values, [s * (s + 1) for s in (particle_n%2)/2:1:particle_n/2])
        end
        println("k=$k")

        all_eig_vecs = zeros(ComplexF64, length(U_values), length(representative_indices))


        targets = Float64[]
        should_project = false

        for (i, U) in enumerate(U_values)
            new_h = new_hopping + new_interaction * U
            E, H_vecs = eigsolve(new_h, rand(ComplexF64, size(new_h)[1]), 5, :SR, ishermitian=true)
            # println(E)

            vec_idx = nothing
            if i == 1
                vec_idx = 1

                # Check for degeneracy: |E0 - E1| < 1e-10
                if length(E) >= 2 && abs(E[1] - E[2]) < 1e-10
                    println("Degeneracy detected at U=$U. Fixing gauge.")
                    should_project = true

                    # Project to the first allowed eigenvalue for each operator
                    # and store them as targets
                    for (op, allowed_vals) in zip(ops, eig_values)
                        # Target the first allowed value (e.g. min Sz)
                        target = allowed_vals[1]
                        push!(targets, target)

                        # Find index of target in allowed_vals
                        target_idx = findfirst(x -> abs(x - target) < 1e-9, allowed_vals)
                        if target_idx === nothing
                            target_idx = 1
                        end

                        H_vecs[1] = project_hermitian(op, H_vecs[1], target_idx, collect(allowed_vals))
                        println(H_vecs[1]' * op * H_vecs[1])
                    end
                end
            else
                # Find vector with highest overlap with previous one
                prev_vec = all_eig_vecs[i-1, :]
                for k in eachindex(H_vecs)
                    if abs(H_vecs[k]' * prev_vec) > 0.9
                        vec_idx = k
                        break
                    end
                end
            end
            if vec_idx === nothing
                # println(E)
                # println([real(H_vecs[k]' * op * H_vecs[k]) for k in eachindex(H_vecs) for op in ops])
                # error("Could not find vector with overlap > 0.9 at U=$U")
                vec_idx = 1
            end

            # If we decided to project at i=1, enforce it at subsequent steps too
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

            push!(all_E[end], E[vec_idx])
            all_eig_vecs[i, :] = H_vecs[vec_idx]

            if i >= 2
                overlap = abs(all_eig_vecs[i, :]' * all_eig_vecs[i-1, :])
                if overlap < 0.9
                    error("error is bad: $overlap")
                end
                println("overlap: $U $overlap $(real(all_eig_vecs[i, :]' * ops[1]* all_eig_vecs[i, :])) $vec_idx")
            end
        end
        push!(all_full_eig_vecs, reconstruct_full_vector(
            all_eig_vecs,
            mapping, s_mapping, representative_indices, magnitude, eig_indices, n_eigs
        ))
    end

    # save data
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
        "E" => all_E,
        "all_full_eig_vecs" => all_full_eig_vecs,
        "indexer" => indexer,
        "mapping" => mapping,
        "s_mapping" => s_mapping,
        "all_eig_indices" => all_eig_indices,
    )


    save_energy_with_metadata(file_name, dict)

    return 0
end