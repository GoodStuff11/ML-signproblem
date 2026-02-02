
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

    lattice_dimension = (4, 3)
    spin_polarized = false
    file_name = "data/N=6_4x3"
    if spin_polarized
        N_up = 2
        N_down = 2
        N = (N_up, N_down)
    else
        N = 6
    end


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
        if subspace.N >= 1
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
        # diagonalize
        for (i, U) in enumerate(U_values)
            new_h = new_hopping + new_interaction * U
            E, H_vecs = eigsolve(new_h, rand(ComplexF64, size(new_h)[1]), 1, :SR, ishermitian=true)

            # here we pick the lowest energy state. It's typically in a S^2 eigenstate,
            # but only when N is odd is it not an Sx eigenstate. Thus we project to Sx.
            if !spin_polarized
                H_vec = project_hermitian(ops[1], H_vecs[1], (N + 1) ÷ 2, collect((-N/2):(N/2)))
            else
                H_vec = H_vecs[1]
            end

            push!(all_E[end], E[1])
            all_eig_vecs[i, :] = H_vec
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