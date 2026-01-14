
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


include("ed_objects.jl")
include("ed_functions.jl")
include("ed_optimization.jl")
include("utility_functions.jl")


function (@main)(ARGS)
    U_values = [0.00001; LinRange(2.1,9,20)]
    U_values = sort([U_values; 10.0 .^LinRange(-3,2,40)])
    eig_indices = [1,1]

    N_up = 3
    N_down = 3
    N =  3

    # lattice = Chain(6, Periodic())
    lattice_dimension = (2,3)
    bc = "periodic"
    # lattice = Chain(6, Periodic())
    lattice = Square(lattice_dimension, if bc == "periodic" Periodic() else Open() end)
    # lattice = Graphs.cycle_graph(3)

    hopping_model = HubbardModel(1.0,0.0,0.0,false)
    interaction_model = HubbardModel(0.0,1.0,0.0,false)

    subspace = HubbardSubspace(N, lattice)
    # subspace = HubbardSubspace(N_up, N_down, lattice)

    H_hopping, indexer = create_Hubbard(hopping_model, subspace; get_indexer=true)
    H_interaction = create_Hubbard(interaction_model, subspace; indexer=indexer)

    mapping = []
    s_mapping = []
    for kind in 1:2
        op = create_operator(subspace,:T, kind=kind)
        r, _, v = findnz(op)
        push!(mapping, r)
        push!(s_mapping, v)
    end


    # create symmetry projected states
    n_eigs = collect(lattice_dimension)

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
        push!(ops,Hermitian(symmetrize(create_operator(subspace,:Sx))))
        push!(eig_values, -particle_n÷2:1:particle_n÷2)
        push!(ops, Hermitian(symmetrize(create_operator(subspace,:S2))))
        push!(eig_values, [s*(s+1) for s in (particle_n%2)/2:1:particle_n/2])
    else
        push!(ops, Hermitian(symmetrize(create_operator(subspace,:S2))))
        particle_n = subspace.N_up + subspace.N_down
        push!(eig_values, [s*(s+1) for s in (particle_n%2)/2:1:particle_n/2])
    end

    # diagonalize
    all_eig_vecs = zeros(ComplexF64, length(U_values), size(new_hopping)[1])
    all_E = []

    for (i,U) in enumerate(U_values)
        println("i=$i")
        new_h = new_hopping + new_interaction * U
        E, H_vecs = eigsolve(new_h, rand(ComplexF64, size(new_h)[1]), 5, :SR)
        H_vec = H_vecs[1]
        push!(all_E, E[1])
        all_eig_vecs[i,:] = H_vec
    end


    # save data
    meta_data = Dict(
        "electron count"=>N, 
        "sites"=>join(lattice_dimension, "x"), 
        "bc"=>bc, 
        "basis"=>"adiabatic",
        "translational symmetry"=>eig_indices,
        "U_values"=>U_values,
        "maxiters"=>200, 
        "optimizer"=>"LBFGS"
        )
    
    dict = Dict(
        "meta_data"=>meta_data,
        "E"=>all_E,
        "all_eig_vecs"=>all_eig_vecs,
        "indexer"=>indexer,
        "mapping"=>mapping, 
        "s_mapping"=>s_mapping, 
        "representative_indices"=>representative_indices, 
        "magnitude"=>magnitude
    )
    
    
    save_energy_with_metadata("data/N=3", dict)

    return 0 
end