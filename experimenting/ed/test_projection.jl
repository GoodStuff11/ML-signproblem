using Lattices
using LinearAlgebra
using Combinatorics
using SparseArrays
using Plots
import Graphs
using LaTeXStrings
using Statistics
using Random
using Zygote
using Optimization, OptimizationOptimisers
using JSON
using OptimizationOptimJL
using ExponentialUtilities
using JLD2
using KrylovKit
# using ExponentialUtilities
include("ed_objects.jl")
include("ed_functions.jl")
include("ed_optimization.jl")
include("utility_functions.jl")

N_up = 4
N_down = 5
N = 6
half_filling = false
# lattice = Chain(6, Periodic())
lattice_dimension = (2, 3)
bc = "periodic"
# lattice = Chain(6, Periodic())
lattice = Square(lattice_dimension, if bc == "periodic"
    Periodic()
else
    Open()
end)
# lattice = Graphs.cycle_graph(3)
hopping_model = HubbardModel(1.0, 0.0, 0.0, false)
interaction_model = HubbardModel(0.0, 1.0, 0.0, false)
subspace = HubbardSubspace(N, lattice)
# subspace = HubbardSubspace(N_up, N_down, lattice)
H_hopping, indexer = create_Hubbard(hopping_model, subspace; get_indexer=true)
H_interaction = create_Hubbard(interaction_model, subspace; indexer=indexer);
n_eigs = collect(lattice_dimension)
function get_reduced_basis(H_hopping, H_interaction, U, subspace, eig_indices, n_eigs)
    mapping = []
    s_mapping = []
    for kind in 1:2
        op = create_operator(subspace, :T, kind=kind)
        r, c, v = findnz(op)
        push!(mapping, r)
        push!(s_mapping, v)
    end
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
        # push!(ops,Hermitian(symmetrize(create_operator(subspace,:∏σx))))
        # push!(eig_values, -1:2:1)
        push!(ops, Hermitian(symmetrize(create_operator(subspace, :S2))))
        push!(eig_values, [s * (s + 1) for s in (particle_n%2)/2:1:particle_n/2])
    else
        push!(ops, Hermitian(symmetrize(create_operator(subspace, :S2))))
        particle_n = subspace.N_up + subspace.N_down
        push!(eig_values, [s * (s + 1) for s in (particle_n%2)/2:1:particle_n/2])
    end
    recon_vector(all_eig_vecs) = reconstruct_full_vector(
        all_eig_vecs,
        mapping, s_mapping, representative_indices, magnitude, eig_indices, n_eigs
    )
    new_h = new_hopping + new_interaction * U
    E, H_vecs = eigsolve(new_h, rand(ComplexF64, size(new_h)[1]), 5, :SR)
    println("eig_indices: $eig_indices $(real.(E))")
    return E, H_vecs, ops, new_h, recon_vector
end
# for eig_indices in Iterators.product(1:2, 1:3)
#     # eig_indices = [1,3]
#     eig_indices = collect(eig_indices)
#     E, H_vecs, ops, new_h, f = get_reduced_basis(H_hopping, H_interaction, 4, subspace, eig_indices, n_eigs)
# end
E, H_vecs, ops, new_h, f = get_reduced_basis(H_hopping, H_interaction, 4, subspace, [2, 2], n_eigs)
println("E=$E")
eig_values = []
particle_n = subspace.N
push!(eig_values, collect(-particle_n÷2:1:particle_n÷2))
# push!(eig_values, [-1,1])
push!(eig_values, [s * (s + 1) for s in (particle_n%2)/2:1:particle_n/2])


for eig_idx in Iterators.product(1:2, 1:3)
    eig_idx = collect(eig_idx)
    v = project_hermitian(ops[1], H_vecs[1], eig_idx[1], eig_values[1])
    # v = project_hermitian(ops[3], v, 2, eig_values[3])
    v = project_hermitian(ops[2], v, eig_idx[2], eig_values[2])
    # println(norm(v))
    if norm(v) ≈ 1
        agreement_found = true
        for i = 1:2
            if !(real(v' * ops[i] * v) ≈ eig_values[i][eig_idx[i]])
                agreement_found = false
            end
        end

        if agreement_found
            for i = 1:2
                println("$eig_idx: $(real.(v' * ops[i] * v)) $(eig_values[i][eig_idx[i]])")
                println("Did it agree?: $(real.(v' * ops[i] * v) ≈ eig_values[i][eig_idx[i]])")
            end
        else
            println("eigenvalue not present")
        end
    else
        println("eigenvalue not present")
    end
    println()
end
