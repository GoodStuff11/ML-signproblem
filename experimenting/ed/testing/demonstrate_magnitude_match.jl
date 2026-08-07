using HDF5
using LinearAlgebra
using Lattices
using SparseArrays

include("../ed_objects.jl")
include("../utility_functions.jl")
include("../ed_functions.jl")

function demonstrate_match()
    folder = "/home/jek354/research/data/new_data/data_new_sign/N=(3, 2)_3x2"
    f = joinpath(folder, "HubbardED_Slater_3x2_(3,2)_t_1_m_2.h5")

    data = h5open(f, "r")
    Lvec = read(data, "metadata/Lvec")
    kvecs = read(data, "metadata/kvecs")
    uvec = read(data, "data/uvec")

    println("==========================================================================")
    println(" DEMONSTRATING EIGENVECTOR COMPONENT MAGNITUDE MATCHING (HDF5 vs Julia)")
    println("==========================================================================")

    for k_idx in 0:1
        k_target = Tuple(kvecs[:, k_idx+1] .+ 1)
        subspace = HubbardSubspace(3, 2, Square(Tuple(Lvec), Periodic()); k=k_target)
        indexer = CombinationIndexer(subspace; order=ColSnake())
        
        # Build Julia Hamiltonian & compute ground state
        H_julia = create_Hubbard(HubbardModel(1.0, uvec[1], 0.0, false), subspace; indexer=indexer, momentum_basis=true, sign_convention=:spin_first)
        vals_j, vecs_j = eigen(Matrix(H_julia))
        julia_gs_evec = vecs_j[:, 1]
        
        # Load HDF5 eigenvector
        evecs_h5 = read(data, "data/evecs/$(k_idx)")
        h5_gs_evec = evecs_h5[:, 1, 1]
        
        # Sort component magnitudes
        mag_julia = sort(abs.(julia_gs_evec))
        mag_h5 = sort(abs.(h5_gs_evec))
        
        max_diff = maximum(abs.(mag_julia .- mag_h5))
        
        println("\nSector index $k_idx | Momentum k = $(kvecs[:, k_idx+1]) | Target k = $k_target:")
        println("  Ground State Energy: ", vals_j[1])
        println("  Max diff in sorted component magnitudes: ", max_diff)
        println("  Top 5 Component Magnitudes:")
        for r in 1:5
            println("    Rank $r: Julia = $(mag_julia[end-r+1]), HDF5 = $(mag_h5[end-r+1])")
        end
    end
    
    close(data)
    println("==========================================================================")
end

demonstrate_match()
