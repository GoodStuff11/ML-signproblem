using HDF5
using LinearAlgebra
using Lattices
using SparseArrays

include("../ed_objects.jl")
include("../utility_functions.jl")
include("../ed_functions.jl")

function run_magnitude_comparison()
    folder = "/home/jek354/research/data/new_data/data_new_sign/N=(3, 2)_3x2"
    f = joinpath(folder, "HubbardED_Slater_3x2_(3,2)_t_1_m_2.h5")
    
    data = h5open(f, "r")
    Lvec = read(data, "metadata/Lvec")
    kvecs = read(data, "metadata/kvecs")
    uvec = read(data, "data/uvec")
    
    k_min = 2 # Sector index in file
    k_target = Tuple(kvecs[:, k_min+1] .+ 1)
    
    subspace = HubbardSubspace(3, 2, Square(Tuple(Lvec), Periodic()); k=k_target)
    indexer = CombinationIndexer(subspace; order=ColSnake())
    
    # 1. Julia Hamiltonian and eigenvector for U=0.25 (uvec[1])
    U_val = uvec[1]
    H_julia = create_Hubbard(HubbardModel(1.0, U_val, 0.0, false), subspace; indexer=indexer, momentum_basis=true, sign_convention=:spin_first)
    vals_julia, vecs_julia = eigen(Matrix(H_julia))
    julia_gs_evec = vecs_julia[:, 1]
    
    # 2. HDF5 eigenvector for U=0.25 (first U index)
    evecs_h5 = read(data, "data/evecs/$(k_min)")
    h5_gs_evec = evecs_h5[:, 1, 1]
    
    close(data)
    
    # Sort absolute magnitudes of components
    mag_julia = sort(abs.(julia_gs_evec))
    mag_h5 = sort(abs.(h5_gs_evec))
    
    max_diff = maximum(abs.(mag_julia .- mag_h5))
    println("--- MAGNITUDE COMPARISON (U = $U_val) ---")
    println("Ground state energy (Julia): ", vals_julia[1])
    println("Vector length: ", length(mag_julia))
    println("Max difference between sorted component magnitudes: ", max_diff)
    println("\nTop 10 component magnitudes (Julia vs HDF5):")
    for i in 1:min(10, length(mag_julia))
        println("  Rank $i: Julia = $(mag_julia[end-i+1]), HDF5 = $(mag_h5[end-i+1])")
    end
    
    if max_diff < 1e-5
        println("\nRESULT: SUCCESS! The component magnitudes MATCH up to permutation.")
    else
        println("\nRESULT: MISMATCH! The component magnitudes DO NOT match.")
    end
end

run_magnitude_comparison()
