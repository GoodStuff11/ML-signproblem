using HDF5
using LinearAlgebra
using Lattices
using SparseArrays
using Printf

include("../ed_objects.jl")
include("../utility_functions.jl")
include("../ed_functions.jl")

function verify_all_sectors()
    folder = "/home/jek354/research/data/new_data/data_new_sign/N=(3, 2)_3x2"
    f = joinpath(folder, "HubbardED_Slater_3x2_(3,2)_t_1_m_2.h5")

    data = h5open(f, "r")
    Lvec = read(data, "metadata/Lvec")
    kvecs = read(data, "metadata/kvecs")
    uvec = read(data, "data/uvec")
    U_val = uvec[1]

    println("==========================================================================")
    println(" VERIFYING :COORDINATE_FIRST + ROW-SNAKE ACROSS ALL MOMENTUM SECTORS")
    println("==========================================================================")
    println("Sector | Momentum k | Dim | Julia GS Energy  | Max Mag Diff (Julia vs HDF5)")
    println("--------------------------------------------------------------------------")

    for k_idx in 0:5
        k_target = Tuple(kvecs[:, k_idx+1] .+ 1)
        raw_h5_evec = read(data, "data/evecs/$(k_idx)")[:, 1, 1]
        
        lattice = Square(Tuple(Lvec), Periodic())
        subspace = HubbardSubspace(3, 2, lattice; k=k_target)
        indexer = CombinationIndexer(subspace; order=RowSnake())

        H_julia = create_Hubbard(HubbardModel(1.0, U_val, 0.0, false), subspace; indexer=indexer, momentum_basis=true, sign_convention=:coordinate_first)
        vals_j, vecs_j = eigen(Matrix(H_julia))
        julia_gs = vecs_j[:, 1]

        max_abs_diff = maximum(abs.(abs.(julia_gs) .- abs.(raw_h5_evec)))

        @printf("%-6d | %-10s | %-3d | %-16.6f | %.4e\n",
            k_idx, string(kvecs[:, k_idx+1]), length(julia_gs), vals_j[1], max_abs_diff)
    end
    close(data)
    println("==========================================================================")
end

verify_all_sectors()
