using HDF5
using LinearAlgebra
using Lattices
using SparseArrays
include("../ed_objects.jl")
include("../utility_functions.jl")
include("../trotter.jl")
include("../ed_functions.jl")

folder_new_sign = "/home/jek354/research/data/new_data/data_new_sign/N=(3, 2)_3x2"
file_path = joinpath(folder_new_sign, "HubbardED_Slater_3x2_(3,2)_t_1_m_2.h5")
data = h5open(file_path, "r")
N = (read(data, "metadata/nup"), read(data, "metadata/ndown"))
Lvec = read(data, "metadata/Lvec")
kvecs = read(data, "metadata/kvecs")

lattice = Square(Tuple(Lvec), Periodic())

println("Julia min energies for each k_target:")
julia_energies = []
for kx in 1:Lvec[1], ky in 1:Lvec[2]
    k_target = (kx, ky)
    subspace = HubbardSubspace(N..., lattice; k=k_target)
    indexer = CombinationIndexer(subspace; order=ColSnake())
    H = create_Hubbard(HubbardModel(1.0, 0.25, 0.0, false), subspace; indexer=indexer, momentum_basis=true, sign_convention=:spin_first)
    evals = eigvals(Hermitian(Matrix(H)))
    min_E = length(evals) > 0 ? evals[1] : NaN
    push!(julia_energies, (k_target, min_E))
    println("k_target = $k_target, E_min = $min_E")
end

println("\nHDF5 min energies for each sector:")
for k in 0:5
    try
        evals = read(data, "data/energies/$k")[:, 1] # First U value
        min_E = evals[1]
        println("Sector $k, kvecs = ", kvecs[:, k+1], ", E_min = $min_E")
    catch
    end
end
close(data)
