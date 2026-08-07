using HDF5
using LinearAlgebra
using Lattices
using SparseArrays
include("../ed_objects.jl")
include("../utility_functions.jl")
include("../ed_functions.jl")

folder_new_sign = "/home/jek354/research/data/new_data/data_new_sign/N=(3, 2)_3x2"
file_path = joinpath(folder_new_sign, "HubbardED_Slater_3x2_(3,2)_t_1_m_2.h5")
data = h5open(file_path, "r")
Lvec = read(data, "metadata/Lvec")
kvecs = read(data, "metadata/kvecs")
k_min = 2

k_target = Tuple(kvecs[:, k_min+1] .+ 1) 
subspace = HubbardSubspace(3, 2, Square(Tuple(Lvec), Periodic()); k=k_target)
indexer = CombinationIndexer(subspace; order=ColSnake())
N_sites = prod(Lvec)

evecs_dataset = read(data, "data/evecs/$(k_min)")
target_vecs = evecs_dataset[:, 1, 1] 

sl_up = read(data, "metadata/slater_labels/2/up")
sl_dn = read(data, "metadata/slater_labels/2/dn")

h5_orbital_to_coord = Dict{Int,Coordinate}()
for o in 0:(N_sites-1)
    kx = o % Lvec[1] + 1
    ky = div(o, Lvec[1]) + 1
    h5_orbital_to_coord[o] = Coordinate(kx, ky)
end

H_native = create_Hubbard(HubbardModel(1.0, 0.25, 0.0, false), subspace; indexer=indexer, momentum_basis=true, sign_convention=:spin_first)
vals, vecs = eigen(Matrix(H_native))
true_evec = vecs[:, 1]

# find global phase
perm_idx_1 = index(indexer, Set([h5_orbital_to_coord[o] for o in sl_up[:, 1]]), Set([h5_orbital_to_coord[o] for o in sl_dn[:, 1]]))
global_phase = true_evec[perm_idx_1] / target_vecs[1]
println("Global phase: ", global_phase)

for h5_idx in 1:10
    up_orbs = sl_up[:, h5_idx]
    dn_orbs = sl_dn[:, h5_idx]
    up_set = Set([h5_orbital_to_coord[o] for o in up_orbs])
    dn_set = Set([h5_orbital_to_coord[o] for o in dn_orbs])
    perm_idx = index(indexer, up_set, dn_set)
    
    expected_sgn = real(true_evec[perm_idx] / (target_vecs[h5_idx] * global_phase))
    println("h5_idx=", h5_idx, " expected_sgn=", expected_sgn)
end
close(data)
