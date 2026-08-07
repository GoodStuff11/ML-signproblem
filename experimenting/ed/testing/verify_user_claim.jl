using HDF5
using LinearAlgebra
using Lattices
using SparseArrays
include("../ed_objects.jl")
include("../utility_functions.jl")
include("../trotter.jl")
include("../ed_functions.jl")
using .Trotter

folder = "/home/jek354/research/data/new_data/data_new_sign/N=(2, 2)_3x2"
valid_files = [f for f in readdir(folder) if occursin("HubbardED", f)]
file_path = joinpath(folder, valid_files[1])

h5open(file_path, "r") do data
    N = (read(data, "metadata/nup"), read(data, "metadata/ndown"))
    Lvec = read(data, "metadata/Lvec")
    U_values = read(data, "data/uvec")
    kvecs = read(data, "metadata/kvecs")
    
    key_labels = [parse(Int, k) for k in keys(data["data/energies"])]
    all_E = [real.(read(data, "data/energies/$(k)"))[1, :] for k in key_labels]
    k_min = find_best_energy_sector(all_E, U_values; labels=key_labels, data=data, su2_symmetry=false)
    
    evecs = read(data, "data/evecs/$(k_min)")
    target_vecs = transpose(evecs[:, 1, :]) # (n_U, H_dim)
    gs_new = target_vecs[2, :] # U=U[2]
    
    sl_all = read(data, "metadata/slater_labels/$k_min")
    # k_sector calculation
    tot_k_sec = zeros(Int, length(Lvec))
    for o in sl_all[:, 1, 1]
        tot_k_sec .+= kvecs[:, o+1]
    end
    for o in sl_all[:, 1, 2]
        tot_k_sec .+= kvecs[:, o+1]
    end
    
    # Reverse Lvec
    Lvec_rev = reverse(Lvec)
    k_sector_rev = tuple([(reverse(tot_k_sec)[d] % Lvec_rev[d]) + 1 for d in 1:length(Lvec_rev)]...)
    
    lattice = Square(Tuple(Lvec_rev), Periodic())
    subspace = HubbardSubspace(N..., lattice; k=k_sector_rev)
    indexer = CombinationIndexer(subspace; order=ColSnake())
    
    new_hopping = create_Hubbard(HubbardModel(1.0,0.0,0.0,false), subspace; indexer=indexer, momentum_basis=true, sign_convention=:spin_first)
    new_interaction = create_Hubbard(HubbardModel(0.0,1.0,0.0,false), subspace; indexer=indexer, momentum_basis=true, sign_convention=:spin_first)
    H = Matrix(new_hopping .+ U_values[2] .* new_interaction)
    
    println("U = ", U_values[2])
    println("Computed Energy: ", real(gs_new'*H*gs_new))
    println("Matrix gs energy: ", eigvals(H)[1])
end
