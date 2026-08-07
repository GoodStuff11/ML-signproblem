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
Lvec_orig = read(data, "metadata/Lvec")
kvecs = read(data, "metadata/kvecs")
k_min = 2

# LOAD RAW EVECS
evecs_dataset = read(data, "data/evecs/$(k_min)")
target_vecs = transpose(evecs_dataset[:, 1, :])
gs_native = target_vecs[1, :]

found = false
for reverse_L in [false, true]
    Lvec = reverse_L ? reverse(Lvec_orig) : Lvec_orig
    lattice = Square(Tuple(Lvec), Periodic())
    
    # Try all k_targets in the lattice
    for kx in 1:Lvec[1], ky in 1:Lvec[2]
        k_target = (kx, ky)
        subspace = HubbardSubspace(N..., lattice; k=k_target)
        
        for order in [ColSnake(), RowSnake()]
            indexer = CombinationIndexer(subspace; order=order)
            
            # If dimension doesn't match, skip
            if length(indexer.inv_comb_dict) != size(target_vecs, 2)
                continue
            end
            
            for sign_conv in [:spin_first, :coordinate_first]
                H = create_Hubbard(HubbardModel(1.0, 0.25, 0.0, false), subspace; indexer=indexer, momentum_basis=true, sign_convention=sign_conv)
                E = real(gs_native' * H * gs_native)
                
                # Check if it's the ground state
                evals = eigvals(Hermitian(Matrix(H)))
                min_E = evals[1]
                
                if isapprox(E, min_E; atol=1e-5)
                    println("FOUND IT! reverse_L=$reverse_L, k_target=$k_target, order=$(typeof(order)), sign_conv=$sign_conv")
                    global found = true
                end
            end
        end
    end
end

if !found
    println("No matching basis found.")
end
close(data)
