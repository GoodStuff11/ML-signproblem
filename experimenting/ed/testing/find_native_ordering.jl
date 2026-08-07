using HDF5
using LinearAlgebra
using Lattices
using SparseArrays

include("../ed_objects.jl")
include("../utility_functions.jl")
include("../ed_functions.jl")

function check_direct_basis_match()
    folder = "/home/jek354/research/data/new_data/data_new_sign/N=(3, 2)_3x2"
    f = joinpath(folder, "HubbardED_Slater_3x2_(3,2)_t_1_m_2.h5")

    data = h5open(f, "r")
    Lvec = read(data, "metadata/Lvec")
    kvecs = read(data, "metadata/kvecs")
    uvec = read(data, "data/uvec")
    
    k_min = 0 # Sector 0 (k=[0,0])
    k_target = Tuple(kvecs[:, k_min+1] .+ 1)

    sl_up = read(data, "metadata/slater_labels/$(k_min)/up")
    sl_dn = read(data, "metadata/slater_labels/$(k_min)/dn")
    H_dim = size(sl_up, 2)
    close(data)

    println("==========================================================================")
    println(" CHECKING DIRECT BASIS MATCH (Without Reordering Vector)")
    println("==========================================================================")

    N_sites = prod(Lvec)

    # 1. Map orbitals to coordinates
    h5_orbital_to_coord = Dict{Int,Coordinate}()
    for o in 0:(N_sites-1)
        h5_orbital_to_coord[o] = Coordinate(o % Lvec[1] + 1, div(o, Lvec[1]) + 1)
    end

    orders = [("ColSnake", ColSnake()), ("RowSnake", RowSnake())]
    L_orders = [("Standard Lvec", Lvec), ("Reversed Lvec", Lvec[end:-1:1])]

    for (order_name, order_type) in orders
        for (l_name, l_vec_val) in L_orders
            subspace = HubbardSubspace(3, 2, Square(Tuple(l_vec_val), Periodic()); k=k_target)
            indexer = CombinationIndexer(subspace; order=order_type)
            
            matches = 0
            for i in 1:H_dim
                up_orbs = sl_up[:, i]
                dn_orbs = sl_dn[:, i]
                up_set = Set([h5_orbital_to_coord[o] for o in up_orbs])
                dn_set = Set([h5_orbital_to_coord[o] for o in dn_orbs])
                
                # Check if indexer's i-th state matches HDF5's i-th state
                julia_conf = indexer.inv_comb_dict[i]
                if julia_conf[1] == up_set && julia_conf[2] == dn_set
                    matches += 1
                end
            end
            
            println("Order: $order_name | $l_name | Direct Basis Index Match Count: $matches / $H_dim")
        end
    end
    println("==========================================================================")
end

check_direct_basis_match()
