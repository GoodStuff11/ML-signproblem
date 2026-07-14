# check_basis_match.jl
using Lattices, HDF5, JLD2, SparseArrays, LinearAlgebra

include("../utility_functions.jl")
include("../ed_objects.jl")
include("../ed_functions.jl")
include("../trotter.jl")
using .Trotter

Lvec = [3, 2]
N_sites = 6
N_elec = (2, 2)
n_up, n_dn = N_elec

lattice = Square(Tuple(Lvec), Periodic())
subspace = HubbardSubspace(n_up, n_dn, lattice; k=(1, 1))
indexer = CombinationIndexer(subspace; order=ColSnake())

# 1. Reconstruct from indexer
function coord_to_site_idx(coord, Lvec)
    c0 = coord.coordinates .- 1
    return Trotter.TamLib.ravel_c(c0, Tuple(Lvec))
end

function coord_set_to_binary(coord_set, Lvec)
    val = zero(UInt)
    for coord in coord_set
        site_idx = coord_to_site_idx(coord, Lvec)
        val |= (one(UInt) << site_idx)
    end
    return val
end

basis_sector_indexer = Vector{UInt}(undef, length(indexer.inv_comb_dict))
for (idx, conf) in enumerate(indexer.inv_comb_dict)
    u_bin = coord_set_to_binary(conf[1], Lvec)
    d_bin = coord_set_to_binary(conf[2], Lvec)
    basis_sector_indexer[idx] = Trotter.combineSpinInts(u_bin, d_bin, N_sites)
end

# 2. Load from HDF5
h5_file = "data_new_sign/N=(2, 2)_3x2/HubbardED_Slater_3x2_(2,2)_t_1_m_2.h5"
basis_sector_h5 = h5open(h5_file, "r") do data
    slater_labels_up = read(data, "metadata/slater_labels/0/up")
    slater_labels_down = read(data, "metadata/slater_labels/0/dn")
    H_dim = size(slater_labels_up, 2)
    
    function h5_to_our_idx(k, Lvec)
        i = k % Lvec[1]
        j = div(k, Lvec[1])
        return i * Lvec[2] + j
    end

    function orbital_indices_to_binary(indices)
        val = zero(UInt)
        for idx in indices
            val |= (one(UInt) << idx)
        end
        return val
    end
    
    res = Vector{UInt}(undef, H_dim)
    for idx in 1:H_dim
        u_mapped = [h5_to_our_idx(o, Lvec) for o in slater_labels_up[:, idx]]
        d_mapped = [h5_to_our_idx(o, Lvec) for o in slater_labels_down[:, idx]]
        u_bin = orbital_indices_to_binary(u_mapped)
        d_bin = orbital_indices_to_binary(d_mapped)
        res[idx] = Trotter.combineSpinInts(u_bin, d_bin, N_sites)
    end
    return res
end

println("Indexer basis size: ", length(basis_sector_indexer))
println("H5 basis size:      ", length(basis_sector_h5))
println("Set difference (Indexer - H5): ", setdiff(Set(basis_sector_indexer), Set(basis_sector_h5)))
println("Set difference (H5 - Indexer): ", setdiff(Set(basis_sector_h5), Set(basis_sector_indexer)))
