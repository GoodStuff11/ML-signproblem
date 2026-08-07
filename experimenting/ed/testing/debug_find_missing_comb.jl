using HDF5, LinearAlgebra, SparseArrays
using Lattices, Combinatorics, KrylovKit

include("../data_path.jl")
include("../utility_functions.jl")
include("../ed_objects.jl")
include("../ed_functions.jl")

function main()
    folder_new_sign = "/home/jek354/research/data/new_data/data_new_sign/N=(3, 2)_3x2"
    valid_files = [f for f in readdir(folder_new_sign) if occursin("HubbardED", f)]
    file_path = joinpath(folder_new_sign, valid_files[1])

    local sl_up, sl_dn, kvecs
    sector_key = "2"
    h5open(file_path, "r") do data
        sl_up = read(data, "metadata/slater_labels/$sector_key/up")
        sl_dn = read(data, "metadata/slater_labels/$sector_key/dn")
        kvecs = read(data, "metadata/kvecs")
    end

    Lvec = [3, 2]
    N = (3, 2)
    lattice = Square(tuple(Lvec...), Periodic())

    k_tuple = (2, 1)
    subspace = HubbardSubspace(N[1], N[2], lattice; k=k_tuple)
    indexer = CombinationIndexer(subspace; order=ColSnake())

    h5_orbital_to_coord = Dict{Int,Coordinate}()
    for o in 0:(size(kvecs, 2)-1)
        h5_orbital_to_coord[o] = Coordinate(kvecs[1, o+1] + 1, kvecs[2, o+1] + 1)
    end

    H_dim = size(sl_up, 2)
    println("Testing all $H_dim Slater determinants in HDF5 sector 2 against indexer for k=(2,1)...")

    missing_cnt = 0
    for h5_idx in 1:H_dim
        up_orbs = sl_up[:, h5_idx]
        dn_orbs = sl_dn[:, h5_idx]

        up_set = Set([h5_orbital_to_coord[o] for o in up_orbs])
        dn_set = Set([h5_orbital_to_coord[o] for o in dn_orbs])

        if !haskey(indexer.comb_dict, (up_set, dn_set))
            println("  Missing h5_idx = $h5_idx: up=$up_orbs, dn=$dn_orbs")
            missing_cnt += 1
        end
    end
    println("Total missing combinations for k=(2,1): $missing_cnt")
end

main()
