using Lattices, HDF5, SparseArrays, LinearAlgebra, Combinatorics, Random
using Zygote, KrylovKit, ExponentialUtilities, Statistics
import Graphs
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using JSON, JLD2

include("../ed_objects.jl"); include("../ed_functions.jl")
include("../ed_optimization.jl"); include("../utility_functions.jl")

lattice = Square((3,2), Periodic())
subspace = HubbardSubspace(2, 2, lattice; k=(1,1))
_, indexer = create_Hubbard(HubbardModel(1.0,0.0,0.0,false), subspace;
                            get_indexer=true, momentum_basis=true)

h5open("/home/jek354/research/ML-signproblem/experimenting/ed/data/N=(2, 2)_3x2/HubbardED_Slater_3x2_(2,2)_t_1.h5","r") do data
    kvecs = read(data["metadata/kvecs"])
    ext_to_coord = Dict(i-1 => Coordinate(kvecs[1,i]+1, kvecs[2,i]+1) for i in 1:size(kvecs,2))

    # Build reverse: Coordinate -> ext linear 0-index
    coord_to_ext = Dict(v => k for (k,v) in ext_to_coord)

    # For each sector, count matches with our k=(1,1) states
    println("=== Sector match counts for our k=(1,1) basis ===")
    for sec in 0:5
        labels = read(data["metadata/slater_labels/$sec"])
        matched = 0
        for i in 1:size(labels,2)
            up_set = Set([ext_to_coord[s] for s in labels[:,i,1]])
            dn_set = Set([ext_to_coord[s] for s in labels[:,i,2]])
            haskey(indexer.comb_dict, (up_set,dn_set)) && (matched += 1)
        end
        println("  sector $sec ($(size(labels,2)) states): $matched matched")
    end
    println()

    # Now find where EACH of our k=(1,1) states appears in the external data
    println("=== For each of our 39 k=(1,1) states, which ext sector contains it? ===")
    for (our_i, conf) in enumerate(indexer.inv_comb_dict)
        found_sec = -1; found_loc = -1
        for sec in 0:5
            labels = read(data["metadata/slater_labels/$sec"])
            for i in 1:size(labels,2)
                up_set = Set([ext_to_coord[s] for s in labels[:,i,1]])
                dn_set = Set([ext_to_coord[s] for s in labels[:,i,2]])
                if up_set == conf[1] && dn_set == conf[2]
                    found_sec = sec; found_loc = i
                    break
                end
            end
            found_sec >= 0 && break
        end
        up_c = sort([s.coordinates for s in conf[1]])
        dn_c = sort([s.coordinates for s in conf[2]])
        println("  our[$our_i] up=$up_c dn=$dn_c -> sector=$found_sec loc=$found_loc")
    end
end
