using HDF5
using LinearAlgebra
using Lattices
using SparseArrays
using Printf

include("../ed_objects.jl")
include("../utility_functions.jl")
include("../ed_functions.jl")

function sanity_check()
    folder = "/home/jek354/research/data/new_data/data_new_sign/N=(3, 2)_3x2"
    f = joinpath(folder, "HubbardED_Slater_3x2_(3,2)_t_1_m_2.h5")

    data = h5open(f, "r")
    Lvec = read(data, "metadata/Lvec")
    kvecs = read(data, "metadata/kvecs")
    uvec = read(data, "data/uvec")
    U_val = uvec[1]

    # Sector 0 (k=[0,0]) - non-degenerate ground state
    k_min = 0 
    k_target = Tuple(kvecs[:, k_min+1] .+ 1)

    evecs_h5 = read(data, "data/evecs/$(k_min)")
    raw_h5_evec = evecs_h5[:, 1, 1]
    close(data)

    println("==========================================================================")
    println(" SANITY CHECK: COMPONENT-BY-COMPONENT VALUE COMPARISON (RowSnake)")
    println("==========================================================================")

    subspace = HubbardSubspace(3, 2, Square(Tuple(Lvec), Periodic()); k=k_target)
    indexer = CombinationIndexer(subspace; order=RowSnake())

    for julia_sign in [:coordinate_first, :spin_first]
        H_julia = create_Hubbard(HubbardModel(1.0, U_val, 0.0, false), subspace; indexer=indexer, momentum_basis=true, sign_convention=julia_sign)
        vals_j, vecs_j = eigen(Matrix(H_julia))
        julia_gs = vecs_j[:, 1]

        # Fix global phase relative to first non-zero element of raw HDF5 vector
        nonzero_idx = findfirst(x -> abs(x) > 1e-4, raw_h5_evec)
        phase = julia_gs[nonzero_idx] / raw_h5_evec[nonzero_idx]
        norm_phase = phase / abs(phase)
        
        aligned_julia_gs = julia_gs ./ norm_phase

        max_val_diff = maximum(abs.(aligned_julia_gs .- raw_h5_evec))
        max_abs_diff = maximum(abs.(abs.(julia_gs) .- abs.(raw_h5_evec)))

        println("\n--- Julia Sign: $julia_sign | Indexer Order: RowSnake ---")
        println("  Max component magnitude diff: ", max_abs_diff)
        println("  Max aligned value diff:       ", max_val_diff)
        println("  Global phase factor applied:  ", norm_phase)
        
        println("\n  First 10 Non-Zero Vector Components (Raw HDF5 vs Aligned Julia):")
        println("  -------------------------------------------------------------------------")
        println("  Idx | Raw HDF5 Component          | Aligned Julia Component     | Match?")
        println("  -------------------------------------------------------------------------")
        
        count = 0
        for i in 1:length(raw_h5_evec)
            if abs(raw_h5_evec[i]) > 1e-4
                count += 1
                diff = abs(aligned_julia_gs[i] - raw_h5_evec[i])
                match_str = diff < 1e-4 ? "EXACT" : "SIGN FLIP"
                @printf("  %3d | %-28s | %-27s | %s\n", 
                    i, string(round(raw_h5_evec[i], digits=6)), string(round(aligned_julia_gs[i], digits=6)), match_str)
                if count >= 10
                    break
                end
            end
        end
        println("  -------------------------------------------------------------------------")
    end
    println("==========================================================================")
end

sanity_check()
