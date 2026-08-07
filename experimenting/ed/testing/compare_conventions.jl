using HDF5
using LinearAlgebra
using Lattices
using SparseArrays

include("../ed_objects.jl")
include("../utility_functions.jl")
include("../ed_functions.jl")

function test_all_conventions()
    folder = "/home/jek354/research/data/new_data/data_new_sign/N=(3, 2)_3x2"
    f = joinpath(folder, "HubbardED_Slater_3x2_(3,2)_t_1_m_2.h5")

    data = h5open(f, "r")
    Lvec = read(data, "metadata/Lvec")
    kvecs = read(data, "metadata/kvecs")
    uvec = read(data, "data/uvec")
    U_val = uvec[1]

    # Test Sector 0 (k=[0,0]) where there is no ground state degeneracy
    k_min = 0 
    k_target = Tuple(kvecs[:, k_min+1] .+ 1)
    
    evecs_h5 = read(data, "data/evecs/$(k_idx_str = string(k_min))")
    h5_gs_evec = evecs_h5[:, 1, 1]
    sl_up = read(data, "metadata/slater_labels/$(k_min)/up")
    sl_dn = read(data, "metadata/slater_labels/$(k_min)/dn")
    H_dim = length(h5_gs_evec)

    close(data)

    println("==========================================================================")
    println(" TESTING ALL CONVENTIONS (Sector $k_min, k = $(kvecs[:, k_min+1]))")
    println("==========================================================================")

    coord_mappings = [
        ("x-fastest (kx = o % Lx + 1, ky = div(o, Lx) + 1)", (o, L) -> Coordinate(o % L[1] + 1, div(o, L[1]) + 1)),
        ("y-fastest (kx = div(o, Ly) + 1, ky = o % Ly + 1)", (o, L) -> Coordinate(div(o, L[2]) + 1, o % L[2] + 1))
    ]

    sign_conventions = [:spin_first, :coordinate_first]
    orders = [ColSnake(), RowSnake()]

    for (map_name, map_func) in coord_mappings
        for julia_sign in sign_conventions
            for cpp_sign in sign_conventions
                for order_type in orders
                    order_name = typeof(order_type).name.name
                    
                    subspace = HubbardSubspace(3, 2, Square(Tuple(Lvec), Periodic()); k=k_target)
                    indexer = CombinationIndexer(subspace; order=order_type)
                    N_sites = prod(Lvec)

                    # Build orbital to coord dict
                    h5_orbital_to_coord = Dict{Int,Coordinate}()
                    for o in 0:(N_sites-1)
                        h5_orbital_to_coord[o] = map_func(o, Lvec)
                    end

                    # Build site ordering
                    coord_to_idx = Dict{Coordinate,Int}()
                    for (i, c) in enumerate(sort(collect(values(h5_orbital_to_coord)); order=order_type))
                        coord_to_idx[c] = i
                    end

                    reordered_h5 = zeros(ComplexF64, length(indexer.inv_comb_dict))

                    for h5_idx in 1:H_dim
                        up_orbs = sl_up[:, h5_idx]
                        dn_orbs = sl_dn[:, h5_idx]
                        up_set = Set([h5_orbital_to_coord[o] for o in up_orbs])
                        dn_set = Set([h5_orbital_to_coord[o] for o in dn_orbs])
                        perm_idx = index(indexer, up_set, dn_set)

                        # C++ mode ordering
                        modes = vcat([(o, 1) for o in up_orbs], [(o, 2) for o in dn_orbs])
                        if cpp_sign == :spin_first
                            sort!(modes, by = x -> (x[2], x[1]))
                        else
                            sort!(modes, by = x -> (x[1], x[2]))
                        end

                        # Map to Julia creation operator indices
                        target_jw = Vector{Int}(undef, length(modes))
                        for (idx, (o, spin)) in enumerate(modes)
                            site_idx = coord_to_idx[h5_orbital_to_coord[o]]
                            if julia_sign == :spin_first
                                target_jw[idx] = (spin == 1) ? site_idx : N_sites + site_idx
                            else
                                target_jw[idx] = 2 * (site_idx - 1) + spin
                            end
                        end

                        sgn = 1 - 2 * permutation_parity(target_jw)
                        reordered_h5[perm_idx] = h5_gs_evec[h5_idx] * sgn
                    end

                    # Compute Julia GS
                    H_julia = create_Hubbard(HubbardModel(1.0, U_val, 0.0, false), subspace; indexer=indexer, momentum_basis=true, sign_convention=julia_sign)
                    vals_j, vecs_j = eigen(Matrix(H_julia))
                    julia_gs = vecs_j[:, 1]

                    # Compare vectors
                    abs_diff = maximum(abs.(abs.(julia_gs) .- abs.(reordered_h5)))
                    
                    # Compute phase factors for non-zero components
                    nonzero_indices = findall(x -> abs(x) > 1e-4, julia_gs)
                    ratios = julia_gs[nonzero_indices] ./ reordered_h5[nonzero_indices]
                    global_phase = ratios[1] / abs(ratios[1])
                    norm_ratios = ratios ./ global_phase
                    
                    signs = real.(norm_ratios)
                    is_pure_sign = all(abs.(imag.(norm_ratios)) .< 1e-4) && all(abs.(abs.(signs) .- 1.0) .< 1e-4)
                    num_pos = count(s -> s > 0.5, signs)
                    num_neg = count(s -> s < -0.5, signs)

                    println("\nConfig: Map=$map_name | JuliaSign=$julia_sign | CppSign=$cpp_sign | Order=$order_name")
                    println("  Max component mag diff: ", abs_diff)
                    if abs_diff < 1e-4
                        println("  Elementwise Magnitude Match: YES")
                        println("  Global Phase: ", global_phase)
                        println("  Sign pattern: +1 count = $num_pos, -1 count = $num_neg (Pure Sign Diff: $is_pure_sign)")
                    else
                        println("  Elementwise Magnitude Match: NO")
                    end
                end
            end
        end
    end
    println("==========================================================================")
end

test_all_conventions()
