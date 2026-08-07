using Lattices
using LinearAlgebra
using SparseArrays
using JLD2
using HDF5
using Test

include(joinpath(@__DIR__, "..", "logging.jl"))
include(joinpath(@__DIR__, "..", "utility_functions.jl"))
using .UtilityFunctions
include(joinpath(@__DIR__, "..", "trotter.jl"))
using .Trotter
include(joinpath(@__DIR__, "..", "ed_objects.jl"))
include(joinpath(@__DIR__, "..", "ed_functions.jl"))

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "debug_row_snake_sign")
    with_logging(log_path) do
        println("=== Debugging RowSnake Site-Order JW Signs ===")

        Lvec = [3, 2]
        N_sites = prod(Lvec)
        dims = Tuple(Lvec)

        # Precompute Row-snake site to Col-snake site mapping
        col_sites_by_row_order = [Trotter.TamFermion.ravel_c(Trotter.TamFermion.unravel_f(r, dims), dims) for r in 0:N_sites-1]

        N_elec = (2, 2)
        u_val = 2.0
        lattice = Square(Tuple(Lvec), Periodic())
        sign_conv = :coordinate_first
        order = RowSnake()

        subspace = HubbardSubspace(N_elec..., lattice; k=[0, 0])
        indexer = CombinationIndexer(subspace; order=order)

        H_trotter, basis_trotter, _ = Trotter.TamFermion.HubbardMomentumBasis(
            1.0, u_val, Tuple(Lvec), N_elec;
            indexer=indexer, sign_convention=sign_conv, lattice_ordering=order
        )
        M_trotter = real.(Matrix(H_trotter))

        new_hopping = create_Hubbard(HubbardModel(1.0, 0.0, 0.0, false), subspace; 
                            indexer=indexer, momentum_basis=true, sign_convention=sign_conv, lattice_ordering=order)
        new_interaction = create_Hubbard(HubbardModel(0.0, 1.0, 0.0, false), subspace; 
                            indexer=indexer, momentum_basis=true, sign_convention=sign_conv, lattice_ordering=order)
        M_create = Matrix(new_hopping .+ u_val .* new_interaction)

        diff = M_trotter .- M_create
        println("Before fix: $(count(x -> abs(x) > 1e-5, diff)) differing matrix elements")

        # Let's test the Row-snake ordered JW state sign formula
        subspace_ints = basis_trotter["ints"]
        d_sub = length(subspace_ints)
        state_signs = Vector{Float64}(undef, d_sub)

        for i in 1:d_sub
            s = subspace_ints[i]
            s_up = UInt(s & ((one(s) << N_sites) - one(s)))
            s_dn = UInt(s >> N_sites)

            swaps = 0
            for r_dn_idx in 0:N_sites-1
                c_dn_idx = col_sites_by_row_order[r_dn_idx + 1]
                if ((s_dn >> c_dn_idx) & 1) == 1
                    # Count up electrons at Row-snake sites r_up > r_dn_idx
                    for r_up_idx in r_dn_idx+1:N_sites-1
                        c_up_idx = col_sites_by_row_order[r_up_idx + 1]
                        if ((s_up >> c_up_idx) & 1) == 1
                            swaps += 1
                        end
                    end
                end
            end
            state_signs[i] = iseven(swaps) ? 1.0 : -1.0
        end

        M_corrected = copy(M_trotter)
        for r in 1:d_sub, c in 1:d_sub
            if abs(M_trotter[r, c]) > 1e-15 && r != c
                # Note: M_trotter originally applied the col-snake sign, so we multiply by col-snake sign and row-snake sign
                s_c = subspace_ints[c]; c_up = UInt(s_c & ((one(s_c) << N_sites) - one(s_c))); c_dn = UInt(s_c >> N_sites)
                s_r = subspace_ints[r]; r_up = UInt(s_r & ((one(s_r) << N_sites) - one(s_r))); r_dn = UInt(s_r >> N_sites)

                # Col-snake sign for c and r
                swaps_c_col = sum(count_ones(c_up >> (d + 1)) for d in 0:N_sites-1 if ((c_dn >> d) & 1) == 1)
                sgn_c_col = iseven(swaps_c_col) ? 1.0 : -1.0

                swaps_r_col = sum(count_ones(r_up >> (d + 1)) for d in 0:N_sites-1 if ((r_dn >> d) & 1) == 1)
                sgn_r_col = iseven(swaps_r_col) ? 1.0 : -1.0

                M_corrected[r, c] *= (sgn_c_col * sgn_r_col) * (state_signs[c] * state_signs[r])
            end
        end

        diff_after = M_corrected .- M_create
        max_abs_diff = maximum(abs.(diff_after))
        println("After RowSnake JW sign correction:")
        println("Max abs diff |M_corrected - M_create|: $max_abs_diff")
        if max_abs_diff < 1e-10
            println("SUCCESS! M_corrected matches M_create 100% EXACTLY!")
        end
    end
end
