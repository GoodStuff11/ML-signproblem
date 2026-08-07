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

# Function to compute exact JW sign difference between spin_first and coordinate_first
# for a transition from state (c_up_int, c_dn_int) to (r_up_int, r_dn_int)
# under up-bilinear c^\dagger_{\uparrow, k1} c_{\uparrow, k2} and down-bilinear c^\dagger_{\downarrow, k3} c_{\downarrow, k4}
function compute_coord_first_jw_sign(c_up_mask::UInt, c_dn_mask::UInt, r_up_mask::UInt, r_dn_mask::UInt, N_sites::Int)
    # Order of spin_first: up electrons at sites, then down electrons at sites
    # Order of coordinate_first: for site s=0..N-1, up electron at s, then down electron at s

    # Let's construct the native permutation of occupied pairs in coordinate_first for c state
    c_pairs_coord = Tuple{Int, Int}[]
    for s in 0:N_sites-1
        if ((c_up_mask >> s) & 1) == 1
            push!(c_pairs_coord, (s, 1))
        end
        if ((c_dn_mask >> s) & 1) == 1
            push!(c_pairs_coord, (s, 2))
        end
    end

    c_pairs_spin = Tuple{Int, Int}[]
    for s in 0:N_sites-1
        if ((c_up_mask >> s) & 1) == 1
            push!(c_pairs_spin, (s, 1))
        end
    end
    for s in 0:N_sites-1
        if ((c_dn_mask >> s) & 1) == 1
            push!(c_pairs_spin, (s, 2))
        end
    end

    # Sign to convert c state from spin_first representation to coordinate_first representation
    perm_c = [findfirst(==(p), c_pairs_coord) for p in c_pairs_spin]
    sgn_c = iseven(permutation_parity(perm_c)) ? 1.0 : -1.0

    # Same for r state
    r_pairs_coord = Tuple{Int, Int}[]
    for s in 0:N_sites-1
        if ((r_up_mask >> s) & 1) == 1
            push!(r_pairs_coord, (s, 1))
        end
        if ((r_dn_mask >> s) & 1) == 1
            push!(r_pairs_coord, (s, 2))
        end
    end

    r_pairs_spin = Tuple{Int, Int}[]
    for s in 0:N_sites-1
        if ((r_up_mask >> s) & 1) == 1
            push!(r_pairs_spin, (s, 1))
        end
    end
    for s in 0:N_sites-1
        if ((r_dn_mask >> s) & 1) == 1
            push!(r_pairs_spin, (s, 2))
        end
    end

    perm_r = [findfirst(==(p), r_pairs_coord) for p in r_pairs_spin]
    sgn_r = iseven(permutation_parity(perm_r)) ? 1.0 : -1.0

    # The conversion factor for the matrix element M(r, c) from spin_first to coordinate_first is:
    # M_coord(r, c) = sgn_r * M_spin(r, c) * sgn_c
    return sgn_r * sgn_c
end

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "test_cross_spin_sign_fix")
    with_logging(log_path) do
        println("=== Testing Cross-Spin JW Sign Correction for HubbardMomentumBasis ===")

        Lvec = [3, 2]
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
        M_trotter = Matrix(H_trotter)

        new_hopping = create_Hubbard(HubbardModel(1.0, 0.0, 0.0, false), subspace; 
                            indexer=indexer, momentum_basis=true, sign_convention=sign_conv, lattice_ordering=order)
        new_interaction = create_Hubbard(HubbardModel(0.0, 1.0, 0.0, false), subspace; 
                            indexer=indexer, momentum_basis=true, sign_convention=sign_conv, lattice_ordering=order)
        M_create = Matrix(new_hopping .+ u_val .* new_interaction)

        diff = M_trotter .- M_create
        diff_indices = findall(x -> abs(x) > 1e-5, diff)
        println("Before fix: $(length(diff_indices)) differing elements")

        # Apply sgn_r * sgn_c correction to M_trotter
        N_sites = prod(Lvec)
        subspace_ints = basis_trotter["ints"]
        d_sub = length(subspace_ints)
        M_corrected = copy(M_trotter)

        for r in 1:d_sub, c in 1:d_sub
            if abs(M_trotter[r, c]) > 1e-15 && r != c
                s_c = subspace_ints[c]
                c_up = s_c & ((UInt(1) << N_sites) - UInt(1))
                c_dn = s_c >> N_sites

                s_r = subspace_ints[r]
                r_up = s_r & ((UInt(1) << N_sites) - UInt(1))
                r_dn = s_r >> N_sites

                corr = compute_coord_first_jw_sign(UInt(c_up), UInt(c_dn), UInt(r_up), UInt(r_dn), N_sites)
                # Note: M_trotter was computed assuming spin_first product of bilinears,
                # so multiplying by corr converts it to coordinate_first basis!
                M_corrected[r, c] = M_trotter[r, c] * corr
            end
        end

        diff_after = M_corrected .- M_create
        max_abs_diff_after = maximum(abs.(diff_after))
        println("After applying sgn_r * sgn_c correction:")
        println("Max absolute difference |M_corrected - M_create|: $max_abs_diff_after")

        if max_abs_diff_after < 1e-10
            println("SUCCESS! M_corrected matches M_create EXACTLY!")
        else
            println("Differing elements remaining: $(count(x -> abs(x) > 1e-5, diff_after))")
        end
    end
end
