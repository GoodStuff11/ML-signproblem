using Lattices
using LinearAlgebra
using SparseArrays
using JLD2
using HDF5

include(joinpath(@__DIR__, "..", "logging.jl"))
include(joinpath(@__DIR__, "..", "utility_functions.jl"))
using .UtilityFunctions
include(joinpath(@__DIR__, "..", "trotter.jl"))
using .Trotter
include(joinpath(@__DIR__, "..", "ed_objects.jl"))
include(joinpath(@__DIR__, "..", "ed_functions.jl"))

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "debug_trotter_33316")
    with_logging(log_path) do
        println("=== Debugging 0.333166 Overlap in test_h5_sign_convention_overlap ===")

        folder_new_sign = "/home/jek354/research/data/new_data/data_h5_fixed/N=(3, 2)_3x3"
        sign_conv = :coordinate_first

        U_vals, vecs, indexer, _, N_elec, _, _, actual_sign = load_ED_data(
            folder_new_sign; sign_convention=sign_conv, verbose=false, use_slater_reference=false
        )

        Lvec = parse_lattice_dimension(folder_new_sign)
        lattice = Square(Tuple(Lvec), Periodic())
        order = RowSnake()
        u_val = U_vals[1]
        gs = vecs[1, :]

        println("indexer.a[1:3] = $(indexer.a[1:3])")
        println("actual_sign = $actual_sign, order = $order")

        # 1. Call Trotter.HubbardMomentumBasis
        H_trotter, basis_trotter, _ = Trotter.HubbardMomentumBasis(
            1.0, u_val, Tuple(Lvec), N_elec;
            indexer=indexer, sign_convention=actual_sign, lattice_ordering=order
        )
        M_trotter = real.(Matrix(H_trotter))
        V_trotter = eigvecs(M_trotter)
        ov_trotter = abs(V_trotter[:, 1]' * gs)

        # 2. Call create_Hubbard
        k = indexer.k
        subspace = HubbardSubspace(N_elec..., lattice; k=k)
        indexer_rebuilt = CombinationIndexer(subspace; order=order)

        new_hopping = create_Hubbard(HubbardModel(1.0, 0.0, 0.0, false), subspace; 
                            indexer=indexer_rebuilt, momentum_basis=true, sign_convention=actual_sign, lattice_ordering=order)
        new_interaction = create_Hubbard(HubbardModel(0.0, 1.0, 0.0, false), subspace; 
                            indexer=indexer_rebuilt, momentum_basis=true, sign_convention=actual_sign, lattice_ordering=order)
        H_create = Matrix(new_hopping .+ u_val .* new_interaction)
        v_create = eigvecs(H_create)
        ov_create = abs(v_create[:, 1]' * gs)

        println("create overlap: $ov_create")
        println("trotter overlap: $ov_trotter")

        diff = M_trotter .- H_create
        println("Max abs diff |M_trotter - H_create|: $(maximum(abs.(diff)))")
        nz_diff = findall(x -> abs(x) > 1e-5, diff)
        println("Number of differing entries: $(length(nz_diff))")
        for idx in nz_diff[1:min(5, end)]
            r, c = idx[1], idx[2]
            println("  ($r, $c): M_trotter = $(M_trotter[r, c]), H_create = $(H_create[r, c])")
            s_r = basis_trotter["ints"][r]
            s_c = basis_trotter["ints"][c]
            println("    row $r: int = $s_r")
            println("    col $c: int = $s_c")
        end
    end
end
