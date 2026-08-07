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
    log_path = make_log_path(@__DIR__, "debug_trotter_9997")
    with_logging(log_path) do
        println("=== Debugging 0.999738 Overlap Mismatch ===")

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

        # 1. H_trotter
        H_trotter, basis_trotter, _ = Trotter.HubbardMomentumBasis(
            1.0, u_val, Tuple(Lvec), N_elec;
            indexer=indexer, sign_convention=actual_sign, lattice_ordering=order
        )
        V_trotter = eigvecs(Matrix(H_trotter))
        E_trotter = eigvals(Matrix(H_trotter))

        # 2. H_create
        k = indexer.k
        subspace = HubbardSubspace(N_elec..., lattice; k=k)
        indexer_rebuilt = CombinationIndexer(subspace; order=order)

        new_hopping = create_Hubbard(HubbardModel(1.0, 0.0, 0.0, false), subspace; 
                            indexer=indexer_rebuilt, momentum_basis=true, sign_convention=actual_sign, lattice_ordering=order)
        new_interaction = create_Hubbard(HubbardModel(0.0, 1.0, 0.0, false), subspace; 
                            indexer=indexer_rebuilt, momentum_basis=true, sign_convention=actual_sign, lattice_ordering=order)
        H_create = Matrix(new_hopping .+ u_val .* new_interaction)
        v_create = eigvecs(H_create)
        E_create = eigvals(H_create)

        println("E_trotter[1:4] = $(E_trotter[1:4])")
        println("E_create[1:4]  = $(E_create[1:4])")

        ov_create = abs(v_create[:, 1]' * gs)
        ov_trotter = abs(V_trotter[:, 1]' * gs)
        ov_between = abs(V_trotter[:, 1]' * v_create[:, 1])

        println("Overlap create vs gs: $ov_create")
        println("Overlap trotter vs gs: $ov_trotter")
        println("Overlap trotter vs create: $ov_between")

        diff_H = maximum(abs.(Matrix(H_trotter) .- H_create))
        println("Max diff |H_trotter - H_create| = $diff_H")

        if diff_H > 1e-10
            diff_pos = findall(x -> abs(x) > 1e-5, Matrix(H_trotter) .- H_create)
            println("Number of differing matrix elements: $(length(diff_pos))")
            for idx in diff_pos[1:min(5, end)]
                r, c = idx[1], idx[2]
                println("  ($r, $c): H_trotter = $(H_trotter[r, c]), H_create = $(H_create[r, c])")
            end
        end
    end
end
