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
    log_path = make_log_path(@__DIR__, "diagnose_tam_fermion_signs")
    with_logging(log_path) do
        println("=== Detailed Comparison of H_trotter vs H_create ===")

        Lvec = [3, 2]
        N_elec = (2, 2)
        u_val = 2.0
        lattice = Square(Tuple(Lvec), Periodic())

        for sign_conv in [:spin_first, :coordinate_first]
            println("\n==================================================")
            println("sign_convention = :$sign_conv")
            println("==================================================")

            order = sign_conv == :spin_first ? ColSnake() : RowSnake()
            subspace = HubbardSubspace(N_elec..., lattice; k=[0, 0])
            indexer = CombinationIndexer(subspace; order=order)

            # 1. H_trotter from TamFermion.HubbardMomentumBasis
            H_trotter, basis_trotter, _ = Trotter.TamFermion.HubbardMomentumBasis(
                1.0, u_val, Tuple(Lvec), N_elec;
                indexer=indexer, sign_convention=sign_conv, lattice_ordering=order
            )
            M_trotter = Matrix(H_trotter)

            # 2. H_create from create_Hubbard
            new_hopping = create_Hubbard(HubbardModel(1.0, 0.0, 0.0, false), subspace; 
                                indexer=indexer, momentum_basis=true, sign_convention=sign_conv, lattice_ordering=order)
            new_interaction = create_Hubbard(HubbardModel(0.0, 1.0, 0.0, false), subspace; 
                                indexer=indexer, momentum_basis=true, sign_convention=sign_conv, lattice_ordering=order)
            M_create = Matrix(new_hopping .+ u_val .* new_interaction)

            # 3. Check elementwise comparison
            diff = M_trotter .- M_create
            max_abs_diff = maximum(abs.(diff))
            println("Max absolute difference |H_trotter - H_create|: $max_abs_diff")

            if max_abs_diff > 1e-10
                # Find indices where they differ
                diff_indices = findall(x -> abs(x) > 1e-5, diff)
                println("Number of differing matrix elements: $(length(diff_indices))")
                println("First 5 differing entries (row, col, M_trotter, M_create):")
                for idx in diff_indices[1:min(5, end)]
                    r, c = idx[1], idx[2]
                    println("  ($r, $c): M_trotter = $(M_trotter[r, c]), M_create = $(M_create[r, c])")
                end
            else
                println("H_trotter and H_create match EXACTLY!")
            end

            E_trotter = eigvals(M_trotter)[1:3]
            E_create = eigvals(M_create)[1:3]
            println("E_trotter[1:3] = $E_trotter")
            println("E_create[1:3]  = $E_create")
        end
    end
end
