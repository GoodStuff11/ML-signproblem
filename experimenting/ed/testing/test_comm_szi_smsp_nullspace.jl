#=
test_comm_szi_smsp_nullspace.jl

Verification script to compute the intersection of nullspaces of the operators
[S^z_i, S_- S_+] for all lattice sites i, using functions in ed_functions.jl.
=#

using Test
using LinearAlgebra
using SparseArrays
using Combinatorics
using Lattices
using Dates

include(joinpath(@__DIR__, "..", "utility_functions.jl"))
include(joinpath(@__DIR__, "..", "ed_objects.jl"))
include(joinpath(@__DIR__, "..", "ed_functions.jl"))
include(joinpath(@__DIR__, "..", "logging.jl"))

function run_nullspace_tests()
    println("================================================================================")
    println("TEST SUITE: Intersection of Nullspaces of [S^z_i, S_- S_+] for all i")
    println("================================================================================")

    @testset "1. Operator Construction and Properties on 2x2 Subspace (2,2)" begin
        println("\n--- 1. Testing Operator Construction on 2x2 with (2,2) Fermions ---")
        lattice_2x2 = Square((2, 2), Periodic())
        Hs_2x2 = HubbardSubspace(2, 2, lattice_2x2)
        dim_2x2 = get_subspace_dimension(Hs_2x2)
        indexer_2x2 = CombinationIndexer(Hs_2x2; order=ColSnake())
        nsites_2x2 = length(indexer_2x2.a)

        # Test create_operator(:Sz)
        Sz_op = create_operator(Hs_2x2, :Sz)
        # Total Sz should be 0 on (2,2)
        @test norm(Sz_op) ≈ 0.0

        # Test create_operator(:SmSp) and create_operator(:SpSm)
        SmSp = create_operator(Hs_2x2, :SmSp)
        SpSm = create_operator(Hs_2x2, :SpSm)
        S2 = create_operator(Hs_2x2, :S2)

        # On (2,2) sector (Sz = 0):
        # S^2 = Sz(Sz+1) + SmSp = SmSp
        # S^2 = Sz(Sz-1) + SpSm = SpSm
        @test norm(SmSp - S2) < 1e-12
        @test norm(SpSm - S2) < 1e-12

        # Test local Szi operators
        Szi_ops = [create_operator(Hs_2x2, :Szi; site=s) for s in 1:nsites_2x2]
        @test length(Szi_ops) == nsites_2x2
        # Sum of local Szi must equal total Sz
        sum_Szi = sum(Szi_ops)
        @test norm(sum_Szi - Sz_op) < 1e-12

        # Test commutator creation directly via create_operator
        comm_ops = [create_operator(Hs_2x2, :comm_Szi_SmSp; site=s) for s in 1:nsites_2x2]
        for s in 1:nsites_2x2
            expected_comm = Szi_ops[s] * SmSp - SmSp * Szi_ops[s]
            @test norm(comm_ops[s] - expected_comm) < 1e-12
        end

        # Sum of [S^z_i, S_- S_+] over all i must be identically zero since [S_total^z, S_- S_+] = 0
        sum_comms = sum(comm_ops)
        @test norm(sum_comms) < 1e-12
        println("Sum of [S^z_i, S_- S_+] over all sites is identically zero (norm = $(norm(sum_comms))).")

        # Compute nullspace intersection using compute_comm_Szi_SmSp_nullspace_intersection
        null_basis, null_dim, retrieved_comms = compute_comm_Szi_SmSp_nullspace_intersection(Hs_2x2)
        println("Subspace dimension: $dim_2x2, Nullspace intersection dimension: $null_dim")
        @test null_dim == 6
        @test size(null_basis) == (dim_2x2, 6)

        # Verify that EVERY vector in the nullspace basis is strictly annihilated by ALL [S^z_i, S_- S_+]
        max_error = 0.0
        for col in 1:null_dim
            v = null_basis[:, col]
            for C in comm_ops
                err = norm(C * v)
                max_error = max(max_error, err)
                @test err < 1e-10
            end
        end
        println("Max ||[S^z_i, S_- S_+] v|| for all basis vectors and sites: $max_error")

        # Verify orthonormal basis
        @test norm(adjoint(null_basis) * null_basis - I(null_dim)) < 1e-12

        # Inspect the 6 basis states
        println("Identified $(null_dim) doubly-occupied singlet states forming the nullspace:")
        for col in 1:null_dim
            v = null_basis[:, col]
            active_idx = findall(abs.(v) .> 1e-4)
            @test length(active_idx) == 1
            conf = indexer_2x2.inv_comb_dict[active_idx[1]]
            up_coords = [(c.coordinates[1]-1, c.coordinates[2]-1) for c in conf[1]]
            dn_coords = [(c.coordinates[1]-1, c.coordinates[2]-1) for c in conf[2]]
            @test conf[1] == conf[2]  # Double occupancy condition: up == dn
            println("  State $col: occupied sites = $up_coords")
        end
    end

    @testset "2. Nullspace Intersection Across Multiple Electron Sectors on 2x2 Lattice" begin
        println("\n--- 2. Testing Multiple Sectors on 2x2 Lattice ---")
        lattice_2x2 = Square((2, 2), Periodic())
        test_cases = [
            # (N_up, N_down, expected_dim, expected_null_dim)
            (0, 0, 1, 1),
            (1, 0, 4, 4),
            (1, 1, 16, 4),   # 4 doubly occupied states
            (2, 0, 6, 6),
            (2, 1, 24, 12),
            (2, 2, 36, 6),   # 6 doubly occupied states
            (3, 0, 4, 4),
            (3, 1, 16, 12),
            (3, 2, 24, 12),
            (3, 3, 16, 4),   # 4 triply doubly occupied states
            (4, 0, 1, 1),
            (4, 4, 1, 1)    # fully filled (1 doubly occupied state)
        ]

        for (nup, ndn, exp_dim, exp_null) in test_cases
            Hs = HubbardSubspace(nup, ndn, lattice_2x2)
            dim = get_subspace_dimension(Hs)
            @test dim == exp_dim

            null_basis, null_dim, comm_ops = compute_comm_Szi_SmSp_nullspace_intersection(Hs)
            @test null_dim == exp_null

            # Numerical annihilation check
            for col in 1:null_dim
                v = null_basis[:, col]
                for C in comm_ops
                    @test norm(C * v) < 1e-10
                end
            end
            println("  Sector ($nup, $ndn): dim = $(rpad(dim, 3)) | nullspace intersection dim = $(rpad(null_dim, 3)) (Passed)")
        end
    end

    @testset "3. 1D 4-Site Chain and 3x2 Lattice Tests" begin
        println("\n--- 3. Testing 1D 4-Site Chain & 3x2 Lattice ---")
        chain_4 = Square((4, 1), Periodic())
        Hs_chain = HubbardSubspace(2, 2, chain_4)
        dim_chain = get_subspace_dimension(Hs_chain)
        null_basis_c, null_dim_c, comms_c = compute_comm_Szi_SmSp_nullspace_intersection(Hs_chain)
        @test dim_chain == 36
        @test null_dim_c == 6
        for col in 1:null_dim_c
            v = null_basis_c[:, col]
            for C in comms_c
                @test norm(C * v) < 1e-10
            end
        end
        println("  4-site chain (2,2): dim = 36 | nullspace intersection dim = $null_dim_c (Passed)")

        # 3x2 lattice with (3,3) fermions
        lattice_3x2 = Square((3, 2), Periodic())
        Hs_3x2 = HubbardSubspace(3, 3, lattice_3x2)
        dim_3x2 = get_subspace_dimension(Hs_3x2)
        println("  3x2 lattice (3,3): total subspace dim = $dim_3x2")
        null_basis_3x2, null_dim_3x2, comms_3x2 = compute_comm_Szi_SmSp_nullspace_intersection(Hs_3x2)
        # For 6 sites with 3 pairs, expected doubly-occupied nullspace states = binomial(6, 3) = 20
        @test null_dim_3x2 == 20
        for col in 1:null_dim_3x2
            v = null_basis_3x2[:, col]
            for C in comms_3x2
                @test norm(C * v) < 1e-10
            end
        end
        println("  3x2 lattice (3,3): dim = 400 | nullspace intersection dim = $null_dim_3x2 (Passed)")
    end

    println("\n================================================================================")
    println("ALL NULLSPACE INTERSECTION TESTS COMPLETED SUCCESSFULLY!")
    println("================================================================================")
end

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "test_comm_szi_smsp_nullspace")
    with_logging(log_path) do
        run_nullspace_tests()
    end
end
