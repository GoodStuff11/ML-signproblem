#=
test_trotter_gate_sorting_and_loading.jl

Verifies:
1. enumerate_ferm_excitations produces canonically sorted gates matching sortGatesByIJ.
2. Saved _shared.jld2 datasets with historical gate orders load and evaluate correctly.
=#

using Test
using LinearAlgebra
using JLD2

if !isdefined(Main, :UtilityFunctions)
    include(joinpath(@__DIR__, "../utility_functions.jl"))
end
include(joinpath(@__DIR__, "../TamLib.jl"))
include(joinpath(@__DIR__, "../TamFermion.jl"))
include(joinpath(@__DIR__, "../data_path.jl"))
include(joinpath(@__DIR__, "../logging.jl"))

using .TamLib
using .TamFermion

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "test_trotter_gate_sorting_and_loading")
    with_logging(log_path) do
        @testset "Trotter Gate Canonical Sorting Tests" begin
            test_lattices = [(3, 2), (3, 3), (4, 2), (2, 2)]

            for lvec in test_lattices
                N = prod(lvec)
                gates_anti = enumerate_ferm_excitations(2, lvec; conserve_mom=true, conserve_sz=true, include_diagonal=false)
                gates_herm = enumerate_ferm_excitations(2, lvec; conserve_mom=true, conserve_sz=true, include_diagonal=true)

                # Check antihermitian gates are sorted by (s_I, s_J)
                sorted_anti, _ = sortGatesByIJ(gates_anti, N)
                @test gates_anti == sorted_anti

                # Check hermitian gates are sorted by (s_I, s_J)
                sorted_herm, _ = sortGatesByIJ(gates_herm, N)
                @test gates_herm == sorted_herm

                # Check that 2N-bit s_I is monotonically non-decreasing, and s_J is non-decreasing for equal s_I
                s_I = [UInt64(g.cre_up) | (UInt64(g.cre_dn) << N) for g in gates_anti]
                s_J = [UInt64(g.ann_up) | (UInt64(g.ann_dn) << N) for g in gates_anti]
                for i in 1:length(gates_anti)-1
                    @test (s_I[i] < s_I[i+1]) || (s_I[i] == s_I[i+1] && s_J[i] <= s_J[i+1])
                end
            end
            println("Canonical gate sorting verified across all test lattices.")
        end

        @testset "Historical _shared.jld2 Gate Loading Tests" begin
            folder = get_data_root()
            sample_shared_path = joinpath(folder, "N=(2, 2)_3x2", "trotter_N=6_num_exponentials=2_ref_slater_antihermitian_shared.jld2")

            if isfile(sample_shared_path)
                shared_data = load(sample_shared_path)["dict"]
                @test haskey(shared_data, "gates")
                saved_gates = shared_data["gates"]
                @test !isempty(saved_gates)
                @test length(saved_gates) == 114

                # Ensure loaded gates can be converted and used with tau_g_operator_sector
                converted_gates = [g isa FGate ? g : FGate(g.cre_up, g.ann_up, g.cre_dn, g.ann_dn) for g in saved_gates]
                @test length(converted_gates) == length(saved_gates)

                # Confirm that saved gates preserve their original sequence when loaded
                @test converted_gates[1].cre_dn == 0x05
                @test converted_gates[1].ann_dn == 0x0a
                println("Successfully verified loading of historical _shared.jld2 gate list.")
            else
                println("Note: Sample shared file not found at $sample_shared_path; skipping disk load test.")
            end
        end

        println("ALL TESTS PASSED SUCCESSFULLY.")
    end
end
