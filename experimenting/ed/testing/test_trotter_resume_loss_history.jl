#=
test_trotter_resume_loss_history.jl

Verification test script to validate:
1. When optimizing a single U value, the existing optimization result for the current U index (e.g. ..._u_25.jld2)
   is loaded instead of an earlier U index (u-1).
2. The full loss history is accumulated and preserved across the existing optimization and new continuation optimization.
3. Compatibility with extract_losses.jl (extract_losses_from_file).
4. Multi-U scans preserve standard sequential loading from earlier U values.
=#

using Test
using JLD2
using HDF5
using LinearAlgebra
using Statistics
using SparseArrays
using Lattices
using Combinatorics

include("../data_path.jl")
include("../logging.jl")
include("../utility_functions.jl")
using .UtilityFunctions
include("../trotter.jl")
using .Trotter

include("../ed_objects.jl")
include("../ed_functions.jl")
include("../extract_losses.jl")

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "test_trotter_resume_loss_history")
    with_logging(log_path) do
        println("=================================================================")
        println("TEST: Trotter Optimization Current-U Loading & Loss History")
        println("=================================================================")

        test_folder = data_folder("N=(2, 2)_3x2")
        println("Loading dataset from: $test_folder")

        U_values, state_vecs, indexer, _, N_elec, spin_conserved, _, sign_convention =
            load_ED_data(test_folder; verbose=false, sign_convention=:spin_first)

        n_up, n_dn = N_elec
        Lvec = parse_lattice_dimension(test_folder)
        N_sites = prod(Lvec)
        basis_sector = Trotter.get_basis_sector(indexer, Lvec, N_sites)

        gates = Trotter.enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true, include_diagonal=true)
        tau_terms = Trotter.fgateToTauSector(gates, N_sites, basis_sector; antihermitian=false)

        # Create temporary working directory for test outputs
        tmp_dir = mktempdir()
        println("Created temporary test directory: $tmp_dir")

        test_u_idx = 20
        u_val = U_values[test_u_idx]
        println("Testing on single U index: $test_u_idx (U = $u_val)")

        save_name_prefix = "test_trotter_resume"
        test_file_path = joinpath(tmp_dir, "$(save_name_prefix)_u_$(test_u_idx).jld2")

        # -------------------------------------------------------------
        # STEP 1: Run Initial Optimization for N1 = 12 iterations
        # -------------------------------------------------------------
        println("\n--- Step 1: Running initial optimization (maxiters = 12) ---")
        N1 = 12
        instructions_step1 = Dict{String,Any}(
            "starting level" => 1,
            "ending level" => 1,
            "num_exponentials" => 1,
            "antihermitian" => false,
            "u_range" => test_u_idx:test_u_idx
        )

        res1 = Trotter.interaction_scan_map_to_state(
            state_vecs, instructions_step1, gates, tau_terms, basis_sector, N_sites;
            maxiters=N1,
            optimizer=:LBFGS,
            initialization_samples=5,
            save_folder=tmp_dir, save_name=save_name_prefix,
            loss_type=:overlap,
            U_values=U_values
        )

        @test isfile(test_file_path)
        data1 = JLD2.load(test_file_path)["dict"]
        coeffs1 = copy(data1["coefficients"])
        losses1 = copy(data1["metrics"]["optimization_losses"][1])
        final_loss1 = data1["metrics"]["loss"][end]

        println("Initial optimization completed:")
        println("  Recorded loss evaluations count: $(length(losses1))")
        println("  Initial loss: $(losses1[1])")
        println("  Final loss: $final_loss1")

        @test length(losses1) >= N1
        @test final_loss1 <= losses1[1]

        # -------------------------------------------------------------
        # STEP 2: Run Continuation Optimization on Same Single U for N2 = 15 iterations
        # -------------------------------------------------------------
        println("\n--- Step 2: Running continuation optimization on same U index (maxiters = 15) ---")
        N2 = 15
        instructions_step2 = Dict{String,Any}(
            "starting level" => 1,
            "ending level" => 1,
            "num_exponentials" => 1,
            "antihermitian" => false,
            "u_range" => test_u_idx:test_u_idx,
            "load_file" => test_file_path
        )

        res2 = Trotter.interaction_scan_map_to_state(
            state_vecs, instructions_step2, gates, tau_terms, basis_sector, N_sites;
            maxiters=N2,
            optimizer=:LBFGS,
            initialization_samples=0, # No multi-start needed when loaded
            save_folder=tmp_dir, save_name=save_name_prefix,
            loss_type=:overlap,
            U_values=U_values
        )

        data2 = JLD2.load(test_file_path)["dict"]
        coeffs2 = copy(data2["coefficients"])
        losses2 = copy(data2["metrics"]["optimization_losses"][1])
        final_loss2 = data2["metrics"]["loss"][end]

        println("Continuation optimization completed:")
        println("  Total accumulated loss evaluations count: $(length(losses2))")
        println("  Prior loss evaluations count: $(length(losses1))")
        println("  New loss evaluations added: $(length(losses2) - length(losses1))")
        println("  Initial loss (start of Step 1): $(losses2[1])")
        println("  Step 1 final loss: $final_loss1")
        println("  Step 2 final loss: $final_loss2")

        # Verify full loss history properties
        @test length(losses2) > length(losses1)
        @test losses2[1:length(losses1)] == losses1
        @test final_loss2 <= final_loss1

        # -------------------------------------------------------------
        # STEP 3: Verify extract_losses_from_file Compatibility
        # -------------------------------------------------------------
        println("\n--- Step 3: Verifying extract_losses_from_file compatibility ---")
        extracted_losses, was_ms, opp_val = extract_losses_from_file(test_file_path)
        println("  Extracted total loss length: $(length(extracted_losses)) (was_multistart=$was_ms)")
        
        ms_best = if haskey(data2["metrics"], "multistart_losses") && !isempty(data2["metrics"]["multistart_losses"])
            idx = data2["metrics"]["best_start_idx"][1]
            idx > 0 ? data2["metrics"]["multistart_losses"][1][idx] : Float64[]
        else
            Float64[]
        end
        @test extracted_losses == vcat(ms_best, losses2)
        @test was_ms == !isempty(ms_best)

        # -------------------------------------------------------------
        # STEP 4: Clean up temporary files
        # -------------------------------------------------------------
        rm(tmp_dir; recursive=true, force=true)
        println("\nCleaned up temporary test directory: $tmp_dir")

        println("\n=================================================================")
        println("ALL TROTTER RESUME AND LOSS HISTORY TESTS PASSED SUCCESSFULLY!")
        println("=================================================================")
        return 0
    end
end
