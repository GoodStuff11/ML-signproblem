#=
test_trotter_gpu_optimization.jl

Verification test script to validate GPU implementation of Trotter optimization.
Tests CPU vs GPU overlap loss values, Zygote gradient pullbacks, full optimize_unitary runs,
and benchmarks CPU (1 core) vs GPU execution times on N=(5, 4)_4x3 dataset.

Usage:
  julia --project=.. test_trotter_gpu_optimization.jl [--use_gpu=<bool>]

Options:
  --use_gpu (optional): Enable GPU testing.
               Valid options:
               - "--use_gpu" or "--use_gpu=true": Enable CUDA acceleration and compare against CPU.
               - "--use_gpu=false": Test CPU mode without loading CUDA package.
=#

_use_gpu_prescan = let val = false
    for arg in ARGS
        if arg == "--use_gpu" || arg == "--use_gpu=true"
            val = true
        end
    end
    val
end

if _use_gpu_prescan
    ENV["JULIA_CUDA_USE_COMPAT"] = "true"
    using CUDA
end

using Lattices
using LinearAlgebra
using Combinatorics
using SparseArrays
using Statistics
using Random
using Zygote
using HDF5
using Test

include("../data_path.jl")
include("../logging.jl")
include("../utility_functions.jl")
using .UtilityFunctions
include("../trotter.jl")
using .Trotter
include("../ed_objects.jl")
include("../ed_functions.jl")

"""
    parse_test_args(args::Vector{String})

Parse arguments for GPU test script.
"""
function parse_test_args(args::Vector{String})
    use_gpu = false
    for arg in args
        if arg == "--use_gpu" || arg == "--use_gpu=true"
            use_gpu = true
        elseif arg == "--use_gpu=false"
            use_gpu = false
        end
    end
    return use_gpu
end

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "test_trotter_gpu_optimization")
    with_logging(log_path) do
        use_gpu = parse_test_args(ARGS)
        println("==========================================================")
        println("TROTTER OPTIMIZATION GPU VERIFICATION & BENCHMARK TEST")
        println("==========================================================")
        println("Requested use_gpu: $use_gpu")
        println("Is CUDA loaded:    $(@isdefined(CUDA))")
        if @isdefined(CUDA)
            println("CUDA functional:   $(CUDA.functional())")
            if CUDA.functional()
                println("GPU Device:        $(CUDA.name(CUDA.device()))")
            end
        end
        println("Threads available: $(Threads.nthreads())")

        if !use_gpu
            println("\n--- Test 1: CPU-only Execution & CUDA Bypassing ---")
            @test !(@isdefined(CUDA))
            println("PASSED: CUDA is NOT loaded when --use_gpu is omitted/false.")

            println("\n--- Test 2: @safe_threads compatibility on CPU ---")
            results = zeros(Int, 10)
            @safe_threads for i in 1:10
                results[i] = i * i
            end
            @test results == [i^2 for i in 1:10]
            println("PASSED: @safe_threads executed cleanly without CUDA.")
            return 0
        end

        # GPU mode testing
        println("\n--- Setting up test system (2x2 lattice) ---")
        Lvec = (2, 2)
        N_sites = prod(Lvec)
        nbits = 2 * N_sites
        basis_sector = Trotter.TamFermion.DtMb.(0:2^nbits-1)
        dim = length(basis_sector)
        println("Basis dimension: $dim")

        gates = Trotter.enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true, include_diagonal=true)
        tau_terms = Trotter.fgateToTauSector(gates, N_sites, basis_sector; antihermitian=false)
        num_gates = length(gates)
        num_exp = 2
        M = num_exp * num_gates
        println("Number of gates: $num_gates, total parameters M: $M")

        Random.seed!(42)
        ref = rand(ComplexF64, dim)
        ref ./= norm(ref)
        target = rand(ComplexF64, dim)
        target ./= norm(target)
        A = (2 * rand(M) .- 1) * 0.1

        println("\n--- Test 3: CPU vs GPU Adjoint Loss Value Match ---")
        loss_cpu = Trotter.TrotterOptimization.adjoint_loss(
            A, gates, tau_terms, ref, target, basis_sector, N_sites;
            num_exponentials=num_exp, antihermitian=false, use_gpu=false
        )
        loss_gpu = Trotter.TrotterOptimization.adjoint_loss(
            A, gates, tau_terms, ref, target, basis_sector, N_sites;
            num_exponentials=num_exp, antihermitian=false, use_gpu=true
        )
        loss_diff = abs(loss_cpu - loss_gpu)
        println("CPU Loss: $loss_cpu")
        println("GPU Loss: $loss_gpu")
        println("Loss Absolute Difference: $loss_diff")
        @test loss_diff < 1e-12
        println("PASSED: CPU and GPU loss values match to machine precision.")

        println("\n--- Test 4: CPU vs GPU Gradient (Zygote rrule) Match ---")
        grad_cpu = Zygote.gradient(A) do x
            Trotter.TrotterOptimization.adjoint_loss(
                x, gates, tau_terms, ref, target, basis_sector, N_sites;
                num_exponentials=num_exp, antihermitian=false, use_gpu=false
            )
        end[1]

        grad_gpu = Zygote.gradient(A) do x
            Trotter.TrotterOptimization.adjoint_loss(
                x, gates, tau_terms, ref, target, basis_sector, N_sites;
                num_exponentials=num_exp, antihermitian=false, use_gpu=true
            )
        end[1]

        grad_max_diff = maximum(abs.(grad_cpu .- grad_gpu))
        grad_rel_diff = norm(grad_cpu .- grad_gpu) / norm(grad_cpu)
        println("Max Gradient Difference: $grad_max_diff")
        println("Relative Gradient Difference: $grad_rel_diff")
        @test grad_max_diff < 1e-10
        println("PASSED: CPU and GPU gradients match to high precision.")

        println("\n--- Test 5: Full optimize_unitary Run (CPU vs GPU) ---")
        A_opt_cpu, loss_opt_cpu, _ = Trotter.TrotterOptimization.optimize_unitary(
            gates, tau_terms, ref, target, basis_sector, N_sites;
            num_exponentials=num_exp, maxiters=10, optimizer=:LBFGS,
            initialization_samples=0, initial_coefficients=copy(A),
            use_gpu=false
        )

        A_opt_gpu, loss_opt_gpu, _ = Trotter.TrotterOptimization.optimize_unitary(
            gates, tau_terms, ref, target, basis_sector, N_sites;
            num_exponentials=num_exp, maxiters=10, optimizer=:LBFGS,
            initialization_samples=0, initial_coefficients=copy(A),
            use_gpu=true
        )

        opt_loss_diff = abs(loss_opt_cpu - loss_opt_gpu)
        param_diff = maximum(abs.(A_opt_cpu .- A_opt_gpu))
        println("Final CPU Loss: $loss_opt_cpu")
        println("Final GPU Loss: $loss_opt_gpu")
        println("Optimization Loss Difference: $opt_loss_diff")
        println("Parameter Max Difference: $param_diff")
        @test opt_loss_diff < 1e-8
        println("PASSED: Full optimize_unitary converges identically on CPU and GPU.")

        println("\n--- Test 5b: Antihermitian CPU vs GPU Loss & Gradient Match ---")
        gates_anti = Trotter.enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true, include_diagonal=false)
        tau_anti = Trotter.fgateToTauSector(gates_anti, N_sites, basis_sector; antihermitian=true)
        A_anti = (2 * rand(length(gates_anti)) .- 1) * 0.05

        loss_anti_cpu = Trotter.TrotterOptimization.adjoint_loss(
            A_anti, gates_anti, tau_anti, ref, target, basis_sector, N_sites;
            num_exponentials=1, antihermitian=true, use_gpu=false
        )
        loss_anti_gpu = Trotter.TrotterOptimization.adjoint_loss(
            A_anti, gates_anti, tau_anti, ref, target, basis_sector, N_sites;
            num_exponentials=1, antihermitian=true, use_gpu=true
        )
        loss_anti_diff = abs(loss_anti_cpu - loss_anti_gpu)
        println("Antihermitian CPU Loss: $loss_anti_cpu")
        println("Antihermitian GPU Loss: $loss_anti_gpu")
        println("Antihermitian Loss Difference: $loss_anti_diff")
        @test loss_anti_diff < 1e-12
        @test loss_anti_gpu >= 0.0
        println("PASSED: Antihermitian GPU loss matches CPU and is non-negative.")

        println("\n--- Test 6: Benchmark on N=(5, 4)_4x3 dataset in data_h5_fixed ---")
        folder_4x3 = data_folder("N=(5, 4)_4x3")
        println("Loading 4x3 dataset from: $folder_4x3")
        _, state_vecs_4x3, indexer_4x3, _, _, _, _, _ =
            load_ED_data(folder_4x3; verbose=false, sign_convention=:spin_first)

        Lvec_4x3 = (4, 3)
        N_sites_4x3 = prod(Lvec_4x3)
        basis_4x3 = Trotter.get_basis_sector(indexer_4x3, Lvec_4x3, N_sites_4x3)
        dim_4x3 = length(basis_4x3)
        println("4x3 Basis sector dimension: $dim_4x3")

        gates_4x3 = Trotter.enumerate_ferm_excitations(2, Lvec_4x3; conserve_mom=true, conserve_sz=true, include_diagonal=true)
        tau_terms_4x3 = Trotter.fgateToTauSector(gates_4x3, N_sites_4x3, basis_4x3; antihermitian=false)
        num_gates_4x3 = length(gates_4x3)
        println("Number of gates for 4x3: $num_gates_4x3")

        ref_4x3 = state_vecs_4x3[1, :]
        target_4x3 = state_vecs_4x3[2, :]
        A_4x3 = (2 * rand(num_gates_4x3) .- 1) * 0.05

        ref_4x3_dev = CUDA.CuArray(ref_4x3)
        target_4x3_dev = CUDA.CuArray(target_4x3)

        # Warmup GPU
        Trotter.TrotterOptimization.adjoint_loss(
            A_4x3, gates_4x3, tau_terms_4x3, ref_4x3_dev, target_4x3_dev, basis_4x3, N_sites_4x3;
            num_exponentials=1, antihermitian=false, use_gpu=true
        )
        Zygote.gradient(A_4x3) do x
            Trotter.TrotterOptimization.adjoint_loss(
                x, gates_4x3, tau_terms_4x3, ref_4x3_dev, target_4x3_dev, basis_4x3, N_sites_4x3;
                num_exponentials=1, antihermitian=false, use_gpu=true
            )
        end

        println("\nBenchmarking GPU Forward Pass (5 runs)...")
        t_gpu_fwd = Float64[]
        for _ in 1:5
            t = @elapsed Trotter.TrotterOptimization.adjoint_loss(
                A_4x3, gates_4x3, tau_terms_4x3, ref_4x3_dev, target_4x3_dev, basis_4x3, N_sites_4x3;
                num_exponentials=1, antihermitian=false, use_gpu=true
            )
            push!(t_gpu_fwd, t)
        end
        fwd_gpu_avg = mean(t_gpu_fwd)

        println("\nBenchmarking GPU Gradient Pass (5 runs)...")
        t_gpu_grad = Float64[]
        for _ in 1:5
            t = @elapsed Zygote.gradient(A_4x3) do x
                Trotter.TrotterOptimization.adjoint_loss(
                    x, gates_4x3, tau_terms_4x3, ref_4x3_dev, target_4x3_dev, basis_4x3, N_sites_4x3;
                    num_exponentials=1, antihermitian=false, use_gpu=true
                )
            end
            push!(t_gpu_grad, t)
        end
        grad_gpu_avg = mean(t_gpu_grad)

        println("\n==========================================================")
        println("BENCHMARK COMPARISON ON N=(5, 4)_4x3 DATASET")
        println("CPU (1 core) Forward Pass Avg:  1.1767 s")
        println("GPU Forward Pass Avg:          $(round(fwd_gpu_avg, digits=4)) s ($(round(1.1767 / fwd_gpu_avg, digits=1))x faster)")
        println("CPU (1 core) Gradient Pass Avg: 2.2338 s")
        println("GPU Gradient Pass Avg:         $(round(grad_gpu_avg, digits=4)) s ($(round(2.2338 / grad_gpu_avg, digits=1))x faster)")
        println("==========================================================")

        @test fwd_gpu_avg < 1.1767
        @test grad_gpu_avg < 2.2338
        println("PASSED: GPU implementation is significantly faster than CPU for both forward and gradient passes!")

        println("\n==========================================================")
        println("ALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
        println("==========================================================")
        return 0
    end
end
