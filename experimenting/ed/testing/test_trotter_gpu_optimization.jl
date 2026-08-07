#=
test_trotter_gpu_optimization.jl

Verification test script to validate GPU implementation of Trotter optimization.
Tests CPU vs GPU overlap loss values, Zygote gradient pullbacks, full optimize_unitary runs,
and verifies that CUDA is not imported when --use_gpu is not passed.

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
using Test

include("../data_path.jl")
include("../logging.jl")
include("../utility_functions.jl")
using .UtilityFunctions
include("../trotter.jl")
using .Trotter

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
        println("TROTTER OPTIMIZATION GPU VERIFICATION TEST")
        println("==========================================================")
        println("Requested use_gpu: $use_gpu")
        println("Is CUDA loaded:    $(@isdefined(CUDA))")
        if @isdefined(CUDA)
            println("CUDA functional:   $(CUDA.functional())")
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

        println("\n==========================================================")
        println("ALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
        println("==========================================================")
        return 0
    end
end
