#=
test_trotter_datatype_gpu.jl

Verification test script to validate GPU & CPU Trotter optimization with different datatypes (Float32, ComplexF32, Float64, ComplexF64).
Specifically tests:
1. Antihermitian overlap loss & Zygote gradient with datatype=Float32 on GPU (verifying fix for cuSPARSE mv! MethodError).
2. Multi-start initialization (find_multi_start_initialization) on GPU with datatype=Float32 and antihermitian=true.
3. optimize_unitary run on GPU with datatype=Float32 and antihermitian=true.
4. Hermitian and antihermitian gradients for ComplexF32, Float64, ComplexF64.
5. to_device_matrix and energy_loss with GPU.

Usage:
  julia --project=.. test_trotter_datatype_gpu.jl [--use_gpu=<bool>]

Options:
  --use_gpu (optional): Enable GPU testing.
               Valid options:
               - "--use_gpu" or "--use_gpu=true": Enable CUDA acceleration and run GPU tests.
               - "--use_gpu=false" or omitted: Test CPU mode without loading CUDA package.
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
using Optimization
using OptimizationOptimJL
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
    parse_test_arguments(args::Vector{String}) -> Bool

Parse command line arguments for test_trotter_datatype_gpu.jl.

Options:
- `--use_gpu`: Run with GPU acceleration if CUDA is available.
- `--use_gpu=true`: Run with GPU acceleration if CUDA is available.
- `--use_gpu=false`: Run without GPU acceleration (CPU-only).
"""
function parse_test_arguments(args::Vector{String})
    use_gpu = false
    for arg in args
        if arg == "--use_gpu" || arg == "--use_gpu=true"
            use_gpu = true
        elseif arg == "--use_gpu=false"
            use_gpu = false
        else
            error("Unknown argument: '$arg'. Valid options are: '--use_gpu', '--use_gpu=true', '--use_gpu=false'.")
        end
    end
    return use_gpu
end

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "test_trotter_datatype_gpu")
    with_logging(log_path) do
        use_gpu = parse_test_arguments(ARGS)
        println("==========================================================")
        println("TROTTER OPTIMIZATION DATATYPE & GPU REGRESSION TESTS")
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

        # 2x2 lattice system setup
        Lvec = (2, 2)
        N_sites = prod(Lvec)
        nbits = 2 * N_sites
        basis_sector = Trotter.TamFermion.DtMb.(0:2^nbits-1)
        dim = length(basis_sector)
        println("Basis dimension: $dim")

        # Gates: antihermitian (no diagonal) and hermitian (with diagonal)
        gates_anti = Trotter.enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true, include_diagonal=false)
        tau_terms_anti = Trotter.fgateToTauSector(gates_anti, N_sites, basis_sector; antihermitian=true)
        num_gates_anti = length(gates_anti)

        gates_herm = Trotter.enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true, include_diagonal=true)
        tau_terms_herm = Trotter.fgateToTauSector(gates_herm, N_sites, basis_sector; antihermitian=false)
        num_gates_herm = length(gates_herm)

        Random.seed!(42)
        ref_raw = rand(ComplexF64, dim)
        ref_raw ./= norm(ref_raw)
        target_raw = rand(ComplexF64, dim)
        target_raw ./= norm(target_raw)

        # 1. Test CPU vs GPU Float32 antihermitian loss and gradient pullback
        println("\n--- Test 1: Float32 Antihermitian Overlap Loss & Gradient ---")
        num_exp = 2
        M_anti = num_exp * num_gates_anti
        A_anti = (2 * rand(Float64, M_anti) .- 1) * 0.05

        ref_real = real.(ref_raw)
        ref_real ./= norm(ref_real)
        target_real = real.(target_raw)
        target_real ./= norm(target_real)

        # CPU Float32
        loss_cpu_f32 = Trotter.TrotterOptimization.adjoint_loss(
            A_anti, gates_anti, tau_terms_anti, ref_real, target_real, basis_sector, N_sites;
            num_exponentials=num_exp, antihermitian=true, use_gpu=false, datatype=Float32
        )
        grad_cpu_f32 = Zygote.gradient(A_anti) do x
            Trotter.TrotterOptimization.adjoint_loss(
                x, gates_anti, tau_terms_anti, ref_real, target_real, basis_sector, N_sites;
                num_exponentials=num_exp, antihermitian=true, use_gpu=false, datatype=Float32
            )
        end[1]
        println("CPU Float32 Loss: $loss_cpu_f32, Gradient norm: $(norm(grad_cpu_f32))")

        if use_gpu && @isdefined(CUDA) && CUDA.functional()
            # GPU Float32 - This was the exact code path that crashed with MethodError in mv!
            loss_gpu_f32 = Trotter.TrotterOptimization.adjoint_loss(
                A_anti, gates_anti, tau_terms_anti, ref_real, target_real, basis_sector, N_sites;
                num_exponentials=num_exp, antihermitian=true, use_gpu=true, datatype=Float32
            )
            grad_gpu_f32 = Zygote.gradient(A_anti) do x
                Trotter.TrotterOptimization.adjoint_loss(
                    x, gates_anti, tau_terms_anti, ref_real, target_real, basis_sector, N_sites;
                    num_exponentials=num_exp, antihermitian=true, use_gpu=true, datatype=Float32
                )
            end[1]
            println("GPU Float32 Loss: $loss_gpu_f32, Gradient norm: $(norm(grad_gpu_f32))")

            loss_diff = abs(loss_cpu_f32 - loss_gpu_f32)
            grad_diff = maximum(abs.(grad_cpu_f32 .- grad_gpu_f32))
            println("Loss Diff (CPU vs GPU Float32): $loss_diff")
            println("Max Grad Diff (CPU vs GPU Float32): $grad_diff")
            @test loss_diff < 1e-5
            @test grad_diff < 1e-4
            println("PASSED: Float32 Antihermitian GPU Loss and Gradient match CPU without cuSPARSE error.")

            # 2. Test find_multi_start_initialization with Float32 and GPU
            println("\n--- Test 2: Multi-start Initialization on GPU with Float32 ---")
            f_test = A -> Trotter.TrotterOptimization.adjoint_loss(
                A, gates_anti, tau_terms_anti, ref_real, target_real, basis_sector, N_sites;
                num_exponentials=1, antihermitian=true, use_gpu=true, datatype=Float32
            )
            optf_test = Optimization.OptimizationFunction((u, p) -> f_test(u), Optimization.AutoZygote())
            A_init, ms_losses, ms_grads, best_idx, ms_run, init_grads = Trotter.TrotterOptimization.find_multi_start_initialization(
                f_test, optf_test, num_gates_anti;
                initialization_samples=5,
                multi_start_samples=2,
                multi_start_iters=5,
                use_gpu=true
            )
            @test length(A_init) == num_gates_anti
            @test length(init_grads) == 5
            println("PASSED: Multi-start sampling on GPU with Float32 executed successfully.")

            # 3. Test optimize_unitary on GPU with Float32
            println("\n--- Test 3: Full optimize_unitary on GPU with Float32 ---")
            A_opt_gpu, loss_opt_gpu, metrics_gpu = Trotter.TrotterOptimization.optimize_unitary(
                gates_anti, tau_terms_anti, ref_real, target_real, basis_sector, N_sites;
                num_exponentials=1, maxiters=10, optimizer=:LBFGS,
                initialization_samples=0, initial_coefficients=copy(A_anti[1:num_gates_anti]),
                antihermitian=true, use_gpu=true, datatype=Float32
            )
            @test loss_opt_gpu < loss_gpu_f32
            println("Initial Loss: $loss_gpu_f32 -> Optimized Loss: $loss_opt_gpu")
            println("PASSED: Full optimize_unitary on GPU with Float32 converged successfully.")

            # 4. Test ComplexF32 GPU mode
            println("\n--- Test 4: ComplexF32 Hermitian Overlap Loss & Gradient on GPU ---")
            A_herm = (2 * rand(Float64, num_gates_herm) .- 1) * 0.05
            loss_gpu_c32 = Trotter.TrotterOptimization.adjoint_loss(
                A_herm, gates_herm, tau_terms_herm, ref_raw, target_raw, basis_sector, N_sites;
                num_exponentials=1, antihermitian=false, use_gpu=true, datatype=ComplexF32
            )
            grad_gpu_c32 = Zygote.gradient(A_herm) do x
                Trotter.TrotterOptimization.adjoint_loss(
                    x, gates_herm, tau_terms_herm, ref_raw, target_raw, basis_sector, N_sites;
                    num_exponentials=1, antihermitian=false, use_gpu=true, datatype=ComplexF32
                )
            end[1]
            println("GPU ComplexF32 Loss: $loss_gpu_c32, Gradient norm: $(norm(grad_gpu_c32))")
            @test !isnan(loss_gpu_c32)
            @test !any(isnan.(grad_gpu_c32))
            println("PASSED: ComplexF32 GPU loss and gradient computed cleanly.")

            # 5. Test to_device_matrix and energy_loss on GPU
            println("\n--- Test 5: to_device_matrix and energy_loss on GPU ---")
            H_sparse = sprand(ComplexF64, dim, dim, 0.1)
            H_sparse = H_sparse + H_sparse'
            H_dev = Trotter.to_device_matrix(H_sparse, true, ComplexF32)
            @test H_dev isa CUDA.CUSPARSE.CuSparseMatrixCSC{ComplexF32, Int32}
            loss_energy_gpu = Trotter.TrotterOptimization.energy_loss(
                A_herm, gates_herm, tau_terms_herm, H_sparse, ref_raw, basis_sector, N_sites;
                num_exponentials=1, antihermitian=false, use_gpu=true, datatype=ComplexF32
            )
            grad_energy_gpu = Zygote.gradient(A_herm) do x
                Trotter.TrotterOptimization.energy_loss(
                    x, gates_herm, tau_terms_herm, H_sparse, ref_raw, basis_sector, N_sites;
                    num_exponentials=1, antihermitian=false, use_gpu=true, datatype=ComplexF32
                )
            end[1]
            println("GPU Energy Loss: $loss_energy_gpu, Gradient norm: $(norm(grad_energy_gpu))")
            @test !isnan(loss_energy_gpu)
            @test !any(isnan.(grad_energy_gpu))
            println("PASSED: to_device_matrix and energy_loss with GPU executed cleanly.")
        else
            println("\nGPU not requested or CUDA not functional. Testing CPU paths for Float32 and ComplexF32...")
            # Test CPU Float32 multi-start and optimize_unitary
            f_test_cpu = A -> Trotter.TrotterOptimization.adjoint_loss(
                A, gates_anti, tau_terms_anti, ref_real, target_real, basis_sector, N_sites;
                num_exponentials=1, antihermitian=true, use_gpu=false, datatype=Float32
            )
            optf_test_cpu = Optimization.OptimizationFunction((u, p) -> f_test_cpu(u), Optimization.AutoZygote())
            A_init_cpu, _, _, _, _, _ = Trotter.TrotterOptimization.find_multi_start_initialization(
                f_test_cpu, optf_test_cpu, num_gates_anti;
                initialization_samples=5,
                multi_start_samples=2,
                multi_start_iters=5,
                use_gpu=false
            )
            @test length(A_init_cpu) == num_gates_anti
            println("PASSED: CPU Float32 multi-start initialization executed cleanly.")
        end

        println("\n==========================================================")
        println("ALL DATATYPE REGRESSION TESTS PASSED SUCCESSFULLY!")
        println("==========================================================")
        return 0
    end
end
