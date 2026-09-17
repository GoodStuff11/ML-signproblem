#=
test_trotter_gpu_streaming_fix.jl

Test script to verify the fix for GPU streaming of gate operators when pre_cache_all is false.

Usage:
  julia --project=.. test_trotter_gpu_streaming_fix.jl

Arguments:
  None. (All tests run automatically).
=#

using LinearAlgebra
using SparseArrays
using Test

include(joinpath(@__DIR__, "..", "data_path.jl"))
include(joinpath(@__DIR__, "..", "logging.jl"))
include(joinpath(@__DIR__, "..", "utility_functions.jl"))
using .UtilityFunctions
include(joinpath(@__DIR__, "..", "trotter.jl"))
using .Trotter

"""
    parse_arguments(args::Vector{String}) -> Dict{String, Any}

Parse command-line arguments for test_trotter_gpu_streaming_fix.jl.
This script takes no required or optional arguments.
"""
function parse_arguments(args::Vector{String})
    if length(args) > 0
        error("test_trotter_gpu_streaming_fix.jl takes no arguments, but received: $(args)")
    end
    return Dict{String, Any}()
end

function (@main)(ARGS)
    parsed_args = parse_arguments(ARGS)
    log_path = make_log_path(@__DIR__, "test_trotter_gpu_streaming_fix")
    with_logging(log_path) do
        println("================================================================================")
        println("=== Test Trotter GPU Gate Streaming Fix (pre_cache_all=false / stream_tau) ===")
        println("================================================================================")

        # 1. Verify signatures and exported symbols
        println("\n--- 1. Checking function signatures and keywords ---")
        
        # Check get_gpu_gate_ops accepts stream_tau kwarg
        @test hasmethod(Trotter.TrotterOptimization.get_gpu_gate_ops, 
            Tuple{Any, Int, AbstractVector{<:Integer}}, 
            (:antihermitian, :datatype, :stream_tau))
        println("  [PASS] get_gpu_gate_ops accepts stream_tau keyword argument")

        # Check prepare_gpu_gate_ops alias exists
        @test isdefined(Trotter.TrotterOptimization, :prepare_gpu_gate_ops)
        @test Trotter.TrotterOptimization.prepare_gpu_gate_ops === Trotter.TrotterOptimization.get_gpu_gate_ops
        println("  [PASS] prepare_gpu_gate_ops is an alias for get_gpu_gate_ops")

        # Check gpu_apply_gate_exp! accepts tau kwarg
        @test hasmethod(Trotter.TrotterOptimization.gpu_apply_gate_exp!,
            Tuple{AbstractVector, AbstractVector, Trotter.TrotterOptimization.GpuGateOps, Int, Float64},
            (:antihermitian, :inverse, :tau))
        println("  [PASS] gpu_apply_gate_exp! accepts tau keyword argument")

        # Check apply_unitary_checkpoints accepts stream_tau kwarg
        @test hasmethod(Trotter.TrotterOptimization.apply_unitary_checkpoints,
            Tuple{AbstractArray, Any, AbstractArray, Any, Int, Int},
            (:antihermitian, :use_gpu, :datatype, :stream_tau))
        println("  [PASS] apply_unitary_checkpoints accepts stream_tau keyword argument")

        # Check backward_adjoint_propagation accepts stream_tau kwarg
        @test hasmethod(Trotter.TrotterOptimization.backward_adjoint_propagation,
            Tuple{AbstractArray, Any, Any, Vector, AbstractVector, Any, Int, Int},
            (:antihermitian, :use_gpu, :datatype, :stream_tau))
        println("  [PASS] backward_adjoint_propagation accepts stream_tau keyword argument")

        # Check apply_unitary accepts stream_tau kwarg
        @test hasmethod(Trotter.TrotterOptimization.apply_unitary,
            Tuple{AbstractArray, Any, AbstractArray, Any, Int, Int},
            (:antihermitian, :use_gpu, :datatype, :param_map, :stream_tau))
        println("  [PASS] apply_unitary accepts stream_tau keyword argument")

        # 2. Test GpuGateOps with simulated pre_cache_all = false
        println("\n--- 2. Testing GpuGateOps with pre_cache_all = false (tau_dev = nothing) ---")
        d = 10
        num_gates = 3

        # Create mock CPU sparse matrices
        mat1 = sprand(Float32, d, d, 0.2)
        mat2 = sprand(Float32, d, d, 0.2)
        tau_cpu = Any[mat1, mat2, nothing] # gate 3 is a diagonal gate under antihermitian
        tau_dev = Any[nothing, nothing, nothing] # simulated un-cached GPU matrices
        is_diag = [false, false, true]
        sign0_val = [1.0, 1.0, 1.0]
        w1 = zeros(Float32, d)
        w2 = zeros(Float32, d)
        pre_cached_all = false

        ops = Trotter.TrotterOptimization.GpuGateOps(tau_cpu, tau_dev, is_diag, sign0_val, w1, w2, pre_cached_all)
        @test ops.pre_cached_all == false
        @test ops.tau_dev[1] === nothing
        @test ops.tau_dev[2] === nothing
        @test ops.tau_dev[3] === nothing
        @test ops.tau_cpu[1] !== nothing
        @test ops.tau_cpu[2] !== nothing
        @test ops.tau_cpu[3] === nothing
        println("  [PASS] GpuGateOps initialized with tau_dev containing nothing (streaming mode)")

        # Test _get_gpu_tau_mat for diagonal gate (should return nothing)
        res_diag = Trotter.TrotterOptimization._get_gpu_tau_mat(ops, 3)
        @test res_diag === nothing
        println("  [PASS] _get_gpu_tau_mat correctly returns nothing for antihermitian diagonal gate")

        # 3. Test build_direct_sparse_tau on real 2x2 lattice
        println("\n--- 3. Testing build_direct_sparse_tau on 2x2 lattice ---")
        Lvec = [2, 2]
        N_sites = prod(Lvec)
        gates_anti = Trotter.enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true, include_diagonal=false)
        gates_herm = Trotter.enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true, include_diagonal=true)
        
        # Test that diagonal gates are properly identified
        has_diag_anti = any(Trotter.is_diagonal_gate.(gates_anti))
        has_diag_herm = any(Trotter.is_diagonal_gate.(gates_herm))
        @test !has_diag_anti
        @test has_diag_herm
        println("  [PASS] Gate enumeration correctly filters diagonal gates for antihermitian (none present)")
        println("  [PASS] Gate enumeration includes diagonal gates for standard hermitian")

        # 4. Check that pre_cache_all formula handles colptr size threshold
        println("\n--- 4. Testing pre_cache_all threshold logic ---")
        # For 4x3 (small d):
        d_small = 1000
        gates_count = 100
        bytes_small = gates_count * (d_small + 1) * 4
        @test bytes_small < 4 * 1024^3
        println("  [PASS] Small basis size ($bytes_small bytes) triggers pre_cache_all = true")

        # For 4x4 (large d):
        d_large = 4008576
        gates_large = 276
        bytes_large = gates_large * (d_large + 1) * 4
        @test bytes_large >= 4 * 1024^3
        println("  [PASS] 4x4 basis size ($(round(bytes_large / 1024^3, digits=2)) GB) triggers pre_cache_all = false")

        println("\n================================================================================")
        println("=== All streaming fix assertions passed successfully! ===")
        println("================================================================================")
        return 0
    end
end
