#=
test_4x3_benchmark.jl

Benchmark script to compare forward and gradient timing of Trotter optimization
on N=(5, 4)_4x3 dataset between CPU (1 core) and GPU.

Usage:
  julia --project=.. test_4x3_benchmark.jl [--use_gpu=<bool>] [--num_exponentials=<int>]
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
using JLD2
using HDF5
using Zygote

include("../data_path.jl")
include("../logging.jl")
include("../utility_functions.jl")
using .UtilityFunctions
include("../trotter.jl")
using .Trotter
include("../ed_objects.jl")
include("../ed_functions.jl")

function parse_benchmark_args(args::Vector{String})
    use_gpu = false
    num_exponentials = 1
    for arg in args
        if arg == "--use_gpu" || arg == "--use_gpu=true"
            use_gpu = true
        elseif arg == "--use_gpu=false"
            use_gpu = false
        elseif startswith(arg, "--num_exponentials=")
            num_exponentials = parse(Int, split(arg, "=", limit=2)[2])
        end
    end
    return use_gpu, num_exponentials
end

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "test_4x3_benchmark")
    with_logging(log_path) do
        use_gpu, num_exp = parse_benchmark_args(ARGS)
        println("==========================================================")
        println("TROTTER OPTIMIZATION 4x3 BENCHMARK")
        println("==========================================================")
        println("Requested use_gpu: $use_gpu")
        println("Num exponentials:  $num_exp")
        println("Threads available: $(Threads.nthreads())")
        if @isdefined(CUDA)
            println("CUDA loaded & functional: $(CUDA.functional())")
            if CUDA.functional()
                println("GPU Device: $(CUDA.name(CUDA.device()))")
            end
        end

        folder = data_folder("N=(5, 4)_4x3")
        println("Loading dataset from: $folder")
        t_load = @elapsed begin
            U_values, state_vecs, indexer, _, N_elec, spin_conserved, _, sign_convention =
                load_ED_data(folder; verbose=false, sign_convention=:spin_first)
        end
        println("ED data loaded in $(round(t_load, digits=3)) s")

        n_up, n_dn = N_elec
        Lvec = (4, 3)
        N_sites = prod(Lvec)

        basis_sector = Trotter.get_basis_sector(indexer, Lvec, N_sites)
        dim = length(basis_sector)
        println("Basis sector dimension: $dim")

        gates = Trotter.enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true, include_diagonal=true)
        tau_terms = Trotter.fgateToTauSector(gates, N_sites, basis_sector; antihermitian=false)
        num_gates = length(gates)
        M = num_exp * num_gates
        println("Number of gates: $num_gates, total parameters M: $M")

        ref = state_vecs[1, :]
        target = state_vecs[2, :]
        Random.seed!(42)
        A = (2 * rand(M) .- 1) * 0.05

        # Warmup run
        println("\nPerforming warmup run...")
        loss_warmup = Trotter.TrotterOptimization.adjoint_loss(
            A, gates, tau_terms, ref, target, basis_sector, N_sites;
            num_exponentials=num_exp, antihermitian=false, use_gpu=use_gpu
        )
        grad_warmup = Zygote.gradient(A) do x
            Trotter.TrotterOptimization.adjoint_loss(
                x, gates, tau_terms, ref, target, basis_sector, N_sites;
                num_exponentials=num_exp, antihermitian=false, use_gpu=use_gpu
            )
        end[1]
        println("Warmup loss: $loss_warmup, grad norm: $(norm(grad_warmup))")

        # Benchmark 5 iterations of forward pass
        println("\nBenchmarking Forward Pass (5 runs)...")
        t_fwd_list = Float64[]
        for r in 1:5
            t = @elapsed begin
                l_val = Trotter.TrotterOptimization.adjoint_loss(
                    A, gates, tau_terms, ref, target, basis_sector, N_sites;
                    num_exponentials=num_exp, antihermitian=false, use_gpu=use_gpu
                )
            end
            push!(t_fwd_list, t)
        end
        fwd_avg = mean(t_fwd_list)
        println("Forward Pass Times (s): ", round.(t_fwd_list, digits=4))
        println("Average Forward Pass Time: $(round(fwd_avg, digits=4)) s")

        # Benchmark 5 iterations of gradient pass (forward + pullback)
        println("\nBenchmarking Gradient Pass (Zygote rrule) (5 runs)...")
        t_grad_list = Float64[]
        for r in 1:5
            t = @elapsed begin
                g_val = Zygote.gradient(A) do x
                    Trotter.TrotterOptimization.adjoint_loss(
                        x, gates, tau_terms, ref, target, basis_sector, N_sites;
                        num_exponentials=num_exp, antihermitian=false, use_gpu=use_gpu
                    )
                end[1]
            end
            push!(t_grad_list, t)
        end
        grad_avg = mean(t_grad_list)
        println("Gradient Pass Times (s): ", round.(t_grad_list, digits=4))
        println("Average Gradient Pass Time: $(round(grad_avg, digits=4)) s")

        println("\n==========================================================")
        println("BENCHMARK COMPLETE")
        println("Mode: $(use_gpu ? "GPU" : "CPU (1 core)")")
        println("Forward Avg:  $(round(fwd_avg, digits=4)) s")
        println("Gradient Avg: $(round(grad_avg, digits=4)) s")
        println("==========================================================")
        return 0
    end
end
