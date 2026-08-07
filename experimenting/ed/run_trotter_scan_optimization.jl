#=
run_trotter_scan_optimization.jl

Run Trotter optimization over a range of U interaction parameters using unitaries mapped from Lanczos ED data,
represented in the momentum basis.

Usage:
  julia --project=.. run_trotter_scan_optimization.jl [folder] [u_start] [u_end] [--maxiters=<number>] [--loss=<type>] [--num_exponentials=<number>] [--antihermitian] [--custom_ref_state=<value>] [--use_gpu=<bool>]

Arguments:
  folder (required): Path to the ED data folder (e.g., "data/N=(2, 2)_2x2").
  u_start (optional): Start index of U values, or direction. Default: 25.
                     Valid options:
                     - "forward": Scan forward from index 26 to the end of U values.
                     - "backward": Scan backward from index 18 down to 1.
                     - [integer]: Run a single specific U-index or the start of a range.
  u_end (optional): End index of U values (if specifying a range). Default: nothing.
  --maxiters=<number> (optional): Maximum number of iterations for optimization. Default: 200.
  --loss=<type> (optional): The loss function to optimize. Default: "overlap".
                     Valid options:
                     - "overlap": Optimize overlap loss (1 - |<ψ'|U|ψ>|^2).
                     - "energy": Optimize energy loss (<ψ|U^† H U|ψ>).
  --num_exponentials=<number> (optional): Number of Trotter layers/steps. Default: 2.
  --antihermitian (optional): Use real-antihermitian generators instead of i * real-hermitian.
  --custom_ref_state=<value> (optional): Use a custom reference state as a Slater determinant.
                     Valid options:
                     - "slater": The Slater determinant ground state of the tight-binding model
                                 (with the lowest kinetic energy and overlap > 0.1).
                     - [integer]: Use the Slater determinant at this specific 1-based basis index.
  --use_gpu=<bool> (optional): Enable GPU acceleration for overlap loss and gradient calculations. Default: false.
                     Valid options:
                     - "--use_gpu" or "--use_gpu=true": Enable CUDA acceleration.
                     - "--use_gpu=false": Disable GPU acceleration (CUDA is not loaded to preserve @safe_threads compatibility).

Examples:
  julia --project=.. run_trotter_scan_optimization.jl "N=(2, 2)_2x2" 25 35
  julia --project=.. run_trotter_scan_optimization.jl "N=(2, 2)_2x2" forward --num_exponentials=3
  julia --project=.. run_trotter_scan_optimization.jl "N=(2, 2)_2x2" 25 40 --loss=energy --antihermitian
  julia --project=.. run_trotter_scan_optimization.jl "N=(2, 2)_2x2" 25 --custom_ref_state=slater
  julia --project=.. run_trotter_scan_optimization.jl "N=(2, 2)_2x2" 25 35 --use_gpu
=#

# Pre-scan ARGS for GPU flag before loading CUDA package
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

include("data_path.jl")
include("logging.jl")
include("utility_functions.jl")
using .UtilityFunctions
include("trotter.jl")
using .Trotter

include("ed_objects.jl")
include("ed_functions.jl")

"""
    parse_arguments(args::Vector{String})

Parse command line arguments for running Trotter scan optimization.
Expected arguments:
1. folder (String): The directory containing exact diagonalization data.
2. u_start (String): The starting U-index or direction ("forward", "backward"). Default: "25".
3. u_end (String): The ending U-index (optional).
4. --maxiters=<number> (Int): Optional maximum iterations parameter. Default: 200.
5. --loss=<type> (String): The loss function to optimize ("overlap", "energy"). Default: "overlap".
6. --num_exponentials=<number> (Int): Optional number of Trotter steps. Default: 1.
7. --antihermitian (flag): Use real-antihermitian generators. Default: false.
8. --custom_ref_state=<value> (String): Use a custom reference state as a Slater determinant.
9. --use_gpu=<bool> (flag/bool): Enable GPU acceleration for Trotter optimization. Default: false.
"""
function parse_arguments(args::Vector{String})
    if isempty(args)
        error("Usage: julia run_trotter_scan_optimization.jl <folder> <u_start> <u_end> [options]")
    end

    folder = data_folder(args[1])
    u_start = args[2]
    u_end = args[3]

    maxiters = 100
    loss_type = :overlap
    num_exponentials = 1
    antihermitian = false
    custom_ref_state_arg = nothing
    use_gpu = false

    for arg in args[4:end]
        if startswith(arg, "--maxiters=")
            val = String(split(arg, "=", limit=2)[2])
            maxiters = parse(Int, val)
        elseif startswith(arg, "--loss=")
            val = String(split(arg, "=", limit=2)[2])
            if val == "overlap"
                loss_type = :overlap
            elseif val == "energy"
                loss_type = :energy
            else
                error("Invalid --loss option: '$val'. Valid options are: 'overlap', 'energy'.")
            end
        elseif startswith(arg, "--num_exponentials=")
            num_exponentials = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg == "--antihermitian"
            antihermitian = true
        elseif startswith(arg, "--custom_ref_state=")
            custom_ref_state_arg = split(arg, "=", limit=2)[2]
        elseif arg == "--use_gpu" || arg == "--use_gpu=true"
            use_gpu = true
        elseif arg == "--use_gpu=false"
            use_gpu = false
        else
            error("Unknown argument: $arg")
        end
    end

    return folder, u_start, u_end, maxiters, loss_type, num_exponentials, antihermitian, custom_ref_state_arg, use_gpu
end

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "run_trotter_scan_optimization")
    with_logging(log_path) do
        folder, u_start, u_end, maxiters, loss_type, num_exponentials, antihermitian, custom_ref_state_arg, use_gpu = parse_arguments(ARGS)
        println("Number of threads: $(Threads.nthreads())")
        println("Use GPU: $use_gpu")
        # 1. Load ED data (loads indexer if JLD2, or we can use it to build the sector basis)
        U_values, state_vecs, indexer, _, N_elec, spin_conserved, _, sign_convention =
            load_ED_data(folder; verbose=true, sign_convention=:spin_first, use_slater_reference=custom_ref_state_arg == "slater")

        n_up, n_dn = N_elec

        # Parse dimension from folder name, default to (3, 3) if fails
        Lvec = parse_lattice_dimension(folder)
        N_sites = prod(Lvec)

        # 2. Computing the basis
        basis_sector = Trotter.get_basis_sector(indexer, Lvec, N_sites)

        # 3. Find the Hamiltonian
        @time H_hop_sector, basis_dict_sector, _ = Trotter.TamFermion.HubbardMomentumBasis(
            1.0, 0.0, Lvec, (n_up, n_dn); indexer=indexer
        )
        @time H_int_sector, _, _ = Trotter.TamFermion.HubbardMomentumBasis(
            0.0, 1.0, Lvec, (n_up, n_dn); indexer=indexer
        )

        # 4. Enumerate Trotter gates and tau terms
        @time gates = Trotter.enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true, include_diagonal=!antihermitian)
        @time tau_terms = Trotter.fgateToTauSector(gates, N_sites, basis_sector; antihermitian=antihermitian)

        # 5. Set up scan range
        scan_instructions = Dict{String,Any}(
            "starting level" => 1,
            "ending level" => 1,
            "num_exponentials" => num_exponentials,
            "antihermitian" => antihermitian
        )

        save_name_prefix = build_save_name_prefix(
            :trotter;
            sites=N_sites,
            electrons=N_elec,
            custom_ref_state_arg=custom_ref_state_arg,
            antihermitian=antihermitian,
            loss_type=loss_type
        )

        v1 = parse(Int, u_start)
        v2 = parse(Int, u_end)
        if v1 > v2
            scan_instructions["u_range"] = v1:-1:v2
            if isfile(joinpath(folder, "$(save_name_prefix)_u_$(v1+1).jld2"))
                scan_instructions["load_file"] = joinpath(folder, "$(save_name_prefix)_u_$(v1+1).jld2")
            end
        else
            scan_instructions["u_range"] = v1:v2
            if isfile(joinpath(folder, "$(save_name_prefix)_u_$(v1-1).jld2"))
                scan_instructions["load_file"] = joinpath(folder, "$(save_name_prefix)_u_$(v1-1).jld2")
            end
        end

        # 6. Run scan optimization
        Trotter.interaction_scan_map_to_state(
            state_vecs, scan_instructions, gates, tau_terms, basis_sector, N_sites;
            maxiters=maxiters,
            optimizer=[:LBFGS, :GradientDescent, :LBFGS],
            initialization_samples=10,
            H_hopping=H_hop_sector, H_interaction=H_int_sector,
            save_folder=folder, save_name=save_name_prefix,
            loss_type=loss_type,
            U_values=U_values,
            antihermitian=antihermitian,
            use_gpu=use_gpu
        )

        return 0
    end
end
