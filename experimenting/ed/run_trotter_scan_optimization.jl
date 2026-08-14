#=
run_trotter_scan_optimization.jl

Run Trotter optimization over a range of U interaction parameters using unitaries mapped from Lanczos ED data,
represented in the momentum basis.

Usage:
  julia --project=.. run_trotter_scan_optimization.jl [folder] [u_start] [u_end] [--maxiters=<number>] [--loss=<type>] [--num_exponentials=<number>] [--antihermitian] [--custom_ref_state=<value>] [--use_gpu=<bool>] [--datatype=<type>] [--grow_from_exponentials=<number>] [--grow_mode=<mode>]

Arguments:
  folder (required): Path to the ED data folder (e.g., "data/N=(2, 2)_2x2").
  u_start (optional): Start index of U values, or direction. Default: "25".
                     Valid options:
                     - "forward": Scan forward from index 26 to the end of U values.
                     - "backward": Scan backward from index 18 down to 1.
                     - [integer]: Run a single specific U-index (if u_end is omitted or equal to u_start) or start of a range.
  u_end (optional): End index of U values (if specifying a range). Default: nothing.
                     Valid options:
                     - [integer]: End of index range. If equal to u_start or omitted, runs a single U index.
                     - omitted/nothing: Optimizes a single U value specified by u_start.
  --maxiters=<number> (optional): Maximum number of iterations for optimization. Default: 100.
  --loss=<type> (optional): The loss function to optimize. Default: "overlap".
                     Valid options:
                     - "overlap": Optimize overlap loss (1 - |<ψ'|U|ψ>|^2).
                     - "energy": Optimize energy loss (<ψ|U^† H U|ψ>).
  --num_exponentials=<number> (optional): Number of Trotter layers/steps. Default: 1.
  --antihermitian (optional): Use real-antihermitian generators instead of i * real-hermitian. Default: false.
  --custom_ref_state=<value> (optional): Use a custom reference state as a Slater determinant.
                     Valid options:
                     - "slater": The Slater determinant ground state of the tight-binding model.
                     - [integer]: Use the Slater determinant at this specific 1-based basis index.
  --use_gpu=<bool> (optional): Enable GPU acceleration for overlap loss and gradient calculations. Default: false.
                     Valid options:
                     - "--use_gpu" or "--use_gpu=true": Enable CUDA acceleration.
                     - "--use_gpu=false": Disable GPU acceleration (CUDA is not loaded to preserve @safe_threads compatibility).
  --datatype=<type> (optional): Data type for GPU vector and matrix operations. Default: ComplexF64.
                     Valid options: ComplexF64, ComplexF32, Float64, Float32.
  --grow_from_exponentials=<number> (optional): Bootstrap this run's (larger) --num_exponentials ansatz
                     from an already-optimized run with this smaller num_exponentials value, instead of
                     starting the newly-added (later) layers from scratch. The existing run's coefficients
                     become the first `grow_from_exponentials` (earlier) layers, unchanged; the new
                     (later) layers are zero-initialized and then optimized normally. Requires
                     --num_exponentials to be greater than this value, and requires a matching saved file
                     (same folder/loss/antihermitian/custom_ref_state settings, but with
                     num_exponentials=<this value>) to already exist for the relevant U index/indices
                     (see --grow_mode). If no such file is found for a given U index, that index falls
                     back to the normal from-scratch initialization instead (a warning is printed).
                     Default: not set (no growing; behaves exactly as before this option existed).
  --grow_mode=<mode> (optional): Only meaningful when --grow_from_exponentials is set. Selects how the
                     grow-from-existing-file bootstrap is applied across a multi-U scan. Default: "chain".
                     Valid options:
                     - "chain": Grow only once, from --grow_from_exponentials's file at the *first* U
                       index processed. Every subsequent U index in the scan then warm-starts from the
                       *previous* U index's just-optimized (already-grown) coefficients, exactly like the
                       normal scan's U-to-U warm start.
                     - "per_u": Grow independently at *every* U index processed, always reloading
                       --grow_from_exponentials's file for that same U index rather than chaining from
                       the neighboring U index's grown result.

Loading Behavior:
  - Single U value (e.g. u_start=25, u_end omitted): If an existing optimization result for the current U index
    exists (`..._u_25.jld2`), its coefficients are loaded as initial values and its full loss trajectory is preserved
    and concatenated with new optimization losses.
  - Range of U values (e.g. u_start=25, u_end=35): Stepping sequentially loads the previous U index (`u-1` or `u+1`).
  This is unaffected by --grow_from_exponentials/--grow_mode: an existing same-num_exponentials file for the
  relevant U index always takes precedence over growing from a smaller ansatz (see --grow_from_exponentials above).

Examples:
  julia --project=.. run_trotter_scan_optimization.jl "N=(2, 2)_2x2" 25
  julia --project=.. run_trotter_scan_optimization.jl "N=(2, 2)_2x2" 25 35
  julia --project=.. run_trotter_scan_optimization.jl "N=(2, 2)_2x2" forward --num_exponentials=3
  julia --project=.. run_trotter_scan_optimization.jl "N=(2, 2)_2x2" 25 40 --loss=energy --antihermitian
  julia --project=.. run_trotter_scan_optimization.jl "N=(2, 2)_2x2" 25 --custom_ref_state=slater
  julia --project=.. run_trotter_scan_optimization.jl "N=(2, 2)_2x2" 25 35 --use_gpu
  julia --project=.. run_trotter_scan_optimization.jl "N=(2, 2)_2x2" forward --num_exponentials=2 --grow_from_exponentials=1
  julia --project=.. run_trotter_scan_optimization.jl "N=(2, 2)_2x2" 25 40 --num_exponentials=2 --grow_from_exponentials=1 --grow_mode=per_u
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
4. --maxiters=<number> (Int): Optional maximum iterations parameter. Default: 100.
5. --loss=<type> (String): The loss function to optimize ("overlap", "energy"). Default: "overlap".
6. --num_exponentials=<number> (Int): Optional number of Trotter steps. Default: 1.
7. --antihermitian (flag): Use real-antihermitian generators. Default: false.
8. --custom_ref_state=<value> (String): Use a custom reference state as a Slater determinant.
9. --use_gpu=<bool> (flag/bool): Enable GPU acceleration for Trotter optimization. Default: false.
10. --datatype=<type> (String): Data type for GPU vector and matrix operations. Default: ComplexF64.
11. --grow_from_exponentials=<number> (Int, optional): Bootstrap --num_exponentials's new (later)
    layers from an existing smaller-num_exponentials run's coefficients instead of from scratch.
    Default: not set (nothing).
12. --grow_mode=<mode> (String): How grow_from_exponentials is applied across a multi-U scan
    ("chain" or "per_u"). Default: "chain".
"""
function parse_arguments(args::Vector{String})
    maxiters = 100
    loss_type = :overlap
    num_exponentials = 1
    antihermitian = false
    custom_ref_state_arg = nothing
    use_gpu = false
    datatype = ComplexF64
    grow_from_exponentials = nothing
    grow_mode = :chain
    filtered_args = String[]

    for arg in args
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
        elseif arg == "--antihermitian" || startswith(arg, "--antihermitian=")
            if occursin("=", arg)
                antihermitian = parse(Bool, split(arg, "=", limit=2)[2])
            else
                antihermitian = true
            end
        elseif startswith(arg, "--custom_ref_state=")
            custom_ref_state_arg = split(arg, "=", limit=2)[2]
        elseif arg == "--use_gpu" || arg == "--use_gpu=true"
            use_gpu = true
        elseif arg == "--use_gpu=false"
            use_gpu = false
        elseif startswith(arg, "--datatype=")
            val = String(split(arg, "=", limit=2)[2])
            if val == "ComplexF64"
                datatype = ComplexF64
            elseif val == "ComplexF32"
                datatype = ComplexF32
            elseif val == "Float64"
                datatype = Float64
            elseif val == "Float32"
                datatype = Float32
            else
                error("Invalid --datatype option: '$val'. Valid options: 'ComplexF64', 'ComplexF32', 'Float64', 'Float32'.")
            end
        elseif startswith(arg, "--grow_from_exponentials=")
            grow_from_exponentials = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--grow_mode=")
            val = String(split(arg, "=", limit=2)[2])
            if val == "chain"
                grow_mode = :chain
            elseif val == "per_u"
                grow_mode = :per_u
            else
                error("Invalid --grow_mode option: '$val'. Valid options are: 'chain', 'per_u'.")
            end
        else
            push!(filtered_args, arg)
        end
    end

    if isempty(filtered_args)
        error("Usage: julia run_trotter_scan_optimization.jl <folder> [u_start] [u_end] [options]")
    end

    if !isnothing(grow_from_exponentials) && grow_from_exponentials >= num_exponentials
        error("--grow_from_exponentials=$grow_from_exponentials must be less than --num_exponentials=$num_exponentials")
    end

    folder = data_folder(filtered_args[1])
    u_start = length(filtered_args) >= 2 ? filtered_args[2] : "25"
    u_end = length(filtered_args) >= 3 ? filtered_args[3] : nothing

    return folder, u_start, u_end, maxiters, loss_type, num_exponentials, antihermitian, custom_ref_state_arg, use_gpu, datatype, grow_from_exponentials, grow_mode
end

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "run_trotter_scan_optimization")
    with_logging(log_path) do
        folder, u_start, u_end, maxiters, loss_type, num_exponentials, antihermitian, custom_ref_state_arg, use_gpu, datatype, grow_from_exponentials, grow_mode = parse_arguments(ARGS)
        println("Number of threads: $(Threads.nthreads())")
        println("Use GPU: $use_gpu")
        println("Data Type: $datatype")
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
            loss_type=loss_type,
            num_exponentials=num_exponentials,
            suffix="u_build"
        )

        grow_from_save_name = if !isnothing(grow_from_exponentials)
            build_save_name_prefix(
                :trotter;
                sites=N_sites,
                electrons=N_elec,
                custom_ref_state_arg=custom_ref_state_arg,
                antihermitian=antihermitian,
                loss_type=loss_type,
                num_exponentials=grow_from_exponentials
            )
        else
            nothing
        end

        if u_end === nothing
            v1 = tryparse(Int, u_start)
            if isnothing(v1)
                if u_start == "forward"
                    println("Forward scan")
                    scan_instructions["u_range"] = 26:length(U_values)
                else
                    println("Backward scan")
                    scan_instructions["u_range"] = 18:-1:1
                end
                scan_instructions["load_file"] = joinpath(folder, "$(save_name_prefix)_u_19.jld2")
                println("Load: $(scan_instructions["load_file"])")
            else
                println("Optimizing single U index: $v1 (U = $(U_values[v1]))")
                scan_instructions["u_range"] = v1:v1
                current_u_file = joinpath(folder, "$(save_name_prefix)_u_$(v1).jld2")
                if isfile(current_u_file)
                    scan_instructions["load_file"] = current_u_file
                    println("Loading existing coefficients and loss history for current U value (u_idx = $v1): $current_u_file")
                end
            end
        else
            v1 = parse(Int, u_start)
            v2 = parse(Int, u_end)
            if v1 > v2
                scan_instructions["u_range"] = v1:-1:v2
                if isfile(joinpath(folder, "$(save_name_prefix)_u_$(v1+1).jld2"))
                    scan_instructions["load_file"] = joinpath(folder, "$(save_name_prefix)_u_$(v1+1).jld2")
                end
            elseif v1 < v2
                scan_instructions["u_range"] = v1:v2
                if isfile(joinpath(folder, "$(save_name_prefix)_u_$(v1-1).jld2"))
                    scan_instructions["load_file"] = joinpath(folder, "$(save_name_prefix)_u_$(v1-1).jld2")
                end
            else
                # v1 == v2 (single U value)
                scan_instructions["u_range"] = v1:v1
                current_u_file = joinpath(folder, "$(save_name_prefix)_u_$(v1).jld2")
                if isfile(current_u_file)
                    scan_instructions["load_file"] = current_u_file
                    println("Loading existing coefficients and loss history for current U value (u_idx = $v1): $current_u_file")
                end
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
            use_gpu=use_gpu,
            datatype=datatype,
            grow_from_num_exponentials=grow_from_exponentials,
            grow_from_save_name=grow_from_save_name,
            grow_mode=grow_mode
        )

        return 0
    end
end
