#=
run_3x3_nu4_nd4_u40.jl

Run Trotter optimization on the 3x3 lattice with 4 up and 4 down electrons (N=(4, 4)_3x3)
at u_idx=40 without saving and without loading any initial coefficient files.

Usage:
  julia --project=.. run_3x3_nu4_nd4_u40.jl [--u_idx=<int>] [--num_exponentials=<int>] [--maxiters=<int>] [--loss=<type>] [--antihermitian=<bool>] [--custom_ref_state=<value>] [--use_gpu=<bool>] [--datatype=<type>]

Options:
  --u_idx=<int> (optional): The U index to optimize. Default: 40.
                     Valid options: Integer between 1 and length(U_values) (e.g. 40 corresponds to U=9.75).
  --num_exponentials=<int> (optional): Number of Trotter layers. Default: 1.
                     Valid options: Positive integer >= 1.
  --maxiters=<int> (optional): Maximum optimization iterations per stage. Default: 100.
                     Valid options: Positive integer >= 1.
  --loss=<type> (optional): Loss function to optimize. Default: "overlap".
                     Valid options:
                     - "overlap": Infidelity loss (1 - |<target|U|ref>|^2).
                     - "energy": Variational Hamiltonian energy loss.
  --antihermitian=<bool> (optional): Use real antihermitian generators. Default: true.
                     Valid options: true, false.
  --custom_ref_state=<value> (optional): Reference state selection. Default: "slater".
                     Valid options: "slater", or an integer basis index.
  --use_gpu=<bool> (optional): Use CUDA GPU acceleration. Default: true.
                     Valid options: true, false.
  --datatype=<type> (optional): Numerical precision. Default: Float32.
                     Valid options: Float32, Float64, ComplexF32, ComplexF64.
=#

# Pre-scan ARGS for GPU flag before loading CUDA package
_use_gpu_prescan = let val = true
    for arg in ARGS
        if arg == "--use_gpu=false"
            val = false
        elseif arg == "--use_gpu" || arg == "--use_gpu=true"
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
using SparseArrays
using JLD2
using HDF5

include(joinpath(@__DIR__, "..", "data_path.jl"))
include(joinpath(@__DIR__, "..", "logging.jl"))
include(joinpath(@__DIR__, "..", "utility_functions.jl"))
using .UtilityFunctions
include(joinpath(@__DIR__, "..", "trotter.jl"))
using .Trotter
include(joinpath(@__DIR__, "..", "ed_objects.jl"))
include(joinpath(@__DIR__, "..", "ed_functions.jl"))

"""
    parse_arguments(args::Vector{String})

Parse command line arguments for the 3x3 (4 up, 4 down) single U run.
"""
function parse_arguments(args::Vector{String})
    u_idx = 40
    num_exponentials = 1
    maxiters = 100
    loss_type = :overlap
    antihermitian = true
    custom_ref_state_arg = "slater"
    use_gpu = true
    datatype = Float32

    for arg in args
        if startswith(arg, "--u_idx=")
            u_idx = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--num_exponentials=")
            num_exponentials = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--maxiters=")
            maxiters = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--loss=")
            val = String(split(arg, "=", limit=2)[2])
            if val == "overlap"
                loss_type = :overlap
            elseif val == "energy"
                loss_type = :energy
            else
                error("Invalid --loss option: '$val'. Valid options are: 'overlap', 'energy'.")
            end
        elseif startswith(arg, "--antihermitian=")
            antihermitian = parse(Bool, split(arg, "=", limit=2)[2])
        elseif arg == "--antihermitian"
            antihermitian = true
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
                error("Invalid --datatype option: '$val'. Valid options are: 'ComplexF64', 'ComplexF32', 'Float64', 'Float32'.")
            end
        end
    end

    if !antihermitian && datatype <: Real
        datatype = ComplexF64
    end

    return u_idx, num_exponentials, maxiters, loss_type, antihermitian, custom_ref_state_arg, use_gpu, datatype
end

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "run_3x3_nu4_nd4_u40")
    with_logging(log_path) do
        u_idx, num_exponentials, maxiters, loss_type, antihermitian, custom_ref_state_arg, use_gpu, datatype = parse_arguments(ARGS)

        println("================================================================================")
        println("Running 3x3 (4 up, 4 down) Trotter optimization (No file load, No file save)")
        println("U index: $u_idx")
        println("Num exponentials: $num_exponentials")
        println("Max iterations: $maxiters")
        println("Loss type: $loss_type")
        println("Antihermitian: $antihermitian")
        println("Custom ref state: $custom_ref_state_arg")
        println("Use GPU: $use_gpu")
        println("Datatype: $datatype")
        println("================================================================================")

        folder = data_folder("N=(4, 4)_3x3")
        
        # Load ED data
        U_values, state_vecs, indexer, _, N_elec, _, _, sign_convention =
            load_ED_data(folder; verbose=true, sign_convention=:spin_first, use_slater_reference=custom_ref_state_arg == "slater")

        n_up, n_dn = N_elec
        Lvec = parse_lattice_dimension(folder)
        N_sites = prod(Lvec)
        target_u = U_values[u_idx]
        println("Optimizing U index $u_idx (U = $target_u)")

        # Compute basis and Hamiltonian sectors
        basis_sector = Trotter.get_basis_sector(indexer, Lvec, N_sites)
        println("Basis sector size: $(length(basis_sector))")

        H_hop_sector, _, _ = Trotter.TamFermion.HubbardMomentumBasis(
            1.0, 0.0, Lvec, (n_up, n_dn); indexer=indexer
        )
        H_int_sector, _, _ = Trotter.TamFermion.HubbardMomentumBasis(
            0.0, 1.0, Lvec, (n_up, n_dn); indexer=indexer
        )

        # Enumerate Trotter gates and tau terms
        gates = Trotter.enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true, include_diagonal=!antihermitian)
        println("Number of gates: $(length(gates))")
        tau_terms = Trotter.fgateToTauSector(gates, N_sites, basis_sector; antihermitian=antihermitian)

        # Explicit instructions with NO load_file and NO save_folder
        scan_instructions = Dict{String,Any}(
            "starting level" => 1,
            "ending level" => 1,
            "num_exponentials" => num_exponentials,
            "antihermitian" => antihermitian,
            "u_range" => u_idx:u_idx,
            "U_values" => U_values
        )

        Trotter.interaction_scan_map_to_state(
            state_vecs, scan_instructions, gates, tau_terms, basis_sector, N_sites;
            maxiters=maxiters,
            optimizer=[:LBFGS, :GradientDescent, :LBFGS],
            initialization_samples=10,
            H_hopping=H_hop_sector, H_interaction=H_int_sector,
            save_folder=nothing, # Explicitly disable saving
            loss_type=loss_type,
            U_values=U_values,
            antihermitian=antihermitian,
            use_gpu=use_gpu,
            datatype=datatype,
            initial_coefficients=nothing # Explicitly start fresh
        )

        println("\nRun finished successfully.")
        return 0
    end
end
