#=
run_trotter_scan_optimization.jl

Run Trotter optimization over a range of U interaction parameters using unitaries mapped from Lanczos ED data,
represented in the momentum basis.

Usage:
  julia --project=.. run_trotter_scan_optimization.jl [folder] [u_start] [u_end] [--maxiters=<number>] [--loss=<type>] [--num_exponentials=<number>] [--antihermitian] [--custom_ref_state=<value>]

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

Examples:
  julia --project=.. run_trotter_scan_optimization.jl "N=(2, 2)_2x2" 25
  julia --project=.. run_trotter_scan_optimization.jl "N=(2, 2)_2x2" 25 35
  julia --project=.. run_trotter_scan_optimization.jl "N=(2, 2)_2x2" forward --num_exponentials=3
  julia --project=.. run_trotter_scan_optimization.jl "N=(2, 2)_2x2" 25 --loss=energy --antihermitian
  julia --project=.. run_trotter_scan_optimization.jl "N=(2, 2)_2x2" 25 --custom_ref_state=slater
  julia --project=.. run_trotter_scan_optimization.jl "N=(2, 2)_2x2" 25 --custom_ref_state=1
=#

using Lattices
using LinearAlgebra
using Combinatorics
using SparseArrays
using Statistics
using Random
using JLD2
using HDF5
using Zygote

# Include utility_functions.jl and trotter.jl
include("data_path.jl")
include("utility_functions.jl")
using .UtilityFunctions
include("trotter.jl")
using .Trotter

include("ed_objects.jl")
include("ed_functions.jl")
include("logging.jl")

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
7. --custom_ref_state=<value> (String): Use a custom reference state as a Slater determinant.
"""
function parse_arguments(args::Vector{String})
    maxiters = 500
    loss_type = :overlap
    num_exponentials = 1
    antihermitian = false
    custom_ref_state_arg = nothing
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
            val = String(split(arg, "=", limit=2)[2])
            num_exponentials = parse(Int, val)
        elseif startswith(arg, "--antihermitian")
            if occursin("=", arg)
                antihermitian = parse(Bool, split(arg, "=", limit=2)[2])
            else
                antihermitian = true
            end
        elseif startswith(arg, "--custom_ref_state=")
            custom_ref_state_arg = String(split(arg, "=", limit=2)[2])
        else
            push!(filtered_args, arg)
        end
    end

    if length(filtered_args) < 1
        error("Please input a folder. Ex: N=(2, 2)_2x2")
    end
    folder = data_folder(filtered_args[1])

    u_start = length(filtered_args) >= 2 ? filtered_args[2] : "25"
    u_end = length(filtered_args) >= 3 ? filtered_args[3] : nothing

    return folder, u_start, u_end, maxiters, loss_type, num_exponentials, antihermitian, custom_ref_state_arg
end

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "run_trotter_scan_optimization")
    with_logging(log_path) do
        folder, u_start, u_end, maxiters, loss_type, num_exponentials, antihermitian, custom_ref_state_arg = parse_arguments(ARGS)

        # 1. Load ED data (loads indexer if JLD2, or we can use it to build the sector basis)
        U_values, state_vecs, indexer, _, N_elec, spin_conserved, _, sign_convention =
            load_ED_data(folder; verbose=true, sign_convention=:spin_first, use_slater_reference=custom_ref_state_arg == "slater")

        n_up, n_dn = N_elec

        # Parse dimension from folder name, default to (3, 3) if fails
        Lvec = parse_lattice_dimension(folder)
        N_sites = prod(Lvec)

        # 2. Computing the basis
        # Convert each (up_coords_set, dn_coords_set) entry in the indexer to a combined
        # 2N-bit integer that fgateToTauSector expects: up bits in the low N bits, dn bits
        # in the upper N bits (via combineSpinInts).
        basis_sector = Trotter.get_basis_sector(indexer, Lvec, N_sites)

        # 3. Find the Hamiltonian
        # Derive the momentum sector from the indexer (same convention as trotter_exp_testing.jl).
        # indexer.k is 1-based coordinate tuple; q_target is the C-order flat index (0-based).
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
        )

        return 0
    end
end
