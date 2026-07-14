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
  julia --project=.. run_trotter_scan_optimization.jl "data/N=(2, 2)_2x2" 25
  julia --project=.. run_trotter_scan_optimization.jl "data/N=(2, 2)_2x2" 25 35
  julia --project=.. run_trotter_scan_optimization.jl "data/N=(2, 2)_2x2" forward --num_exponentials=3
  julia --project=.. run_trotter_scan_optimization.jl "data/N=(2, 2)_2x2" 25 --loss=energy --antihermitian
  julia --project=.. run_trotter_scan_optimization.jl "data/N=(2, 2)_2x2" 25 --custom_ref_state=slater
  julia --project=.. run_trotter_scan_optimization.jl "data/N=(2, 2)_2x2" 25 --custom_ref_state=1
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
if !isdefined(Main, :UtilityFunctions)
    include("utility_functions.jl")
end
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
    maxiters = 300
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
        error("Please input a folder. Ex: data/N=(2, 2)_2x2")
    end
    folder = filtered_args[1]

    u_start = length(filtered_args) >= 2 ? filtered_args[2] : "25"
    u_end = length(filtered_args) >= 3 ? filtered_args[3] : nothing

    return folder, u_start, u_end, maxiters, loss_type, num_exponentials, antihermitian, custom_ref_state_arg
end

"""
    construct_custom_ref_state(custom_ref_state_arg::Union{String,Nothing}, folder::String, basis_sector::Vector{UInt}, U_values::Vector{Float64})

Construct a custom Slater determinant reference state vector if requested by the command line argument.
Returns a `Vector{ComplexF64}` or `nothing`.
"""
function construct_custom_ref_state(custom_ref_state_arg::Union{String,Nothing}, folder::String, basis_sector::Vector{UInt}, U_values::Vector{Float64})
    if isnothing(custom_ref_state_arg)
        return nothing
    end

    local slater_idx
    if custom_ref_state_arg == "slater"
        jld2_path = joinpath(folder, "meta_data_and_E.jld2")
        if isfile(jld2_path)
            println("Finding Slater ground state index from JLD2 file...")
            dic = load_saved_dict(jld2_path)
            all_E = dic["E"]
            U_values_for_sector = dic["meta_data"]["U_values"]
            k_min = find_best_energy_sector(all_E, U_values_for_sector; data=dic)
            slater_idx = get_slater_ground_state(dic, k_min)
        else
            println("Finding Slater ground state index from HDF5 file...")
            valid_files = [f for f in readdir(folder) if occursin("HubbardED", f)]
            if isempty(valid_files)
                error("No HubbardED HDF5 file found in folder: $folder")
            end
            h5_file = joinpath(folder, valid_files[1])
            slater_idx = h5open(h5_file, "r") do data
                key_labels = [parse(Int, k) for k in keys(data["data/energies"])]
                all_E = [real.(read(data, "data/energies/$(k)"))[:, 1] for k in key_labels]
                k_min = find_best_energy_sector(all_E, U_values; labels=key_labels)
                return get_slater_ground_state(data, k_min)
            end
        end
        println("Slater ground state index found: $slater_idx")
    else
        try
            slater_idx = parse(Int, custom_ref_state_arg)
        catch e
            error("Invalid --custom_ref_state value: '$custom_ref_state_arg'. Must be 'slater' or an integer index.")
        end
        if slater_idx < 1 || slater_idx > length(basis_sector)
            error("Parsed Slater index $slater_idx is out of bounds (1 to $(length(basis_sector))).")
        end
        println("Using user-specified Slater index: $slater_idx")
    end

    if slater_idx == -1
        error("No Slater ground state could be found in the current sector.")
    end

    custom_ref = zeros(ComplexF64, length(basis_sector))
    custom_ref[slater_idx] = 1.0
    return custom_ref
end

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "run_trotter_scan_optimization")
    with_logging(log_path) do
        folder, u_start, u_end, maxiters, loss_type, num_exponentials, antihermitian, custom_ref_state_arg = parse_arguments(ARGS)

        # 1. Load ED data (loads indexer if JLD2, or we can use it to build the sector basis)
        if isnothing(custom_ref_state_arg)
            U_values, state_vecs, indexer, _, N_elec, spin_conserved, _, sign_convention =
                load_ED_data(folder; verbose=true, sign_convention=:spin_first)
        else
            U_values, state_vecs, indexer, _, N_elec, spin_conserved, _, sign_convention =
                load_ED_data(folder; verbose=true, sign_convention=:spin_first, use_slater_reference=false)
        end

        n_up, n_dn = N_elec

        # Parse dimension from folder name, default to (3, 3) if fails
        Lvec = [3, 3]
        try
            m_dim = match(r"_(?<W>\d+)x(?<H>\d+)", folder)
            if !isnothing(m_dim)
                Lvec = [parse(Int, m_dim[:W]), parse(Int, m_dim[:H])]
            end
        catch e
            @warn "Could not parse dimensions from folder path, using default [3, 3]"
        end
        N_sites = prod(Lvec)

        # Helper to convert Coordinate to 0-based site index
        function coord_to_site_idx(coord, Lvec)
            c0 = coord.coordinates .- 1
            return Trotter.ravel_c(c0, Tuple(Lvec))
        end

        # Helper to convert Coordinate set to binary representation (the configuration stored in indexer)
        function coord_set_to_binary(coord_set, Lvec)
            val = zero(UInt)
            for coord in coord_set
                site_idx = coord_to_site_idx(coord, Lvec)
                val |= (one(UInt) << site_idx)
            end
            return val
        end

        # 2. Reconstruct the sector basis integers
        local basis_sector
        if !isnothing(indexer)
            println("Reconstructing basis sector from indexer...")
            basis_sector = Vector{UInt}(undef, length(indexer.inv_comb_dict))
            for (idx, conf) in enumerate(indexer.inv_comb_dict)
                u_bin = coord_set_to_binary(conf[1], Lvec)
                d_bin = coord_set_to_binary(conf[2], Lvec)
                basis_sector[idx] = Trotter.combineSpinInts(u_bin, d_bin, N_sites)
            end
        else
            println("Reconstructing basis sector from HDF5 file...")
            valid_files = [f for f in readdir(folder) if occursin("HubbardED", f)]
            if isempty(valid_files)
                error("No indexer loaded, and no HubbardED HDF5 file found in folder: $folder")
            end
            h5_file = joinpath(folder, valid_files[1])

            basis_sector = h5open(h5_file, "r") do data
                key_labels = [parse(Int, k) for k in keys(data["data/energies"])]
                all_E = [real.(read(data, "data/energies/$(k)"))[:, 1] for k in key_labels]
                k_min = find_best_energy_sector(all_E, U_values; labels=key_labels)

                separate_spins_stored = (read(data, "metadata/slater_labels/$k_min") isa Dict)
                if !separate_spins_stored
                    slater_labels = read(data, "metadata/slater_labels/$k_min")
                    H_dim = size(slater_labels, 2)
                else
                    slater_labels_up = read(data, "metadata/slater_labels/$k_min/up")
                    slater_labels_down = read(data, "metadata/slater_labels/$k_min/dn")
                    H_dim = size(slater_labels_up, 2)
                end

                # Helper to convert orbital indices to binary representation
                function orbital_indices_to_binary(indices, N)
                    val = zero(UInt)
                    for idx in indices
                        val |= (one(UInt) << idx)
                    end
                    return val
                end

                basis_sector_h5 = Vector{UInt}(undef, H_dim)
                for idx in 1:H_dim
                    up_indices = separate_spins_stored ? slater_labels_up[:, idx] : slater_labels[:, idx, 1]
                    dn_indices = separate_spins_stored ? slater_labels_down[:, idx] : slater_labels[:, idx, 2]

                    u_bin = orbital_indices_to_binary(up_indices, N_sites)
                    d_bin = orbital_indices_to_binary(dn_indices, N_sites)

                    # Combine up and dn spins
                    basis_sector_h5[idx] = Trotter.combineSpinInts(u_bin, d_bin, N_sites)
                end

                return basis_sector_h5
            end
        end
        # 3. Supply Hamiltonian components in the momentum sector directly
        # Determine the target momentum sector index q_target
        local q_target
        if isnothing(indexer)
            # Find the sector from HDF5 file
            valid_files = [f for f in readdir(folder) if occursin("HubbardED", f)]
            if isempty(valid_files)
                error("No indexer loaded, and no HubbardED HDF5 file found in folder: $folder")
            end
            h5_file = joinpath(folder, valid_files[1])
            q_target = h5open(h5_file, "r") do data
                key_labels = [parse(Int, k) for k in keys(data["data/energies"])]
                all_E = [real.(read(data, "data/energies/$(k)"))[:, 1] for k in key_labels]
                k_min = find_best_energy_sector(all_E, U_values; labels=key_labels)
                kvecs = read(data, "metadata/kvecs")
                k_tuple = tuple((kvecs[:, k_min+1] .+ 1)...)
                return Trotter.ravel_c(k_tuple .- 1, Tuple(Lvec))
            end
        else
            q_target = Trotter.ravel_c(indexer.k .- 1, Tuple(indexer.lattice_dims))
        end

        println("Constructing sector Hamiltonians directly using HubbardMomentumBasis (q_target = $q_target)...")
        @time H_hop_mom, basis_dict, _ = TamFermion.HubbardMomentumBasis(1.0, 0.0, Lvec, (n_up, n_dn); q_target=q_target)
        @time H_int_mom, _, _ = TamFermion.HubbardMomentumBasis(0.0, 1.0, Lvec, (n_up, n_dn); q_target=q_target)

        # 4. Map basis_sector to basis_dict["ints"] and find the permutation mapping
        state_to_idx = Dict(val => idx for (idx, val) in enumerate(basis_dict["ints"]))
        for val in basis_sector
            if !haskey(state_to_idx, val)
                error("State $val from basis_sector not found in HubbardMomentumBasis sector ints!")
            end
        end
        perm = [state_to_idx[val] for val in basis_sector]

        H_hop_sector = H_hop_mom[perm, perm]
        H_int_sector = H_int_mom[perm, perm]

        # 5. Enumerate Trotter gates and tau terms
        @time gates = Trotter.enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true, include_diagonal=true)
        @time tau_terms = Trotter.fgateToTauSector(gates, N_sites, basis_sector; antihermitian=antihermitian)

        # 6. Set up scan range
        scan_instructions = Dict{String,Any}(
            "starting level" => 1,
            "ending level" => 1,
            "num_exponentials" => num_exponentials,
            "antihermitian" => antihermitian
        )

        save_name_prefix = "trotter_N=$N_sites"
        if !isnothing(custom_ref_state_arg)
            save_name_prefix *= "_ref_$(custom_ref_state_arg)"
        end
        if antihermitian
            save_name_prefix *= "_antihermitian"
        end
        if loss_type == :energy
            save_name_prefix *= "_loss_energy"
        end

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

        custom_ref = construct_custom_ref_state(custom_ref_state_arg, folder, basis_sector, U_values)

        # 7. Run scan optimization
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
            custom_ref_state=custom_ref
        )

        return 0
    end
end
