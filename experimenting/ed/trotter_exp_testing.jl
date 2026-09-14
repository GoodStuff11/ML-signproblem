#=
trotter_exp_testing.jl

Compare Hamiltonian energies across three methods:
  1. Exact exponential  – stored coefficients from unitary_map_energy_symmetry=false files
                          applied as a single unitary (assumes energy optimization).
  2. Trotterized        – same stored coefficients repeated P times, each copy divided
                          by P (approximating the exact exponential at increasing order P).
  3. Trotter-optimized  – overlap-optimized Trotter coefficients from
                          trotter_N=<N_sites>_u files.

Usage:
  julia --project=.. trotter_exp_testing.jl [folders...] [options]

Arguments/Options:
  folders (positional, optional): Zero, one, or more paths to ED data folders (or system size suffixes like "3x2", "3x3").
                                 - If zero are provided: defaults to "N=(2, 2)_3x2".
                                 - If one is provided and no --u option is specified: runs a U-value sweep analysis
                                   for that single system size, saving the plot to <folder>/<output_file>.png.
                                 - If multiple are provided (or one folder with the --u option is specified): runs a
                                   system-size comparison analysis at the specified U value.
                                   Saves the plot to <first_folder>/<output_file>_system_size.png.

  --u=<float> or --U=<float> (optional): Specify a single U value to perform system size comparison at.
                                         This argument is REQUIRED if multiple folders/system sizes are specified.
                                         If multiple system sizes are inputted but this option is missing, the script
                                         raises an error.

  --trotter_orders=<list> (optional): Comma-separated list of positive integers
                     specifying the Trotterization repetition counts P to compare.
                     For each P the exact-exp coefficient vector is repeated P times
                     and divided by P before applying the unitary.
                     Default: "1,2,4,8".

  --n_up=<int> (optional): Number of spin-up electrons. Default: 4.

  --n_dn=<int> (optional): Number of spin-down electrons. Default: 4.

  --lvec=<WxH> (optional): Lattice dimensions in the format WxH (e.g. "3x3").
                     Default: "3x3".

  --output=<string> (optional): Name of png to output (will have .png or _system_size.png appended). Default: trotter_order_comparison

  --antihermitian (optional): Whether to use antihermitian operators. Can be true/false or --antihermitian. Default: auto-detect from shared.jld2 files.

  --loss=<string> (optional): Loss type. Valid options are:
                              - "overlap": overlap-optimized loss
                              - "energy": energy-optimized loss
                              Default: "overlap".

  --custom_ref_state=<string> (optional): Custom reference state label. Default: nothing (use Slater determinant).

Examples:
  1. Sweep U values for a single system:
     julia --project=.. trotter_exp_testing.jl
     julia --project=.. trotter_exp_testing.jl "N=(2, 2)_3x2"

  2. Compare trotterized energies across system sizes at a single U value:
     julia --project=.. trotter_exp_testing.jl "N=(2, 2)_3x2" "N=(3, 3)_3x3" --u=2.0

  3. System size comparison specifying short system size suffixes:
     julia --project=.. trotter_exp_testing.jl 3x2 3x3 --u=4.0 --trotter_orders=1,2,4
=#

include("trotter_analysis_lib.jl")

# ═══════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ═══════════════════════════════════════════════════════════════════════════════


"""
    resolve_system_folder(input::String) -> String

Resolves the positional input to a full directory path inside get_data_root().
Supports absolute paths, folders within the data root, or strings that end with or match system names.
"""
function resolve_system_folder(input::String)
    if isabspath(input) && isdir(input)
        return input
    end
    root = get_data_root()
    p1 = joinpath(root, input)
    if isdir(p1)
        return p1
    end
    # Search for matching folders
    for item in readdir(root)
        if isdir(joinpath(root, item))
            if item == input || endswith(item, "_" * input)
                return joinpath(root, item)
            end
        end
    end
    error("Could not resolve system size / folder: '$input' in data root '$root'")
end

"""
    parse_arguments(args::Vector{String}) -> (folders, trotter_orders, n_up, n_dn, lvec, output, antihermitian, loss_type, custom_ref_state_arg, u_val)

Parse command-line arguments for the trotter_exp_testing script.

Returns:
  - folders        (Vector{String}) : resolved paths to ED data folders.
  - trotter_orders (Vector{Int})    : Trotter repetition counts to sweep over.
  - n_up           (Int)            : number of spin-up electrons.
  - n_dn           (Int)            : number of spin-down electrons.
  - lvec           (Vector{Int})    : lattice dimensions [W, H].
  - output         (String)         : output filename
  - antihermitian  (Union{Bool,Nothing}) : antihermitian flag or nothing
  - loss_type      (Symbol)         : loss type (:overlap or :energy)
  - custom_ref_state_arg (Union{String,Nothing}) : custom reference state name or nothing
  - u_val          (Union{Float64,Nothing}) : specified U value or nothing
"""
function parse_arguments(args::Vector{String})
    output = "trotter_order_comparison"
    trotter_orders = [1, 2, 4, 8]
    n_up = 4
    n_dn = 4
    lvec = [3, 3]
    antihermitian = nothing
    loss_type = :overlap
    custom_ref_state_arg = nothing
    u_val = nothing
    positional = String[]

    for arg in args
        if startswith(arg, "--trotter_orders=")
            val = split(arg, "=", limit=2)[2]
            trotter_orders = [parse(Int, s) for s in split(val, ",")]
        elseif startswith(arg, "--n_up=")
            n_up = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--n_dn=")
            n_dn = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--lvec=")
            parts = split(split(arg, "=", limit=2)[2], "x")
            length(parts) == 2 || error("--lvec must be in the form WxH, e.g. 3x3")
            lvec = [parse(Int, parts[1]), parse(Int, parts[2])]
        elseif startswith(arg, "--output")
            output = String(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--antihermitian")
            if occursin("=", arg)
                antihermitian = parse(Bool, split(arg, "=", limit=2)[2])
            else
                antihermitian = true
            end
        elseif startswith(arg, "--loss=")
            val = String(split(arg, "=", limit=2)[2])
            if val == "overlap"
                loss_type = :overlap
            elseif val == "energy"
                loss_type = :energy
            else
                error("Invalid --loss option: '$val'. Valid options are: 'overlap', 'energy'.")
            end
        elseif startswith(arg, "--custom_ref_state=")
            custom_ref_state_arg = String(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--u=") || startswith(arg, "--U=")
            u_val = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--")
            error("Unknown option: $arg")
        else
            push!(positional, arg)
        end
    end

    folders = String[]
    if isempty(positional)
        push!(folders, resolve_system_folder("N=(2, 2)_3x2"))
    else
        for pos in positional
            push!(folders, resolve_system_folder(pos))
        end
    end

    if length(folders) > 1 && isnothing(u_val)
        error("If multiple system sizes are specified, a single U value must be provided via the --u parameter (e.g. --u=2.0).")
    end

    return folders, trotter_orders, n_up, n_dn, lvec, output, antihermitian, loss_type, custom_ref_state_arg, u_val
end
# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "trotter_exp_testing")
    with_logging(log_path) do
        folders, trotter_orders, n_up, n_dn, lvec, output_file, antihermitian, loss_type, custom_ref_state_arg, u_val = parse_arguments(ARGS)

        println("=== Trotter experiment testing ===")
        println("  folders        = $folders")
        println("  trotter_orders = $trotter_orders")
        println("  loss_type      = $loss_type")
        println("  custom_ref     = $custom_ref_state_arg")
        println("  u_val          = $u_val")
        println()

        if !isnothing(u_val)
            # Run system size computation for the given folders
            system_names, num_sites, gs_energies, exact_exp_energies, trotter_energies, trotter_opt_energies =
                compute_trotterized_energies_system_size(
                    folders, u_val, trotter_orders, custom_ref_state_arg, antihermitian, loss_type
                )

            # Plot the results
            println("\nBuilding and saving system size comparison plot...")
            fig = build_system_size_comparison_plot(
                system_names, num_sites, gs_energies, exact_exp_energies,
                trotter_energies, trotter_opt_energies,
                trotter_orders, u_val, loss_type
            )

            # Save the plot in the first folder (or a default path)
            out_png = joinpath(folders[1], "$(output_file)_system_size.png")
            out_pdf = joinpath(folders[1], "$(output_file)_system_size.pdf")
            save(out_png, fig)
            save(out_pdf, fig)
            println("  Saved → $out_png, $out_pdf")
        else
            # Run U sweep for the single folder
            U_values, gs_energies, exact_exp_energies, exact_exp_overlaps, trotter_energies, trotter_overlaps, trotter_opt_energies, trotter_opt_overlaps, trotter_to_exact_overlaps, n_up_loaded, n_dn_loaded, lvec =
                compute_trotterized_energies_u_sweep(
                    folders[1], trotter_orders, custom_ref_state_arg, antihermitian, loss_type
                )

            # Plot energy comparison
            println("\nBuilding and saving energy comparison plot...")
            fig_energy = build_comparison_plot(
                U_values, gs_energies, exact_exp_energies,
                trotter_energies, trotter_opt_energies,
                trotter_orders, n_up_loaded, n_dn_loaded, lvec;
                custom_ref_state_arg=custom_ref_state_arg,
                loss_type=loss_type
            )

            out_png = joinpath(folders[1], "$output_file.png")
            out_pdf = joinpath(folders[1], "$output_file.pdf")
            save(out_png, fig_energy)
            save(out_pdf, fig_energy)
            println("  Saved energy plot → $out_png, $out_pdf")

            # Plot overlap comparison (ground state infidelity)
            println("\nBuilding and saving overlap comparison plot...")
            fig_overlap = build_overlap_comparison_plot(
                U_values, exact_exp_overlaps,
                trotter_overlaps, trotter_opt_overlaps,
                trotter_orders, n_up_loaded, n_dn_loaded, lvec;
                trotter_to_exact_overlaps=trotter_to_exact_overlaps,
                custom_ref_state_arg=custom_ref_state_arg,
                loss_type=loss_type,
                metric=:infidelity
            )

            out_ovlp_png = joinpath(folders[1], "$(output_file)_overlap.png")
            out_ovlp_pdf = joinpath(folders[1], "$(output_file)_overlap.pdf")
            save(out_ovlp_png, fig_overlap)
            save(out_ovlp_pdf, fig_overlap)
            println("  Saved overlap plot → $out_ovlp_png, $out_ovlp_pdf")

            # Also save Trotter discretization error plot
            fig_trotter_err = build_overlap_comparison_plot(
                U_values, exact_exp_overlaps,
                trotter_overlaps, trotter_opt_overlaps,
                trotter_orders, n_up_loaded, n_dn_loaded, lvec;
                trotter_to_exact_overlaps=trotter_to_exact_overlaps,
                custom_ref_state_arg=custom_ref_state_arg,
                loss_type=loss_type,
                metric=:trotter_error,
                legend_position=:rb
            )
            out_err_png = joinpath(folders[1], "$(output_file)_trotter_error.png")
            out_err_pdf = joinpath(folders[1], "$(output_file)_trotter_error.pdf")
            save(out_err_png, fig_trotter_err)
            save(out_err_pdf, fig_trotter_err)
            println("  Saved trotter error plot → $out_err_png, $out_err_pdf")
        end

        return 0
    end
end
