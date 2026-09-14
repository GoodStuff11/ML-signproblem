#=
test_infidelity_and_dimH_plots.jl

Test script to verify the generation of the two-panel figure requested for final_analysis.ipynb:
- Plot 1: Infidelity (1 - overlap^2) vs U for all systems in FILE_LABEL_PAIRS (solid curves, no baseline dashed lines).
- Plot 2: Infidelity vs Hilbert space dimension for multiple selected U values (e.g. U = 4.0, 8.0, 12.0),
  with matching y-axis and denoted in a dedicated legend.

Usage:
  julia --project=.. test_infidelity_and_dimH_plots.jl [options]

Options:
  --selected_u=<vals> (optional): Comma-separated U values for Plot 2. Default: "4.0,8.0,12.0".
                      Valid options: Comma-separated floats representing U (e.g. "4.0,8.0,12.0").
=#

using Lattices
using LinearAlgebra
using Combinatorics
using SparseArrays
using CairoMakie
import Graphs
using LaTeXStrings
using Statistics
using Random
using Zygote
using Optimization, OptimizationOptimisers
using JSON
using OptimizationOptimJL
using JLD2
using ExponentialUtilities
using LsqFit
using HDF5

if !isdefined(Main, :UtilityFunctions)
    include("../utility_functions.jl")
end
include("../ed_objects.jl")
include("../ed_functions.jl")
include("../ed_optimization.jl")
include("../data_path.jl")
include("../logging.jl")

cmap1(L) = [Makie.ColorSchemes.roma[z] for z in range(0, 1, length=L)]
cmap2(L) = [Makie.ColorSchemes.managua[z] for z in range(0, 1, length=L)]

const FILE_LABEL_PAIRS = [
    (L"3\times 2\;(2,2)",   "N=(2, 2)_3x2", (2,2)),
    (L"3\times 2,\;(3,2)",  "N=(3, 2)_3x2", (3,2)),
    (L"3\times 2,\;(3,3)",  "N=(3, 3)_3x2", (3,3)),
    (L"3\times3,\;(3,2)",   "N=(3, 2)_3x3", (3,2)),
    (L"4\times2,\;(3,3)",   "N=(3, 3)_4x2", (3,3)),
    (L"3\times3,\;(3,3)",   "N=(3, 3)_3x3", (3,3)),
    (L"3\times3,\;(4,3)",   "N=(4, 3)_3x3", (4,3)),
    (L"3\times3,\;(4,4)",   "N=(4, 4)_3x3", (4,4)),
    (L"3\times3,\;(5,4)",   "N=(5, 4)_3x3", (5,4)),
    (L"4\times3,\;(5,4)",   "N=(5, 4)_4x3", (5,4)),
]

const FOLDER = get_data_root()

const THICKNESS_SCALE = 1.3
const BASE_FONTSIZE   = 16
const BASE_LINEWIDTH  = 2.5
const BASE_MARKERSIZE = 12

function get_figsize(num_cols::Int, height_mm::Float64=55.0)
    px_per_mm = 96 / 25.4
    if num_cols == 1
        width_mm = 89
    elseif num_cols == 2
        width_mm = 180
    else
        error("Can only have num_cols == 1 or 2 (given: $num_cols)")
    end
    width_px = width_mm * px_per_mm
    height_px = height_mm * px_per_mm
    return (width_px, height_px)
end

function create_fig(num_cols::Int, height_mm::Float64; kwargs...)
    fig = Figure(size=get_figsize(num_cols, height_mm), figure_padding=5)
    ax = Axis(fig[1, 1]; 
        aspect = (1+sqrt(5))/2,
        kwargs...
    )
    return fig, ax
end

const LEGEND_ARGS = Dict(:rowgap=>-8, :padding=>(3,3,0,0))
set_theme!(theme_latexfonts(), fontsize=10)

#=
    parse_cli_args(args) -> Dict{Symbol, Any}

Parse command-line arguments for test_infidelity_and_dimH_plots.
=#
function parse_cli_args(args)
    config = Dict{Symbol, Any}(
        :selected_u_values => [4.0, 8.0, 12.0],
    )

    for arg in args
        if startswith(arg, "--selected_u=")
            val_str = split(arg, "=", limit=2)[2]
            config[:selected_u_values] = parse.(Float64, split(val_str, ","))
        else
            error("Unknown command line argument: $arg")
        end
    end

    return config
end

#=
    generate_infidelity_and_dimH_figure(selected_U_values::Vector{Float64}; output_dir::String) -> Figure

Generates the 2-panel figure:
- Plot 1 (left): 1 - overlap^2 vs U for all systems in FILE_LABEL_PAIRS (solid lines only).
- Plot 2 (right): 1 - overlap^2 vs Hilbert space dimension for multiple selected U values,
  matching the y-axis of Plot 1 and denoted in a dedicated legend.
=#
function generate_infidelity_and_dimH_figure(selected_U_values::Vector{Float64}; output_dir::String)
    fig, ax_loss = create_fig(2, 70.0;
        xlabel = L"U",
        ylabel = L"1 - |\langle E_0(U)|\mathcal{U}|E_0(0)\rangle|^2",
        yscale = log10,
        limits = ((0, 15), (1e-15, 1.0)),
    )

    ax_dimH = Axis(fig[1, 2];
        aspect = (1+sqrt(5))/2,
        xlabel = L"\dim \mathcal{H}",
        ylabel = L"1 - |\langle E_0(U)|\mathcal{U}|E_0(0)\rangle|^2",
        xscale = log10,
        yscale = log10,
        limits = ((20, 60000), (1e-15, 1.0)),
    )

    u_indices = 2:60
    palette_colors = cmap2(length(FILE_LABEL_PAIRS))

    dimH_list = Float64[]
    # losses_by_u[k] will store the losses across all systems for selected_U_values[k]
    num_u = length(selected_U_values)
    losses_by_u = [Float64[] for _ in 1:num_u]
    processed_systems = String[]

    # Distinct linestyles and markers for each U value
    u_linestyles = [:dot, :solid, :dash, :dashdot]
    u_markers = [:circle, :diamond, :rect, :utriangle]
    u_line_colors = [(:gray40, 0.7), (:black, 0.7), (:gray40, 0.7), (:gray60, 0.7)]

    for (sys_idx, (display_label, file_label, electron_counts)) in enumerate(FILE_LABEL_PAIRS)
        sys_dir = joinpath(FOLDER, file_label)
        if !isdir(sys_dir)
            @warn "Directory for system $file_label does not exist: $sys_dir. Skipping."
            continue
        end

        num_sites = prod(parse_lattice_dimension(file_label))
        prefix = build_save_name_prefix(:trotter;
            electrons = electron_counts, sites = num_sites,
            antihermitian = true, custom_ref_state_arg = "slater")

        meta_file = joinpath(sys_dir, "meta_data_and_E.jld2")
        hilbert_space_size = nothing
        interaction_data = nothing
        if isfile(meta_file)
            metadata_dict = load(meta_file)["dict"]
            hilbert_space_size = size(metadata_dict["all_full_eig_vecs"][1], 2)
            interaction_data = metadata_dict["meta_data"]["U_values"]
        else
            valid_files = [f for f in readdir(sys_dir) if occursin("HubbardED", f)]
            if isempty(valid_files)
                @warn "No meta_data_and_E.jld2 or HubbardED HDF5 files found in $sys_dir. Skipping."
                continue
            end
            h5open(joinpath(sys_dir, valid_files[1]), "r") do h5data
                interaction_data = read(h5data["data/uvec"])
                hilbert_space_size = length(read(h5data["data/evecs/0"])[:, 1, 1])
            end
        end

        optimized_losses = Float64[]
        valid_u_indices = Int[]
        for u_idx in u_indices
            filepath = joinpath(sys_dir, "$(prefix)_u_$(u_idx).jld2")
            if !isfile(filepath)
                continue
            end
            result_dict = load_saved_dict(filepath)

            push!(valid_u_indices, u_idx)
            # 1 - overlap^2 corresponds to the optimized loss
            push!(optimized_losses, result_dict["metrics"]["loss"][2])
        end

        if isempty(valid_u_indices)
            @warn "No valid optimization files found for $file_label. Skipping."
            continue
        end

        # Plot 1: 1 - overlap^2 vs U (solid lines only, strictly no baseline dashed lines)
        lines!(ax_loss, interaction_data[valid_u_indices], optimized_losses;
            color = palette_colors[sys_idx], linestyle = :solid, label = string(display_label))

        push!(dimH_list, Float64(hilbert_space_size))
        push!(processed_systems, file_label)

        # Collect data for Plot 2 across each selected U value
        for (k, u_target) in enumerate(selected_U_values)
            target_idx = argmin(abs.(interaction_data[valid_u_indices] .- u_target))
            push!(losses_by_u[k], optimized_losses[target_idx])
            println("System $file_label | dimH = $hilbert_space_size | U = $u_target -> loss = $(optimized_losses[target_idx])")
        end
    end

    # Plot 2: For each U value, plot connecting trend line and scatter points
    order = sortperm(dimH_list)
    dimH_sorted = dimH_list[order]

    for (k, u_val) in enumerate(selected_U_values)
        ls = u_linestyles[mod1(k, length(u_linestyles))]
        mk = u_markers[mod1(k, length(u_markers))]
        lc = u_line_colors[mod1(k, length(u_line_colors))]

        # Trend line connecting sorted points
        lines!(ax_dimH, dimH_sorted, losses_by_u[k][order];
            color = lc, linestyle = ls, linewidth = 1.5)

        # Scatter points for each system (colored by system to match Plot 1)
        scatter!(ax_dimH, dimH_list, losses_by_u[k];
            color = palette_colors, markersize = 8, marker = mk)
    end

    # U-values legend on Plot 2 (in-axis legend at lower-right)
    u_legend_elements = [
        [LineElement(color = :black, linestyle = u_linestyles[mod1(k, length(u_linestyles))], linewidth = 1.5),
         MarkerElement(marker = u_markers[mod1(k, length(u_markers))], color = :black, markersize = 8)]
        for k in 1:num_u
    ]
    u_legend_labels = [L"U = %$(round(Int, u))" for u in selected_U_values]
    axislegend(ax_dimH, u_legend_elements, u_legend_labels;
        position = :rb, backgroundcolor = (:white, 0.8), LEGEND_ARGS...)

    # Systems legend below both panels
    Legend(fig[2, 1:2], ax_loss, orientation = :horizontal, nbanks = 2; LEGEND_ARGS...)
    rowgap!(fig.layout, 5)
    colgap!(fig.layout, 20)

    if !isdir(output_dir)
        mkpath(output_dir)
    end
    png_path = joinpath(output_dir, "loss_and_dimH.png")
    pdf_path = joinpath(output_dir, "loss_and_dimH.pdf")
    save(png_path, fig)
    save(pdf_path, fig)
    println("Saved figure to: $png_path and $pdf_path")

    # Verification assertions
    @assert length(processed_systems) == length(FILE_LABEL_PAIRS) "Expected $(length(FILE_LABEL_PAIRS)) systems, got $(length(processed_systems))"
    for k in 1:num_u
        @assert all(0.0 .<= losses_by_u[k] .<= 1.0) "All loss values at U=$(selected_U_values[k]) must be in [0, 1]"
    end
    @assert isfile(png_path) && filesize(png_path) > 0 "PNG file must exist and be non-empty"
    @assert isfile(pdf_path) && filesize(pdf_path) > 0 "PDF file must exist and be non-empty"

    return fig
end

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "test_infidelity_and_dimH_plots")
    with_logging(log_path) do
        println("=== Starting test_infidelity_and_dimH_plots ===")
        config = parse_cli_args(ARGS)
        selected_U_values = config[:selected_u_values]
        println("Selected U values: $selected_U_values")

        output_dir = joinpath(@__DIR__, "..", "good_images", "final")
        fig = generate_infidelity_and_dimH_figure(selected_U_values; output_dir=output_dir)

        @assert fig isa Figure "Output must be a Makie Figure"
        println("=== TEST PASSED SUCCESSFULLY: test_infidelity_and_dimH_plots ===")
    end
end
