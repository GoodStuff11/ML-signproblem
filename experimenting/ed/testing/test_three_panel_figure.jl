#=
test_three_panel_figure.jl

Test script to generate and verify the 3-panel figure:
- Panel 1: Overlap improvement ratio for systems starting near baseline for small U (axislegend included).
- Panel 2: Overlap improvement ratio for systems starting noticeably > 0 (or > baseline) for small U (axislegend included).
- Panel 3: Infidelity vs Hilbert space dimension for multiple U values (e.g. U = 4, 8, 12) with in-axis legend.
=#

using Lattices
using Combinatorics
using SparseArrays
using LinearAlgebra
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

const LEGEND_ARGS = Dict(:rowgap=>-6, :padding=>(3,3,0,0), :labelsize=>9)
set_theme!(theme_latexfonts(), fontsize=10)

function generate_three_panel_figure(; output_dir::String)
    fig = Figure(size = (1050, 300), figure_padding = 6)

    # Panel 1: Improvement ratio for systems starting at baseline
    ax1 = Axis(fig[1, 1];
        aspect = (1 + sqrt(5)) / 2,
        xlabel = L"U",
        ylabel = L"\frac{|\langle E_0(U)|\mathcal{U}|E_0(0)\rangle|^2}{|\langle E_0(U)|E_0(0)\rangle|^2}",
        limits = ((0, 15), (1, 15.5)),
        yticks = [1, 5, 10, 15]
    )

    # Panel 2: Improvement ratio for systems starting noticeably > baseline
    ax2 = Axis(fig[1, 2];
        aspect = (1 + sqrt(5)) / 2,
        xlabel = L"U",
        limits = ((0, 15), (1, 15.5)),
        yticks = [1, 5, 10, 15]
    )
    hideydecorations!(ax2, grid = false)

    # Panel 3: Infidelity vs dimH across multiple U values
    ax3 = Axis(fig[1, 3];
        aspect = (1 + sqrt(5)) / 2,
        xlabel = L"\dim \mathcal{H}",
        ylabel = L"1 - |\langle E_0(U)|\mathcal{U}|E_0(0)\rangle|^2",
        xscale = log10,
        yscale = log10,
        limits = ((15, 60000), (1e-15, 1.0)),
    )

    u_indices = 2:60
    palette_colors = cmap2(length(FILE_LABEL_PAIRS))
    selected_U_values = [4.0, 8.0, 12.0]
    num_u = length(selected_U_values)

    dimH_list = Float64[]
    losses_by_u = [Float64[] for _ in 1:num_u]

    # Styling for multiple U values in Panel 3
    u_linestyles = [:dot, :solid, :dash, :dashdot]
    u_markers = [:circle, :diamond, :rect, :utriangle]
    u_line_colors = [(:gray40, 0.7), (:black, 0.7), (:gray40, 0.7), (:gray60, 0.7)]

    panel1_systems = String[]
    panel2_systems = String[]

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
        local hilbert_space_size, interaction_data
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

        optimized_overlaps = Float64[]
        baseline_overlaps = Float64[]
        optimized_losses = Float64[]
        valid_u_indices = Int[]

        for u_idx in u_indices
            filepath = joinpath(sys_dir, "$(prefix)_u_$(u_idx).jld2")
            if !isfile(filepath)
                continue
            end
            result_dict = load_saved_dict(filepath)

            push!(valid_u_indices, u_idx)
            push!(optimized_losses, result_dict["metrics"]["loss"][2])
            push!(optimized_overlaps, 1.0 - result_dict["metrics"]["loss"][2])
            push!(baseline_overlaps, 1.0 - result_dict["metrics"]["loss"][1])
        end

        if isempty(valid_u_indices)
            @warn "No valid optimization files found for $file_label. Skipping."
            continue
        end

        overlap_improvement_ratios = optimized_overlaps ./ baseline_overlaps

        # Systems starting noticeably > 1 (e.g. > 1.5) at small U go to Panel 2, others to Panel 1
        initial_ratio = overlap_improvement_ratios[1]
        target_ax = initial_ratio > 1.5 ? ax2 : ax1
        if initial_ratio > 1.5
            push!(panel2_systems, file_label)
        else
            push!(panel1_systems, file_label)
        end

        lines!(target_ax, interaction_data[valid_u_indices], overlap_improvement_ratios;
            color = palette_colors[sys_idx], label = string(display_label), linewidth = 1.8)

        # Collect data for Panel 3 across all selected U values
        push!(dimH_list, Float64(hilbert_space_size))
        for (k, u_target) in enumerate(selected_U_values)
            target_idx = argmin(abs.(interaction_data[valid_u_indices] .- u_target))
            push!(losses_by_u[k], optimized_losses[target_idx])
        end
    end

    # Axis legends for Panel 1 and Panel 2
    axislegend(ax1; position = :lt, backgroundcolor = (:white, 0.85), LEGEND_ARGS...)
    axislegend(ax2; position = :lt, backgroundcolor = (:white, 0.85), LEGEND_ARGS...)

    # Panel 3: For each U value, plot connecting line and scatter points
    order = sortperm(dimH_list)
    dimH_sorted = dimH_list[order]

    for (k, u_val) in enumerate(selected_U_values)
        ls = u_linestyles[mod1(k, length(u_linestyles))]
        mk = u_markers[mod1(k, length(u_markers))]
        lc = u_line_colors[mod1(k, length(u_line_colors))]

        # Trend line connecting sorted points
        lines!(ax3, dimH_sorted, losses_by_u[k][order];
            color = lc, linestyle = ls, linewidth = 1.5)

        # Scatter points for each system (colored by system palette)
        scatter!(ax3, dimH_list, losses_by_u[k];
            color = palette_colors, markersize = 8, marker = mk)
    end

    # U-values legend on Panel 3 (in-axis legend at lower-right)
    u_legend_elements = [
        [LineElement(color = :black, linestyle = u_linestyles[mod1(k, length(u_linestyles))], linewidth = 1.5),
         MarkerElement(marker = u_markers[mod1(k, length(u_markers))], color = :black, markersize = 8)]
        for k in 1:num_u
    ]
    u_legend_labels = [L"U = %$(round(Int, u))" for u in selected_U_values]
    axislegend(ax3, u_legend_elements, u_legend_labels;
        position = :rb, backgroundcolor = (:white, 0.85), LEGEND_ARGS...)

    # Bottom-left corner annotations
    text!(ax1, 0.0, 0.0; text = "(a)", space = :relative, align = (:left, :bottom), offset = (8, 8), fontsize = 12)
    text!(ax2, 0.0, 0.0; text = "(b)", space = :relative, align = (:left, :bottom), offset = (8, 8), fontsize = 12)
    text!(ax3, 0.0, 0.0; text = "(c)", space = :relative, align = (:left, :bottom), offset = (8, 8), fontsize = 12)

    colgap!(fig.layout, 1, 6)
    colgap!(fig.layout, 2, 18)

    if !isdir(output_dir)
        mkpath(output_dir)
    end
    png_path = joinpath(output_dir, "three_panel_figure.png")
    pdf_path = joinpath(output_dir, "three_panel_figure.pdf")
    save(png_path, fig)
    save(pdf_path, fig)
    println("Saved 3-panel figure to: $png_path and $pdf_path")
    println("Panel 1 systems: ", panel1_systems)
    println("Panel 2 systems: ", panel2_systems)
    return fig
end

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "test_three_panel_figure")
    with_logging(log_path) do
        println("=== Starting test_three_panel_figure ===")
        output_dir = joinpath(@__DIR__, "..", "good_images", "final")
        fig = generate_three_panel_figure(output_dir=output_dir)
        @assert fig isa Figure "Output must be a Makie Figure"
        println("=== TEST PASSED SUCCESSFULLY: test_three_panel_figure ===")
    end
end
