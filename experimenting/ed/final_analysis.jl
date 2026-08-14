"""
    final_analysis.jl

Produces publication-quality figures using CairoMakie for ED optimization analysis:
  - Loss curves comparing reused vs. random initial coefficients.
  - Relative state overlaps across interaction strengths U for multiple system sizes.
  - Overlap improvement ratio subplots.
  - Coefficient value trajectories and histograms at selected U values.

Usage:
    julia --project=.. final_analysis.jl
"""

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
    include("utility_functions.jl")
end
include("ed_objects.jl")
include("ed_functions.jl")
include("ed_optimization.jl")
include("data_path.jl")
include("logging.jl")

# ---------------------------------------------------------------------------
# Global Theme — equivalent to Plots.jl thickness_scaling = 1.3
# ---------------------------------------------------------------------------
const THICKNESS_SCALE = 1.3
const BASE_FONTSIZE   = 16
const BASE_LINEWIDTH  = 1.5
const BASE_MARKERSIZE = 12

set_theme!(
    fontsize    = round(Int, BASE_FONTSIZE * THICKNESS_SCALE),
    linewidth   = BASE_LINEWIDTH * THICKNESS_SCALE,
    markersize  = BASE_MARKERSIZE * THICKNESS_SCALE,
    Axis = (
        xticksize      = 6 * THICKNESS_SCALE,
        yticksize      = 6 * THICKNESS_SCALE,
        xminorticksize = 4 * THICKNESS_SCALE,
        yminorticksize = 4 * THICKNESS_SCALE,
        spinewidth     = 1 * THICKNESS_SCALE,
    ),
)

# ---------------------------------------------------------------------------
# Colormap Helpers
# ---------------------------------------------------------------------------

"""
    cmap1(num_colors::Int) -> Vector{RGBAf}

Return `num_colors` evenly-spaced colors from the `:linear_blue_5_95_c73_n256` colormap.
"""
function cmap1(num_colors::Int)
    return [Makie.ColorSchemes.linear_blue_5_95_c73_n256[z] for z in range(0, 1, length=num_colors)]
end

"""
    cmap2(num_colors::Int) -> Vector{RGBAf}

Return `num_colors` evenly-spaced colors from the `:managua` colormap.
"""
function cmap2(num_colors::Int)
    return [Makie.ColorSchemes.managua[z] for z in range(0, 1, length=num_colors)]
end

# ---------------------------------------------------------------------------
# System Configurations: (Display Label, Data Subfolder, Electron Counts)
# ---------------------------------------------------------------------------
const FILE_LABEL_PAIRS = [
    (L"3\times 2\;(2,2)",   "N=(2, 2)_3x2", (2, 2)),
    (L"3\times 2,\;(3,2)",  "N=(3, 2)_3x2", (3, 2)),
    (L"3\times3,\;(3,2)",   "N=(3, 2)_3x3", (3, 2)),
    (L"3\times3,\;(3,3)",   "N=(3, 3)_3x3", (3, 3)),
    (L"4\times2,\;(3,3)",   "N=(3, 3)_4x2", (3, 3)),
    (L"3\times3,\;(4,3)",   "N=(4, 3)_3x3", (4, 3)),
    (L"3\times3,\;(4,4)",   "N=(4, 4)_3x3", (4, 4)),
    (L"4\times3,\;(4,4)",   "N=(4, 4)_4x3", (5, 4)),
]

# ---------------------------------------------------------------------------
# Figure 1: Loss curves — reused vs. random initial coefficients
# ---------------------------------------------------------------------------

"""
    plot_loss_curves()

Plot optimization loss curves for (N↑, N↓) = (4,4) on the 3x3 lattice comparing
reused coefficients from prior U values against randomly generated initial coefficients.
"""
function plot_loss_curves()
    electron_counts = (4, 4)
    site_dim = (3, 3)
    u_indices = 2:7:60

    fig = Figure()
    ax = Axis(fig[1, 1];
        xlabel = "Iteration",
        ylabel = L"1-|\langle E_0(U)|\mathcal{U}|\psi_{ref}\rangle|^2",
        yscale = log10,
    )

    palette_colors = cmap1(length(u_indices))

    # Reused initial coefficients (solid lines)
    for (color_idx, u_idx) in enumerate(u_indices)
        subfolder = "N=$(electron_counts)_$(join(site_dim, "x"))"
        prefix = build_save_name_prefix(:trotter; sites=prod(site_dim), antihermitian=true, custom_ref_state_arg="slater")
        filepath = data_folder(joinpath(subfolder, "$(prefix)_u_$(u_idx).jld2"))
        if isfile(filepath)
            result_dict = load_saved_dict(filepath)
            lines!(ax, result_dict["metrics"]["optimization_losses"][1];
                color = palette_colors[color_idx], linewidth = 2)
        end
    end

    # Random initial coefficients (dotted lines)
    for (color_idx, u_idx) in enumerate(u_indices)
        subfolder = "N=$(electron_counts)_$(join(site_dim, "x"))_separate"
        prefix = build_save_name_prefix(:trotter; sites=prod(site_dim), antihermitian=true, custom_ref_state_arg="slater")
        filepath = data_folder(joinpath(subfolder, "$(prefix)_u_$(u_idx).jld2"))
        if isfile(filepath)
            result_dict = load_saved_dict(filepath)
            lines!(ax, result_dict["metrics"]["optimization_losses"][1];
                color = palette_colors[color_idx], linewidth = 2, linestyle = :dot)
        end
    end

    # Legend entries
    for (color_idx, u_idx) in enumerate(u_indices)
        lines!(ax, [NaN], [NaN];
            color = palette_colors[color_idx], linewidth = 2,
            label = L"U=%$((u_idx - 1) * 0.25)")
    end
    lines!(ax, [NaN], [NaN]; color = :black, linewidth = 2, linestyle = :solid, label = "Reused coefficients")
    lines!(ax, [NaN], [NaN]; color = :black, linewidth = 2, linestyle = :dot, label = "Random coefficients")

    axislegend(ax; position = :rt, backgroundcolor = (:white, 0.8))

    mkpath("good_images/final/extra")
    save("good_images/final/extra/loss_curve_(4,4)_3x3.png", fig)
    save("good_images/final/extra/loss_curve_(4,4)_3x3.pdf", fig)
    display(fig)
    return fig
end

# ---------------------------------------------------------------------------
# Figure 2: Main Overlap Analysis and Improvement Ratios
# ---------------------------------------------------------------------------

"""
    plot_main_analysis()

Plot state overlaps and overlap improvement ratios across interaction strengths U
for all system configurations in `FILE_LABEL_PAIRS`. Also generates coefficient histograms
and trajectory plots.
"""
function plot_main_analysis()
    data_root = get_data_root()

    fig_overlap = Figure()
    ax_overlap = Axis(fig_overlap[1, 1];
        xlabel = L"U",
        ylabel = L"|\langle E_0(U)|\mathcal{U}|E_0(\epsilon)\rangle|^2",
        limits = ((0, 15), nothing),
    )

    fig_improvement = Figure()
    ax_improvement = Axis(fig_improvement[1, 1];
        xlabel = L"U",
        ylabel = L"\frac{|\langle E_0(U)|\mathcal{U}|E_0(\epsilon)\rangle|^2}{|\langle E_0(U)|E_0(\epsilon)\rangle|^2}",
        limits = ((0, 15), (1, 15)),
    )

    hilbert_space_sizes = []
    final_performances = []
    num_tuning_parameters = []

    u_indices = 2:60
    palette_colors = cmap1(length(FILE_LABEL_PAIRS))

    for (sys_idx, (display_label, file_label, electron_counts)) in enumerate(FILE_LABEL_PAIRS)
        sys_dir = joinpath(data_root, file_label)
        if !isdir(sys_dir)
            @warn "Directory for system $file_label does not exist: $sys_dir. Skipping."
            continue
        end

        num_sites = prod(parse_lattice_dimension(file_label))
        prefix = build_save_name_prefix(:trotter;
            electrons = electron_counts, sites = num_sites,
            antihermitian = true, custom_ref_state_arg = "slater")

        meta_file = joinpath(sys_dir, "meta_data_and_E.jld2")
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
            interaction_data = nothing
            hilbert_space_size = nothing
            h5open(joinpath(sys_dir, valid_files[1]), "r") do h5data
                interaction_data = read(h5data["data/uvec"])
                hilbert_space_size = length(read(h5data["data/evecs/0"])[:, 1, 1])
            end
        end

        coefficients_list = []
        optimized_overlaps = Float64[]
        second_order_overlaps = Float64[]
        baseline_overlaps = Float64[]

        valid_u_indices = Int[]
        for u_idx in u_indices
            filepath = joinpath(sys_dir, "$(prefix)_u_$(u_idx).jld2")
            if !isfile(filepath)
                continue
            end
            result_dict = load_saved_dict(filepath)

            push!(valid_u_indices, u_idx)
            push!(coefficients_list, result_dict["coefficients"][2])
            push!(optimized_overlaps, 1 - result_dict["metrics"]["loss"][2])
            if length(result_dict["metrics"]["loss"]) > 2
                push!(second_order_overlaps, 1 - result_dict["metrics"]["loss"][3])
            end
            push!(baseline_overlaps, 1 - result_dict["metrics"]["loss"][1])
        end

        if isempty(valid_u_indices)
            @warn "No valid optimization files found for $file_label. Skipping."
            continue
        end

        coef_matrix = reduce(hcat, coefficients_list)

        selected_u_indices = [20, 30, 45]

        # Coefficient trajectory plot
        fig_trajectory = Figure()
        ax_trajectory = Axis(fig_trajectory[1, 1];
            xlabel = L"U",
            ylabel = L"A^{(2)} \;\textrm{value}",
            limits = ((0, 10), (-0.5, 0.5)),
        )
        for col in eachcol(coef_matrix)
            lines!(ax_trajectory, interaction_data[valid_u_indices], col; color = (:royalblue1, 0.06))
        end
        for (color_idx, u_idx) in enumerate(selected_u_indices)
            if u_idx in valid_u_indices
                vlines!(ax_trajectory, [interaction_data[u_idx]];
                    color = palette_colors[color_idx], linestyle = :dash,
                    label = L"U=%$(round(interaction_data[u_idx], digits=2))")
            end
        end
        axislegend(ax_trajectory; position = :rt, backgroundcolor = (:white, 0.8))

        # Main overlap curves
        lines!(ax_overlap, interaction_data[valid_u_indices], baseline_overlaps;
            linewidth = 1, color = palette_colors[sys_idx], linestyle = :dash)
        lines!(ax_overlap, interaction_data[valid_u_indices], optimized_overlaps;
            linewidth = 1, color = palette_colors[sys_idx], label = string(display_label))

        # Improvement ratios
        overlap_improvement_ratios = optimized_overlaps ./ baseline_overlaps
        target_u_val = 10
        scatterlines!(ax_improvement, interaction_data[valid_u_indices], overlap_improvement_ratios;
            color = palette_colors[sys_idx], marker = :circle, label = string(display_label))

        target_idx = argmin(abs.(interaction_data .- target_u_val)) + valid_u_indices[1] - 1
        push!(final_performances, overlap_improvement_ratios[min(target_idx, length(overlap_improvement_ratios))])
        push!(hilbert_space_sizes, hilbert_space_size)

        dimension_match = match(r"(?<N>\d+)[xX](?<M>\d+)", file_label)
        push!(num_tuning_parameters, get_num_2nd_order_coef(
            parse(Int, dimension_match[:N]),
            parse(Int, dimension_match[:M])
        ))

        # Histogram at selected U values
        for (color_idx, u_idx) in enumerate(selected_u_indices)
            filepath = joinpath(sys_dir, "$(prefix)_u_$(u_idx).jld2")
            if !isfile(filepath)
                continue
            end
            result_dict = load_saved_dict(filepath)
            coef_values = result_dict["coefficients"]
            if length(coef_values) < 5
                coef_values = coef_values[2]
            end

            fig_histogram = Figure()
            ax_histogram = Axis(fig_histogram[1, 1];
                xlabel = L"A^{(2)}\;\textrm{value}",
                ylabel = L"\textrm{Count}",
                limits = ((-0.5, 0.5), nothing),
            )
            hist!(ax_histogram, coef_values;
                bins = LinRange(-0.5, 0.5, 50),
                color = palette_colors[color_idx],
                label = L"U=%$(round(interaction_data[u_idx], digits=2))")
            axislegend(ax_histogram; position = :rt, backgroundcolor = (:white, 0.8))
            display(fig_histogram)

            mkpath("good_images/antihermitian")
            save("good_images/antihermitian/U=$(round(interaction_data[u_idx], digits=2))_$(file_label)_histogram.png", fig_histogram)
            save("good_images/antihermitian/U=$(round(interaction_data[u_idx], digits=2))_$(file_label)_histogram.pdf", fig_histogram)
        end
    end

    # Style legend entries for main overlap figure
    lines!(ax_overlap, [NaN], [NaN];
        color = :black, linewidth = 1, linestyle = :solid,
        label = L"\textrm{Optimized}\;\,A^{(2)} ")
    lines!(ax_overlap, [NaN], [NaN];
        color = :black, linewidth = 1, linestyle = :dash,
        label = L"A^{(2)}=0")

    axislegend(ax_overlap; position = :lb, backgroundcolor = (:white, 0.8))
    axislegend(ax_improvement; position = :lt, backgroundcolor = (:white, 0.8))

    display(fig_overlap)
    display(fig_improvement)

    mkpath("good_images/antihermitian")
    save("good_images/antihermitian/relative_loss.png", fig_overlap)
    save("good_images/antihermitian/relative_loss.pdf", fig_overlap)
    save("good_images/antihermitian/loss_improvement2.png", fig_improvement)
    save("good_images/antihermitian/loss_improvement2.pdf", fig_improvement)

    return fig_overlap, fig_improvement
end

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "final_analysis")
    with_logging(log_path) do
        plot_loss_curves()
        plot_main_analysis()
        return 0
    end
end
