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
# using CUDA
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
# Colormap Helpers
# ---------------------------------------------------------------------------

"""
    cmap1(L::Int) -> Vector{RGBAf}

Return `L` evenly-spaced colors from the `:roma` colormap.
"""
cmap1(L::Int) = [Makie.ColorSchemes.roma[z] for z in range(0, 1, length=L)]

"""
    cmap2(L::Int) -> Vector{RGBAf}

Return `L` evenly-spaced colors from the `:managua` colormap.
"""
cmap2(L::Int) = [Makie.ColorSchemes.managua[z] for z in range(0, 1, length=L)]

# ---------------------------------------------------------------------------
# System Configurations: (Display Label, Data Subfolder, Electron Counts)
# ---------------------------------------------------------------------------
const FILE_LABEL_PAIRS = [
    (L"3\times 2\;(2,2)",   "N=(2, 2)_3x2", (2, 2)),
    (L"3\times 2,\;(3,2)",  "N=(3, 2)_3x2", (3, 2)),
    (L"3\times 2,\;(3,3)",  "N=(3, 3)_3x2", (3, 3)),
    (L"3\times3,\;(3,2)",   "N=(3, 2)_3x3", (3, 2)),
    (L"4\times2,\;(3,3)",   "N=(3, 3)_4x2", (3, 3)),
    (L"3\times3,\;(3,3)",   "N=(3, 3)_3x3", (3, 3)),
    (L"3\times3,\;(4,3)",   "N=(4, 3)_3x3", (4, 3)),
    (L"3\times3,\;(4,4)",   "N=(4, 4)_3x3", (4, 4)),
    (L"3\times3,\;(5,4)",   "N=(5, 4)_3x3", (5, 4)),
    # (L"4\times3,\;(4,4)",   "N=(4, 4)_4x3", (4, 4)),
    (L"4\times3,\;(5,4)",   "N=(5, 4)_4x3", (5, 4)),
]

const FOLDER = get_data_root()

const THICKNESS_SCALE = 1.3
const BASE_FONTSIZE   = 16    # Makie default
const BASE_LINEWIDTH  = 1.5   # Makie default
const BASE_MARKERSIZE = 12    # Makie default

# ---------------------------------------------------------------------------
# Authentic LaTeX Typography (CMSY10 Calligraphic Font for \mathcal)
# ---------------------------------------------------------------------------
import CairoMakie.MathTeXEngine as MTE
import Makie.FreeTypeAbstraction as FreeTypeAbstraction

const CMSY_FONT_PATHS = [
    "/usr/share/texlive/texmf-dist/fonts/type1/public/amsfonts/cm/cmsy10.pfb",
    "/usr/share/fonts/type1/texlive-fonts-recommended/cmsy10.pfb"
]
const CMSY_FONT_PATH = first(filter(isfile, CMSY_FONT_PATHS))

if isfile(CMSY_FONT_PATH)
    const CMSY_FONT = FreeTypeAbstraction.FTFont(CMSY_FONT_PATH)
    const MATHCAL_INV_MAP = Dict{Char, Char}(v => k for (k, v) in MTE.latex_symbols[raw"\mathcal"])

    const _orig_texelements = MTE.texelements

    function MTE.texelements(doc::LaTeXString, fontinfo)
        elements, bbox = _orig_texelements(doc, fontinfo)
        new_elements = MTE.TeXElement[]
        for el in elements
            if el isa MTE.TeXChar && haskey(MATHCAL_INV_MAP, el.represented_char)
                ascii_char = MATHCAL_INV_MAP[el.represented_char] # 'U'
                push!(new_elements, MTE.TeXChar(ascii_char, el.position, CMSY_FONT, el.scale, :cal))
            else
                push!(new_elements, el)
            end
        end
        return new_elements, bbox
    end

    function MTE.texelements(doc::LaTeXString)
        MTE.texelements(doc, MTE.default_fonts())
    end
end

set_theme!(
    fontsize       = round(Int, BASE_FONTSIZE * THICKNESS_SCALE),
    linewidth      = BASE_LINEWIDTH * THICKNESS_SCALE,
    markersize     = BASE_MARKERSIZE * THICKNESS_SCALE,
    Axis = (
        xticksize      = 6 * THICKNESS_SCALE,
        yticksize      = 6 * THICKNESS_SCALE,
        xminorticksize = 4 * THICKNESS_SCALE,
        yminorticksize = 4 * THICKNESS_SCALE,
        spinewidth     = 1 * THICKNESS_SCALE,
    ),
)

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
for all system configurations in `FILE_LABEL_PAIRS`. Also generates coefficient trajectory plots.
"""
function plot_main_analysis()
    fig_overlap = Figure(size=(1200, 500))
    ax_overlap = Axis(fig_overlap[1, 1];
        xlabel = L"U",
        ylabel = L"|\langle E_0(U)|\mathcal{U}|E_0(0)\rangle|^2",
        limits = ((0, 15), (-0.05, 1.05)),
    )

    ax_improvement = Axis(fig_overlap[1, 2];
        xlabel = L"U",
        ylabel = L"\frac{|\langle E_0(U)|\mathcal{U}|E_0(0)\rangle|^2}{|\langle E_0(U)|E_0(0)\rangle|^2}",
        limits = ((0, 15), (1, 15)),
    )

    hilbert_space_sizes = []
    final_performances = []
    num_tuning_parameters = []

    u_indices = 2:60
    palette_colors = cmap2(length(FILE_LABEL_PAIRS))

    selected_u_indices = [20, 30, 45]
    palette_colors2 = cmap2(length(selected_u_indices))

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
        if isfile(meta_file)
            interaction_data = nothing
            hilbert_space_size = nothing
            jldopen(meta_file, "r") do f
                interaction_data = f["dict/meta_data/U_values"]
                if haskey(f["dict"], "all_E")
                    hilbert_space_size = size(f["dict/all_E"], 1)
                elseif haskey(f["dict"], "all_full_eig_vecs")
                    hilbert_space_size = size(f["dict/all_full_eig_vecs"][1], 2)
                end
            end
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
                    color = palette_colors2[color_idx], linestyle = :dash,
                    label = L"U=%$(round(interaction_data[u_idx], digits=2))")
            end
        end
        axislegend(ax_trajectory; position = :rt, backgroundcolor = (:white, 0.8))

        # Main overlap curves
        lines!(ax_overlap, interaction_data[valid_u_indices], baseline_overlaps;
            color = palette_colors[sys_idx], linestyle = :dash)
        lines!(ax_overlap, interaction_data[valid_u_indices], optimized_overlaps;
            color = palette_colors[sys_idx])

        # Improvement ratios
        overlap_improvement_ratios = optimized_overlaps ./ baseline_overlaps
        target_u_val = 10
        lines!(ax_improvement, interaction_data[valid_u_indices], overlap_improvement_ratios;
            color = palette_colors[sys_idx], label = string(display_label))

        target_idx = argmin(abs.(interaction_data .- target_u_val)) + valid_u_indices[1] - 1
        push!(final_performances, overlap_improvement_ratios[min(target_idx, length(overlap_improvement_ratios))])
        push!(hilbert_space_sizes, hilbert_space_size)

        dimension_match = match(r"(?<N>\d+)[xX](?<M>\d+)", file_label)
        push!(num_tuning_parameters, get_num_2nd_order_coef(
            parse(Int, dimension_match[:N]),
            parse(Int, dimension_match[:M])
        ))
    end

    # Style legend entries for main overlap figure
    lines!(ax_overlap, [NaN], [NaN];
        color = :black, linestyle = :solid,
        label = L"\textrm{Optimized}\;\,A^{(2)} ")
    lines!(ax_overlap, [NaN], [NaN];
        color = :black, linestyle = :dash,
        label = L"A^{(2)}=0")

    Legend(fig_overlap[1, 3], ax_improvement)
    axislegend(ax_overlap; position = :lb, backgroundcolor = (:white, 0.8))

    display(fig_overlap)

    out_dir = joinpath(@__DIR__, "good_images", "final")
    mkpath(out_dir)
    save(joinpath(out_dir, "loss_curve.png"), fig_overlap)
    save(joinpath(out_dir, "loss_curve.pdf"), fig_overlap)
    println("Saved loss_curve.png and loss_curve.pdf to: ", out_dir)

    return fig_overlap
end

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "final_analysis")
    with_logging(log_path) do
        plot_main_analysis()
        return 0
    end
end
