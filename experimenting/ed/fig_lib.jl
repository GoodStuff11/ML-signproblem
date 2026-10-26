#=
fig_lib.jl

Shared plotting library for the paper notebooks (figure2_3.ipynb, figure4.ipynb,
trotter_exp_testing.ipynb) and the scripts that make the same figures
(trotter_exp_testing.jl, testing/test_*.jl). It holds the general plotting tools
only, nothing specific to one figure:
  * the figure conventions: journal-width figure sizes, the LaTeX-font theme, the
    colour maps and the legend arguments;
  * panel labels "(a)", "(b)", ... and free-text panel annotations ("U = 8",
    "F >= 0.95", ...).
Figure 4's data, computations and plot builders live in fig4_lib.jl, which loads
this file itself. This file loads only CairoMakie and LaTeXStrings and no ED code.
A notebook or script must `include("fig_lib.jl")` and never re-define any of the
functions or constants below; change them here so every figure stays consistent.
This file defines no `(@main)` entry point and is never run directly.

Workflow for making a figure
----------------------------
 1. Load the library once, at the top of the notebook / script:
        include("fig_lib.jl")          # also sets the theme (LaTeX fonts, 10 pt)
    or, for figure 4's data, computations and plot builders as well,
        include("fig4_lib.jl")         # loads fig_lib.jl too
 2. Create the figure at the journal width, with its first panel:
        fig, ax1 = create_fig(2, 55.0; xlabel=L"U", ylabel=L"E")
    `num_cols` is 1 (89 mm, single column) or 2 (180 mm, full page width) and the
    second argument is the height in mm. Keyword arguments go to the first Axis.
 3. Add any further panels to the figure's grid, as usual in Makie:
        ax2 = Axis(fig[1, 2]; xlabel=L"\dim \mathcal{H}")
 4. Plot. Use cmap1(L) ... cmap4(L) for L-colour palettes and splat LEGEND_ARGS into
    every legend:
        axislegend(ax1; position=:rb, LEGEND_ARGS...)
 5. Add free-text annotations to individual panels (optional):
        annotate_panel!(ax1, L"U = 8")                          # bottom-left corner
        annotate_panel!(ax2, L"\mathcal{F} \geq 0.95"; corner=:rt)
        annotate_panel!(ax2, "peak"; at=(3.0, 0.4), space=:data, align=(:left, :center))
 6. Label the panels last, once every panel exists:
        label_panels!(fig)             # (a), (b), ... in reading order
    Axes are found automatically and ordered row by row, left to right, from their
    grid positions (nested GridLayouts included). By default each label sits in the
    margin just above-left of its panel, where it can never cover data (the layout
    shrinks the panels slightly to make room). `placement=:inside` puts it inside
    the plot area instead (no layout change), `corner=:rt` moves it to another
    corner, `start='c'` starts at (c), and `axes=[ax2, ax1]` labels exactly the
    given axes in the given order (e.g. to skip an inset).
 7. Save as vector PDF (plus PNG for previews):
        save("name.pdf", fig); save("name.png", fig)
    CairoMakie writes every label, tick and annotation as real text with embedded
    fonts, so it stays selectable and searchable in the PDF. Do not rasterize the
    figure (e.g. by converting the PDF to an image) or the text is lost.
=#

using CairoMakie
using LaTeXStrings

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE CONVENTIONS
# ═══════════════════════════════════════════════════════════════════════════════

# MATLAB's parula, 64 entries, from https://stackoverflow.com/a/43264077
const PARULA = Makie.ColorSchemes.ColorScheme([Makie.RGB(c...) for c in [
    (0.2081, 0.1663, 0.5292), (0.2116238095, 0.1897809524, 0.5776761905), (0.212252381, 0.2137714286, 0.6269714286),
    (0.2081, 0.2386, 0.6770857143), (0.1959047619, 0.2644571429, 0.7279), (0.1707285714, 0.2919380952, 0.779247619),
    (0.1252714286, 0.3242428571, 0.8302714286), (0.0591333333, 0.3598333333, 0.8683333333), (0.0116952381, 0.3875095238, 0.8819571429),
    (0.0059571429, 0.4086142857, 0.8828428571), (0.0165142857, 0.4266, 0.8786333333), (0.032852381, 0.4430428571, 0.8719571429),
    (0.0498142857, 0.4585714286, 0.8640571429), (0.0629333333, 0.4736904762, 0.8554380952), (0.0722666667, 0.4886666667, 0.8467),
    (0.0779428571, 0.5039857143, 0.8383714286), (0.079347619, 0.5200238095, 0.8311809524), (0.0749428571, 0.5375428571, 0.8262714286),
    (0.0640571429, 0.5569857143, 0.8239571429), (0.0487714286, 0.5772238095, 0.8228285714), (0.0343428571, 0.5965809524, 0.819852381),
    (0.0265, 0.6137, 0.8135), (0.0238904762, 0.6286619048, 0.8037619048), (0.0230904762, 0.6417857143, 0.7912666667),
    (0.0227714286, 0.6534857143, 0.7767571429), (0.0266619048, 0.6641952381, 0.7607190476), (0.0383714286, 0.6742714286, 0.743552381),
    (0.0589714286, 0.6837571429, 0.7253857143), (0.0843, 0.6928333333, 0.7061666667), (0.1132952381, 0.7015, 0.6858571429),
    (0.1452714286, 0.7097571429, 0.6646285714), (0.1801333333, 0.7176571429, 0.6424333333), (0.2178285714, 0.7250428571, 0.6192619048),
    (0.2586428571, 0.7317142857, 0.5954285714), (0.3021714286, 0.7376047619, 0.5711857143), (0.3481666667, 0.7424333333, 0.5472666667),
    (0.3952571429, 0.7459, 0.5244428571), (0.4420095238, 0.7480809524, 0.5033142857), (0.4871238095, 0.7490619048, 0.4839761905),
    (0.5300285714, 0.7491142857, 0.4661142857), (0.5708571429, 0.7485190476, 0.4493904762), (0.609852381, 0.7473142857, 0.4336857143),
    (0.6473, 0.7456, 0.4188), (0.6834190476, 0.7434761905, 0.4044333333), (0.7184095238, 0.7411333333, 0.3904761905),
    (0.7524857143, 0.7384, 0.3768142857), (0.7858428571, 0.7355666667, 0.3632714286), (0.8185047619, 0.7327333333, 0.3497904762),
    (0.8506571429, 0.7299, 0.3360285714), (0.8824333333, 0.7274333333, 0.3217), (0.9139333333, 0.7257857143, 0.3062761905),
    (0.9449571429, 0.7261142857, 0.2886428571), (0.9738952381, 0.7313952381, 0.266647619), (0.9937714286, 0.7454571429, 0.240347619),
    (0.9990428571, 0.7653142857, 0.2164142857), (0.9955333333, 0.7860571429, 0.196652381), (0.988, 0.8066, 0.1793666667),
    (0.9788571429, 0.8271428571, 0.1633142857), (0.9697, 0.8481380952, 0.147452381), (0.9625857143, 0.8705142857, 0.1309),
    (0.9588714286, 0.8949, 0.1132428571), (0.9598238095, 0.9218333333, 0.0948380952), (0.9661, 0.9514428571, 0.0755333333),
    (0.9763, 0.9831, 0.0538)]])

# Parula and plasma end in a bright yellow that barely shows on white paper, so their
# palettes stop at this fraction of the colour map (gold/orange) instead of at 1.
const YELLOW_CUTOFF = 0.85

cmap1(L) = [Makie.ColorSchemes.roma[z] for z in range(0, 1, length=L)]
cmap2(L) = [PARULA[z] for z in range(0, YELLOW_CUTOFF, length=L)]
cmap3(L) = [Makie.ColorSchemes.plasma[z] for z in range(0, YELLOW_CUTOFF, length=L)]
cmap4(L) = [Makie.ColorSchemes.RdYlBu[z] for z in range(1, 0, length=L)]  # RdYlBu_r: blue -> red

"""
    get_figsize(num_cols, height_mm=55.0) -> (width_px, height_px)

Figure size for a single-column (`num_cols=1`, 89 mm) or full-width (`num_cols=2`,
180 mm) figure, in Makie px at 96 px/inch -- which CairoMakie's default PDF export
(0.75 pt per px) turns back into exactly those millimetres.
"""
function get_figsize(num_cols::Int, height_mm::Float64=55.0)
    px_per_mm = 96 / 25.4
    if num_cols == 1
        width_mm = 89 # mm
    elseif num_cols == 2
        width_mm = 180 # mm
    else
        error("Can only have num_cols == 1 or 2 (given: $num_cols)")
    end
    width_px = width_mm * px_per_mm
    height_px = height_mm * px_per_mm
    return (width_px, height_px)
end

"""
    create_fig(num_cols, height_mm=55.0; aspect=nothing, kwargs...) -> (Figure, Axis)

Journal-sized figure (see `get_figsize`) with its first panel at `fig[1, 1]`. `aspect`
is the axis aspect ratio (`nothing` or `:auto` lets it fill the cell); the remaining
keyword arguments are passed to that `Axis`. Add further panels with
`Axis(fig[i, j]; ...)` and label them all with `label_panels!(fig)`.
"""
function create_fig(num_cols::Int, height_mm::Float64=55.0; aspect=nothing, kwargs...)
    fig = Figure(size=get_figsize(num_cols, height_mm), figure_padding=5)
    ax_aspect = (aspect == :auto || isnothing(aspect)) ? nothing : aspect
    ax = Axis(fig[1, 1];
        aspect=ax_aspect,
        kwargs...
    )
    return fig, ax
end

const LEGEND_ARGS = Dict(:rowgap => -8, :padding => (3, 3, 0, 0))
set_theme!(theme_latexfonts(), fontsize=10)
# set_theme!(font = "Arial", fontsize = 10)

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL LABELS AND ANNOTATIONS
# ═══════════════════════════════════════════════════════════════════════════════
# Both are drawn as text (text! inside the axis, Label in the layout margin), so they
# come out as real, selectable text in the saved PDF. Corners use axislegend's
# convention: a horizontal letter (l/c/r) then a vertical one (b/c/t), e.g. :lt is
# the left-top corner.

const PANEL_AXIS_TYPES = Union{Axis,Axis3,PolarAxis}

# Grid position of `block` as a path of (row, col) pairs from the figure's top-level
# layout down to the cell holding it, so that sorting these paths orders panels row by
# row, left to right, also when panels sit in nested GridLayouts.
function grid_path(block)
    path = Tuple{Int,Int}[]
    gc = block.layoutobservables.gridcontent[]
    while !isnothing(gc)
        pushfirst!(path, (first(gc.span.rows), first(gc.span.cols)))
        gc = gc.parent.layoutobservables.gridcontent[]
    end
    return path
end

"""
    panel_axes(fig) -> Vector

Every Axis / Axis3 / PolarAxis of `fig`, in reading order (row by row, left to right,
by grid position). This is the order `label_panels!` assigns (a), (b), ... in.
"""
panel_axes(fig::Figure) = sort([b for b in fig.content if b isa PANEL_AXIS_TYPES]; by=grid_path)

# (anchor in relative axis coordinates, text alignment, offset in px pointing into the axis)
function corner_geometry(corner::Symbol, pad::Real)
    c = string(corner)
    (length(c) == 2 && c[1] in "lcr" && c[2] in "bct") ||
        error("corner must be one of :lb, :lc, :lt, :cb, :cc, :ct, :rb, :rc, :rt (given: :$corner)")
    h = Dict('l' => (0.0, :left, pad), 'c' => (0.5, :center, 0), 'r' => (1.0, :right, -pad))[c[1]]
    v = Dict('b' => (0.0, :bottom, pad), 'c' => (0.5, :center, 0), 't' => (1.0, :top, -pad))[c[2]]
    return (h[1], v[1]), (h[2], v[2]), (h[3], v[3])
end

# Draw `str` at `corner` of `ax`: inside the plot area (`placement=:inside`, a text!
# plot that does not change the layout or the axis limits) or in the layout margin just
# outside that corner (`placement=:outside`, a Label that the layout makes room for).
function place_corner_text!(ax, str, corner::Symbol, placement::Symbol, pad::Real; kwargs...)
    if placement == :inside
        anchor, align, offset = corner_geometry(corner, pad)
        plt = text!(ax, anchor...; text=str, space=:relative, align=align, offset=offset, kwargs...)
        translate!(plt, 0, 0, 10)  # draw above the data, even data plotted afterwards
        return plt
    elseif placement == :outside
        sides = Dict(:lt => (TopLeft(), :right, :bottom, (0, pad, pad, 0)),
                     :rt => (TopRight(), :left, :bottom, (pad, 0, pad, 0)),
                     :lb => (BottomLeft(), :right, :top, (0, pad, 0, pad)),
                     :rb => (BottomRight(), :left, :top, (pad, 0, 0, pad)))
        haskey(sides, corner) || error("placement=:outside needs corner :lt, :rt, :lb or :rb (given: :$corner)")
        side, halign, valign, padding = sides[corner]
        gc = ax.layoutobservables.gridcontent[]
        isnothing(gc) && error("placement=:outside needs an axis placed in a GridLayout")
        return Label(gc.parent[gc.span.rows, gc.span.cols, side], str;
            halign=halign, valign=valign, padding=padding, kwargs...)
    else
        error("placement must be :inside or :outside (given: :$placement)")
    end
end

"""
    label_panels!(fig; axes=panel_axes(fig), labels=nothing, start='a', format=l -> "(\$l)",
                  corner=:lt, placement=:outside, pad=4, kwargs...) -> Vector

Label the panels of `fig` "(a)", "(b)", ... in reading order. Call it after every
panel has been added. Returns the created text plots (`:inside`) or Labels (`:outside`).

- `axes`: the panels to label, in labelling order. Default: every axis of `fig`, row by
  row, left to right (see `panel_axes`). Pass a subset to skip e.g. an inset.
- `labels`: explicit label strings, one per axis (overrides `start` and `format`).
- `start`: letter of the first panel, e.g. `start='c'` continues a figure split over
  two files.
- `format`: turns the letter into the label, e.g. `l -> "\$l)"` or `l -> uppercase(l)`.
- `corner`: corner of each panel (a Symbol, or one Symbol per axis) -- :lt, :rt, :lb,
  :rb, ... (see `corner_geometry`).
- `placement`: `:outside` (default) in the layout margin beside the corner, so the label
  never covers data (corners :lt, :rt, :lb, :rb only), or `:inside` the plot area,
  which leaves the layout untouched but can overlap curves or legends.
- `pad`: distance in px from the corner.
- `kwargs`: text attributes, e.g. `fontsize=12`, `font=:bold`, `color=:gray30`.
"""
function label_panels!(fig::Figure; axes=panel_axes(fig), labels=nothing, start::Char='a',
    format=l -> "($l)", corner=:lt, placement::Symbol=:outside, pad::Real=4, kwargs...)
    n = length(axes)
    texts = isnothing(labels) ? [format(start + (i - 1)) for i in 1:n] : labels
    length(texts) == n || error("got $(length(texts)) labels for $n panels")
    corners = corner isa Symbol ? fill(corner, n) : corner
    length(corners) == n || error("got $(length(corners)) corners for $n panels")
    return [place_corner_text!(ax, t, c, placement, pad; kwargs...) for (ax, t, c) in zip(axes, texts, corners)]
end

"""
    annotate_panel!(ax, str; corner=:lb, placement=:inside, pad=6,
                    at=nothing, space=:relative, align=(:left, :bottom), kwargs...)

Write a free-text annotation, e.g. `L"U = 8"` or `L"\\mathcal{F} \\geq 0.95"`, on the
panel `ax`. By default it goes inside the plot area at a corner (see `label_panels!`
for `corner`, `placement` and `pad`; the default corner :lb keeps clear of the panel
label at :lt). With `at=(x, y)` it goes at that point instead, in relative axis coordinates
(`space=:relative`, (0, 0) is bottom-left and (1, 1) top-right) or in data coordinates
(`space=:data`), with `align` choosing which point of the text sits at `at`.
`kwargs` are text attributes, e.g. `fontsize=12`, `color=:gray30`.
"""
function annotate_panel!(ax, str; corner::Symbol=:lb, placement::Symbol=:inside, pad::Real=6,
    at=nothing, space::Symbol=:relative, align=(:left, :bottom), kwargs...)
    isnothing(at) && return place_corner_text!(ax, str, corner, placement, pad; kwargs...)
    plt = text!(ax, at...; text=str, space=space, align=align, kwargs...)
    translate!(plt, 0, 0, 10)
    return plt
end
