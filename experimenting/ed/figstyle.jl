#=
figstyle.jl -- shared figure style for the manuscript (palette v2 + line identity).

Makie port of the matplotlib `figstyle.py` the figure spec was written against; the hex lists are
copied verbatim. Include it after fig_lib.jl (it uses Makie, LineElement, MarkerElement):

    include("figstyle.jl")
    system_line!(ax, U, y, i)                       # i = rank of the system by dim H, 1 = smallest
    Legend(pos, system_legend_entries(1:n), labels) # one entry per system, same look as the lines

Units: line widths, marker sizes and stroke widths are the spec's numbers in Makie units, as the
figures already used (e.g. linewidth = 1.8 for data lines).
=#

_hex(s) = parse(Makie.Colors.Colorant, s)

# ---- palettes ----------------------------------------------------------------------------------
# Systems ordered by Hilbert-space dimension, SMALL -> LARGE (index 1 = smallest)
const SIZE = _hex.(["#fdaa24", "#fe890d", "#f8682b", "#e74e48", "#d13758",
                    "#b82265", "#9a1176", "#760785", "#520288", "#2e027e"])
# Trotter steps m = 1, 2, 3, 4, 6, 8 (light -> dark); exact exponential (m = infinity); ansatz
const TROTTER = _hex.(["#10cafd", "#05a8e9", "#0688d4", "#0065c6", "#1f44aa", "#26208d"])
const EXACT = _hex("#d75b0c")      # m = infinity, dashed
const ANSATZ = :black              # exponential product, thick solid
const REF = _hex("#444444")        # threshold / guide lines, thin dotted
# Figure 1 schematic (Okabe-Ito based)
const BLUE, ORANGE, GREEN = _hex("#0072B2"), _hex("#D55E00"), _hex("#007A59")

# matplotlib 'o', 's', '^', 'D', 'v'; cycles with rank, so neighbours differ
const SHAPES = [:circle, :rect, :utriangle, :diamond, :dtriangle]

const DATA_LW = 1.8        # data lines
const ANSATZ_LW = 2.8      # the ansatz (exponential product) in Fig. 4
const REF_KW = (color=REF, linestyle=:dot, linewidth=1.0)   # reference elements: thresholds, markers of U, guides
const DASH_ODD = Makie.Linestyle([0.0, 5.0, 7.0])           # matplotlib (0, (5, 2)): 5 on, 2 off

"""
    line_kw(i; n_markevery=24, base_offset=7, step=3) -> NamedTuple

Style of the i-th system, i = 1 for the smallest dim H (figstyle.py's i is 0-based: i0 = i - 1).
Even i0: solid line, filled markers; odd i0: dashed (5, 2) line, white-filled markers. Markers at
every `n_markevery`-th point starting at `base_offset + step * i0` (0-based), staggered by rank.
"""
function line_kw(i::Integer; n_markevery::Integer=24, base_offset::Integer=7, step::Integer=3)
    i0 = i - 1
    solid = iseven(i0)
    c = SIZE[i]
    return (color=c, linestyle=solid ? :solid : DASH_ODD, linewidth=DATA_LW,
        marker=SHAPES[mod(i0, 5)+1], markersize=6.5, markercolor=solid ? c : :white,
        strokecolor=c, strokewidth=1.4, markevery=(base_offset + step * i0, n_markevery),
        z=10 - 0.1 * i0)
end

"""
    system_line!(ax, x, y, i; label=nothing, kw=line_kw(i)) -> (line, markers)

Draws system i's data line with its markers (every kw.markevery[2]-th point from kw.markevery[1],
0-based) in the figstyle look. Larger systems are drawn lower in the stack (kw.z).
"""
function system_line!(ax, x, y, i::Integer; label=nothing, kw=line_kw(i))
    ln = lines!(ax, x, y; color=kw.color, linestyle=kw.linestyle, linewidth=kw.linewidth, label=label)
    idx = (kw.markevery[1]+1):kw.markevery[2]:length(x)
    sc = scatter!(ax, collect(x)[idx], collect(y)[idx]; marker=kw.marker, markersize=kw.markersize,
        color=kw.markercolor, strokecolor=kw.strokecolor, strokewidth=kw.strokewidth)
    translate!(ln, 0, 0, kw.z); translate!(sc, 0, 0, kw.z + 0.05)
    return ln, sc
end

"""
    system_legend_entry(i; kw=line_kw(i)) -> Vector

Legend entry (line + marker) that looks like system i's line.
"""
system_legend_entry(i::Integer; kw=line_kw(i)) = [
    LineElement(color=kw.color, linestyle=kw.linestyle, linewidth=kw.linewidth),
    MarkerElement(marker=kw.marker, markersize=kw.markersize, color=kw.markercolor,
        strokecolor=kw.strokecolor, strokewidth=kw.strokewidth)]

system_legend_entries(is) = [system_legend_entry(i) for i in is]
