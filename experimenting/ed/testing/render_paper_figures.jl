#=
render_paper_figures.jl

Re-make the paper's Figs. 2, 3 and 4 from the notebooks that define them (figure2_3.ipynb and
figure4.ipynb), without opening Jupyter, and write the paper PDFs to a directory of your choice.
The notebook cells are run exactly as written, except that every occurrence of the paper's
hard-coded figure directory (PAPER_FIGURES below, which the notebooks save figure2.pdf and
figure3.pdf into) is replaced by --out_dir, so a render never overwrites the paper checkout
unless you point --out_dir at it. Side outputs the cells write as usual (good_images/final/*,
data-summary/*.csv) are also written, relative to experimenting/ed of the checkout this
script lives in.

Usage:
  julia --project=.. testing/render_paper_figures.jl --figure=<figure> --out_dir=<dir>
  (run from experimenting/ed)

Options:
  --figure=<figure> (required): which figures to make. Valid options:
      - "2_3": figure2_3.ipynb cells 0 (setup), 11 (Fig. 2), 14 and 15 (Fig. 3: loads the
               pruning analyses, fits them and plots). Writes <out_dir>/figure2.pdf and
               <out_dir>/figure3.pdf. Takes a few minutes and must run on a compute node.
      - "4":   figure4.ipynb cells 1 (setup), 3 (configuration), 9 (plot builders), 19 (loads
               data-summary/fig4.csv -- nothing is recomputed) and 22 (Fig. 4, saved as
               good_images/final/energy_error_vs_dimH.pdf), then copies that PDF to
               <out_dir>/figure4.pdf. Takes about a minute and a half.
     The two run in separate Julia processes because the notebooks define clashing constants.
  --out_dir=<dir> (required): directory the paper PDFs are written to; created if missing.

Examples:
  julia --project=.. testing/render_paper_figures.jl --figure=4 --out_dir=/tmp/figs
  julia --project=.. testing/render_paper_figures.jl --figure=2_3 \
      --out_dir=/home/jek354/research/68f6bef0eaaf5c0928a922c6/Figures
=#

using JSON

include(joinpath(@__DIR__, "..", "logging.jl"))

const ED_DIR = normpath(joinpath(@__DIR__, ".."))
const PAPER_FIGURES = "/home/jek354/research/68f6bef0eaaf5c0928a922c6/Figures"
const NOTEBOOK_CELLS = Dict(
    "2_3" => ("figure2_3.ipynb", [0, 11, 14, 15]),
    "4" => ("figure4.ipynb", [1, 3, 9, 19, 22]),
)

"""
    parse_cli_args(args) -> (figure::String, out_dir::String)

Parse `--figure=` (one of the keys of NOTEBOOK_CELLS) and `--out_dir=`; both are required.
"""
function parse_cli_args(args)
    figure, out_dir = nothing, nothing
    for arg in args
        if startswith(arg, "--figure=")
            figure = split(arg, "=", limit=2)[2]
            haskey(NOTEBOOK_CELLS, figure) ||
                error("--figure must be one of $(sort(collect(keys(NOTEBOOK_CELLS)))) (given: $figure)")
        elseif startswith(arg, "--out_dir=")
            out_dir = abspath(split(arg, "=", limit=2)[2])
        else
            error("Unknown command line argument: $arg")
        end
    end
    isnothing(figure) && error("--figure is required")
    isnothing(out_dir) && error("--out_dir is required")
    return figure, out_dir
end

"""
    run_notebook_cells(notebook, cells, out_dir)

Evaluate the given code cells (0-based, as Jupyter numbers them in the JSON) of `notebook`
in Main, in order, with PAPER_FIGURES replaced by `out_dir`. Each cell is evaluated as if it
were a file in experimenting/ed, so `@__DIR__` resolves as it does in the notebook.
"""
function run_notebook_cells(notebook::String, cells::Vector{Int}, out_dir::String)
    nb = JSON.parsefile(joinpath(ED_DIR, notebook))
    for i in cells
        cell = nb["cells"][i+1]
        cell["cell_type"] == "code" || error("$notebook cell $i is not a code cell")
        src = replace(join(cell["source"]), PAPER_FIGURES => out_dir)
        println("=== running $notebook cell $i ===")
        Base.include_string(Main, src, joinpath(ED_DIR, "$(splitext(notebook)[1])_cell$(i).jl"))
    end
end

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "render_paper_figures")
    with_logging(log_path) do
        figure, out_dir = parse_cli_args(ARGS)
        mkpath(out_dir)
        cd(ED_DIR)
        notebook, cells = NOTEBOOK_CELLS[figure]
        figure == "4" && Core.eval(Main, :(NOTEBOOK_DIR = $ED_DIR))
        run_notebook_cells(notebook, cells, out_dir)
        if figure == "4"
            cp(joinpath(ED_DIR, "good_images", "final", "energy_error_vs_dimH.pdf"),
                joinpath(out_dir, "figure4.pdf"); force=true)
        end
        for f in (figure == "4" ? ["figure4.pdf"] : ["figure2.pdf", "figure3.pdf"])
            p = joinpath(out_dir, f)
            println(p, isfile(p) ? "  written, $(filesize(p)) bytes, $(Libc.strftime(mtime(p)))" : "  MISSING")
        end
    end
end
