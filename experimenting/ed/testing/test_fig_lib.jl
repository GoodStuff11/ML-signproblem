#=
test_fig_lib.jl

Verifies fig_lib.jl's panel-label / annotation API and that every notebook and script
that makes a figure with `create_fig` takes it from fig_lib.jl:

 1. API on a synthetic figure: `panel_axes` orders the panels row by row, left to right
    (nested GridLayouts included) whatever order they were created in; `label_panels!`
    writes "(a)", "(b)", ... in that order, by default in the margin just above-left of
    each panel (checked on the solved layout: every label lies outside its plot area);
    the `start`, `labels`, `corner`, `placement=:inside` options; `annotate_panel!` in a
    corner and at a data point, without changing the axis limits; bad corners/placements
    raise errors.
 2. Selectable text: the saved PDF's text is extracted with Ghostscript (txtwrite), and
    every panel label and annotation (including L"U=8" and L"\mathcal{F}\geq 0.95") must
    be found in it as text.
 3. Static audit of the library split: fig_lib.jl defines exactly the general plotting
    tools (conventions, panel labels, annotations), loads only CairoMakie and
    LaTeXStrings and includes no other file; everything figure-4 -- data, computations
    and plot builders -- is in fig4_lib.jl, which includes fig_lib.jl; the two define no
    name in common. Every .ipynb / .jl file under experimenting/ed that calls
    `create_fig` loads fig_lib.jl (directly or through fig4_lib.jl), every one that
    calls a fig4_lib.jl function includes fig4_lib.jl (directly, or through
    trotter_exp_testing.jl), and none redefines a top-level function or constant of
    either library. Then each library is loaded on its own in a fresh Julia process,
    and fig4_lib.jl's three plot builders are run on synthetic data there and saved.
 4. (only with --render_notebooks) Runs the notebooks' setup cells, each notebook in a
    fresh Julia process (in Main, as the Jupyter kernel does), to check fig_lib.jl loads
    alongside their code; then runs the
    figure2_3.ipynb cells that build its three labelled multi-panel figures on the real
    data and checks their PDFs contain (a), (b), (c) and the U=8 annotation as text.

Usage:
  julia --project=../.. test_fig_lib.jl [options]       (from experimenting/ed/testing)

Options:
  --output_dir=<path> (optional): Directory the test figures (PDF + PNG) are written to.
                      Created if missing. Default: a fresh temporary directory.
  --render_notebooks  (optional): Also run part 4. Needs the ED data (the data root from
                      data_path.jl, e.g. ED_DATA_ROOT=/home/jek354/research/data/new_data/data_h5_fixed)
                      and takes a few minutes. The notebook cells' own `save`/`CSV.write`
                      calls write into <output_dir>/good_images/final and
                      <output_dir>/data-summary, never into the repository.
  --pypdf_python=<path> (optional): A Python interpreter with the pypdf package. Part 2
                      then also extracts the PDF text with pypdf and checks the
                      calligraphic letters (\mathcal{F}, \mathcal{H}) copy out as 𝓕, 𝓗.
                      Ghostscript's txtwrite (always used) garbles characters outside
                      Unicode's basic plane, so it cannot check those.
Requires the `gs` (Ghostscript) executable on PATH for parts 2 and 4.
=#

using Test
using JSON

include(joinpath(@__DIR__, "..", "logging.jl"))
include(joinpath(@__DIR__, "..", "fig_lib.jl"))

const ED_DIR = normpath(joinpath(@__DIR__, ".."))

"""
    parse_cli_args(args) -> NamedTuple

Parse the options documented in this file's header.
"""
function parse_cli_args(args)
    output_dir = nothing
    render_notebooks = false
    pypdf_python = nothing
    for arg in args
        if startswith(arg, "--output_dir=")
            output_dir = split(arg, "=", limit=2)[2]
        elseif arg == "--render_notebooks"
            render_notebooks = true
        elseif startswith(arg, "--pypdf_python=")
            pypdf_python = split(arg, "=", limit=2)[2]
        else
            error("Unknown command line argument: $arg")
        end
    end
    output_dir = isnothing(output_dir) ? mktempdir() : abspath(output_dir)
    mkpath(output_dir)
    return (; output_dir, render_notebooks, pypdf_python)
end

"""
    pdf_text(path) -> String

Text of a PDF as a PDF viewer would select/copy it, via Ghostscript's txtwrite device.
"""
function pdf_text(path)
    isnothing(Sys.which("gs")) && error("Ghostscript (gs) not found on PATH")
    return read(`gs -q -dNOPAUSE -dBATCH -dSAFER -sDEVICE=txtwrite -sOutputFile=- $path`, String)
end

squash(s) = filter(!isspace, s)
scalar(x) = x isa AbstractVector ? only(x) : x  # text! stores `text` as a one-element vector

"""
    pypdf_text(python, path) -> String

Text of a PDF extracted by pypdf, run with the given Python interpreter. Unlike
Ghostscript's txtwrite it decodes characters outside Unicode's basic plane, such as the
calligraphic letters of \\mathcal{F} and \\mathcal{H}.
"""
pypdf_text(python, path) = read(`$python -I -c "import pypdf, sys; print(''.join(p.extract_text() for p in pypdf.PdfReader(sys.argv[1]).pages))" $path`, String)

# ─────────────────────────────────────────────────────────────────────────────
# 1 + 2: API on a synthetic figure, and selectable text in its PDF
# ─────────────────────────────────────────────────────────────────────────────
function test_api(output_dir, pypdf_python)
    # panels created deliberately out of reading order, one row nested in a GridLayout
    fig, ax_a = create_fig(2, 100.0; xlabel=L"U", ylabel=L"E - E_0")
    inner = GridLayout(fig[2, 2])
    ax_e = Axis(inner[1, 2]; xlabel=L"U")
    ax_c = Axis(fig[2, 1]; xlabel=L"\dim \mathcal{H}", ylabel=L"1-\mathcal{F}", xscale=log10, yscale=log10)
    ax_b = Axis(fig[1, 2]; xlabel=L"U")
    ax_d = Axis(inner[1, 1]; xlabel=L"U", ylabel=L"\mathcal{F}")
    expected = [ax_a, ax_b, ax_c, ax_d, ax_e]

    x = range(0, 15, length=60)
    for (i, ax) in enumerate(expected)
        ax === ax_c && continue
        lines!(ax, x, @. 1 - exp(-x / (2i)); color=cmap2(5)[i], label=L"P = %$i")
    end
    lines!(ax_c, 10 .^ (1:6), 10.0 .^ -(1:6); color=:black)
    axislegend(ax_a; position=:rb, LEGEND_ARGS...)

    @testset "panel_axes reading order" begin
        @test panel_axes(fig) == expected
        @test [grid_path(ax) for ax in expected] ==
              [[(1, 1)], [(1, 2)], [(2, 1)], [(2, 2), (1, 1)], [(2, 2), (1, 2)]]
    end

    @testset "annotate_panel! leaves the axis limits alone" begin
        reset_limits!(ax_a); reset_limits!(ax_b)
        lims_a, lims_b = ax_a.finallimits[], ax_b.finallimits[]
        ann_u = annotate_panel!(ax_a, L"U=8")
        ann_f = annotate_panel!(ax_b, L"\mathcal{F}\geq 0.95"; corner=:rt, fontsize=12)
        ann_pt = annotate_panel!(ax_d, "knee"; at=(5.0, 0.5), space=:data, align=(:left, :center))
        reset_limits!(ax_a); reset_limits!(ax_b)
        @test ax_a.finallimits[] == lims_a
        @test ax_b.finallimits[] == lims_b
        @test ann_u isa Makie.Text && ann_u.space[] == :relative && ann_u.align[] == (:left, :bottom)
        @test ann_f.align[] == (:right, :top) && only(ann_f[1][]) == Point2(1.0, 1.0)
        @test ann_pt.space[] == :data && only(ann_pt[1][]) == Point2(5.0, 0.5)
        println("annotations: U=8 at ax_a :lb, F>=0.95 at ax_b :rt, \"knee\" at data (5, 0.5) of ax_d")
    end

    labels = label_panels!(fig)
    @testset "label_panels! default: (a)-(e), outside the top-left corners" begin
        @test length(labels) == 5
        @test all(l -> l isa Label, labels)
        @test [l.text[] for l in labels] == ["(a)", "(b)", "(c)", "(d)", "(e)"]
        for (l, ax) in zip(labels, expected)
            gl, ga = l.layoutobservables.gridcontent[], ax.layoutobservables.gridcontent[]
            @test gl.parent === ga.parent && gl.span == ga.span
            @test gl.side isa Makie.GridLayoutBase.TopLeft
        end
    end

    pdf = joinpath(output_dir, "fig_lib_api_test.pdf")
    save(pdf, fig)
    save(joinpath(output_dir, "fig_lib_api_test.png"), fig)

    @testset "labels lie outside the plot areas on the solved layout" begin
        for (l, ax) in zip(labels, expected)
            lb, axb = l.layoutobservables.computedbbox[], ax.layoutobservables.computedbbox[]
            label_bottom = minimum(lb)[2]
            label_right = maximum(lb)[1]
            ax_top, ax_left = maximum(axb)[2], minimum(axb)[1]
            println("  $(l.text[]): label bottom $(round(label_bottom, digits=1)) px >= panel top $(round(ax_top, digits=1)) px, ",
                    "label right $(round(label_right, digits=1)) px <= panel left $(round(ax_left, digits=1)) px")
            @test label_bottom >= ax_top - 0.5
            @test label_right <= ax_left + 0.5
        end
    end

    @testset "PDF text is selectable" begin
        txt = pdf_text(pdf)
        println("--- text extracted from $pdf by Ghostscript ---")
        println(txt)
        println("--- end of extracted text ---")
        flat = squash(txt)
        for s in ["(a)", "(b)", "(c)", "(d)", "(e)", "U=8", "≥0.95", "knee"]
            found = occursin(s, flat)
            println("  \"$s\" found as text: $found")
            @test found
        end
        if !isnothing(pypdf_python)
            flat_py = squash(pypdf_text(pypdf_python, pdf))
            println("--- text extracted by pypdf ---\n", pypdf_text(pypdf_python, pdf), "--- end ---")
            for s in ["(a)", "(b)", "(c)", "(d)", "(e)", "U=8", "𝓕≥0.95", "dim𝓗", "1−𝓕", "knee"]
                found = occursin(s, flat_py)
                println("  pypdf: \"$s\" found as text: $found")
                @test found
            end
        end
    end

    @testset "options: start, labels, corner, placement=:inside" begin
        fig2, ax1 = create_fig(1, 55.0)
        ax2 = Axis(fig2[1, 2])
        ls = label_panels!(fig2; start='c', placement=:inside, corner=[:rt, :lb], fontsize=12)
        @test [scalar(l.text[]) for l in ls] == ["(c)", "(d)"]
        @test all(l -> l isa Makie.Text && l.space[] == :relative, ls)
        @test ls[1].align[] == (:right, :top) && ls[2].align[] == (:left, :bottom)
        @test ls[1].offset[][1:2] == [-4, -4] && ls[2].offset[][1:2] == [4, 4]
        @test ls[1].fontsize[] == 12
        @test ls[1].parent === ax1.scene && ls[2].parent === ax2.scene
        @test translation(ls[1])[][3] == 10
        ls2 = label_panels!(fig2; axes=[ax2], labels=["(i)"], placement=:outside, corner=:rt)
        @test ls2[1].text[] == "(i)" && ls2[1].layoutobservables.gridcontent[].side isa Makie.GridLayoutBase.TopRight
        ls3 = label_panels!(fig2; format=l -> "$(uppercase(l))", axes=[ax1])
        @test ls3[1].text[] == "A"
    end

    @testset "bad arguments raise" begin
        fig3, ax = create_fig(1)
        @test_throws ErrorException label_panels!(fig3; corner=:xy)
        @test_throws ErrorException label_panels!(fig3; corner=:ct, placement=:outside)
        @test_throws ErrorException label_panels!(fig3; placement=:middle)
        @test_throws ErrorException label_panels!(fig3; labels=["(a)", "(b)"])
        @test_throws ErrorException annotate_panel!(ax, "x"; corner=:top)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# 3: static audit of the notebooks / scripts that use create_fig
# ─────────────────────────────────────────────────────────────────────────────

# names defined at the top level of `code` (functions, short-form functions, consts, globals)
function toplevel_definitions(code::AbstractString)
    names = Set{Symbol}()
    ex = Meta.parseall(code)
    function visit(e)
        e isa Expr || return
        if e.head in (:toplevel, :block)
            foreach(visit, e.args)
        elseif e.head == :const
            visit(e.args[1])
        elseif e.head == :function || e.head == :(=)
            lhs = e.args[1]
            while lhs isa Expr && lhs.head in (:where, :(::))
                lhs = lhs.args[1]
            end
            if lhs isa Symbol
                push!(names, lhs)
            elseif lhs isa Expr && lhs.head == :call && lhs.args[1] isa Symbol
                push!(names, lhs.args[1])
            end
        elseif e.head == :macrocall && length(e.args) >= 3
            visit(e.args[end])  # e.g. a docstring wrapping a definition
        end
    end
    visit(ex)
    return names
end

notebook_code(path) = join((join(c["source"]) for c in JSON.parsefile(path)["cells"] if c["cell_type"] == "code"), "\n")

function test_audit()
    fig_src = read(joinpath(ED_DIR, "fig_lib.jl"), String)
    fig4_src = read(joinpath(ED_DIR, "fig4_lib.jl"), String)
    lib_names = toplevel_definitions(fig_src)
    fig4_names = toplevel_definitions(fig4_src)

    @testset "fig_lib.jl general plotting only, everything figure-4 in fig4_lib.jl" begin
        @test sort(collect(lib_names)) == sort([:cmap1, :cmap2, :get_figsize, :create_fig, :LEGEND_ARGS,
            :PANEL_AXIS_TYPES, :grid_path, :panel_axes, :corner_geometry, :place_corner_text!,
            :label_panels!, :annotate_panel!])
        println("  fig_lib.jl defines: $(sort(collect(lib_names)))")
        fig4_plotting = [:EXACT_EXP_KW, :TROTTER_OPT_KW, :plot_curve_group!, :create_u_fig, :excess_over,
            :infidelity, :build_comparison_plot, :build_overlap_comparison_plot, :build_system_size_comparison_plot]
        @test all(n -> n in fig4_names, fig4_plotting)
        @test all(n -> n in fig4_names, [:load_shared_dict, :build_exact_to_trotter_mapping,
            :compute_gs_states, :load_trotter_opt_energies_and_overlaps, :setup_system,
            :compute_trotterized_energies_u_sweep, :compute_trotterized_energies_system_size])
        @test isempty(intersect(lib_names, fig4_names))
        @test occursin(r"include\(\"fig_lib\.jl\"\)", fig4_src)  # fig4_lib.jl brings the general tools along
        uses = [m[1] for m in eachmatch(r"(?m)^using ([A-Za-z.]+)", fig_src)]
        fig_code = replace(replace(fig_src, r"#=.*?=#"s => ""), r"#.*" => "")  # drop comments (the header's examples)
        code_includes = [m.match for m in eachmatch(r"\binclude\(.*", fig_code)]
        println("  fig_lib.jl `using`: $uses, `include`s in code: $(isempty(code_includes) ? "none" : code_includes)")
        @test sort(uses) == ["CairoMakie", "LaTeXStrings"]
        @test isempty(code_includes)
    end

    files = String[]
    for (root, _, fs) in walkdir(ED_DIR)
        occursin("/.ipynb_checkpoints", root) && continue
        for f in fs
            (endswith(f, ".ipynb") || endswith(f, ".jl")) && push!(files, joinpath(root, f))
        end
    end
    fig4_call = Regex("\\b(" * join(sort([String(n) for n in fig4_names if occursin(r"^[a-z]", String(n))]; by=length, rev=true), "|") * ")\\(")
    users = String[]
    fig4_users = String[]
    for path in sort(files)
        path in (joinpath(ED_DIR, "fig_lib.jl"), joinpath(ED_DIR, "fig4_lib.jl"), @__FILE__()) && continue
        code = endswith(path, ".ipynb") ? notebook_code(path) : read(path, String)
        via_tet = occursin(r"include\((joinpath\(.*)?\"(\.\./)?trotter_exp_testing\.jl\"\)", code)
        loads_fig4 = occursin(r"include\(.*fig4_lib\.jl\"", code) || via_tet
        loads_fig = occursin(r"include\(.*fig_lib\.jl\"", code) || loads_fig4  # fig4_lib.jl includes fig_lib.jl
        calls_fig = occursin(r"\bcreate_fig\(", code)
        calls_fig4 = (loads_fig || loads_fig4) && occursin(fig4_call, code)
        (calls_fig || calls_fig4) || continue
        name = relpath(path, ED_DIR)
        calls_fig && push!(users, name)
        calls_fig4 && push!(fig4_users, name)
        defs = try
            toplevel_definitions(code)
        catch err
            @warn "could not parse $name; using a regex for its definitions" err
            Set(Symbol.(m[1] for m in eachmatch(r"(?m)^(?:function |const )?([A-Za-z_][A-Za-z0-9_!]*)\s*(?:\(|=)", code)))
        end
        clash = sort(collect(intersect(defs, union(lib_names, fig4_names))))
        println("  $(rpad(name, 55)) plots: $(calls_fig) includes fig_lib.jl: $loads_fig   ",
                "uses fig4_lib.jl functions: $(calls_fig4) includes fig4_lib.jl: $loads_fig4   ",
                "redefines library names: $(isempty(clash) ? "none" : clash)")
        calls_fig && @test loads_fig
        calls_fig4 && @test loads_fig4
        @test isempty(clash)
    end
    @test sort(users) == sort(["figure2_3.ipynb", "figure4.ipynb", "trotter_exp_testing.ipynb",
        "testing/test_infidelity_and_dimH_plots.jl", "testing/test_plot_pruning_and_sparsity_threshold.jl",
        "testing/test_three_panel_figure.jl"])
    @test issubset(["figure4.ipynb", "trotter_exp_testing.ipynb", "testing/verify_trotter_opt_run_label.jl"], fig4_users)
    # trotter_exp_testing.jl itself must load both (its notebook and test reach them through it)
    tet = read(joinpath(ED_DIR, "trotter_exp_testing.jl"), String)
    @test occursin("include(\"fig4_lib.jl\")", tet)
end

# fig4_lib.jl's plot builders, called on synthetic data in a fresh process that loads only
# fig4_lib.jl: each must return a Figure that saves to PDF. ARGS = [fig4_lib.jl, output dir].
const BUILDER_CHECK = raw"""
include(ARGS[1])
U = collect(0.25:0.25:15.0); n = length(U); orders = [1, 2, 4]
gs = -1.0 .- U ./ 10
exact = gs .+ 1e-3 .* (1 .+ U)
trot = Dict(P => exact .+ 1e-2 ./ P^2 for P in orders)
opt = gs .+ 5e-4 .* (1 .+ U)
ov(x) = clamp.(1 .- x, 0, 1)
figs = [
    "comparison" => build_comparison_plot(U, gs, exact, trot, opt, orders, 2, 2, [3, 2]),
    "overlap" => build_overlap_comparison_plot(U, ov(exact .- gs), Dict(P => ov(trot[P] .- gs) for P in orders),
        ov(opt .- gs), orders, 2, 2, [3, 2]; trotter_to_exact_overlaps=Dict(P => ov(1e-3 ./ P^2 .* ones(n)) for P in orders)),
    "system_size" => build_system_size_comparison_plot(["3x2", "3x3", "4x3"], [6, 9, 12], [-1.0, -2.0, -3.0],
        [-0.99, -1.98, -2.95], Dict(P => [-0.98, -1.95, -2.9] for P in orders), [-0.995, -1.99, -2.97], orders, 8.0, :overlap),
]
for (name, fig) in figs
    path = joinpath(ARGS[2], "fig4_builder_$(name).pdf")
    save(path, fig)
    ok = fig isa Figure && filesize(path) > 0
    println("  fig4_lib.jl builder $name: returns a Figure, saved $(basename(path)): $ok")
    ok || exit(4)
end
"""

# Each library on its own, in a fresh process: fig_lib.jl must load without any ED code or
# figure-4 code, and fig4_lib.jl must bring its computations, its plot builders and fig_lib.jl.
function test_standalone(output_dir)
    project = normpath(joinpath(ED_DIR, ".."))
    checks = [
        ("fig_lib.jl", "all(n -> isdefined(Main, n), (:create_fig, :label_panels!, :annotate_panel!)) && " *
                       "!any(n -> isdefined(Main, n), (:build_comparison_plot, :load_saved_dict, :build_save_name_prefix, :compute_gs_states))"),
        ("fig4_lib.jl", "all(n -> isdefined(Main, n), (:compute_gs_states, :setup_system, :load_saved_dict, " *
                        ":build_comparison_plot, :build_system_size_comparison_plot, :create_fig, :label_panels!))"),
    ]
    for (lib, cond) in checks
        code = "include($(repr(joinpath(ED_DIR, lib)))); ok = $cond; println(\"  $lib alone in a fresh process: \", ok ? \"loads, defines what it should\" : \"WRONG\"); exit(ok ? 0 : 3)"
        @test success(pipeline(`$(Base.julia_cmd()) --project=$project -e $code`; stdout=stdout, stderr=stderr))
    end
    @test success(pipeline(`$(Base.julia_cmd()) --project=$project -e $BUILDER_CHECK $(joinpath(ED_DIR, "fig4_lib.jl")) $output_dir`;
        stdout=stdout, stderr=stderr))
end

# ─────────────────────────────────────────────────────────────────────────────
# 4: run the notebooks' own code
# ─────────────────────────────────────────────────────────────────────────────
function cell_source(nb_path, cell_id)
    cells = [c for c in JSON.parsefile(nb_path)["cells"] if get(c, "id", nothing) == cell_id]
    length(cells) == 1 || error("cell $cell_id not found in $nb_path")
    return join(cells[1]["source"])
end

# Julia code run in a fresh `julia --project=..` process, which evaluates the given cells
# of a notebook in Main, in order, with soft scope, as the IJulia kernel would: relative includes and
# @__DIR__ resolve to experimenting/ed, relative save paths to the process's working
# directory. ARGS = [notebook path, rewrite flag ("to_output" or "none"), cell ids...].
const CELL_RUNNER = raw"""
using JSON
import REPL
nb_path, rewrite, ids = ARGS[1], ARGS[2], ARGS[3:end]
cells = Dict(get(c, "id", "") => join(c["source"]) for c in JSON.parsefile(nb_path)["cells"])
if rewrite == "to_output"
    # figure2_3.ipynb cells 6 and 11 assign these inside a top-level `for` and only reach
    # the globals (soft scope) once the globals exist -- i.e. on a re-run in a used kernel
    global interaction_data = nothing
    global hilbert_space_size = nothing
end
task_local_storage(:SOURCE_PATH, nb_path) do
    for id in ids
        println("  running $(basename(nb_path)) cell $id")
        code = cells[id]
        if rewrite == "to_output"  # cells that save under @__DIR__ (the repo) save under pwd() instead
            code = replace(code, "joinpath(@__DIR__, " => "joinpath(pwd(), ")
        end
        include_string(REPL.softscope, Main, code, nb_path)  # IJulia's soft scope for cells
    end
end
for name in (:create_fig, :label_panels!, :annotate_panel!)
    println("  $(basename(nb_path)): $name defined in Main: ", isdefined(Main, name))
    isdefined(Main, name) || exit(2)
end
exit(0)  # a kernel never calls the `(@main)` an included script (trotter_exp_testing.jl) defines
"""

function run_cells(nb_name, cell_ids; dir=ED_DIR, rewrite="none")
    cmd = `$(Base.julia_cmd()) --project=$(normpath(joinpath(ED_DIR, ".."))) -e $CELL_RUNNER $(joinpath(ED_DIR, nb_name)) $rewrite $cell_ids`
    return success(pipeline(Cmd(cmd; dir=dir); stdout=stdout, stderr=stderr))
end

function test_notebooks(output_dir)
    @testset "notebook setup cells load fig_lib.jl (fresh process each)" begin
        for (nb, setup) in [("figure2_3.ipynb", ["f87e3c53"]), ("figure4.ipynb", ["82e2c0f7"]),
                            ("trotter_exp_testing.ipynb", ["3dd31cf9"])]
            @test run_cells(nb, setup)
        end
    end

    @testset "figure2_3.ipynb labelled figures" begin
        mkpath(joinpath(output_dir, "good_images", "final"))
        mkpath(joinpath(output_dir, "data-summary"))
        @test run_cells("figure2_3.ipynb", ["f87e3c53", "bb8aae21", "90ef3f88", "e01caf5a", "3f6ea54d"];
            dir=output_dir, rewrite="to_output")
        final = joinpath(output_dir, "good_images", "final")
        pruning_pdf = only(filter(f -> startswith(f, "pruning_and_sparsity_threshold") && endswith(f, ".pdf"), readdir(final)))
        for (pdf, want) in [("loss_curve.pdf", ["(a)", "(b)"]),
                            ("three_panel_figure.pdf", ["(a)", "(b)", "(c)"]),
                            (pruning_pdf, ["(a)", "(b)", "U=8", "threshold"])]
            flat = squash(pdf_text(joinpath(final, pdf)))
            for s in want
                found = occursin(s, flat)
                println("  $pdf: \"$s\" found as text: $found")
                @test found
            end
        end
    end
end

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "test_fig_lib")
    with_logging(log_path) do
        config = parse_cli_args(ARGS)
        println("Writing test figures to: $(config.output_dir)")
        @testset "fig_lib.jl" begin
            @testset "API + selectable PDF text" test_api(config.output_dir, config.pypdf_python)
            @testset "notebooks/scripts use fig_lib.jl / fig4_lib.jl" test_audit()
            @testset "each library loads on its own" test_standalone(config.output_dir)
            config.render_notebooks && @testset "notebook figures" test_notebooks(config.output_dir)
        end
        println("Test figures are in: $(config.output_dir)")
    end
end
