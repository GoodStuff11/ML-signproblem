#=
test_notebook_conversion.jl

Test suite validating the conversion of plot_dimH_and_barren_analysis.jl into
plot_dimH_and_barren_analysis.ipynb and checking the alternative overlap vs dimH plot.
=#

using Test
using JSON
using CairoMakie
using Statistics

# Step 1: Validate Notebook File Structure and JSON
@testset "Notebook File Structure and JSON Integrity" begin
    nb_path = joinpath(@__DIR__, "..", "plot_dimH_and_barren_analysis.ipynb")
    @test isfile(nb_path)

    raw_content = read(nb_path, String)
    nb_json = JSON.parse(raw_content)

    @test haskey(nb_json, "nbformat")
    @test nb_json["nbformat"] == 4
    @test haskey(nb_json, "cells")
    @test length(nb_json["cells"]) == 22

    cell_types = [c["cell_type"] for c in nb_json["cells"]]
    @test count(==( "markdown"), cell_types) == 11
    @test count(==( "code"), cell_types) == 11

    # Ensure kernel metadata specifies Julia
    @test haskey(nb_json, "metadata")
    @test haskey(nb_json["metadata"], "kernelspec")
    @test nb_json["metadata"]["kernelspec"]["language"] == "julia"
end

# Step 2: Validate Syntax of All Code Cells
@testset "Notebook Code Cell Syntax" begin
    nb_path = joinpath(@__DIR__, "..", "plot_dimH_and_barren_analysis.ipynb")
    nb_json = JSON.parse(read(nb_path, String))

    code_cells = [c for c in nb_json["cells"] if c["cell_type"] == "code"]
    for (i, cell) in enumerate(code_cells)
        code_str = join(cell["source"])
        parsed_expr = Meta.parse("begin\n" * code_str * "\nend")
        @test parsed_expr.head == :block
    end
end

# Step 3: Evaluate Definition Cells Directly from the Notebook & Test Functionality
module NotebookTestEnv
    using Test
    using CairoMakie
    using Statistics
    using JSON

    const NOTEBOOK_DIR = normpath(joinpath(@__DIR__, ".."))
    cd(NOTEBOOK_DIR)

    nb_path = joinpath(NOTEBOOK_DIR, "plot_dimH_and_barren_analysis.ipynb")
    nb_json = JSON.parse(read(nb_path, String))

    # Evaluate the import, configuration, helper, and plotting code cells (indices 2, 4, 6, 8, 10, 12)
    def_cell_indices = [2, 4, 6, 8, 10, 12]
    for idx in def_cell_indices
        code_str = join(nb_json["cells"][idx]["source"])
        expr = Meta.parse("begin\n" * code_str * "\nend")
        Base.eval(NotebookTestEnv, expr)
    end
end

# Also verify that plot_dimH_and_barren_analysis.jl compiles cleanly inside an isolated module
module ScriptTestEnv
    using Test, CairoMakie, Statistics
    include(joinpath(@__DIR__, "..", "plot_dimH_and_barren_analysis.jl"))
end

@testset "Notebook and Script Evaluated Functions & Plot Generation" begin
    # Test fit_power_law from the notebook
    x_test = [1.0, 2.0, 4.0, 8.0]
    c_true = 3.2
    alpha_true = 1.75
    y_test = c_true .* x_test .^ (-alpha_true)
    c_fit, alpha_fit = NotebookTestEnv.fit_power_law(x_test, y_test)
    @test isapprox(c_fit, c_true; rtol=1e-4)
    @test isapprox(alpha_fit, alpha_true; rtol=1e-4)

    # Test Plotting Functions with test data containing both energies and overlaps
    test_data = [
        (
            name="sys1", dimH=39, gs_energy=-7.46, exact_exp_energy=-7.45,
            exact_exp_overlap=0.985,
            opt_trotter_energy=-7.44, opt_trotter_overlap=0.992,
            trotter_before_energies=Dict(1 => -7.30, 2 => -7.40),
            trotter_before_overlaps=Dict(1 => 0.95, 2 => 0.97),
            ucc_gnorms=[0.5, 0.4, 0.6], vqe_gnorms=[0.2, 0.15, 0.25],
        ),
        (
            name="sys2", dimH=120, gs_energy=-10.0, exact_exp_energy=-9.95,
            exact_exp_overlap=0.980,
            opt_trotter_energy=-9.92, opt_trotter_overlap=0.988,
            trotter_before_energies=Dict(1 => -9.80, 2 => -9.90),
            trotter_before_overlaps=Dict(1 => 0.94, 2 => 0.96),
            ucc_gnorms=[0.45, 0.42], vqe_gnorms=[0.18, 0.16],
        ),
    ]

    @test haskey(test_data[1], :opt_trotter_energy)
    @test haskey(test_data[1], :opt_trotter_overlap)

    test_out_dir = mktempdir()

    # (a) Energy Error Plot
    fig_a = NotebookTestEnv.build_energy_error_vs_dimH_plot(test_data, [1, 2])
    @test fig_a isa Figure
    save(joinpath(test_out_dir, "test_fig_a.png"), fig_a)
    @test isfile(joinpath(test_out_dir, "test_fig_a.png"))

    # (a-alt) Overlap vs dimH Plot (linear overlap scale)
    fig_a_overlap = NotebookTestEnv.build_overlap_vs_dimH_plot(test_data, [1, 2])
    @test fig_a_overlap isa Figure
    save(joinpath(test_out_dir, "test_fig_a_overlap.png"), fig_a_overlap)
    @test isfile(joinpath(test_out_dir, "test_fig_a_overlap.png"))

    # Also test that the script module version works identically
    fig_script_overlap = ScriptTestEnv.build_overlap_vs_dimH_plot(test_data, [1, 2])
    @test fig_script_overlap isa Figure

    # (a-alt infidelity) Infidelity vs dimH Plot (log-log scale)
    fig_a_infid = NotebookTestEnv.build_overlap_vs_dimH_plot(test_data, [1, 2]; use_infidelity=true)
    @test fig_a_infid isa Figure
    save(joinpath(test_out_dir, "test_fig_a_infid.png"), fig_a_infid)
    @test isfile(joinpath(test_out_dir, "test_fig_a_infid.png"))

    # (b) Trotter Order Sweep Plot
    orders_b = [1, 2, 3, 4]
    energies_after = [-7.3, -7.38, -7.42, -7.44]
    overlaps_after = [0.85, 0.92, 0.96, 0.98]
    fig_b = NotebookTestEnv.build_overlap_energy_vs_trotter_order_plot(
        -7.45, 0.99, -7.40, 0.95, orders_b, energies_after, overlaps_after,
    )
    @test fig_b isa Figure
    save(joinpath(test_out_dir, "test_fig_b.png"), fig_b)
    @test isfile(joinpath(test_out_dir, "test_fig_b.png"))

    # (c) Barren Plateau Plot
    fig_c = NotebookTestEnv.build_barren_plateau_vs_dimH_plot(test_data, 4)
    @test fig_c isa Figure
    save(joinpath(test_out_dir, "test_fig_c.png"), fig_c)
    @test isfile(joinpath(test_out_dir, "test_fig_c.png"))

    # Clean up
    rm(test_out_dir; recursive=true)
end

println("All notebook verification tests passed successfully!")
