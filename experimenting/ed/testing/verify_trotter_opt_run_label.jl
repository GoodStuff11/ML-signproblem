#=
verify_trotter_opt_run_label.jl

Checks that fig4_lib.jl's `load_trotter_opt_energies_and_overlaps` -- the function that
supplies the red "Optimized trotter" curve in figure4.ipynb's panels (a)/(a-alt)/(b) --
actually resolves to the un-truncated `--maxiters=1000` re-runs
(`--run_label=barren_study_mi1000`) rather than silently returning NaN.

Background: the function previously had `suffix="noreg"` hardcoded. No
`trotter_*_noreg_u_*.jld2` files exist, so it returned all-NaN for every system and
`compute_one_system_data` fell back to the original `barren_study` runs -- which were made
at the driver's default `--maxiters=100` and stopped at a 303-iteration cap. The plot
therefore showed truncated Trotter data no matter how many times the notebook was re-run.

This calls the loader with exactly the arguments the notebook passes (a single-element
U vector at U=8.0, ref_slater, antihermitian, overlap loss) and compares the returned
overlap against the infidelity stored in the `_mi1000` JLD2 file read directly.

Usage:
  julia --project=.. testing/verify_trotter_opt_run_label.jl
=#

include(joinpath(@__DIR__, "..", "fig4_lib.jl"))
using JLD2, Printf

const SYSTEMS = [
    ("N=(2, 2)_3x2", 6), ("N=(3, 2)_3x2", 6), ("N=(3, 2)_3x3", 9),
    ("N=(3, 3)_3x2", 6), ("N=(3, 3)_3x3", 9), ("N=(3, 3)_4x2", 8),
    ("N=(4, 3)_3x3", 9), ("N=(4, 4)_3x3", 9), ("N=(5, 4)_3x3", 9),
    ("N=(5, 4)_4x3", 12),
]

function (@main)(ARGS)
    println("TROTTER_OPT_RUN_LABEL = ", repr(TROTTER_OPT_RUN_LABEL))
    root = get_data_root()
    failures = String[]
    @printf("%-18s %-13s %-13s %s\n", "system", "loader_infid", "file_infid", "status")
    for (name, nsites) in SYSTEMS
        folder = joinpath(root, name)

        # Exactly how the notebook's compute_one_system_data calls it.
        _, o_opt = load_trotter_opt_energies_and_overlaps(
            folder, nsites, [8.0];
            custom_ref_state_arg="slater", antihermitian=true, loss_type=:overlap,
        )
        loader_infid = isempty(o_opt) ? NaN : 1.0 - o_opt[1]

        # Ground truth: read the _mi1000 file directly.
        f = joinpath(folder,
            "trotter_N=$(nsites)_ref_slater_antihermitian_barren_study_mi1000_u_33.jld2")
        file_infid = isfile(f) ? load(f)["dict"]["metrics"]["loss"][end] : NaN

        ok = isfinite(loader_infid) && isfinite(file_infid) &&
             isapprox(loader_infid, file_infid; rtol=1e-8, atol=1e-16)
        ok || push!(failures, name)
        @printf("%-18s %-13.4e %-13.4e %s\n", name, loader_infid, file_infid,
                ok ? "OK" : "MISMATCH/NaN")
    end

    println()
    if isempty(failures)
        println("PASS: all $(length(SYSTEMS)) systems resolve to the maxiters=1000 re-runs.")
    else
        println("FAIL: $(length(failures)) system(s) did not resolve: ", join(failures, ", "))
        error("verification failed")
    end
end
