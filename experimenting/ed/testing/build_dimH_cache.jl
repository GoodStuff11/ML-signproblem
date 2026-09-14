#=
build_dimH_cache.jl

One-off caching harness for debugging plot_dimH_and_barren_analysis.jl. That script's
own dimH sweep (compute_dimH_sweep_data over the 10-system DIMH_SWEEP_SYSTEMS list) is
the expensive part (setup_system + get_exact_coefficients + matrix-exponential energy/
overlap evaluation per system, several minutes total) -- but everything downstream of it
(the build_*_plot functions, and manual inspection of the raw per-system numbers) is
cheap and is what actually needs debugging.

This script runs compute_dimH_sweep_data ONCE with the script's default arguments and
saves the result to a JLD2 cache file. Re-run this only when the underlying data files
(barren_study / unitary_map / trotter_opt jld2 outputs) change; otherwise just re-run
debug_dimH_plot.jl (or load the cache directly in a REPL), which loads the cache in
under a second and skips the sweep entirely.

Includes dimH_barren_analysis_lib.jl, NOT plot_dimH_and_barren_analysis.jl itself:
that script's `(@main)(ARGS)` entry point auto-runs at the end of top-level execution
whenever the file defining it is `include`d, by ANY means (a plain `include("...")`
from another script, or even `julia -e 'include("...")'`) -- so including the script
directly here would silently re-run its whole default sweep on top of this one. See
dimH_barren_analysis_lib.jl's header docstring for details.

Usage:
  julia --project=../.. build_dimH_cache.jl
=#

include("../dimH_barren_analysis_lib.jl")

const CACHE_PATH = joinpath(@__DIR__, "dimH_debug_cache.jld2")

println("=== Computing dimH-sweep data (this is the expensive part; cached afterwards) ===")
sweep_data, order_sweep_result = compute_dimH_sweep_data(
    DIMH_SWEEP_SYSTEMS;
    trotter_orders_a=[1, 2, 3, 4, 6, 8],
    P_vqe=4,
    order_sweep_system=SINGLE_SYSTEM_FOR_TROTTER_ORDER_SWEEP,
    trotter_orders_b=collect(1:10),
    profile=true,
)

# NamedTuples (with nested Dicts) round-trip through JLD2 fine, but we convert to
# plain Dicts here so the cache doesn't depend on the NamedTuple field order/types
# matching exactly between the save and load Julia sessions.
sweep_data_dicts = [Dict(String(k) => v for (k, v) in pairs(d)) for d in sweep_data]

JLD2.jldsave(CACHE_PATH; sweep_data=sweep_data_dicts, order_sweep_result=order_sweep_result)
println("\nCached dimH-sweep data (10 systems) + order-sweep result to:\n  $CACHE_PATH")
