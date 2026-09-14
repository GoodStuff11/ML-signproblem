#=
debug_dimH_plot.jl

Fast iteration environment for debugging plot_dimH_and_barren_analysis.jl's figures
without re-running its expensive dimH sweep. Run build_dimH_cache.jl once first to
produce dimH_debug_cache.jld2; this script just loads that cache (instant) and gives
you sweep_data / order_sweep_result plus a diagnostic-table helper in a live REPL.

Usage (from experimenting/ed/testing/):
  julia --project=../.. -i debug_dimH_plot.jl

Then, at the REPL prompt, e.g.:
  fig_a = build_energy_error_vs_dimH_plot(sweep_data, [1, 2, 3, 4, 6, 8]); display(fig_a)
  print_diagnostic_table(sweep_data, [1, 2, 3, 4, 6, 8])

If you edit trotter_analysis_lib.jl or dimH_barren_analysis_lib.jl itself (e.g. a
build_*_plot function) while iterating, re-include this file (`include("debug_dimH_plot.jl")`
from the same REPL, or start the REPL under Revise.jl) to pick up the change -- sweep_data
itself does not need to be recomputed.

Includes dimH_barren_analysis_lib.jl, NOT plot_dimH_and_barren_analysis.jl itself:
the latter's `(@main)(ARGS)` auto-runs its whole (expensive, default-argument) sweep at
the end of top-level execution whenever the file defining it is `include`d by any means
-- including this one -- which would defeat the entire point of this cache. See
dimH_barren_analysis_lib.jl's header docstring for details.
=#

include("../dimH_barren_analysis_lib.jl")

const CACHE_PATH = joinpath(@__DIR__, "dimH_debug_cache.jld2")
isfile(CACHE_PATH) || error(
    "No cache at $CACHE_PATH -- run `julia --project=../.. build_dimH_cache.jl` first " *
    "(from this testing/ directory) to build it.",
)

raw = JLD2.load(CACHE_PATH)
sweep_data = [(; (Symbol(k) => v for (k, v) in d)...) for d in raw["sweep_data"]]
order_sweep_result = raw["order_sweep_result"]
exact_energy, exact_overlap, opt_trotter_energy, opt_trotter_overlap, orders_b, energies_after, overlaps_after =
    order_sweep_result

"""
    print_diagnostic_table(data, trotter_orders_a)

Per-system printout of dimH, true ground-state energy, the optimized exact-exp
energy/error/overlap, the post-hoc "Trotterized exact exp" energy/error/overlap at each
P in trotter_orders_a (the quantity plots (a)/(a-alt) now show -- should only ever be
>= the exact-exp error, converging down to it as P grows), and -- for reference only,
not plotted -- the directly re-optimized Trotter ansatz's own energy/error/overlap at
each P (trotter_before_energies/_overlaps; CAN legitimately beat exact-exp, since it's
an independently-optimized ansatz on a different variational manifold). Sorted by dimH,
ascending. A "<!-- VIOLATION" flag on a Trotterized-exact-exp line would mean an actual
bug (that line is supposed to be a strict upper bound on the exact-exp error).
"""
function print_diagnostic_table(data::Vector{<:NamedTuple}, trotter_orders_a::Vector{Int})
    order = sortperm([d.dimH for d in data])
    for d in data[order]
        err_exact = abs(d.exact_exp_energy - d.gs_energy)
        println("\n$(d.name): dimH=$(d.dimH)  gs_energy=$(d.gs_energy)")
        println("  exact_exp:              E=$(d.exact_exp_energy)  err=$err_exact  overlap=$(d.exact_exp_overlap)")
        for P in trotter_orders_a
            e = get(d.trotterized_exact_exp_energies, P, NaN)
            o = get(d.trotterized_exact_exp_overlaps, P, NaN)
            err = abs(e - d.gs_energy)
            flag = (isfinite(err) && isfinite(err_exact) && err < err_exact - 1e-10) ? "  <-- VIOLATION (should be >= exact_exp err)" : ""
            println("  trotterized_exact_exp P=$P: E=$e  err=$err  overlap=$o$flag")
        end
        for P in trotter_orders_a
            e = get(d.trotter_before_energies, P, NaN)
            o = get(d.trotter_before_overlaps, P, NaN)
            err = abs(e - d.gs_energy)
            flag = (isfinite(err) && isfinite(err_exact) && err < err_exact) ? "  (beats exact_exp -- OK, different ansatz)" : ""
            println("  [ref] directly-optimized trotter P=$P: E=$e  err=$err  overlap=$o$flag")
        end
    end
end

println("Loaded sweep_data ($(length(sweep_data)) systems) and order_sweep_result from cache.")
println("Try: print_diagnostic_table(sweep_data, [1, 2, 3, 4, 6, 8])")
