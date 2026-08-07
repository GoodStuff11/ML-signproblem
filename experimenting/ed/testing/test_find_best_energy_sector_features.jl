"""
    test_find_best_energy_sector_features.jl

Test and verify:
1. `find_best_energy_sector` works without providing `U_values` (positional arg optional).
2. `find_best_energy_sector` tallies all approximately degenerate ground states at each U value
   and picks the k value that was a ground state most often.

Usage:
    julia --project=.. testing/test_find_best_energy_sector_features.jl
"""

using Lattices
using LinearAlgebra
using SparseArrays
using JLD2
using HDF5
using Test

include(joinpath(@__DIR__, "..", "logging.jl"))
include(joinpath(@__DIR__, "..", "utility_functions.jl"))
using .UtilityFunctions
include(joinpath(@__DIR__, "..", "ed_objects.jl"))
include(joinpath(@__DIR__, "..", "ed_functions.jl"))

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "test_find_best_energy_sector_features")
    with_logging(log_path) do
        println("=== Running find_best_energy_sector Features Tests ===")

        # Test 1: Optional U_values positional argument
        all_E = [[-5.0, -4.0], [-5.0, -3.0]]
        labels = [10, 20]

        # Call without U_values
        k_min1 = find_best_energy_sector(all_E; labels=labels, verbose=true)
        println("Result without U_values: $k_min1")
        @test k_min1 == 10 || k_min1 == 20

        # Call with U_values
        U_vals = [0.0, 2.0]
        k_min2 = find_best_energy_sector(all_E, U_vals; labels=labels, verbose=true)
        println("Result with U_values: $k_min2")
        @test k_min2 == 10

        # Test 2: Approximate degeneracy tallying
        # Sector 1: [-5.0, -4.0, -3.000000001]
        # Sector 2: [-5.0, -4.0, -3.0]
        # Sector 3: [-4.0, -3.0, -3.0]
        # At U=1 (idx 1): Sectors 1 and 2 are degenerate at -5.0. Tally: Sec 1 +1, Sec 2 +1.
        # At U=2 (idx 2): Sectors 1 and 2 are degenerate at -4.0. Tally: Sec 1 +1, Sec 2 +1.
        # At U=3 (idx 3): Sectors 1, 2, 3 are degenerate at -3.0 within 1e-6. Tally: Sec 1 +1, Sec 2 +1, Sec 3 +1.
        # Total counts: Sec 1: 3, Sec 2: 3, Sec 3: 1.
        # If Sector 2 was also ground state at another point, say:
        # Sector 1: [-5.0, -4.0, -2.0]
        # Sector 2: [-5.0, -4.0, -3.0]
        # Sector 3: [-4.0, -4.0, -3.0]
        # U=1: Sec 1 & Sec 2 both -5.0 (tally 1 & 2)
        # U=2: Sec 1, Sec 2, Sec 3 all -4.0 (tally 1, 2, 3)
        # U=3: Sec 2 & Sec 3 both -3.0 (tally 2 & 3)
        # Total counts: Sec 1 = 2, Sec 2 = 3, Sec 3 = 2.
        # Sec 2 should win because it was ground state (or degenerate GS) 3 times!

        E_deg = [
            [-5.0, -4.0, -2.0],  # sector 1
            [-5.0, -4.0, -3.0],  # sector 2
            [-4.0, -4.0, -3.0]   # sector 3
        ]
        sec_labels = [:sec1, :sec2, :sec3]

        best_sec = find_best_energy_sector(E_deg; labels=sec_labels, verbose=true, atol=1e-6)
        println("Selected degenerate sector: $best_sec")
        @test best_sec == :sec2

        println("=== All Tests Passed ===")
    end
end
