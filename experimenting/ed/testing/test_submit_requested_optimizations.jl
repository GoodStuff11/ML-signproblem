#=
test_submit_requested_optimizations.jl

Unit test to verify the 16 combinations of systems and options.
=#

using Test

include("submit_requested_optimizations.jl")

@testset "Requested Optimizations Setup" begin
    runs = get_requested_runs()
    @test length(runs) == 16

    systems_expected = ["N=(5, 4)_3x3", "N=(4, 4)_3x3", "N=(3, 3)_3x2", "N=(3, 2)_3x2"]
    options_expected = ["trotter_slater", "trotter_default", "exact_slater", "exact_default"]

    for sys in systems_expected
        sys_runs = filter(r -> r.system == sys, runs)
        @test length(sys_runs) == 4
        for opt in options_expected
            matching = filter(r -> r.label == opt, sys_runs)
            @test length(matching) == 1
            r = matching[1]
            if startswith(opt, "trotter")
                @test r.script == "run_trotter_scan_optimization.jl"
            else
                @test r.script == "run_lanczos_scan_optimization.jl"
            end
            if endswith(opt, "slater")
                @test any(startswith(x, "--custom_ref_state=slater") for x in r.cli_extra)
            else
                @test !any(startswith(x, "--custom_ref_state") for x in r.cli_extra)
            end
        end
    end
end
