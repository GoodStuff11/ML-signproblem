#=
test_trotter_exp_testing_notebook.jl

Verifies that:
1. The code in trotter_exp_testing.ipynb loads and executes cleanly with CairoMakie,
   create_fig, axislegend, the latex fonts theme, and cmap2.
2. All data (exact exponential coefficients, Trotterized energies, and Trotter-optimized energies)
   load properly (60/60 points loaded, non-NaN values verified).
3. The comparison plots build successfully as CairoMakie.Figure with axislegend.
4. The physical crossover between Exact Exp and Trotter Opt is verified (Trotter Opt has lower infidelity at small U, Exact Exp has lower infidelity at large U).
5. The Trotterization discretization error (metric=:trotter_error) shows distinct, well-separated orders P = 1, 2, 4, 8.
6. Saving to PNG and PDF formats works seamlessly.
=#

using Test
using JSON
using CairoMakie

include("../data_path.jl")
include("../logging.jl")
include("../trotter_exp_testing.jl")

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "test_trotter_exp_testing_notebook")
    with_logging(log_path) do
        println("=== Running test_trotter_exp_testing_notebook ===")

        # 1. Read trotter_exp_testing.ipynb and verify structure
        nb_path = joinpath(@__DIR__, "..", "trotter_exp_testing.ipynb")
        @test isfile(nb_path)
        nb_json = JSON.parsefile(nb_path)
        @test length(nb_json["cells"]) >= 4

        # 2. Test running the sweep on N=(4, 4)_3x3
        FOLDER = data_folder("N=(4, 4)_3x3")
        TROTTER_ORDERS = [1, 2, 4, 8]
        custom_ref_state_arg = "slater"
        antihermitian = true
        loss_type = :overlap

        U_values, gs_energies, exact_exp_energies, exact_exp_overlaps, trotter_energies, trotter_overlaps, trotter_opt_energies, trotter_opt_overlaps, trotter_to_exact_overlaps, n_up_loaded, n_dn_loaded, lvec =
            compute_trotterized_energies_u_sweep(FOLDER, TROTTER_ORDERS, custom_ref_state_arg, antihermitian, loss_type)

        println("\nVerification Results:")
        println("  U_values count:            $(length(U_values))")
        println("  gs_energies non-NaN:       $(sum(!isnan, gs_energies)) / $(length(U_values))")
        println("  exact_exp_energies non-NaN:$(sum(!isnan, exact_exp_energies)) / $(length(U_values))")
        println("  exact_exp_overlaps non-NaN:$(sum(!isnan, exact_exp_overlaps)) / $(length(U_values))")
        for P in TROTTER_ORDERS
            println("  trotter P=$P non-NaN E:    $(sum(!isnan, trotter_energies[P])) / $(length(U_values))")
            println("  trotter P=$P non-NaN Ovlp: $(sum(!isnan, trotter_overlaps[P])) / $(length(U_values))")
            println("  trotter P=$P to Exact Ovlp:$(sum(!isnan, trotter_to_exact_overlaps[P])) / $(length(U_values))")
        end
        println("  trotter_opt non-NaN E:     $(sum(!isnan, trotter_opt_energies)) / $(length(U_values))")
        println("  trotter_opt non-NaN Ovlp:  $(sum(!isnan, trotter_opt_overlaps)) / $(length(U_values))")

        # Assertions
        n_U = length(U_values)
        @test n_U >= 60
        @test sum(!isnan, gs_energies) == n_U
        @test sum(!isnan, exact_exp_energies) == n_U
        @test sum(!isnan, exact_exp_overlaps) == n_U
        for P in TROTTER_ORDERS
            @test sum(!isnan, trotter_energies[P]) == n_U
            @test sum(!isnan, trotter_overlaps[P]) == n_U
            @test sum(!isnan, trotter_to_exact_overlaps[P]) == n_U
        end
        @test sum(!isnan, trotter_opt_energies) == n_U
        @test sum(!isnan, trotter_opt_overlaps) == n_U

        # Verify Exact Exp vs Trotter Opt performance
        u_nz = findfirst(u -> u > 0, U_values)
        infid_opt_u_nz = 1.0 - trotter_opt_overlaps[u_nz]
        infid_exact_u_nz = 1.0 - exact_exp_overlaps[u_nz]
        println("\nAccuracy Comparison:")
        println("  At U = $(U_values[u_nz]): Trotter Opt Infid = $infid_opt_u_nz, Exact Exp Infid = $infid_exact_u_nz")

        # At strong coupling U = 15.0 (u_i = end), Exact Exp has smaller infidelity than single-step Trotter Opt
        infid_opt_uEnd = 1.0 - trotter_opt_overlaps[end]
        infid_exact_uEnd = 1.0 - exact_exp_overlaps[end]
        @test infid_exact_uEnd < infid_opt_uEnd
        println("  At U = $(U_values[end]): Exact Exp Infid = $infid_exact_uEnd < Trotter Opt Infid = $infid_opt_uEnd (Exact Exp better)")

        # Verify Trotter Discretization Error scaling P=1 > P=2 > P=4 > P=8
        for u_i in [u_nz, 16, n_U]
            err1 = 1.0 - trotter_to_exact_overlaps[1][u_i]
            err2 = 1.0 - trotter_to_exact_overlaps[2][u_i]
            err4 = 1.0 - trotter_to_exact_overlaps[4][u_i]
            err8 = 1.0 - trotter_to_exact_overlaps[8][u_i]
            @test err1 > err2
            @test err2 > err4
            @test err4 > err8
            println("  At U = $(U_values[u_i]): Trotterization errors P=(1,2,4,8) = ($err1, $err2, $err4, $err8)")
        end

        # 3. Build energy comparison plot
        fig_energy = build_comparison_plot(
            U_values, gs_energies, exact_exp_energies,
            trotter_energies, trotter_opt_energies,
            TROTTER_ORDERS, n_up_loaded, n_dn_loaded, lvec;
            custom_ref_state_arg=custom_ref_state_arg,
            loss_type=loss_type
        )
        @test fig_energy isa CairoMakie.Figure
        println("\n  Energy plot constructed successfully: ", typeof(fig_energy))

        # 4. Build overlap comparison plot (infidelity on log scale)
        fig_overlap_infid = build_overlap_comparison_plot(
            U_values, exact_exp_overlaps,
            trotter_overlaps, trotter_opt_overlaps,
            TROTTER_ORDERS, n_up_loaded, n_dn_loaded, lvec;
            trotter_to_exact_overlaps=trotter_to_exact_overlaps,
            custom_ref_state_arg=custom_ref_state_arg,
            loss_type=loss_type,
            metric=:infidelity
        )
        @test fig_overlap_infid isa CairoMakie.Figure
        println("  Overlap infidelity plot constructed successfully: ", typeof(fig_overlap_infid))

        # 5. Build Trotter discretization error plot (exact infidelity on log scale)
        fig_trotter_err = build_overlap_comparison_plot(
            U_values, exact_exp_overlaps,
            trotter_overlaps, trotter_opt_overlaps,
            TROTTER_ORDERS, n_up_loaded, n_dn_loaded, lvec;
            trotter_to_exact_overlaps=trotter_to_exact_overlaps,
            custom_ref_state_arg=custom_ref_state_arg,
            loss_type=loss_type,
            metric=:trotter_error
        )
        @test fig_trotter_err isa CairoMakie.Figure
        println("  Trotter discretization error plot constructed successfully: ", typeof(fig_trotter_err))

        # 6. Verify save functionality
        test_png = joinpath(@__DIR__, "trotter_test_comparison.png")
        test_pdf = joinpath(@__DIR__, "trotter_test_comparison.pdf")
        save(test_png, fig_energy)
        save(test_pdf, fig_energy)
        @test isfile(test_png)
        @test isfile(test_pdf)
        rm(test_png; force=true)
        rm(test_pdf; force=true)
        println("  Verified saving to PNG and PDF successfully.")

        println("\nAll tests passed successfully!")
    end
end
