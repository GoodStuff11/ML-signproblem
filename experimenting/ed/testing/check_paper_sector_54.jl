"""
    check_paper_sector_54.jl

Which momentum sector were the paper's 4x3 (5,4) one-layer results
(`trotter_N=12_ref_slater_antihermitian_u_<idx>.jld2`, written 2026-08) trained in?
Sectors 1, q=(0,1), and 2, q=(0,2), hold exactly degenerate ground states.

For u_idx in (2, 30, 61) the saved coefficients are evaluated in each sector (target = that
sector's ground state, reference = that sector's Slater reference) in both gate orders: the
current sorted one and the pre-2026-08-31 unsorted one (files before that date were saved in
the unsorted order; see align_warm_start). The training sector/order is the one that reproduces
the saved final loss; the other combinations should be far off.

Usage (from experimenting/ed):  julia --project=.. testing/check_paper_sector_54.jl
"""

using Lattices, LinearAlgebra, Combinatorics, SparseArrays, Statistics, Random, JLD2, HDF5, Zygote, Test
include(joinpath(@__DIR__, "..", "data_path.jl"))
include(joinpath(@__DIR__, "..", "logging.jl"))
include(joinpath(@__DIR__, "..", "utility_functions.jl"))
using .UtilityFunctions
include(joinpath(@__DIR__, "..", "trotter.jl"))
using .Trotter
include(joinpath(@__DIR__, "..", "ed_objects.jl"))
include(joinpath(@__DIR__, "..", "ed_functions.jl"))

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "check_paper_sector_54")
    with_logging(log_path) do
        folder = data_folder("N=(5, 4)_4x3")
        Lvec = [4, 3]; N = 12
        saved = Dict(u => JLD2.load(joinpath(folder, "trotter_N=12_ref_slater_antihermitian_u_$(u).jld2"))["dict"] for u in (2, 30, 61))
        best = Dict{Int,Any}()
        @testset "paper 4x3 (5,4) sector" begin
            for k in (1, 2)
                U, sv, idxr, _, _, _, _, _ = load_ED_data(folder; verbose=false, sign_convention=:spin_first,
                    use_slater_reference=true, sector=k)
                basis = Trotter.get_basis_sector(idxr, Lvec, N)
                gates = Trotter.enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true, include_diagonal=false)
                tau = Trotter.fgateToTauSector(gates, N, basis; antihermitian=true)
                cur = Trotter.gate_keys(gates)
                leg = Trotter.gate_keys(Trotter.enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true,
                    include_diagonal=false, sort_gates=false))
                p_leg = Trotter.gate_permutation(leg, cur)
                println("sector $k: indexer k=$(idxr.k) (1-based q+1), dim=$(size(sv, 2)), reference index $(argmax(abs2.(sv[1, :])))")
                for u in sort(collect(keys(saved))); d = saved[u]
                    A = d["coefficients"]
                    saved_loss = d["metrics"]["loss"][end]
                    for (name, g, t) in (("sorted (current)", gates, tau), ("unsorted (pre-08-31)", gates[p_leg], tau[p_leg]))
                        L = Trotter.adjoint_loss(A, g, t, sv[1, :], sv[u, :], basis, N; num_exponentials=1, antihermitian=true)
                        L0 = 1 - abs2(dot(sv[u, :], sv[1, :]))
                        println("  u_idx=$u (U=$(U[u])): order $name -> infidelity $(round(L, sigdigits=5))   " *
                                "[saved final loss $(round(saved_loss, sigdigits=5)), reference-only infidelity $(round(L0, sigdigits=4))]")
                        if abs(L - saved_loss) < 1e-6
                            best[u] = (k, name)
                        end
                    end
                end
            end
            println("\nreproduces the saved loss: ", best)
            @test length(best) == 3
            @test length(unique(first.(values(best)))) == 1
        end
    end
    return 0
end
