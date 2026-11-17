"""
    check_paper_sectors.jl

For every system whose ground state is exactly degenerate between momentum sectors AND that has
saved one-layer Slater-reference results (`trotter_N=<N>_ref_slater_antihermitian_u_<idx>.jld2`),
find the sector those results were trained in: the saved coefficients are evaluated at three U
indices in every tied sector (target = that sector's ground state, reference = its Slater
reference), in both gate orders (current sorted / pre-2026-08-31 unsorted). The training sector is
the one where the saved final loss is reproduced (|ΔL| < 1e-6) at all three U; it must be unique.

Usage (from experimenting/ed):  julia --project=.. testing/check_paper_sectors.jl
Writes nothing; prints one line per (system, sector, U, order) and a summary.
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

# system => sectors tied for the ground state at every U (testing/test_sector_tiebreak.jl)
const SYSTEMS = ["N=(3, 2)_3x2" => [2, 4], "N=(3, 2)_3x3" => [1, 2, 3, 6], "N=(4, 3)_3x3" => [1, 2, 3, 6],
                 "N=(5, 4)_3x3" => [1, 2, 3, 6], "N=(5, 4)_4x3" => [1, 2]]

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "check_paper_sectors")
    with_logging(log_path) do
        found = Dict{String,Any}()
        @testset "paper sectors" begin
            for (name, sectors) in SYSTEMS
                folder = data_folder(name)
                Lvec = parse_lattice_dimension(folder); N = prod(Lvec)
                prefix = "trotter_N=$(N)_ref_slater_antihermitian_u_"
                idxs = sort([parse(Int, m[1]) for f in readdir(folder) for m in (match(r"^trotter_N=\d+_ref_slater_antihermitian_u_(\d+)\.jld2$", f),) if m !== nothing])
                us = unique([idxs[1], idxs[cld(end, 2)], idxs[end]])
                saved = Dict(u => JLD2.load(joinpath(folder, prefix * "$(u).jld2"))["dict"] for u in us)
                hits = Set{Int}()
                println("\n== $name: tied sectors $sectors, checking u_idx $us")
                for k in sectors
                    U, sv, idxr, _, _, _, _, _ = load_ED_data(folder; verbose=false, sign_convention=:spin_first,
                        use_slater_reference=true, sector=k)
                    basis = Trotter.get_basis_sector(idxr, Lvec, N)
                    gates = Trotter.enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true, include_diagonal=false)
                    tau = Trotter.fgateToTauSector(gates, N, basis; antihermitian=true)
                    leg = Trotter.gate_keys(Trotter.enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true,
                        include_diagonal=false, sort_gates=false))
                    p_leg = Trotter.gate_permutation(leg, Trotter.gate_keys(gates))
                    ok_all = Dict("sorted" => true, "unsorted" => true)
                    for u in us
                        d = saved[u]; A = d["coefficients"]; Ls = d["metrics"]["loss"][end]
                        for (oname, g, t) in (("sorted", gates, tau), ("unsorted", gates[p_leg], tau[p_leg]))
                            L = Trotter.adjoint_loss(A, g, t, sv[1, :], sv[u, :], basis, N; num_exponentials=1, antihermitian=true)
                            ok = abs(L - Ls) < 1e-6
                            ok_all[oname] &= ok
                            println("  sector $k (q=$(idxr.k .- 1)) u_idx=$u U=$(U[u]) $oname: loss $(round(L, sigdigits=5)) vs saved $(round(Ls, sigdigits=5))" * (ok ? "  MATCH" : ""))
                        end
                    end
                    any(values(ok_all)) && push!(hits, k)
                    any(values(ok_all)) && (found[name] = (sector=k, q=idxr.k .- 1, order=[o for (o, v) in ok_all if v]))
                end
                println("  -> reproduced in sectors $(collect(hits))")
                @test length(hits) == 1
            end
        end
        println("\nSUMMARY (training sector of the saved results):")
        for (name, _) in SYSTEMS
            println("  $name: ", get(found, name, "NOT FOUND"))
        end
    end
    return 0
end
