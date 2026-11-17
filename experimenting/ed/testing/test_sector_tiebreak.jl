"""
    test_sector_tiebreak.jl

Checks the tie-break of `find_best_energy_sector` (ties in the per-U ground-state tally go to the
smallest sector index) and lists, for every ground-state folder in data_h5_fixed, the sector the
OLD rule (Dict hash order, reimplemented here exactly) and the NEW rule select, following the same
two paths as `load_h5_ED_data` (one file with all sectors / one file per sector). Then, for every
folder with a `momentum_sector.toml` record, a DEFAULT load_ED_data (no sector=) must land in the
recorded sector and reproduce the saved paper loss at u_idx 31; a folder without a record
(N=(4, 4)_4x4) must follow the smallest-index rule.

Usage (from experimenting/ed):  julia --project=.. testing/test_sector_tiebreak.jl
"""

using Lattices, LinearAlgebra, Combinatorics, SparseArrays, JLD2, HDF5, Test, TOML
include(joinpath(@__DIR__, "..", "data_path.jl"))
include(joinpath(@__DIR__, "..", "logging.jl"))
include(joinpath(@__DIR__, "..", "utility_functions.jl"))
using .UtilityFunctions
include(joinpath(@__DIR__, "..", "trotter.jl"))
using .Trotter
include(joinpath(@__DIR__, "..", "ed_objects.jl"))
include(joinpath(@__DIR__, "..", "ed_functions.jl"))

const ROOT = "/home/jek354/research/data/new_data/data_h5_fixed"

"""Old selection: identical tally, then `argmax` over the Dict (hash iteration order)."""
function old_pick(all_E, labels; atol=1e-6)
    length(all_E) == 1 && return labels[1]
    counts = Dict{Int,Int}()
    for u in eachindex(all_E[1])
        e = [real(all_E[k][u]) for k in 1:length(all_E)]
        p = sortperm(e)
        for i in p
            isapprox(e[i], e[p[1]]; atol=atol) || break
            counts[i] = get(counts, i, 0) + 1
        end
    end
    return labels[argmax(counts)]
end

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "test_sector_tiebreak")
    with_logging(log_path) do
        @testset "sector tie-break" begin
            # synthetic: sectors 7 and 3 tie at every U; 3 must win whatever the order
            E = [fill(-1.0, 5), fill(-2.0, 5), fill(-2.0, 5)]
            @test find_best_energy_sector(E; labels=[1, 7, 3]) == 3
            @test find_best_energy_sector(E[[3, 2, 1]]; labels=[3, 7, 1]) == 3
            # majority still beats a smaller index
            E2 = [[-2.0, -2.0, -1.0], [-1.0, -1.0, -2.0]]
            @test find_best_energy_sector(E2; labels=[9, 0]) == 9
            # non-integer labels use tiebreak_keys
            @test find_best_energy_sector(E; labels=["a", "b", "c"], tiebreak_keys=[1, 7, 3]) == "c"
            println("synthetic checks done")

            println("\nfolder | candidates tied for most tallies | OLD (hash order) | NEW (smallest index)")
            changed = String[]
            for folder in sort(readdir(ROOT; join=true))
                b = basename(folder)
                occursin("excited", b) && continue
                files = sort([f for f in readdir(folder) if occursin("HubbardED", f) && endswith(f, ".h5")])
                isempty(files) && continue
                if length(files) == 1
                    old, new, tied = h5open(joinpath(folder, files[1])) do d
                        labs = [parse(Int, k) for k in keys(d["data/energies"])]
                        allE = [real.(read(d, "data/energies/$k"))[1, :] for k in labs]
                        o = old_pick(allE, labs); n = find_best_energy_sector(allE; labels=labs)
                        mins = [minimum(abs.(allE[i] .- [minimum(e[u] for e in allE) for u in eachindex(allE[1])])) for i in eachindex(labs)]
                        (o, n, sort(labs[[i for i in eachindex(labs) if all(abs.(allE[i] .- [minimum(e[u] for e in allE) for u in eachindex(allE[1])]) .<= 1e-6)]]))
                    end
                else
                    keyof(f) = h5open(d -> minimum(parse(Int, k) for k in keys(d["data/energies"])), joinpath(folder, f))
                    labs = [keyof(f) for f in files]
                    allE = [h5open(d -> [minimum(real.(read(d, "data/energies/$k"))[1, u] for k in keys(d["data/energies"])) for u in 1:length(read(d, "data/uvec"))],
                                   joinpath(folder, f)) for f in files]
                    old = keyof(old_pick(allE, files))
                    new = keyof(find_best_energy_sector(allE; labels=files, tiebreak_keys=labs))
                    gsmin = [minimum(e[u] for e in allE) for u in eachindex(allE[1])]
                    tied = sort(labs[[i for i in eachindex(labs) if all(abs.(allE[i] .- gsmin) .<= 1e-6)]])
                end
                println("$b | $tied | $old | $new" * (old == new ? "" : "   <-- CHANGED"))
                old == new || push!(changed, b)
                @test new == (isempty(tied) ? new : minimum(tied))   # the new pick is the smallest always-degenerate sector
            end
            println("\ndefault sector changes in: $(isempty(changed) ? "none" : changed)")

            # end to end: folders with momentum_sector.toml load their recorded sector by default, and the
            # saved paper coefficients reproduce their saved loss through that DEFAULT load
            println("\nrecorded folders, default load_ED_data (no sector=):")
            for folder in sort(readdir(ROOT; join=true))
                rec = joinpath(folder, "momentum_sector.toml")
                isfile(rec) || continue
                k = TOML.parsefile(rec)["sector"]
                U, sv, idxr, _, _, _, _, _ = load_ED_data(folder; verbose=false, sign_convention=:spin_first, use_slater_reference=true)
                Lvec = parse_lattice_dimension(folder); N = prod(Lvec)
                f = joinpath(folder, "trotter_N=$(N)_ref_slater_antihermitian_u_31.jld2")
                d = JLD2.load(f)["dict"]
                basis = Trotter.get_basis_sector(idxr, Lvec, N)
                gates = Trotter.enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true, include_diagonal=false)
                tau = Trotter.fgateToTauSector(gates, N, basis; antihermitian=true)
                leg = Trotter.gate_keys(Trotter.enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true, include_diagonal=false, sort_gates=false))
                p = Trotter.gate_permutation(leg, Trotter.gate_keys(gates))
                L = Trotter.adjoint_loss(d["coefficients"], gates[p], tau[p], sv[1, :], sv[31, :], basis, N; num_exponentials=1, antihermitian=true)
                Ls = d["metrics"]["loss"][end]
                println("  $(basename(folder)): record sector $k, loaded q=$(idxr.k .- 1), u_idx=31 loss $(round(L, sigdigits=6)) vs saved $(round(Ls, sigdigits=6))")
                @test idxr.k .- 1 == Tuple(TOML.parsefile(rec)["q"])
                @test abs(L - Ls) < 1e-6
            end
            # a folder without a record still follows the smallest-index rule
            r = load_h5_ED_data(joinpath(ROOT, "N=(4, 4)_4x4"); use_slater_reference=false)
            println("  N=(4, 4)_4x4 (no record, tied sectors 0,2,8): loaded indexer k = $(r[3].k) (1-based q+1)")
            @test r[3].k == (1, 1)
        end
    end
    return 0
end
