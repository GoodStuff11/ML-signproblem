"""
    test_excited_level4_54.jl

Checks the 4x3 (5,4) excited-state target `N=(5, 4)_4x3_excited_level4` (4th excited level of the
ground-state sector 2, q=(0,2), tracked in U; UCC_Hubbard TrackHubbardED_XDiag.jl --level 4),
loaded exactly as the optimization loads it (target_state=:tracked, reference_target=:weak_connected):
sector, eigen-residual in the ML momentum basis, stored energies, seed energy, consecutive-U
overlaps, recorded tracking diagnostics, and the reference Slater determinant.

Usage (from experimenting/ed):  julia --project=.. testing/test_excited_level4_54.jl
"""

using Lattices, LinearAlgebra, Combinatorics, SparseArrays, Statistics, Random, JLD2, HDF5, Zygote, Test, TOML
include(joinpath(@__DIR__, "..", "data_path.jl"))
include(joinpath(@__DIR__, "..", "logging.jl"))
include(joinpath(@__DIR__, "..", "utility_functions.jl"))
using .UtilityFunctions
include(joinpath(@__DIR__, "..", "trotter.jl"))
using .Trotter
include(joinpath(@__DIR__, "..", "ed_objects.jl"))
include(joinpath(@__DIR__, "..", "ed_functions.jl"))

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "test_excited_level4_54")
    with_logging(log_path) do
        F = data_folder("N=(5, 4)_4x3_excited_level4")
        Lvec = [4, 3]; N = (5, 4)
        @testset "4x3 (5,4) level-4 tracked target" begin
            U, sv, idx, _, _, _, _, _ = load_ED_data(F; verbose=false, sign_convention=:spin_first, use_slater_reference=true,
                target_state=:tracked, reference_target=:weak_connected)
            println("loaded: $(size(sv, 1) - 1) targets, dim $(size(sv, 2)), sector q = $(idx.k .- 1)")
            @test idx.k .- 1 == (0, 2)
            Hh, _, _ = Trotter.TamFermion.HubbardMomentumBasis(1.0, 0.0, Lvec, N; indexer=idx)
            Hi, _, _ = Trotter.TamFermion.HubbardMomentumBasis(0.0, 1.0, Lvec, N; indexer=idx)
            fh = joinpath(F, only(filter(f -> startswith(f, "HubbardED"), readdir(F))))
            Es, rank, mo = h5open(fh) do d
                vec(real.(read(d, "data/energies/2"))), read(d, "data/track_rank"), read(d, "data/track_min_overlap")
            end
            worst_r = 0.0; worst_dE = 0.0
            for u in 2:length(U)
                v = sv[u, :]; H = Hh + U[u] * Hi
                E = real(dot(v, H * v)); worst_r = max(worst_r, norm(H * v - E * v)); worst_dE = max(worst_dE, abs(E - Es[u-1]))
            end
            println("eigen-residual in the ML momentum basis: max $worst_r;  max |<H> - E_stored| = $worst_dE")
            @test worst_r < 1e-8 && worst_dE < 1e-9
            cons = [abs(dot(sv[u, :], sv[u+1, :])) for u in 2:length(U)-1]
            println("consecutive-U overlaps of the targets: min $(round(minimum(cons), digits=5)) (U $(U[1+argmin(cons)])→$(U[2+argmin(cons)])), median $(round(median(cons), digits=5))")
            println("per-substep tracking overlap: min $(minimum(mo)); rank in sector spectrum over U: $(Int.(unique(rank))) " *
                    "(first U where each appears: $([U[1+findfirst(==(r), rank)] for r in unique(rank)]))")
            println("energies at U = 0.25, 4, 8, 15: $(Es[[1, 16, 32, 60]])")
            # seed = 4th excited level of sector 2 at U=0.25: the stored ground-state data gives levels 0,1;
            # an independent dense-free check: the ground state of sector 2 (existing data) lies below
            E0 = h5open(d -> real.(read(d, "data/energies/2"))[1, 1], joinpath(dirname(F), "N=(5, 4)_4x3", "HubbardED_XDiag_momSlater_4x3_nu_5_nd_4_t_1_m_2_sectors_(2).h5"))
            println("seed E(U=0.25) = $(Es[1]); sector-2 ground state E0(U=0.25) = $E0; excitation = $(Es[1] - E0)")
            @test isapprox(Es[1], -15.59288098; atol=1e-7)
            ref = argmax(abs2.(sv[1, :]))
            println("reference Slater determinant (weak_connected): index $ref, weight in the U=0.25 target $(round(abs2(sv[2, ref]), digits=4)); " *
                    "reference fidelity at U=15: $(round(abs2(sv[end, ref]), digits=4))")
            @test ref == argmax(abs2.(sv[2, :]))
            @test minimum(cons) > 0.9
        end
    end
    return 0
end
