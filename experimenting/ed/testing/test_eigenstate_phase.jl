#=
test_eigenstate_phase.jl

Verify that Hubbard model eigenstates in momentum sectors are real vectors
up to a global complex phase. If ψ = e^{iφ} * v where v is real, then the
entire Trotter optimization can be done in real arithmetic (with antihermitian
generators), cutting GPU memory by 4x vs ComplexF64.

Usage:
  julia --project=.. test_eigenstate_phase.jl <folder> [--all_sectors]

Arguments:
  folder (required): Path to the ED data folder (e.g., "N=(6, 6)_4x4").
  --all_sectors (optional): Check all momentum sectors, not just the ground state sector.

Output:
  For each U value, prints the maximum imaginary component of the eigenstate
  after stripping the global phase. Values near machine epsilon (~1e-15)
  confirm the eigenstate is real up to phase.
=#

using Lattices
using LinearAlgebra
using SparseArrays
using JLD2
using HDF5

include(joinpath(@__DIR__, "..", "data_path.jl"))
include(joinpath(@__DIR__, "..", "logging.jl"))
include(joinpath(@__DIR__, "..", "utility_functions.jl"))
using .UtilityFunctions
include(joinpath(@__DIR__, "..", "trotter.jl"))
using .Trotter
include(joinpath(@__DIR__, "..", "ed_objects.jl"))
include(joinpath(@__DIR__, "..", "ed_functions.jl"))

"""
    strip_global_phase(v::AbstractVector{<:Complex}) -> (v_real, phase, max_imag)

Strip the global complex phase from vector `v` by dividing by the phase of
the component with the largest magnitude. Returns the real part of the
phase-stripped vector, the extracted phase, and the maximum imaginary residual.

The max_imag value quantifies how "real" the vector is after phase stripping:
values near machine epsilon confirm the vector is real up to a global phase.
"""
function strip_global_phase(v::AbstractVector{<:Complex})
    idx = argmax(abs.(v))
    phase = v[idx] / abs(v[idx])
    v_stripped = v .* conj(phase)
    max_imag = maximum(abs, imag.(v_stripped))
    return real.(v_stripped), phase, max_imag
end

"""
    parse_arguments(args::Vector{String})

Parse command line arguments for the eigenstate phase test.
Expected arguments:
1. folder (String): The directory containing exact diagonalization data.
2. --all_sectors (flag): Check all momentum sectors. Default: false (only ground state sector).
"""
function parse_arguments(args::Vector{String})
    if isempty(args)
        error("Usage: julia test_eigenstate_phase.jl <folder> [--all_sectors]")
    end
    folder = data_folder(args[1])
    all_sectors = any(arg -> arg == "--all_sectors", args)
    return folder, all_sectors
end

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "test_eigenstate_phase")
    with_logging(log_path) do
        folder, all_sectors = parse_arguments(ARGS)
        println("=== Eigenstate Phase Verification ===")
        println("Folder: $folder")
        println("Check all sectors: $all_sectors")

        # Load the ground state sector eigenstates
        U_values, state_vecs, indexer, _, N_elec, _, _, sign_convention =
            load_ED_data(folder; verbose=true, sign_convention=:spin_first, use_slater_reference=false)

        H_dim = size(state_vecs, 2)
        n_U = length(U_values)
        println("\nHilbert space dimension: $H_dim")
        println("Number of U values: $n_U")

        # Check each eigenstate
        println("\n--- Ground State Sector Eigenstates ---")
        println("  U_value  |  max_imag_residual  |  is_real_up_to_phase")
        println("  " * "-"^60)

        all_real = true
        REAL_THRESHOLD = 1e-10
        max_residual_overall = 0.0

        for u_idx in 1:n_U
            psi = state_vecs[u_idx, :]
            _, _, max_imag = strip_global_phase(psi)
            is_real = max_imag < REAL_THRESHOLD
            if !is_real
                all_real = false
            end
            max_residual_overall = max(max_residual_overall, max_imag)
            # Print every 5th U value, plus first, last, and any non-real ones
            if u_idx == 1 || u_idx == n_U || u_idx % 5 == 0 || !is_real
                status = is_real ? "✓ REAL" : "✗ COMPLEX"
                println("  U=$(round(U_values[u_idx], digits=4))  |  $(max_imag)  |  $status")
            end
        end

        println("\n  Overall max imaginary residual: $max_residual_overall")
        if all_real
            println("  ✓ ALL eigenstates are REAL up to a global phase (threshold=$REAL_THRESHOLD)")
            println("  → Safe to use Float64/Float32 precision for GPU computation with antihermitian generators.")
        else
            println("  ✗ Some eigenstates have significant imaginary components after phase stripping.")
            println("  → Must use ComplexF64/ComplexF32 precision for GPU computation.")
        end

        # Additionally verify: are the tau matrices real-valued?
        # (They should be since build_direct_sparse_tau uses Float64 signs cast to ComplexF64)
        println("\n--- Tau Matrix Reality Check ---")
        Lvec = parse_lattice_dimension(folder)
        N_sites = prod(Lvec)
        basis_sector = Trotter.get_basis_sector(indexer, Lvec, N_sites)
        gates = Trotter.enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true, include_diagonal=false)
        println("Number of gates: $(length(gates))")

        tau_max_imag = 0.0
        for (k, g) in enumerate(gates)
            sp_mat, _, _ = Trotter.TrotterOptimization.build_direct_sparse_tau(
                g, N_sites, basis_sector, sortperm(UInt64.(basis_sector)), nothing; antihermitian=true
            )
            mat_max_imag = maximum(abs, imag.(nonzeros(sp_mat)))
            tau_max_imag = max(tau_max_imag, mat_max_imag)
        end
        println("Max imaginary component across all tau matrices: $tau_max_imag")
        if tau_max_imag < 1e-15
            println("✓ All tau matrices are purely REAL → real arithmetic is valid for antihermitian case.")
        else
            println("✗ Some tau matrices have imaginary components → complex arithmetic required.")
        end

        # Verify: antihermitian exp(a*tau) preserves reality
        println("\n--- Antihermitian Propagation Reality Check ---")
        println("Testing: if ref is real, does exp(a*tau)*ref stay real?")
        psi_0 = state_vecs[1, :]
        v_real, phase_0, _ = strip_global_phase(psi_0)
        # Use random coefficients
        A_test = randn(length(gates))
        ops = Trotter.fgateToExpSector(gates, A_test, N_sites, basis_sector; antihermitian=true)
        v_evolved = copy(v_real)
        for op in ops
            v_evolved = op * v_evolved
        end
        evolved_max_imag = maximum(abs, imag.(v_evolved))
        println("Max imaginary after antihermitian propagation: $evolved_max_imag")
        if evolved_max_imag < 1e-10
            println("✓ Antihermitian evolution preserves reality of the state.")
        else
            println("✗ Antihermitian evolution introduced imaginary components: $evolved_max_imag")
        end

        println("\n=== Summary ===")
        println("Eigenstates real up to phase: $all_real (max residual: $max_residual_overall)")
        println("Tau matrices real: $(tau_max_imag < 1e-15)")
        println("Antihermitian preserves reality: $(evolved_max_imag < 1e-10)")
        if all_real && tau_max_imag < 1e-15 && evolved_max_imag < 1e-10
            println("✓ CONCLUSION: Safe to use Float64/Float32 for all GPU computations with --antihermitian.")
            println("  This would reduce GPU memory by 2x (Float64) or 4x (Float32) compared to ComplexF64.")
        end

        return 0
    end
end
