#=
run_repeat_optimization_experiment.jl

Repeat the *same* single-U optimization many times from independent random
initializations, for both ansatz families (Trotter product ansatz and the exact
matrix-exponential ansatz) and both loss functions (overlap and energy), and
record the final loss of every individual run to a CSV.

This is a statistics-over-random-restarts experiment, not a U scan: the ED data,
momentum-sector basis, Trotter gate set and Hamiltonians are all built once, and
only the optimization is repeated.

Usage:
  julia --project=.. run_repeat_optimization_experiment.jl <folder> <u_idx> [options]

Arguments:
  folder (required): ED data sub-folder, e.g. "N=(3, 2)_3x2".
  u_idx  (required): 1-based index into the data's U values (e.g. 32 => U = 8.0).

Options:
  --runs=<n>                 Number of independent repeats per (ansatz, loss). Default: 100.
  --run_start=<n>            First run index to execute. Default: 1.
  --run_end=<n>              Last run index to execute. Default: --runs. Together with
                             --run_start this lets disjoint run ranges be farmed out to
                             separate (e.g. SLURM array) tasks; run index r always uses
                             seed = --seed + r, so the ranges stay reproducible and the
                             per-task CSVs concatenate into one consistent table.
  --maxiters=<n>             Maximum iterations per optimizer stage. Default: 1000.
  --initialization_samples=<n>  Multi-start random samples drawn per run. Default: 1.
  --custom_ref_state=<value> Reference state ("slater" or a basis index). Default: "slater".
  --antihermitian[=<bool>]   Real-antihermitian generators. Default: true.
  --num_exponentials=<n>     Trotter layers. Default: 1.
  --regularization=<x>       L2 penalty coefficient on the exact-ansatz loss
                             (`+ 0.5 * x * ||t||^2`, see `REGULARIZATION_STRENGTH` in
                             ed_optimization.jl). Default: 1e-3, the value that file has
                             always used. Pass 0 to optimize the bare infidelity/energy,
                             which is the like-for-like comparison against the Trotter
                             ansatz -- the Trotter losses carry no such penalty. Ignored
                             by the Trotter ansatz.
  --ansatz=<list>            Comma-separated subset of "trotter,exact". Default: both.
  --loss=<list>              Comma-separated subset of "overlap,energy". Default: both.
  --seed=<n>                 Base RNG seed. Run r of a given (ansatz, loss) uses seed + r. Default: 20260915.
  --out=<path>               Output CSV path. Default: repeat_optimization_<folder>_u<idx>.csv in cwd.

The CSV is written incrementally (one row appended and flushed per completed run),
so a partially-finished job still leaves usable data.
=#

using LinearAlgebra
using Combinatorics
using SparseArrays
using Statistics
using Random
using Printf
using Dates
using JLD2
using HDF5
using Zygote
using Lattices
using Optimization
using OptimizationOptimJL
using OptimizationOptimisers
using KrylovKit

include("data_path.jl")
include("logging.jl")
include("utility_functions.jl")
using .UtilityFunctions

# Trotter stays namespaced: `Trotter.optimize_unitary` and the exact-ansatz
# `optimize_unitary` from ed_optimization.jl share a name, so we never `using` it.
include("trotter.jl")

include("ed_objects.jl")
include("ed_functions.jl")
include("ed_optimization.jl")
include("nn_strategy.jl")

"""
    parse_arguments(args) -> NamedTuple

Parse the command line for this experiment. See the file header for the option list.
"""
function parse_arguments(args::Vector{String})
    runs = 100
    run_start = nothing
    run_end = nothing
    maxiters = 1000
    initialization_samples = 1
    custom_ref_state_arg = "slater"
    antihermitian = true
    num_exponentials = 1
    regularization = 1.0e-3
    ansatze = [:trotter, :exact]
    losses = [:overlap, :energy]
    seed = 20260915
    out_path = nothing
    positional = String[]

    for arg in args
        if startswith(arg, "--runs=")
            runs = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--run_start=")
            run_start = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--run_end=")
            run_end = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--maxiters=")
            maxiters = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--initialization_samples=")
            initialization_samples = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--custom_ref_state=")
            custom_ref_state_arg = String(split(arg, "=", limit=2)[2])
        elseif arg == "--antihermitian"
            antihermitian = true
        elseif startswith(arg, "--antihermitian=")
            antihermitian = parse(Bool, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--num_exponentials=")
            num_exponentials = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--regularization=")
            regularization = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--ansatz=")
            ansatze = Symbol[]
            for v in split(split(arg, "=", limit=2)[2], ",")
                v = strip(v)
                v in ("trotter", "exact") || error("Invalid --ansatz entry: '$v'. Valid: 'trotter', 'exact'.")
                push!(ansatze, Symbol(v))
            end
        elseif startswith(arg, "--loss=")
            losses = Symbol[]
            for v in split(split(arg, "=", limit=2)[2], ",")
                v = strip(v)
                v in ("overlap", "energy") || error("Invalid --loss entry: '$v'. Valid: 'overlap', 'energy'.")
                push!(losses, Symbol(v))
            end
        elseif startswith(arg, "--seed=")
            seed = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--out=")
            out_path = String(split(arg, "=", limit=2)[2])
        else
            push!(positional, arg)
        end
    end

    length(positional) >= 2 || error("Usage: julia run_repeat_optimization_experiment.jl <folder> <u_idx> [options]")
    folder_arg = positional[1]
    folder = data_folder(folder_arg)
    u_idx = parse(Int, positional[2])

    run_start = isnothing(run_start) ? 1 : run_start
    run_end = isnothing(run_end) ? runs : run_end
    run_start >= 1 || error("--run_start must be >= 1, got $run_start")
    run_end >= run_start || error("--run_end ($run_end) must be >= --run_start ($run_start)")

    if isnothing(out_path)
        safe = replace(folder_arg, r"[^A-Za-z0-9]" => "_")
        out_path = "repeat_optimization_$(safe)_u$(u_idx).csv"
    end

    return (; folder, folder_arg, u_idx, runs, run_start, run_end, maxiters, initialization_samples,
        custom_ref_state_arg, antihermitian, num_exponentials, regularization, ansatze, losses, seed, out_path)
end

const CSV_HEADER = "ansatz,loss_type,run,seed,final_loss,infidelity,energy,n_params,initial_loss,elapsed_s,regularization"

"""
    csv_field(x) -> String

Format one CSV cell, writing non-finite floats as empty rather than `NaN`/`Inf` text.
"""
csv_field(x::Float64) = isfinite(x) ? @sprintf("%.17g", x) : ""
csv_field(x) = string(x)

function append_row!(io, row)
    println(io, join(csv_field.(row), ","))
    flush(io)
end

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "run_repeat_optimization_experiment")
    with_logging(log_path) do
        cfg = parse_arguments(ARGS)

        println("=== Repeat-optimization experiment ===")
        println("Folder:                 $(cfg.folder)")
        println("U index:                $(cfg.u_idx)")
        println("Runs per config:        $(cfg.runs) (executing $(cfg.run_start):$(cfg.run_end))")
        println("maxiters:               $(cfg.maxiters)")
        println("initialization_samples: $(cfg.initialization_samples)")
        println("Reference state:        $(cfg.custom_ref_state_arg)")
        println("antihermitian:          $(cfg.antihermitian)")
        println("num_exponentials:       $(cfg.num_exponentials)")
        println("regularization (exact): $(cfg.regularization)")
        println("Ansatze:                $(cfg.ansatze)")
        println("Losses:                 $(cfg.losses)")
        println("Output CSV:             $(cfg.out_path)")
        println("Threads:                $(Threads.nthreads())")

        REGULARIZATION_STRENGTH[] = cfg.regularization

        use_slater_ref = cfg.custom_ref_state_arg == "slater" ? true :
                         (tryparse(Int, cfg.custom_ref_state_arg) !== nothing ? parse(Int, cfg.custom_ref_state_arg) : true)

        # ---- Shared setup (done once) ----------------------------------------
        U_values, state_vecs, indexer, precomputed_structures, N_elec, spin_conserved, use_symmetry, sign_convention =
            load_ED_data(cfg.folder; verbose=true, sign_convention=:spin_first, use_slater_reference=use_slater_ref)

        n_up, n_dn = N_elec
        Lvec = parse_lattice_dimension(cfg.folder)
        N_sites = prod(Lvec)
        target_u = U_values[cfg.u_idx]
        println("U value at index $(cfg.u_idx): $target_u")

        # `load_ED_data` with a Slater reference prepends the reference state as row 1,
        # so target row indices shift by one (same convention as the scan drivers).
        has_prepended_ref = (state_vecs isa AbstractMatrix) && (size(state_vecs, 1) == length(U_values) + 1)
        ref_row = 1
        target_row = has_prepended_ref ? cfg.u_idx + 1 : cfg.u_idx
        state1 = state_vecs isa AbstractMatrix ? state_vecs[ref_row, :] : state_vecs[ref_row]
        state2 = state_vecs isa AbstractMatrix ? state_vecs[target_row, :] : state_vecs[target_row]
        println("Hilbert space dimension: $(length(state1))")

        # Trotter-side setup: momentum-sector basis, Hamiltonians, gates, tau terms.
        basis_sector = Trotter.get_basis_sector(indexer, Lvec, N_sites)
        H_hop_sector, _, _ = Trotter.TamFermion.HubbardMomentumBasis(1.0, 0.0, Lvec, (n_up, n_dn); indexer=indexer)
        H_int_sector, _, _ = Trotter.TamFermion.HubbardMomentumBasis(0.0, 1.0, Lvec, (n_up, n_dn); indexer=indexer)
        H_trotter = H_hop_sector + target_u * H_int_sector

        gates = Trotter.enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true,
            include_diagonal=!cfg.antihermitian)
        tau_terms = Trotter.fgateToTauSector(gates, N_sites, basis_sector; antihermitian=cfg.antihermitian)
        println("Trotter gates: $(length(gates))")

        ed_energy = real(dot(state2, H_trotter * state2))
        println("Exact ED ground energy at U=$target_u: $ed_energy")

        # Exact-ansatz setup: the same Hamiltonian built in the full (non-sector)
        # representation that `optimize_unitary(state1, state2, indexer)` works in.
        H_exact = if :exact in cfg.ansatze
            subspace = reconstruct_subspace(indexer, spin_conserved)
            H_hop_e, H_int_e = create_hubbard_matrices(subspace; indexer=indexer, get_indexer=false,
                sign_convention=sign_convention, lattice_ordering=ColSnake())
            H_hop_e + target_u * H_int_e
        else
            nothing
        end

        # ---- Run loop --------------------------------------------------------
        fresh = !isfile(cfg.out_path)
        io = open(cfg.out_path, "a")
        fresh && (println(io, CSV_HEADER); flush(io))

        try
            for ansatz in cfg.ansatze, loss_type in cfg.losses
                println("\n######## ansatz=$ansatz loss=$loss_type ########")
                for run in cfg.run_start:cfg.run_end
                    seed = cfg.seed + run
                    Random.seed!(seed)
                    println("\n---- $ansatz / $loss_type / run $run of $(cfg.runs) (seed $seed) ----")
                    t0 = time()

                    final_loss = NaN
                    infidelity = NaN
                    energy = NaN
                    n_params = 0
                    initial_loss = NaN

                    if ansatz == :trotter
                        opt_target = loss_type == :energy ? H_trotter : state2
                        A_opt, floss, metrics = Trotter.optimize_unitary(
                            gates, tau_terms, state1, opt_target, basis_sector, N_sites;
                            loss_type=loss_type,
                            H=H_trotter,
                            state2=state2,
                            num_exponentials=cfg.num_exponentials,
                            maxiters=cfg.maxiters,
                            optimizer=[:LBFGS, :GradientDescent, :LBFGS],
                            initialization_samples=cfg.initialization_samples,
                            antihermitian=cfg.antihermitian,
                            use_gpu=false,
                            datatype=ComplexF64
                        )
                        final_loss = floss
                        n_params = length(A_opt)
                        initial_loss = isempty(metrics["loss"]) ? NaN : Float64(metrics["loss"][1])

                        psi = Array(Trotter.apply_unitary(A_opt, gates, state1, basis_sector, N_sites,
                            cfg.num_exponentials; antihermitian=cfg.antihermitian, use_gpu=false, datatype=ComplexF64))
                        infidelity = 1.0 - abs2(dot(state2, psi))
                        energy = real(dot(psi, H_trotter * psi))
                    else
                        _, _, coeffs, _, _, metrics, _ = optimize_unitary(
                            state1, state2, indexer;
                            spin_conserved=spin_conserved,
                            use_symmetry=use_symmetry,
                            maxiters=cfg.maxiters,
                            optimization_scheme=[2],
                            gradient=:adjoint_gradient,
                            antihermitian=cfg.antihermitian,
                            optimizer=[:GradientDescent, :LBFGS, :GradientDescent, :LBFGS],
                            perturb_optimization=0.0,
                            initialization_samples=cfg.initialization_samples,
                            multi_start_iters=50,
                            multi_start_samples=5,
                            precomputed_structures=precomputed_structures,
                            sign_convention=sign_convention,
                            lattice_ordering=ColSnake(),
                            max_time_ratio=50.0,
                            loss_type=loss_type,
                            H=H_exact,
                            use_gpu=false,
                            num_exponentials=cfg.num_exponentials
                        )
                        # metrics["loss"] holds the true (unregularized) loss of the
                        # reconstructed state, one entry per optimization order.
                        final_loss = Float64(metrics["loss"][end])
                        initial_loss = length(metrics["optimization_losses"]) >= 1 &&
                                       !isempty(metrics["optimization_losses"][1]) ?
                                       Float64(metrics["optimization_losses"][1][1]) : NaN
                        n_params = sum(isnothing(c) ? 0 : length(c) for c in coeffs)
                        if loss_type == :overlap
                            infidelity = final_loss
                            energy = isempty(get(metrics, "energy", [])) ? NaN : Float64(metrics["energy"][end])
                        else
                            energy = final_loss
                            infidelity = isempty(get(metrics, "overlap", [])) ? NaN : Float64(metrics["overlap"][end])
                        end
                    end

                    elapsed = time() - t0
                    println("  final_loss=$final_loss  infidelity=$infidelity  energy=$energy  ($(round(elapsed, digits=1))s)")
                    # The Trotter losses carry no L2 penalty, so the knob only ever
                    # describes the exact-ansatz rows.
                    row_reg = ansatz == :exact ? cfg.regularization : 0.0
                    append_row!(io, (ansatz, loss_type, run, seed, final_loss, infidelity, energy,
                        n_params, initial_loss, elapsed, row_reg))
                end
            end
        finally
            close(io)
        end

        println("\nWrote $(cfg.out_path)")
        return 0
    end
end
