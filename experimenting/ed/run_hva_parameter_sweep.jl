#=
run_hva_parameter_sweep.jl

Sweep the HVA ansatz's fidelity against its parameter count, at a single U index,
with the same protocol as `run_repeat_optimization_experiment.jl` so the rows
concatenate with `benchmarks/repeat_optimization_*.csv`.

The HVA carries far fewer free parameters per layer than the standard
momentum-conserving Trotter ansatz (a whole Hamiltonian term shares one
coefficient), so "how good is it" is only meaningful as a curve against
`n_params`. This script walks a set of layer counts for each tying scheme and
records the final infidelity of every individual run.

Because the HVA is a *real-space* ansatz, everything runs in the full real-space
occupation basis (see `trotter_realspace.jl`): the ED target states are lifted
out of their momentum sector and the lift is verified against the ED energy
before any optimization happens.

Usage:
  julia --project=.. run_hva_parameter_sweep.jl <folder> <u_idx> [options]

Arguments:
  folder (required): ED data sub-folder, e.g. "N=(3, 2)_3x2".
  u_idx  (required): 1-based index into the data's U values (e.g. 32 => U = 7.75).

Options:
  --tie=<list>               Comma-separated subset of "full,spin,none". Each are independently
                             run. Default: all three.
  --layers=<list>            Comma-separated HVA layer counts. Default: the counts whose
                             parameter totals bracket the standard ansatz's, i.e.
                             [1/4, 1/2, 1, 2, 4] x hva_layers_matching_dof(...).
  --runs=<n>                 Independent repeats per (tie, layers, loss). Default: 3.
  --run_start=<n>            First run index to execute. Default: 1.
  --run_end=<n>              Last run index. Default: --runs. Run r uses seed = --seed + r,
                             so disjoint ranges can be farmed out and still concatenate.
  --maxiters=<n>             Maximum iterations per optimizer stage. Default: 1000.
  --initialization_samples=<n>  Multi-start random samples per run. Default: 1.
  --loss=<list>              Comma-separated subset of "overlap,energy". Default: overlap.
  --hva_pbc=<bool>           Periodic bonds in the ansatz. Default: true when every lattice
                             axis is even (or length <= 2), false otherwise -- a periodic
                             axis of odd length > 2 cannot be 2-coloured and would throw.
  --ref=<slater|eigenstate>  Which state the circuit starts from. Default: slater.
                             "slater"     -- a single Slater determinant, prepended as row 1
                                             of the ED data (U_values then starts at 0.0).
                             "eigenstate" -- the ED ground state at the LOWEST U in the data
                                             (no row is prepended). NOTE this shifts the U
                                             indexing by one: with "slater" u_idx=9 is U=2.0,
                                             with "eigenstate" u_idx=8 is U=2.0. The `U` column
                                             in the CSV records the actual value either way.
  --custom_ref_state=<value> Reference state ("slater" or a basis index). Default: "slater".
  --control                  Also run the standard momentum-conserving Trotter ansatz
                             (antihermitian, 1 layer) as a calibration row. This should
                             reproduce the stored repeat_optimization_*.csv numbers.
  --seed=<n>                 Base RNG seed. Default: 20260915.
  --use_gpu[=<bool>]         Run the circuit evolution and adjoint gradients on a CUDA GPU.
                             CUDA is only loaded when this is set. Default: false.
  --out=<path>               Output CSV. Default:
                             benchmarks/hva_parameter_sweep_<folder>_u<idx>.csv

The CSV schema extends `run_repeat_optimization_experiment.jl` with three columns,
`n_layers` (the HVA layer count P, 1 for the standard-ansatz control) and
`n_params_per_layer` (shared parameters per layer under the active tying scheme),
so `n_params == n_layers * n_params_per_layer`, and appends a trailing `U` column
holding the interaction strength `U_values[u_idx]`, then `max_rss_mb`: the process's
peak resident host memory (MiB, `Sys.maxrss()`) when the row was written. It is a
process-wide high-water mark, so it is exact per row only when a job runs a single
(tie, layers, loss, run); GPU memory is not included. Everything else matches, so
`benchmarks/collect_repeat_optimization_csv.py` still works. The tying scheme and
layer count are also encoded in the `ansatz` column as `hva_<tie>_P<layers>`,
which keeps the (ansatz, loss_type, run) uniqueness that collector checks.
=#

# Pre-scan ARGS for the GPU flag so CUDA is only loaded when it is requested.
_use_gpu_prescan = let val = false
    for arg in ARGS
        if arg == "--use_gpu" || arg == "--use_gpu=true"
            val = true
        elseif arg == "--use_gpu=false"
            val = false
        end
    end
    val
end

if _use_gpu_prescan
    ENV["JULIA_CUDA_USE_COMPAT"] = "true"
    using CUDA
end

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

include("data_path.jl")
include("logging.jl")
include("utility_functions.jl")
using .UtilityFunctions

include("trotter.jl")

include("ed_objects.jl")
include("ed_functions.jl")

const CSV_HEADER = "ansatz,loss_type,run,seed,final_loss,infidelity,energy,n_params,n_layers,n_params_per_layer,initial_loss,elapsed_s,U,max_rss_mb"
const OPTIMIZER_CHAIN = [:LBFGS, :GradientDescent, :LBFGS]

function parse_arguments(args::Vector{String})
    ties = [:full, :spin, :none]
    layers = Int[]
    runs = 3
    run_start = nothing
    run_end = nothing
    maxiters = 1000
    initialization_samples = 1
    losses = [:overlap]
    hva_pbc = nothing
    custom_ref_state_arg = "slater"
    ref_mode = nothing
    control = false
    seed = 20260915
    use_gpu = false
    out_path = nothing
    positional = String[]

    for arg in args
        if startswith(arg, "--tie=")
            vals = split(split(arg, "=", limit=2)[2], ",")
            ties = map(vals) do v
                v in ("full", "spin", "none") ||
                    error("Invalid --tie entry: '$v'. Valid: full, spin, none.")
                Symbol(v)
            end
        elseif startswith(arg, "--layers=")
            layers = parse.(Int, split(split(arg, "=", limit=2)[2], ","))
        elseif startswith(arg, "--runs=")
            runs = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--run_start=")
            run_start = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--run_end=")
            run_end = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--maxiters=")
            maxiters = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--initialization_samples=")
            initialization_samples = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--loss=")
            vals = split(split(arg, "=", limit=2)[2], ",")
            losses = map(vals) do v
                v in ("overlap", "energy") ||
                    error("Invalid --loss entry: '$v'. Valid: overlap, energy.")
                Symbol(v)
            end
        elseif arg == "--hva_pbc" || startswith(arg, "--hva_pbc=")
            hva_pbc = occursin("=", arg) ? parse(Bool, split(arg, "=", limit=2)[2]) : true
        elseif startswith(arg, "--ref=")
            v = String(split(arg, "=", limit=2)[2])
            v in ("slater", "eigenstate") ||
                error("Invalid --ref entry: '$v'. Valid: slater, eigenstate.")
            ref_mode = Symbol(v)
        elseif startswith(arg, "--custom_ref_state=")
            custom_ref_state_arg = String(split(arg, "=", limit=2)[2])
            # elseif arg == "--control"
            #     control = true
        elseif startswith(arg, "--seed=")
            seed = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg == "--use_gpu" || startswith(arg, "--use_gpu=")
            use_gpu = occursin("=", arg) ? parse(Bool, split(arg, "=", limit=2)[2]) : true
        elseif startswith(arg, "--out=")
            out_path = String(split(arg, "=", limit=2)[2])
        else
            push!(positional, arg)
        end
    end

    length(positional) >= 2 ||
        error("Usage: julia run_hva_parameter_sweep.jl <folder> <u_idx> [options]")

    folder_arg = positional[1]
    folder = data_folder(folder_arg)
    u_idx = parse(Int, positional[2])

    # --ref wins; otherwise fall back to the old --custom_ref_state spelling.
    ref_mode = isnothing(ref_mode) ?
               (custom_ref_state_arg == "slater" ? :slater : :eigenstate) : ref_mode

    run_start = isnothing(run_start) ? 1 : run_start
    run_end = isnothing(run_end) ? runs : run_end

    if isnothing(out_path)
        safe = replace(folder_arg, r"[^A-Za-z0-9]+" => "_")
        out_path = joinpath(@__DIR__, "benchmarks", "hva_parameter_sweep_$(safe)_u$(u_idx).csv")
    end

    return (; folder, folder_arg, u_idx, ties, layers, runs, run_start, run_end,
        maxiters, initialization_samples, losses, hva_pbc, custom_ref_state_arg,
        ref_mode, control, seed, use_gpu, out_path)
end

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "run_hva_parameter_sweep")
    with_logging(log_path) do
        cfg = parse_arguments(ARGS)

        println("Number of threads: $(Threads.nthreads())")
        println("folder:                 $(cfg.folder_arg)")
        println("u_idx:                  $(cfg.u_idx)")
        println("ties:                   $(cfg.ties)")
        println("losses:                 $(cfg.losses)")
        println("maxiters:               $(cfg.maxiters)")
        println("initialization_samples: $(cfg.initialization_samples)")
        println("runs:                   $(cfg.run_start):$(cfg.run_end) of $(cfg.runs)")
        println("reference:              $(cfg.ref_mode)")
        println("output CSV:             $(cfg.out_path)")
        if cfg.use_gpu
            CUDA.functional() || error("--use_gpu was given but CUDA is not functional on this node")
            println("use_gpu:                true ($(CUDA.name(CUDA.device())))")
        else
            println("use_gpu:                false")
        end

        U_values, state_vecs, indexer, _, N_elec, spin_conserved, _, sign_convention =
            load_ED_data(cfg.folder; verbose=true, sign_convention=:spin_first,
                use_slater_reference=cfg.ref_mode === :slater)

        n_up, n_dn = N_elec
        Lvec = parse_lattice_dimension(cfg.folder)
        N_sites = prod(Lvec)
        target_u = U_values[cfg.u_idx]
        println("Lvec = $Lvec, nvec = ($n_up, $n_dn), U = $target_u")
        if cfg.ref_mode === :eigenstate
            println("reference = ED ground state at U = $(U_values[1]) " *
                    "(no Slater row prepended, so u_idx is shifted by one vs --ref=slater)")
        end

        hva_pbc = if !isnothing(cfg.hva_pbc)
            cfg.hva_pbc
        else
            auto = all(L -> L <= 2 || iseven(L), Lvec)
            println("--hva_pbc not given; using $auto " *
                    (auto ? "(every axis is even or length <= 2)" :
                     "(an odd axis > 2 admits no 2-colouring into commuting bond groups)"))
            auto
        end

        has_prepended_ref = (state_vecs isa AbstractMatrix) &&
                            (size(state_vecs, 1) == length(U_values) + 1)
        ref_row = 1
        target_row = has_prepended_ref ? cfg.u_idx + 1 : cfg.u_idx

        # ---- momentum sector: the standard ansatz's home, and the control -----
        basis_mom = Trotter.get_basis_sector(indexer, Lvec, N_sites)
        H_hop_m, _, _ = Trotter.TamFermion.HubbardMomentumBasis(1.0, 0.0, Lvec, (n_up, n_dn); indexer=indexer)
        H_int_m, _, _ = Trotter.TamFermion.HubbardMomentumBasis(0.0, 1.0, Lvec, (n_up, n_dn); indexer=indexer)
        H_mom = H_hop_m + target_u * H_int_m
        s1_m = state_vecs[ref_row, :]
        s2_m = state_vecs[target_row, :]
        println("momentum sector dimension: $(length(s1_m))")

        gates_std = Trotter.enumerate_ferm_excitations(2, Lvec; conserve_mom=true,
            conserve_sz=true, include_diagonal=false)
        println("standard Trotter gates (antihermitian): $(length(gates_std))")

        # ---- real space: the HVA's home --------------------------------------
        basis_real, _, _ = Trotter.realspace_basis(Lvec, (n_up, n_dn))
        H_hop_r = Trotter.TamFermion.HubbardRealSpace(1.0, 0.0, Lvec, (n_up, n_dn);
            use_pbc=true, returnBasis=false)
        H_int_r = Trotter.TamFermion.HubbardRealSpace(0.0, 1.0, Lvec, (n_up, n_dn);
            use_pbc=true, returnBasis=false)
        H_real = H_hop_r + target_u * H_int_r

        vecs_r = Trotter.momentum_sector_to_realspace(state_vecs, basis_mom, Lvec, (n_up, n_dn))
        s1_r = vecs_r[ref_row, :]
        s2_r = vecs_r[target_row, :]
        println("real-space dimension: $(length(s1_r))")

        for (label, vm, vr) in (("reference", s1_m, s1_r), ("target", s2_m, s2_r))
            chk = Trotter.check_realspace_transform(vm, vr, H_mom, H_real; label=label)
            @printf("  transform check (%s): |psi| %.12f -> %.12f ; <H> %.10f -> %.10f\n",
                label, chk.norm_mom, chk.norm_real, chk.energy_mom, chk.energy_real)
        end

        ed_energy = real(dot(s2_m, H_mom * s2_m))
        println("Exact ED ground energy at U=$target_u: $ed_energy")

        # ---- CSV -------------------------------------------------------------
        mkpath(dirname(cfg.out_path))
        fresh = !isfile(cfg.out_path)
        io = open(cfg.out_path, "a")
        fresh && (println(io, CSV_HEADER); flush(io))

        function record(ansatz, loss_type, run, seed, floss, infid, energy, n_params,
            n_layers, n_params_per_layer, init_loss, dt)
            @printf(io, "%s,%s,%d,%d,%.17g,%.17g,%.17g,%d,%d,%d,%.17g,%.3f,%.17g,%.1f\n",
                ansatz, loss_type, run, seed, floss, infid, energy, n_params,
                n_layers, n_params_per_layer, init_loss, dt, target_u, Sys.maxrss() / 2^20)
            flush(io)
            @printf("  %-18s %-8s run %-3d n_params=%-5d layers=%-4d params/layer=%-5d infidelity=%.4e  (%.1f s)\n",
                ansatz, loss_type, run, n_params, n_layers, n_params_per_layer, infid, dt)
        end

        try
            #     # ---- control: the standard momentum-conserving ansatz -------------
            #     if cfg.control
            #         taus_std = Trotter.fgateToTauSector(gates_std, N_sites, basis_mom; antihermitian=true)
            #         for loss_type in cfg.losses
            #             println("\n######## control: standard Trotter, loss=$loss_type ########")
            #             opt_target = loss_type == :energy ? H_mom : s2_m
            #             for run in cfg.run_start:cfg.run_end
            #                 seed = cfg.seed + run
            #                 Random.seed!(seed)
            #                 t0 = time()
            #                 A, floss, metrics = Trotter.optimize_unitary(
            #                     gates_std, taus_std, s1_m, opt_target, basis_mom, N_sites;
            #                     loss_type=loss_type, H=H_mom, state2=s2_m, num_exponentials=1,
            #                     maxiters=cfg.maxiters, optimizer=OPTIMIZER_CHAIN,
            #                     initialization_samples=cfg.initialization_samples,
            #                     antihermitian=true, use_gpu=false, datatype=ComplexF64)
            #                 psi = Array(Trotter.apply_unitary(A, gates_std, s1_m, basis_mom, N_sites, 1;
            #                     antihermitian=true, use_gpu=false, datatype=ComplexF64))
            #                 record("trotter", loss_type, run, seed, floss,
            #                     1.0 - abs2(dot(s2_m, psi)), real(dot(psi, H_mom * psi)),
            #                     length(A), 1, length(A),
            #                     isempty(metrics["loss"]) ? NaN : Float64(metrics["loss"][1]),
            #                     time() - t0)
            #             end
            #         end
            #     end

            # ---- HVA sweep ----------------------------------------------------
            for tie in cfg.ties
                gates, pmap = Trotter.enumerate_ferm_excitations_HVA(Lvec; use_pbc=hva_pbc, tie=tie)
                taus = Trotter.fgateToTauSector(gates, N_sites, basis_real; antihermitian=false)
                n_per_layer = Trotter.num_shared_params(pmap, length(gates))

                layer_set = if !isempty(cfg.layers)
                    cfg.layers
                else
                    match = Trotter.hva_layers_matching_dof(pmap, length(gates), gates_std, 1).layers
                    sort(unique(max.(1, [match ÷ 4, match ÷ 2, match, 2 * match, 4 * match])))
                end

                println("\n######## HVA tie=$tie: $(length(gates)) gates, " *
                        "$n_per_layer params/layer, layers $layer_set ########")

                for loss_type in cfg.losses, P in layer_set
                    opt_target = loss_type == :energy ? H_real : s2_r
                    label = "hva_$(tie)_P$(P)"
                    for run in cfg.run_start:cfg.run_end
                        seed = cfg.seed + run
                        Random.seed!(seed)
                        t0 = time()
                        A, floss, metrics = Trotter.optimize_unitary(
                            gates, taus, s1_r, opt_target, basis_real, N_sites;
                            loss_type=loss_type, H=H_real, state2=s2_r, num_exponentials=P,
                            param_map=pmap, maxiters=cfg.maxiters, optimizer=OPTIMIZER_CHAIN,
                            initialization_samples=cfg.initialization_samples,
                            antihermitian=false, use_gpu=cfg.use_gpu, datatype=ComplexF64)
                        psi = Array(Trotter.apply_unitary(A, gates, s1_r, basis_real, N_sites, P;
                            antihermitian=false, use_gpu=cfg.use_gpu, datatype=ComplexF64, param_map=pmap))
                        record(label, loss_type, run, seed, floss,
                            1.0 - abs2(dot(s2_r, psi)), real(dot(psi, H_real * psi)),
                            length(A), P, n_per_layer,
                            isempty(metrics["loss"]) ? NaN : Float64(metrics["loss"][1]),
                            time() - t0)
                    end
                end
            end
        finally
            close(io)
        end

        println("\nWrote $(cfg.out_path)")
        println("Log: $log_path")
        return 0
    end
end
