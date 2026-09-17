#=
benchmark_timings.jl

Benchmark forward-pass and gradient wall-clock times for the two UCC ansatz
implementations, WITHOUT running any optimization. No coefficients, metrics or
other run artifacts are saved: the only outputs are the timings CSV and the
usual tee'd stdout log under `logs/<date>/` that every script here writes.

  --code=trotter : Trotter/disentangled circuit
                   (`Trotter.adjoint_loss` / `Trotter.energy_loss` in trotter_loss.jl)
  --code=exact   : single full matrix exponential
                   (`adjoint_loss` / `adjoint_energy_loss` in ed_optimization.jl)

For each (system, loss, num_exponentials) the script builds exactly the objects the
production runners build (same ED data, same gate set / operator structure, same
reference and target states), draws one random coefficient vector, and then times

  * the forward pass  : f(A)
  * the gradient call : Zygote.gradient(f, A)   (forward + backward)

`backward` is reported as (gradient - forward), since Zygote's gradient call
necessarily re-runs the forward pass.

The two codes must be benchmarked in SEPARATE julia processes: trotter.jl and
ed_optimization.jl both define `adjoint_loss`, and `@safe_threads` silently falls
back to serial execution whenever CUDA is loaded, so a GPU run and an 80-thread CPU
run also have to be separate processes.

Usage:
  julia --project=.. benchmark_timings.jl --code=<trotter|exact> [options]

Options:
  --code=<trotter|exact>   (required) Which implementation to benchmark.
  --use_gpu[=<bool>]       Load CUDA and run on the GPU. Default: false.
  --systems=<a;b;c>        Semicolon-separated ED data sub-folders.
                           Default: "N=(5, 4)_3x3;N=(5, 4)_4x3;N=(4, 3)_3x3".
  --losses=<a,b>           Comma-separated: overlap, energy. Default: "overlap,energy".
  --exponentials=<a,b,..>  Trotter layer counts (--code=trotter only). Default: "1,2,4,8".
  --reps=<n>               Timed repetitions per configuration. Default: 5.
  --warmup=<n>             Warm-up calls before timing (excludes JIT). Default: 1.
  --u-index=<n>            Index into U_values for the target state. Default: 33.
  --magnitude=<x>          Scale of the random coefficients, matching the middle of
                           the production multi-start range [1e-7, 1e-1]. Default: 0.01.
  --antihermitian=<bool>   Real antihermitian generators. Default: true.
  --custom_ref_state=<v>   "slater", an integer basis index, or "none". Default: "slater".
  --datatype=<type>        ComplexF64 | ComplexF32 | Float64 | Float32 (trotter only).
                           Default: ComplexF64.
  --out=<path>             CSV output path. Default: benchmarks/timings_<code>_<device>.csv
  --tag=<string>           Free-form label written into every row. Default: "".

Examples:
  julia --project=.. benchmark_timings.jl --code=trotter --use_gpu
  julia -t 80 --project=.. benchmark_timings.jl --code=exact --losses=overlap
=#

# ── Pre-scan ARGS for the GPU flag before loading CUDA ────────────────────────
# @safe_threads runs serially whenever CUDA is loaded, so CUDA must NOT be loaded
# for the CPU-threaded runs.
const _USE_GPU = let val = false
    for arg in ARGS
        if arg == "--use_gpu" || arg == "--use_gpu=true"
            val = true
        elseif arg == "--use_gpu=false"
            val = false
        end
    end
    val
end

const _CODE = let val = nothing
    for arg in ARGS
        if startswith(arg, "--code=")
            val = Symbol(split(arg, "=", limit=2)[2])
        end
    end
    if val === nothing
        error("--code=<trotter|exact> is required")
    elseif val ∉ (:trotter, :exact)
        error("Invalid --code=$val. Valid options are: 'trotter', 'exact'.")
    end
    val
end

if _USE_GPU
    ENV["JULIA_CUDA_USE_COMPAT"] = "true"
    using CUDA
    if !CUDA.functional()
        @info "CUDA not functional yet — trying local_toolkit mode"
        CUDA.set_runtime_version!(local_toolkit=true)
    end
    CUDA.functional() || error("--use_gpu was requested but CUDA is not functional.")
    @info "GPU available: $(CUDA.name(CUDA.CuDevice(0)))"
end

using Lattices
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

include("data_path.jl")
include("logging.jl")
include("utility_functions.jl")
using .UtilityFunctions
include("ed_objects.jl")
include("ed_functions.jl")

if _CODE == :trotter
    include("trotter.jl")
    import .Trotter
else
    using Optimization, OptimizationOptimJL, OptimizationOptimisers
    using KrylovKit
    include("ed_optimization.jl")
end

# ── Argument parsing ──────────────────────────────────────────────────────────

# Ordered smallest Hilbert space first, so that a job that runs out of time still
# leaves the cheap systems in the CSV (rows are flushed as they are produced).
const DEFAULT_SYSTEMS = ["N=(4, 3)_3x3", "N=(5, 4)_3x3", "N=(5, 4)_4x3"]

function parse_arguments(args::Vector{String})
    opts = Dict{Symbol,Any}(
        :systems => DEFAULT_SYSTEMS,
        :losses => [:overlap, :energy],
        :exponentials => [1, 2, 4, 8],
        :reps => 5,
        :warmup => 1,
        :u_index => 33,
        :magnitude => 0.01,
        :antihermitian => true,
        :custom_ref_state => "slater",
        :datatype => ComplexF64,
        :out => nothing,
        :tag => "",
    )

    for arg in args
        if arg == "--code" || startswith(arg, "--code=") ||
           arg == "--use_gpu" || startswith(arg, "--use_gpu=")
            continue # handled in the pre-scan
        elseif startswith(arg, "--systems=")
            opts[:systems] = String.(filter(!isempty, strip.(split(split(arg, "=", limit=2)[2], ";"))))
        elseif startswith(arg, "--losses=")
            vals = strip.(split(split(arg, "=", limit=2)[2], ","))
            for v in vals
                v in ("overlap", "energy") || error("Invalid --losses entry: '$v'. Valid options are: 'overlap', 'energy'.")
            end
            opts[:losses] = Symbol.(vals)
        elseif startswith(arg, "--exponentials=")
            opts[:exponentials] = parse.(Int, strip.(split(split(arg, "=", limit=2)[2], ",")))
        elseif startswith(arg, "--reps=")
            opts[:reps] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--warmup=")
            opts[:warmup] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--u-index=") || startswith(arg, "--u_index=")
            opts[:u_index] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--magnitude=")
            opts[:magnitude] = parse(Float64, split(arg, "=", limit=2)[2])
        elseif arg == "--antihermitian"
            opts[:antihermitian] = true
        elseif startswith(arg, "--antihermitian=")
            opts[:antihermitian] = parse(Bool, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--custom_ref_state=")
            opts[:custom_ref_state] = String(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--datatype=")
            v = String(split(arg, "=", limit=2)[2])
            opts[:datatype] = v == "ComplexF64" ? ComplexF64 :
                              v == "ComplexF32" ? ComplexF32 :
                              v == "Float64" ? Float64 :
                              v == "Float32" ? Float32 :
                              error("Invalid --datatype option: '$v'. Valid options are: 'ComplexF64', 'ComplexF32', 'Float64', 'Float32'.")
        elseif startswith(arg, "--out=")
            opts[:out] = String(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--tag=")
            opts[:tag] = String(split(arg, "=", limit=2)[2])
        else
            error("Unrecognized argument: $arg")
        end
    end

    if !opts[:antihermitian] && opts[:datatype] <: Real
        @warn "Hermitian generators require complex arithmetic. Overriding datatype to ComplexF64."
        opts[:datatype] = ComplexF64
    end
    opts[:reps] >= 1 || error("--reps must be >= 1")
    opts[:warmup] >= 0 || error("--warmup must be >= 0")

    if isnothing(opts[:out])
        device = _USE_GPU ? "gpu" : "cpu$(Threads.nthreads())"
        opts[:out] = joinpath(@__DIR__, "benchmarks", "timings_$(_CODE)_$(device).csv")
    end
    return opts
end

# ── Timing helpers ────────────────────────────────────────────────────────────

_sync() = _USE_GPU ? CUDA.synchronize() : nothing

"""
    time_calls(f, x; warmup, reps) -> NamedTuple

Time `f(x)` (forward) and `Zygote.gradient(f, x)` (forward + backward). The first
call of each is timed separately and reported as `compile_*`, since it includes
JIT compilation; the reported samples come from calls made after `warmup` further
calls.
"""
function time_calls(f, x; warmup::Int=1, reps::Int=5)
    compile_fwd = @elapsed begin
        f(x)
        _sync()
    end
    for _ in 1:warmup
        f(x)
        _sync()
    end
    fwd = Float64[]
    for _ in 1:reps
        t0 = time_ns()
        f(x)
        _sync()
        push!(fwd, (time_ns() - t0) / 1e9)
    end

    compile_grad = @elapsed begin
        Zygote.gradient(f, x)
        _sync()
    end
    for _ in 1:warmup
        Zygote.gradient(f, x)
        _sync()
    end
    grad = Float64[]
    for _ in 1:reps
        t0 = time_ns()
        Zygote.gradient(f, x)
        _sync()
        push!(grad, (time_ns() - t0) / 1e9)
    end

    return (compile_fwd=compile_fwd, fwd=fwd, compile_grad=compile_grad, grad=grad)
end

_std1(v) = length(v) > 1 ? std(v) : 0.0

# ── CSV output ────────────────────────────────────────────────────────────────

const CSV_COLUMNS = [
    "code", "system", "lattice", "n_up", "n_dn", "dim", "loss", "num_exponentials",
    "device", "threads", "gpu", "n_params", "u_index", "U", "loss_value",
    "forward_min_s", "forward_median_s", "forward_mean_s", "forward_std_s",
    "gradient_min_s", "gradient_median_s", "gradient_mean_s", "gradient_std_s",
    "backward_median_s", "gradient_over_forward",
    "compile_forward_s", "compile_gradient_s", "setup_s",
    "reps", "warmup", "magnitude", "datatype", "antihermitian", "ref_state",
    "julia_version", "hostname", "timestamp", "tag",
]

function csv_escape(x)
    s = string(x)
    return (occursin(',', s) || occursin('"', s)) ? '"' * replace(s, '"' => "\"\"") * '"' : s
end

function open_csv(path::String)
    mkpath(dirname(path))
    fresh = !isfile(path) || filesize(path) == 0
    io = open(path, "a")
    fresh && println(io, join(CSV_COLUMNS, ","))
    flush(io)
    return io
end

function write_row!(io, row::Dict{String,Any})
    println(io, join((csv_escape(get(row, c, "")) for c in CSV_COLUMNS), ","))
    flush(io)
end

# ── Shared setup ──────────────────────────────────────────────────────────────

"""
    load_system(folder, custom_ref_state)

Load the ED data for one system exactly the way the production runners do, and
resolve the reference (state1) / target (state2) pair for a given U index.
"""
function load_system(folder::String, custom_ref_state::String)
    use_slater_ref = custom_ref_state == "none" ? false :
                     custom_ref_state == "slater" ? true :
                     (tryparse(Int, custom_ref_state) !== nothing ? parse(Int, custom_ref_state) : true)

    U_values, state_vecs, indexer, precomputed_structures, N_elec, spin_conserved, use_symmetry, sign_convention =
        load_ED_data(folder; verbose=true, sign_convention=:spin_first, use_slater_reference=use_slater_ref)

    Lvec = parse_lattice_dimension(folder)
    has_prepended_ref = (state_vecs isa AbstractMatrix) && (size(state_vecs, 1) == length(U_values) + 1)

    return (; U_values, state_vecs, indexer, precomputed_structures, N_elec, spin_conserved,
        use_symmetry, sign_convention, Lvec, has_prepended_ref)
end

_row_of(vecs, i) = vecs isa AbstractMatrix ? vecs[i, :] : vecs[i]

function ref_and_target(sys, u_index::Int)
    ref = _row_of(sys.state_vecs, 1)
    target = _row_of(sys.state_vecs, sys.has_prepended_ref ? u_index + 1 : u_index)
    return ref, target
end

# ── Trotter benchmark ─────────────────────────────────────────────────────────

function benchmark_trotter(io, opts)
    for system in opts[:systems]
        folder = data_folder(system)
        isdir(folder) || (@warn "Skipping missing data folder: $folder"; continue)
        println("\n", "="^78, "\n[trotter] $system\n", "="^78)

        setup_s = @elapsed begin
            sys = load_system(folder, opts[:custom_ref_state])
            n_up, n_dn = sys.N_elec
            N_sites = prod(sys.Lvec)
            basis_sector = Trotter.get_basis_sector(sys.indexer, sys.Lvec, N_sites)
            H_hop, _, _ = Trotter.TamFermion.HubbardMomentumBasis(1.0, 0.0, sys.Lvec, (n_up, n_dn); indexer=sys.indexer)
            H_int, _, _ = Trotter.TamFermion.HubbardMomentumBasis(0.0, 1.0, sys.Lvec, (n_up, n_dn); indexer=sys.indexer)
            gates = Trotter.enumerate_ferm_excitations(2, sys.Lvec; conserve_mom=true, conserve_sz=true,
                include_diagonal=!opts[:antihermitian])
            tau_terms = Trotter.fgateToTauSector(gates, N_sites, basis_sector; antihermitian=opts[:antihermitian])
        end

        u_index = opts[:u_index]
        1 <= u_index <= length(sys.U_values) ||
            error("--u-index=$u_index is out of range for $system (1:$(length(sys.U_values)))")
        U = sys.U_values[u_index]
        ref, target = ref_and_target(sys, u_index)
        dim = length(basis_sector)

        datatype = opts[:datatype]
        ref_prep = (datatype <: Real) ? Trotter.strip_global_phase(ref)[1] : ref
        target_prep = (datatype <: Real) ? Trotter.strip_global_phase(target)[1] : target
        H = H_hop + U * H_int

        @printf("  dim=%d  gates=%d  U[%d]=%.6g  setup=%.2fs\n", dim, length(gates), u_index, U, setup_s)

        for loss in opts[:losses], P in opts[:exponentials]
            M = length(gates) * P
            Random.seed!(1234)
            A = (2 * rand(M) .- 1) * opts[:magnitude]

            f = if loss == :overlap
                A -> Trotter.adjoint_loss(A, gates, tau_terms, ref_prep, target_prep, basis_sector, N_sites;
                    num_exponentials=P, antihermitian=opts[:antihermitian], use_gpu=_USE_GPU, datatype=datatype)
            else
                A -> Trotter.energy_loss(A, gates, tau_terms, H, ref_prep, basis_sector, N_sites;
                    num_exponentials=P, antihermitian=opts[:antihermitian], use_gpu=_USE_GPU, datatype=datatype)
            end

            println("\n--- trotter | $system | loss=$loss | P=$P | params=$M ---")
            t = time_calls(f, A; warmup=opts[:warmup], reps=opts[:reps])
            loss_value = f(A)
            report_and_write!(io, opts, t; code="trotter", system=system, lattice=join(sys.Lvec, "x"),
                n_up=n_up, n_dn=n_dn, dim=dim, loss=loss, num_exponentials=P, n_params=M,
                u_index=u_index, U=U, loss_value=loss_value, setup_s=setup_s)

            _USE_GPU && CUDA.reclaim()
        end
    end
end

# ── Exact matrix-exponential benchmark ────────────────────────────────────────

function benchmark_exact(io, opts)
    for system in opts[:systems]
        folder = data_folder(system)
        isdir(folder) || (@warn "Skipping missing data folder: $folder"; continue)
        println("\n", "="^78, "\n[exact] $system\n", "="^78)

        sys = nothing
        H_hop = H_int = nothing
        setup_s = @elapsed begin
            sys = load_system(folder, opts[:custom_ref_state])
            subspace = reconstruct_subspace(sys.indexer, sys.spin_conserved)
            H_hop, H_int = create_hubbard_matrices(subspace; indexer=sys.indexer, get_indexer=false,
                sign_convention=:spin_first, lattice_ordering=ColSnake())
        end

        u_index = opts[:u_index]
        1 <= u_index <= length(sys.U_values) ||
            error("--u-index=$u_index is out of range for $system (1:$(length(sys.U_values)))")
        U = sys.U_values[u_index]
        state1, state2 = ref_and_target(sys, u_index)
        n_up, n_dn = sys.N_elec
        dim = length(sys.indexer.inv_comb_dict)
        H = H_hop + U * H_int

        @printf("  dim=%d  U[%d]=%.6g  use_symmetry=%s  setup=%.2fs\n",
            dim, u_index, U, sys.use_symmetry, setup_s)

        for loss in opts[:losses]
            # Mirrors optimize_unitary: the operator structure's initial-magnitude
            # estimate depends on the loss at the identity unitary.
            loss0 = loss == :energy ? real(dot(state1, H * state1)) :
                    max(0.0, 1 - abs2(state1' * state2))
            init_mag = loss == :energy ? 0.01 + 0im : loss0 * 100

            struct_s = @elapsed sd = ensure_operator_structure!(
                2, Dict{Int,Dict{Symbol,Any}}(), sys.indexer, sys.spin_conserved, sys.use_symmetry,
                false, :spin_first, ColSnake(), sys.precomputed_structures, opts[:antihermitian], init_mag)

            use_gpu_flag, ops_gpu, state1_gpu, state2_gpu, H_gpu =
                setup_gpu_resources(_USE_GPU ? true : false, state1, state2, H, loss, sd[:ops])
            if _USE_GPU && !use_gpu_flag
                error("--use_gpu was requested but setup_gpu_resources fell back to CPU.")
            end

            _, _, _, f_adjoint, f_adjoint_gpu = setup_loss_functions(
                loss, 2, sd[:ops], sd[:rows], sd[:cols], sd[:signs], sd[:param_index_map],
                sd[:parameter_mapping], sd[:parity], dim, state1, state2, H,
                sys.use_symmetry, opts[:antihermitian], use_gpu_flag,
                ops_gpu, state1_gpu, state2_gpu, H_gpu, 1)

            f_raw = use_gpu_flag ? f_adjoint_gpu : f_adjoint
            f = t -> f_raw(t, nothing)

            P = sys.use_symmetry ? length(sd[:sym_data][1]) : length(sd[:t_keys])
            Random.seed!(1234)
            t_vals = sys.use_symmetry ? real(rand(typeof(sd[:signs][1]), P) * opts[:magnitude]) :
                     (2 * rand(P) .- 1) * opts[:magnitude]

            println("\n--- exact | $system | loss=$loss | params=$P (structure built in $(round(struct_s, digits=2))s) ---")
            t = time_calls(f, t_vals; warmup=opts[:warmup], reps=opts[:reps])
            loss_value = f(t_vals)
            report_and_write!(io, opts, t; code="exact", system=system, lattice=join(sys.Lvec, "x"),
                n_up=n_up, n_dn=n_dn, dim=dim, loss=loss, num_exponentials=1, n_params=P,
                u_index=u_index, U=U, loss_value=loss_value, setup_s=setup_s + struct_s)

            _USE_GPU && CUDA.reclaim()
        end
    end
end

# ── Reporting ─────────────────────────────────────────────────────────────────

function report_and_write!(io, opts, t; code, system, lattice, n_up, n_dn, dim, loss,
    num_exponentials, n_params, u_index, U, loss_value, setup_s)

    fmed, gmed = median(t.fwd), median(t.grad)
    @printf("    forward : min %.4fs  median %.4fs  mean %.4fs  std %.4fs\n",
        minimum(t.fwd), fmed, mean(t.fwd), _std1(t.fwd))
    @printf("    gradient: min %.4fs  median %.4fs  mean %.4fs  std %.4fs\n",
        minimum(t.grad), gmed, mean(t.grad), _std1(t.grad))
    @printf("    backward (gradient - forward) median: %.4fs   ratio gradient/forward: %.2fx\n",
        gmed - fmed, fmed > 0 ? gmed / fmed : NaN)
    @printf("    first call (incl. JIT): forward %.2fs  gradient %.2fs\n", t.compile_fwd, t.compile_grad)

    write_row!(io, Dict{String,Any}(
        "code" => code,
        "system" => system,
        "lattice" => lattice,
        "n_up" => n_up,
        "n_dn" => n_dn,
        "dim" => dim,
        "loss" => loss,
        "num_exponentials" => num_exponentials,
        "device" => _USE_GPU ? "gpu" : "cpu",
        "threads" => Threads.nthreads(),
        "gpu" => _USE_GPU ? CUDA.name(CUDA.device()) : "",
        "n_params" => n_params,
        "u_index" => u_index,
        "U" => U,
        "loss_value" => loss_value,
        "forward_min_s" => minimum(t.fwd),
        "forward_median_s" => fmed,
        "forward_mean_s" => mean(t.fwd),
        "forward_std_s" => _std1(t.fwd),
        "gradient_min_s" => minimum(t.grad),
        "gradient_median_s" => gmed,
        "gradient_mean_s" => mean(t.grad),
        "gradient_std_s" => _std1(t.grad),
        "backward_median_s" => gmed - fmed,
        "gradient_over_forward" => fmed > 0 ? gmed / fmed : NaN,
        "compile_forward_s" => t.compile_fwd,
        "compile_gradient_s" => t.compile_grad,
        "setup_s" => setup_s,
        "reps" => opts[:reps],
        "warmup" => opts[:warmup],
        "magnitude" => opts[:magnitude],
        "datatype" => opts[:datatype],
        "antihermitian" => opts[:antihermitian],
        "ref_state" => opts[:custom_ref_state],
        "julia_version" => VERSION,
        "hostname" => gethostname(),
        "timestamp" => Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
        "tag" => opts[:tag],
    ))
end

# ── Entry point ───────────────────────────────────────────────────────────────

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "benchmark_timings")
    with_logging(log_path) do
        run_benchmark(ARGS)
    end
    return 0
end

function run_benchmark(ARGS)
    opts = parse_arguments(ARGS)

    println("Code:        $_CODE")
    println("Device:      ", _USE_GPU ? "GPU ($(CUDA.name(CUDA.device())))" : "CPU")
    println("Threads:     $(Threads.nthreads())")
    println("BLAS threads:$(LinearAlgebra.BLAS.get_num_threads())")
    println("Systems:     $(opts[:systems])")
    println("Losses:      $(opts[:losses])")
    _CODE == :trotter && println("Exponentials:$(opts[:exponentials])")
    println("Reps:        $(opts[:reps]) (after $(opts[:warmup]) warm-up call(s))")
    println("U index:     $(opts[:u_index])")
    println("Magnitude:   $(opts[:magnitude])")
    println("Output CSV:  $(opts[:out])")

    io = open_csv(opts[:out])
    try
        _CODE == :trotter ? benchmark_trotter(io, opts) : benchmark_exact(io, opts)
    finally
        close(io)
    end

    println("\nWrote timings to $(opts[:out])")
    return 0
end

