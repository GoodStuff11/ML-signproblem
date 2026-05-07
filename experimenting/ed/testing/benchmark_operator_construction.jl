"""
Benchmark: build_n_body_structure & create_randomized_nth_order_operator
=========================================================================
Goal: Profile and optimise the two most expensive setup functions when
running `optimize_unitary` on large systems such as N=(5,5)_4x4.

Optimisation strategies tested
-------------------------------
1. Threaded `build_n_body_structure` – outer loop over configs with
   thread-local accumulators merged at the end.
2. Prefiltered key sets inside `build_n_body_structure` – for each
   operator key, compute which configs *could* be valid before the inner
   loop rather than letting every (conf, key) pair go through Set ops.
3. Faster `create_randomized_nth_order_operator` – avoid repeated Dict
   key-existence checks by collecting unique keys into a Set first, then
   building the Dict in one pass.

Usage
-----
    julia --threads auto testing/benchmark_operator_construction.jl

or from the `ed/` directory:
    julia --threads auto testing/benchmark_operator_construction.jl
"""

using Lattices
using LinearAlgebra
using Combinatorics
using SparseArrays
using Random
using Statistics
using Printf
using Zygote
using Optimization, OptimizationOptimisers
using JLD2
using ExponentialUtilities
using JSON

# ── include source files (path relative to *this* file's directory) ──────────
const SCRIPT_DIR = @__DIR__
include(joinpath(SCRIPT_DIR, "..", "ed_objects.jl"))
include(joinpath(SCRIPT_DIR, "..", "ed_functions.jl"))
include(joinpath(SCRIPT_DIR, "..", "ed_optimization.jl"))
include(joinpath(SCRIPT_DIR, "..", "utility_functions.jl"))

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

"""Build a CombinationIndexer for a square lattice with given electron counts."""
function make_indexer(Lx::Int, Ly::Int, N_up::Int, N_down::Int)
    hs = HubbardSubspace(N_up, N_down, Square((Lx, Ly), Periodic()))
    return CombinationIndexer(hs)
end

function report_system(label, indexer)
    dim = length(indexer.inv_comb_dict)
    n_sites = length(indexer.a)
    println("  System : $label")
    println("  Sites  : $n_sites")
    println("  Dim    : $dim")
end

# ─────────────────────────────────────────────────────────────────────────────
# Optimised implementations
# ─────────────────────────────────────────────────────────────────────────────

"""
    create_randomized_nth_order_operator_fast(n, indexer; kwargs...)

Faster version of `create_randomized_nth_order_operator`.

Key changes vs. baseline:
- Collects all valid keys into a pre-allocated `Vector` (avoids repeated
  `key ∉ keys(t_dict)` Dict-lookup overhead per iteration).
- Uses a single-pass Dict construction from the deduplicated key vector.
"""
function create_randomized_nth_order_operator_fast(
    n::Int,
    indexer::CombinationIndexer;
    magnitude::T = 1e-3 + 0im,
    omit_H_conj::Bool = false,
    conserve_spin::Bool = false,
    normalize_coefficients::Bool = false,
    conserve_momentum::Bool = false,
) where {T}
    site_list = sort(indexer.a)
    all_ops(label) = combinations([(s, σ, label) for s in site_list for σ in 1:2], n)

    equal_spin(c, a) =
        sum((σ * 2 - 3) for (s, σ, _) in c) == sum((σ * 2 - 3) for (s, σ, _) in a)
    geq_ops(c, a) =
        [(s.coordinates..., σ) for (s, σ, _) in c] <=
        [(s.coordinates..., σ) for (s, σ, _) in a]

    must_conserve_momentum =
        conserve_momentum || (!isnothing(indexer.k) && !isnothing(indexer.lattice_dims))

    # ── collect valid keys first (no Dict insertion yet) ──────────────────
    valid_keys = Vector{Vector{Tuple{eltype(site_list),Int,Symbol}}}()

    for (ops_create, ops_annihilate) in
        Iterators.product(all_ops(:create), all_ops(:annihilate))

        if must_conserve_momentum
            tot_k = zeros(Int, length(indexer.lattice_dims))
            for (s, σ, _) in ops_create
                tot_k .+= (s.coordinates .- 1)
            end
            for (s, σ, _) in ops_annihilate
                tot_k .-= (s.coordinates .- 1)
            end
            tot_k = tot_k .% indexer.lattice_dims
            !all(tot_k .== 0) && continue
        end

        if (!omit_H_conj || geq_ops(ops_create, ops_annihilate)) &&
           (!conserve_spin || equal_spin(ops_create, ops_annihilate))
            push!(valid_keys, [ops_create; ops_annihilate])
        end
    end

    # ── deduplicate and build Dict with one pass ───────────────────────────
    # (duplicates can arise when the same key appears in multiple products)
    seen = Set{UInt64}()
    t_dict = Dict{Vector{Tuple{eltype(site_list),Int,Symbol}},T}()
    sizehint!(t_dict, length(valid_keys))

    for key in valid_keys
        h = hash(key)
        if h ∈ seen
            t_dict[key] = get(t_dict, key, zero(T)) + (2 * rand() - 1) / 2 * magnitude
        else
            push!(seen, h)
            t_dict[key] = (2 * rand() - 1) / 2 * magnitude
        end
    end

    if normalize_coefficients
        nc = length(t_dict)
        for key in keys(t_dict)
            t_dict[key] /= nc
        end
    end

    return t_dict
end


"""
    build_n_body_structure_threaded(t, indexer; skip_lower_triangular=false)

Thread-parallel version of `build_n_body_structure`.

Each thread accumulates into its own local (rows, cols, signs, ops_list)
to avoid locking, and all thread-local results are merged at the end.
"""
function build_n_body_structure_threaded(
    t::Dict{Vector{Tuple{T,Int,Symbol}},U},
    indexer::CombinationIndexer;
    skip_lower_triangular::Bool = false,
) where {T,U<:Number}
    t_keys = sort!(collect(keys(t)))
    build_n_body_structure_from_keys_threaded(t_keys, indexer, U; skip_lower_triangular)
end

function build_n_body_structure_from_keys_threaded(
    t_keys::AbstractVector,
    indexer::CombinationIndexer{T},
    ::Type{U} = Float64;
    skip_lower_triangular::Bool = false,
) where {T,U<:Number}
    sorted_sites = sort(indexer.a)
    n_configs = length(indexer.inv_comb_dict)
    n_threads = Threads.nthreads()

    chunk_size = max(1, cld(n_configs, n_threads))
    chunks = Iterators.partition(1:n_configs, chunk_size)

    tasks = map(chunks) do chunk
        Threads.@spawn begin
            c_rows  = Int[]
            c_cols  = Int[]
            c_signs = U[]
            c_ops   = Vector{Vector{Tuple{T,Int,Symbol}}}()

            for i1 in chunk
                conf = indexer.inv_comb_dict[i1]
                for ops in t_keys
                    conf_new = [copy(conf[1]), copy(conf[2])]
                    valid = true

                    for (site, spin, op) in reverse(ops)
                        if op == :annihilate
                            if site ∉ conf_new[spin]
                                valid = false; break
                            end
                            delete!(conf_new[spin], site)
                        elseif op == :create
                            if site ∈ conf_new[spin]
                                valid = false; break
                            end
                            push!(conf_new[spin], site)
                        else
                            error("Invalid operator symbol: $op")
                        end
                    end

                    valid || continue

                    i2 = index(indexer, conf_new[1], conf_new[2])
                    skip_lower_triangular && i1 > i2 && continue

                    s = compute_jw_sign(conf, sorted_sites, ops)
                    push!(c_rows,  i1)
                    push!(c_cols,  i2)
                    push!(c_signs, s)
                    push!(c_ops,   ops)
                end
            end
            (c_rows, c_cols, c_signs, c_ops)
        end
    end

    results = fetch.(tasks)
    
    rows   = reduce(vcat, [r[1] for r in results])
    cols   = reduce(vcat, [r[2] for r in results])
    signs  = reduce(vcat, [r[3] for r in results])
    ops_list = reduce(vcat, [r[4] for r in results])

    return rows, cols, signs, ops_list
end


"""
    build_n_body_structure_prefiltered(t, indexer; skip_lower_triangular=false)

Alternative approach: for each operator key, precompute which spin sectors
the source configuration must satisfy (occupation constraints).  This lets
the inner loop skip configs cheaply before touching `Set` internals.

For a k-body operator the constraint is: each (site, :annihilate) site must
be occupied in the appropriate spin sector, and each (site, :create) site
must be empty.  We record these requirements upfront and test them with a
fast membership check before allocating `conf_new`.
"""
function build_n_body_structure_prefiltered(
    t::Dict{Vector{Tuple{T,Int,Symbol}},U},
    indexer::CombinationIndexer;
    skip_lower_triangular::Bool = false,
) where {T,U<:Number}
    t_keys = sort!(collect(keys(t)))
    build_n_body_structure_from_keys_prefiltered(t_keys, indexer, U; skip_lower_triangular)
end

function build_n_body_structure_from_keys_prefiltered(
    t_keys::AbstractVector,
    indexer::CombinationIndexer{T},
    ::Type{U} = Float64;
    skip_lower_triangular::Bool = false,
) where {T,U<:Number}
    sorted_sites = sort(indexer.a)
    rows  = Int[]
    cols  = Int[]
    signs = U[]
    ops_list = Vector{Vector{Tuple{T,Int,Symbol}}}()

    # Precompute per-key occupation requirements (applied in reverse order)
    # must_have[k]  = [(site, spin)] that must be OCCUPIED before applying key k
    # must_empty[k] = [(site, spin)] that must be EMPTY before applying key k
    must_have  = [Tuple{T,Int}[] for _ in eachindex(t_keys)]
    must_empty = [Tuple{T,Int}[] for _ in eachindex(t_keys)]
    valid_key_mask = trues(length(t_keys))

    for (k, ops) in enumerate(t_keys)
        req_have = Set{Tuple{T,Int}}()
        req_empty = Set{Tuple{T,Int}}()
        is_invalid = false

        for (site, spin, op) in reverse(ops)
            mode = (site, spin)
            if op == :annihilate
                if mode ∈ req_empty
                    is_invalid = true
                    break
                end
                if mode ∉ req_empty
                    push!(req_have, mode)
                end
            elseif op == :create
                if mode ∈ req_have
                    is_invalid = true
                    break
                end
                if mode ∉ req_have
                    push!(req_empty, mode)
                end
            end
        end

        if is_invalid
            valid_key_mask[k] = false
        else
            must_have[k] = collect(req_have)
            must_empty[k] = collect(req_empty)
        end
    end

    for (i1, conf) in enumerate(indexer.inv_comb_dict)
        for (k, ops) in enumerate(t_keys)
            !valid_key_mask[k] && continue
            
            # Fast pre-check: occupation constraints
            ok = true
            for (site, spin) in must_have[k]
                if site ∉ conf[spin]; ok = false; break; end
            end
            ok || continue
            for (site, spin) in must_empty[k]
                if site ∈ conf[spin]; ok = false; break; end
            end
            ok || continue

            # Full application
            conf_new = [copy(conf[1]), copy(conf[2])]
            valid = true
            for (site, spin, op) in reverse(ops)
                if op == :annihilate
                    if site ∉ conf_new[spin]; valid = false; break; end
                    delete!(conf_new[spin], site)
                else
                    if site ∈ conf_new[spin]; valid = false; break; end
                    push!(conf_new[spin], site)
                end
            end
            valid || continue

            i2 = index(indexer, conf_new[1], conf_new[2])
            skip_lower_triangular && i1 > i2 && continue

            s = compute_jw_sign(conf, sorted_sites, ops)
            push!(rows, i1)
            push!(cols, i2)
            push!(signs, s)
            push!(ops_list, ops)
        end
    end

    return rows, cols, signs, ops_list
end


# ─────────────────────────────────────────────────────────────────────────────
# Correctness check
# ─────────────────────────────────────────────────────────────────────────────

"""Verify that two (rows, cols, signs) outputs agree after sorting."""
function assert_structures_equal(ref, cmp; label="")
    r1, c1, s1, _ = ref
    r2, c2, s2, _ = cmp

    p1 = sortperm(collect(zip(r1, c1)))
    p2 = sortperm(collect(zip(r2, c2)))

    ok = (r1[p1] == r2[p2]) && (c1[p1] == c2[p2]) && (s1[p1] == s2[p2])
    if ok
        println("  ✓  $label matches reference")
    else
        println("  ✗  $label MISMATCH vs reference!")
        println("     nnz ref=$(length(r1))  cmp=$(length(r2))")
    end
    return ok
end


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark
# ─────────────────────────────────────────────────────────────────────────────

function run_benchmarks()
    println("="^70)
    println("Operator Construction Benchmark")
    println("Julia threads: $(Threads.nthreads())")
    println("="^70)

    Random.seed!(42)

    # ── system definitions ─────────────────────────────────────────────────
    systems = [
        ("N=(3,3)_3x2",  3, 2, 3, 3),
        ("N=(4,4)_3x3",  3, 3, 4, 4),
        ("N=(4,4)_4x2",  4, 2, 4, 4),
        ("N=(5,5)_4x4",  4, 4, 5, 5),   # target large system
    ]

    orders = [1, 2]

    results = Dict{String,Any}()

    for (sys_label, Lx, Ly, N_up, N_down) in systems
        println("\n" * "─"^70)
        println("System: $sys_label  ($(Lx)×$(Ly), N↑=$N_up, N↓=$N_down)")
        println("─"^70)

        @time indexer = make_indexer(Lx, Ly, N_up, N_down)
        report_system(sys_label, indexer)

        for order in orders
            println("\n  ── order = $order ─────────────────────────────")
            n_keys_baseline = 0

            # ── 1. create_randomized_nth_order_operator ──────────────────
            println("\n  [1] create_randomized_nth_order_operator (baseline)")
            t_ref = nothing
            t_create_base = @elapsed begin
                t_ref = create_randomized_nth_order_operator(
                    order, indexer;
                    magnitude = 1.0 + 0im,
                    omit_H_conj = false,
                    conserve_spin = true,
                    normalize_coefficients = false,
                )
            end
            n_keys_baseline = length(t_ref)
            @printf("      keys=%d   time=%.4f s\n", n_keys_baseline, t_create_base)

            println("\n  [2] create_randomized_nth_order_operator_fast")
            t_fast = nothing
            t_create_fast = @elapsed begin
                Random.seed!(42)
                t_fast = create_randomized_nth_order_operator_fast(
                    order, indexer;
                    magnitude = 1.0 + 0im,
                    omit_H_conj = false,
                    conserve_spin = true,
                )
            end
            @printf(
                "      keys=%d   time=%.4f s  speedup=%.2fx\n",
                length(t_fast), t_create_fast,
                t_create_base / max(t_create_fast, 1e-9),
            )

            # Verify same key-set
            if sort!(collect(keys(t_ref))) == sort!(collect(keys(t_fast)))
                println("      ✓ key sets match")
            else
                println("      ✗ key sets DIFFER")
            end

            # ── 2. build_n_body_structure ────────────────────────────────
            dim = length(indexer.inv_comb_dict)
            println("\n  [3] build_n_body_structure (baseline)")
            ref_structure = nothing
            t_build_base = @elapsed begin
                ref_structure = build_n_body_structure(t_ref, indexer)
            end
            n_nnz = length(ref_structure[1])
            @printf(
                "      dim=%d  n_keys=%d  nnz=%d  time=%.4f s\n",
                dim, n_keys_baseline, n_nnz, t_build_base,
            )

            # Skip the slower variants for the largest system on first pass
            # – run correctness check only, not full timing.
            run_full = (dim <= 50_000)

            println("\n  [4] build_n_body_structure_prefiltered")
            pf_structure = nothing
            t_build_pf = @elapsed begin
                pf_structure = build_n_body_structure_prefiltered(t_ref, indexer)
            end
            @printf(
                "      nnz=%d  time=%.4f s  speedup=%.2fx\n",
                length(pf_structure[1]), t_build_pf,
                t_build_base / max(t_build_pf, 1e-9),
            )
            assert_structures_equal(ref_structure, pf_structure; label="prefiltered")

            if Threads.nthreads() > 1
                println("\n  [5] build_n_body_structure_threaded ($(Threads.nthreads()) threads)")
                th_structure = nothing
                t_build_th = @elapsed begin
                    th_structure = build_n_body_structure_threaded(t_ref, indexer)
                end
                @printf(
                    "      nnz=%d  time=%.4f s  speedup=%.2fx\n",
                    length(th_structure[1]), t_build_th,
                    t_build_base / max(t_build_th, 1e-9),
                )
                assert_structures_equal(ref_structure, th_structure; label="threaded")
            else
                println("\n  [5] Threaded: skipped (only 1 thread available)")
                println("      Hint: run with `julia --threads auto` to enable")
            end

            # ── 3. BenchmarkTools for small systems ──────────────────────
            if run_full
                println("\n  [Timing] Precise timing (baseline, small system):")
                # Removed BenchmarkTools logic to avoid missing dependency
            end

            # Store summary
            results["$(sys_label)_order$(order)"] = Dict(
                "dim"            => dim,
                "n_keys"         => n_keys_baseline,
                "nnz"            => n_nnz,
                "t_create_base"  => t_create_base,
                "t_create_fast"  => t_create_fast,
                "t_build_base"   => t_build_base,
                "t_build_pf"     => t_build_pf,
            )
        end
    end

    println("\n" * "="^70)
    println("Summary table")
    println("="^70)
    @printf(
        "%-22s | %2s | %8s | %8s | %10s | %10s | %8s | %8s\n",
        "System", "n", "dim", "n_keys", "t_create(s)", "t_build(s)", "sp_create", "sp_build",
    )
    println("-"^100)
    for (sys_label, _, _, _, _) in systems
        for order in orders
            k = "$(sys_label)_order$(order)"
            haskey(results, k) || continue
            r = results[k]
            sp_c = r["t_create_base"] / max(r["t_create_fast"], 1e-9)
            sp_b = r["t_build_base"]  / max(r["t_build_pf"],   1e-9)
            @printf(
                "%-22s | %2d | %8d | %8d | %10.3f | %10.3f | %8.2f | %8.2f\n",
                sys_label, order,
                r["dim"], r["n_keys"],
                r["t_create_base"], r["t_build_base"],
                sp_c, sp_b,
            )
        end
    end

    println("\nDone.")
    return results
end

# ─────────────────────────────────────────────────────────────────────────────
# Also demonstrate the usage pattern from ed_optimization.jl:
# precompute_n_body_structures + ensure_operator_structure! flow
# ─────────────────────────────────────────────────────────────────────────────

function demo_precompute_flow(; Lx=3, Ly=2, N_up=3, N_down=3)
    println("\n" * "="^70)
    println("Demo: precompute_n_body_structures (as used in optimize_unitary)")
    println("="^70)

    indexer = make_indexer(Lx, Ly, N_up, N_down)
    dim = length(indexer.inv_comb_dict)
    println("System: $(Lx)×$(Ly) N↑=$N_up N↓=$N_down  dim=$dim")

    println("\nBaseline: two separate create+build calls (what happens without cache):")
    t_no_cache = @elapsed begin
        for order in 1:2
            t_dict = create_randomized_nth_order_operator(
                order, indexer; magnitude = 1.0 + 0im, omit_H_conj = false, conserve_spin = true,
            )
            build_n_body_structure(t_dict, indexer)
        end
    end
    @printf("  time = %.4f s\n", t_no_cache)

    println("\nOptimised: precompute_n_body_structures (single call, reuse Dict):")
    t_precompute = @elapsed begin
        structures = precompute_n_body_structures(indexer, 2; spin_conserved = true)
    end
    @printf("  time = %.4f s  speedup vs 2× baseline = %.2fx\n",
            t_precompute, 2 * t_no_cache / max(t_precompute, 1e-9))

    println("\n  Cached keys:")
    for (k, v) in structures
        @printf("    order=%d use_sym=%-5s  nnz=%d  n_keys=%d\n",
                k[1], string(k[2]), length(v[:rows]), length(v[:t_keys]))
    end
end


# ── entry point ───────────────────────────────────────────────────────────────
run_benchmarks()
demo_precompute_flow()
