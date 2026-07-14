using Flux
using JLD2

# ─────────────────────────────────────────────────────────────
#  Abstract strategy interface
# ─────────────────────────────────────────────────────────────

"""
    CoefficientStrategy

Abstract supertype for all coefficient interpolation/prediction strategies.
Every concrete subtype must implement:

    interpolate_coefficients(strategy, labels, dim::Vector{Int}; kwargs...) -> Vector

which maps a list of scattering labels on a lattice `dim` to predicted coefficient values.
"""
abstract type CoefficientStrategy end

"""
    interpolate_coefficients(strategy::CoefficientStrategy, labels, dim; kwargs...) -> Vector

Unified entry point for coefficient prediction. Dispatch to the concrete strategy.
"""
function interpolate_coefficients end

"""
    LegendreContext

Holds the physical context needed to evaluate a `LegendreStrategy` at a specific point in
parameter space.  Mirrors `NeuralNetContext` so that both strategies share the same
calling convention.

Fields
------
- `U_value` : Hubbard interaction value for this prediction.  The strategy will select
              the interpolator whose training U value is closest to this.
"""
struct LegendreContext
    U_value::Float64
end

"""
    NeuralNetContext

Holds the physical inputs needed to evaluate a trained `NeuralNetStrategy` at a specific
point in parameter space.  These are deliberately *not* part of the strategy itself because
they describe an observation (a particular U value and electron filling), not the network
architecture.

Fields
------
- `U_value`  : Hubbard interaction value for this prediction.
- `electrons` : Tuple `(n_up, n_down)` for the system being predicted.
- `U_max`    : Normalization constant used during training (must match the trained strategy).
"""
struct NeuralNetContext
    U_value::Float64
    electrons::Tuple{Int,Int}
    U_max::Float64
    k_steps::Int
    l_step::Int
    k_max::Int
end

NeuralNetContext(U_value, electrons, U_max) = NeuralNetContext(U_value, electrons, U_max, 1, 1, 10)


struct CoefficientInterpolator{T}
    interpolated_c::Vector{T}
    basis_degrees::Vector{Vector{Int}}
    d::Int
end

# ─────────────────────────────────────────────────────────────
#  Strategy 1: Legendre polynomial projection (per-U pipeline)
# ─────────────────────────────────────────────────────────────

"""
    LegendreStrategy <: CoefficientStrategy

Wraps a collection of `CoefficientInterpolator`s — one per training U value — as a
`CoefficientStrategy`.  At prediction time a `LegendreContext` supplies the target
U value; the strategy selects the interpolator whose U value is closest.

Fields
------
- `interps`   : Dict mapping `u_idx` → `CoefficientInterpolator`.
- `U_values`  : Vector of U values, indexed the same as `interps` keys.
"""
struct LegendreStrategy <: CoefficientStrategy
    interps::Dict{Int,CoefficientInterpolator}
    U_values::Vector{Float64}
end

"""
    LegendreStrategy(coefficients_by_u, labels, dim, U_values; tol=1e-10, filter_spin=true)

Build a `LegendreStrategy` from a Dict of per-U coefficient vectors (as returned by
`load_data_coefficients`).  One `CoefficientInterpolator` is fitted for each `u_idx`
present in `coefficients_by_u`.  Optionally filters to symmetry-unique labels
(first spin == 1) before fitting.
"""
function LegendreStrategy(coefficients_by_u::Dict{Int,<:Any}, labels, dim::Vector{Int},
    U_values; tol::Float64=1e-10, filter_spin::Bool=true)

    # Pre-compute the shared spin filter and basis degrees once.
    if filter_spin
        valid = [label_to_k(l, dim)[5][1] == 1 for l in labels]
        labels_f = labels[valid]
    else
        valid = trues(length(labels))
        labels_f = labels
    end
    bd = generate_basis_degrees(length(labels_f), dim; labels=labels_f)

    # Compute P and its pseudoinverse once since they only depend on labels, basis degrees, and dim.
    println("Building Legendre interpolator matrix P and computing its pseudoinverse once...")
    P = build_P_matrix(labels_f, bd, dim)
    println("Rank of P: $(rank(P))  dimension of P: $(size(P))")
    F = svd(P)
    cond_P = iszero(F.S[end]) ? Inf : F.S[1] / F.S[end]
    println("Info: Matrix P condition number: ", cond_P)

    S_inv = [s > tol * F.S[1] ? 1 / s : 0.0 for s in F.S]
    P_inv = F.V * Diagonal(S_inv) * F.U'

    interps = Dict{Int,CoefficientInterpolator}()
    for (u_idx, coeffs) in sort(collect(coefficients_by_u), by=first)
        c = filter_spin ? coeffs[valid] : coeffs
        interpolated_c = P_inv * c
        interps[u_idx] = CoefficientInterpolator{eltype(interpolated_c)}(interpolated_c, bd, length(dim))
    end

    return LegendreStrategy(interps, collect(Float64, U_values))
end

"""
    interpolate_coefficients(s::LegendreStrategy, ctx::LegendreContext, labels, dim) -> Vector

Evaluate the Legendre interpolator for the U value in `ctx` at each entry of `labels`
on lattice `dim`.  Selects the stored interpolator whose training U value is closest to
`ctx.U_value`.
"""
function interpolate_coefficients(s::LegendreStrategy, ctx::LegendreContext,
    labels, dim::Vector{Int})
    # Find the u_idx whose stored U value is closest to the requested one.
    best_u_idx = argmin(u_idx -> abs(s.U_values[u_idx] - ctx.U_value), keys(s.interps))
    interp = s.interps[best_u_idx]
    return interpolate_coefficients(labels, dim, interp)   # existing low-level method
end

"""
    label_to_k(label, dim::Vector{Int})

Converts a discrete lattice scattering label containing coordinates and spins into
multi-dimensional continuous wavevectors. It scales the discrete site index `(n - 1)`
to an exact momentum mapping `k = (n - 1) * 2π / dim`.
Returns the 4 wavevectors (`k1`, `k2`, `k3`, `k4`) and their integer `spins`.
"""
function label_to_k(label, dim::Vector{Int})
    k1 = 2 * pi ./ dim .* (collect(label[1][1].coordinates) .- 1)
    k2 = 2 * pi ./ dim .* (collect(label[2][1].coordinates) .- 1)
    k3 = 2 * pi ./ dim .* (collect(label[3][1].coordinates) .- 1)
    k4 = 2 * pi ./ dim .* (collect(label[4][1].coordinates) .- 1)
    spins = [label[1][2], label[2][2], label[3][2], label[4][2]]
    return k1, k2, k3, k4, spins
end


# ─────────────────────────────────────────────────────────────
#  NN helpers — featurization (uses shared label_to_k)
# ─────────────────────────────────────────────────────────────

"""
    nn_feature(k::Vector{Float64}, spin::Int) -> Vector{Float32}

Encode one leg of a scattering event as a normalized Float32 feature vector.
Maps k ∈ [0, 2π] → k/π - 1 ∈ [-1, 1], and spin ∈ {1,2} → {-1f0, +1f0}.
"""
nn_feature(k::Vector{Float64}, spin::Int) =
    Float32[k ./ π .- 1.0..., Float32(2 * (spin - 1) - 1)]

normalize_U(U, U_max) = Float32(2 * U / U_max - 1)
normalize_dim(d, d_max) = Float32.(2 .* d ./ d_max .- 1)
normalize_electrons(el, n_max) = Float32[2*el[1]/n_max-1, 2*el[2]/n_max-1]

"""
    featurize_entry(coefficients, labels, dim, U_value, electrons; kwargs...)

Convert one raw data entry into normalized Float32 matrices `(X1, X2, X3, X4, Ctx, Y)`.
Each matrix has `length(coefficients)` columns (one per label).

Context vector always contains `U`; optionally appends lattice `dim` (2 values) and
electron counts (2 values) when `include_dim` / `include_electrons` are `true`.
"""
function featurize_entry(coefficients, labels, dim, U_value, electrons;
    U_max,
    include_dim=false, dim_max=nothing,
    include_electrons=false)
    ctx_base = Float32[normalize_U(U_value, U_max)]
    if include_dim
        @assert !isnothing(dim_max) "dim_max required when include_dim=true"
        append!(ctx_base, normalize_dim(dim, dim_max))
    end
    if include_electrons
        append!(ctx_base, normalize_electrons(electrons, prod(dim) ÷ 2))
    end

    n_ctx = length(ctx_base)
    N = length(coefficients)

    X1 = Matrix{Float32}(undef, length(dim) + 1, N)
    X2 = Matrix{Float32}(undef, length(dim) + 1, N)
    X3 = Matrix{Float32}(undef, length(dim) + 1, N)
    X4 = Matrix{Float32}(undef, length(dim) + 1, N)
    Ctx = repeat(reshape(ctx_base, n_ctx, 1), 1, N)
    Y = Float32.(coefficients)

    for j in eachindex(labels)
        k1, k2, k3, k4, spins = label_to_k(labels[j], dim)
        X1[:, j] = nn_feature(k1, spins[1])
        X2[:, j] = nn_feature(k2, spins[2])
        X3[:, j] = nn_feature(k3, spins[3])
        X4[:, j] = nn_feature(k4, spins[4])
    end
    return X1, X2, X3, X4, Ctx, Y
end

"""
    featurize_all(raw_data; U_max, include_dim, dim_max, include_electrons, use_scale_head)

Parallel featurization of pre-loaded raw data into normalized Float32 matrices.

When `use_scale_head=false` (original mode): Y contains raw Float32 coefficients.
When `use_scale_head=true`  (scaled mode):   Y is per-entry RMS-normalized; Y_log_scale
  holds log10(RMS) per sample for scale-head supervision.

Always returns `(X1, X2, X3, X4, Ctx, Y, Y_log_scale)` — in original mode Y_log_scale
is a zero vector (unused by the training loop).
"""
function featurize_all(raw_data;
    U_max,
    include_dim=false, dim_max=nothing,
    include_electrons=false,
    use_scale_head::Bool=true,
    include_trotter::Bool=false,
    k_max::Int=10)
    n_feat = length(raw_data[1][3]) + 1   # d + 1 (k components + spin)
    n_ctx = 1 + (include_dim ? 2 : 0) + (include_electrons ? 2 : 0) + (include_trotter ? 2 : 0)
    n_per = [length(d[1]) for d in raw_data]
    offsets = [0; cumsum(n_per)]
    N = offsets[end]

    X1 = Matrix{Float32}(undef, n_feat, N)
    X2 = Matrix{Float32}(undef, n_feat, N)
    X3 = Matrix{Float32}(undef, n_feat, N)
    X4 = Matrix{Float32}(undef, n_feat, N)
    Ctx = Matrix{Float32}(undef, n_ctx, N)
    Y = Vector{Float32}(undef, N)
    Y_log_scale = zeros(Float32, N)   # only filled when use_scale_head=true
    Y_opt_loss = zeros(Float32, N)

    @safe_threads for ti in eachindex(raw_data)
        entry = raw_data[ti]
        coefficients = entry[1]
        labels = entry[2]
        dim = entry[3]
        U_value = entry[4]
        electrons = entry[5]
        opt_loss = entry[6]
        k = length(entry) >= 8 ? entry[7] : 1
        l = length(entry) >= 8 ? entry[8] : 1
        
        ctx_base = if use_scale_head
            val = Float32(log10(max(U_value, 1e-4)))
            log_min = -4.0f0
            log_max = Float32(log10(U_max))
            Float32[2f0 * (val - log_min) / (log_max - log_min) - 1f0]
        else
            Float32[normalize_U(U_value, U_max)]
        end
        include_dim && append!(ctx_base, normalize_dim(dim, dim_max))
        include_electrons && append!(ctx_base, normalize_electrons(electrons, prod(dim) ÷ 2))

        if include_trotter
            k_norm = Float32(2.0 * k / k_max - 1.0)
            l_norm = Float32(2.0 * l / k - 1.0)
            push!(ctx_base, k_norm)
            push!(ctx_base, l_norm)
        end

        c32 = Float32.(coefficients)

        # Per-entry scale (used only in scaled mode).
        entry_rms = sqrt(sum(c32 .^ 2) / length(c32))
        entry_log_scale = (use_scale_head && entry_rms > 0) ? log10(entry_rms) : 0.0f0
        inv_rms = (use_scale_head && entry_rms > 0) ? 1.0f0 / entry_rms : 1.0f0

        col_start = offsets[ti] + 1
        for j in eachindex(coefficients)
            col = col_start + j - 1
            k1, k2, k3, k4, spins = label_to_k(labels[j], dim)
            X1[:, col] = nn_feature(k1, spins[1])
            X2[:, col] = nn_feature(k2, spins[2])
            X3[:, col] = nn_feature(k3, spins[3])
            X4[:, col] = nn_feature(k4, spins[4])
            Ctx[:, col] = ctx_base
            Y[col] = c32[j] * inv_rms   # raw when inv_rms=1, normalized otherwise
            Y_log_scale[col] = entry_log_scale
            Y_opt_loss[col] = opt_loss
        end
    end
    if use_scale_head
        println("Y_log_scale range: [$(minimum(Y_log_scale)), $(maximum(Y_log_scale))]")
    end
    return X1, X2, X3, X4, Ctx, Y, Y_log_scale, Y_opt_loss
end

# ─────────────────────────────────────────────────────────────
#  Data loading helpers (mirrored from coefficient_prediction.jl)
# ─────────────────────────────────────────────────────────────

function load_folder_header_nn(electrons, system_size; file_label="")
    folder = "data/N=$(electrons)_$(system_size[1])x$(system_size[2])$file_label"
    e_metadata = load_saved_dict(joinpath(folder, "meta_data_and_E.jld2"))
    dim = [parse(Int, x) for x in split(e_metadata["meta_data"]["sites"], "x")]
    shared = load_saved_dict(joinpath(folder,
        "unitary_map_energy_symmetry=false_N=$(electrons)_shared.jld2"))
    labels = shared["coefficient_labels"][2]
    return folder, e_metadata, dim, labels
end

function load_u_coefficients_nn(folder, electrons, u_idx, e_metadata)
    U_value = e_metadata["meta_data"]["U_values"][u_idx]
    dic = load_saved_dict(joinpath(folder,
        "unitary_map_energy_symmetry=false_N=$(electrons)_u_$(u_idx).jld2"))
    opt_loss = (haskey(dic, "metrics") && haskey(dic["metrics"], "loss")) ? Float32(dic["metrics"]["loss"][end]) : 0.0f32
    return dic["coefficients"][2], U_value, opt_loss
end

function load_data_nn(electrons, system_size, u_indices; file_label="")
    folder, e_metadata, dim, labels = load_folder_header_nn(
        electrons, system_size; file_label)
    result = Tuple[]
    for ui in u_indices
        u_file = joinpath(folder, "unitary_map_energy_symmetry=false_N=$(electrons)_u_$(ui).jld2")
        if !isfile(u_file)
            continue  # skip missing indices (e.g. folders with fewer u-files than the range)
        end
        coefficients, U_value, opt_loss = load_u_coefficients_nn(folder, electrons, ui, e_metadata)
        if !isnothing(coefficients)
            push!(result, (coefficients, labels, dim, U_value, opt_loss))
        end
    end
    if isempty(result)
        @warn "No u-files found for N=$(electrons) $(system_size[1])x$(system_size[2])$(file_label) in range $(collect(u_indices)[[1,end]])"
    end
    return result
end

function fgate_to_label(g, Lvec)
    # Dynamically resolve functions to handle different inclusion scopes
    tf = @isdefined(Trotter) ? Trotter.TamFermion : Main.Trotter.TamFermion
    tl = @isdefined(Trotter) ? Trotter.TamLib : Main.Trotter.TamLib

    N_sites = prod(Lvec)
    cre_up_sites = tf.int2occ(g.cre_up, N_sites)
    cre_dn_sites = tf.int2occ(g.cre_dn, N_sites)
    ann_up_sites = tf.int2occ(g.ann_up, N_sites)
    ann_dn_sites = tf.int2occ(g.ann_dn, N_sites)

    function make_ops(up_sites, dn_sites, op_type)
        ops = Any[]
        for s in up_sites
            coords0 = tl.unravel_c(s - 1, Tuple(Lvec))
            push!(ops, (Coordinate((coords0 .+ 1)...), 1, op_type))
        end
        for s in dn_sites
            coords0 = tl.unravel_c(s - 1, Tuple(Lvec))
            push!(ops, (Coordinate((coords0 .+ 1)...), 2, op_type))
        end
        return ops
    end

    ops_create = make_ops(cre_up_sites, cre_dn_sites, :create)
    ops_annihilate = make_ops(ann_up_sites, ann_dn_sites, :annihilate)

    site_idx(c) = tl.ravel_c(collect(c.coordinates) .- 1, Tuple(Lvec))
    
    sort!(ops_create, by = op -> (site_idx(op[1]), op[2]))
    sort!(ops_annihilate, by = op -> (site_idx(op[1]), op[2]))

    result = Vector{Tuple{Coordinate{length(Lvec),Int},Int,Symbol}}()
    for op in ops_create
        push!(result, (op[1], op[2], op[3]))
    end
    for op in ops_annihilate
        push!(result, (op[1], op[2], op[3]))
    end
    return result
end

function load_data_trotter_nn(electrons, system_size, u_indices; file_label="", loss_type=:overlap)
    folder = "data/N=$(electrons)_$(system_size[1])x$(system_size[2])$file_label"
    if !isdir(folder)
        @warn "Directory $folder not found."
        return Tuple[]
    end

    N_sites = prod(system_size)
    prefix = loss_type == :energy ? "trotter_N=$(N_sites)_loss_energy" : "trotter_N=$(N_sites)"

    meta_file = joinpath(folder, "meta_data_and_E.jld2")
    if !isfile(meta_file)
        @warn "Metadata file $meta_file not found in $folder."
        return Tuple[]
    end
    e_metadata = load_saved_dict(meta_file)
    U_values = e_metadata["meta_data"]["U_values"]

    shared_file = joinpath(folder, "$(prefix)_shared.jld2")
    if !isfile(shared_file)
        @warn "Shared Trotter file $shared_file not found in $folder."
        return Tuple[]
    end
    shared_dict = load_saved_dict(shared_file)
    gates = shared_dict["gates"]

    labels = [fgate_to_label(g, system_size) for g in gates]

    result = Tuple[]
    for ui in u_indices
        u_file = joinpath(folder, "$(prefix)_u_$(ui).jld2")
        if !isfile(u_file)
            continue
        end
        dic = load_saved_dict(u_file)
        coefficients = dic["coefficients"]
        if isnothing(coefficients)
            continue
        end
        U_value = U_values[ui]
        opt_loss = (haskey(dic, "metrics") && haskey(dic["metrics"], "loss")) ? Float32(dic["metrics"]["loss"][end]) : 0.0f32
        push!(result, (coefficients, labels, system_size, U_value, opt_loss))
    end
    return result
end

function load_all_data_nn(folder_specs; is_trotter=false, loss_type=:overlap)
    all_raw = Any[]
    for (electrons, system_size, u_indices, file_label) in folder_specs
        if is_trotter
            trotter_data = load_data_trotter_nn(electrons, system_size, u_indices; file_label, loss_type)
            for (coefficients, labels, dim, U_value, opt_loss) in trotter_data
                N_s = length(labels)
                k_steps = length(coefficients) ÷ N_s
                for l in 1:k_steps
                    coeff_slice = coefficients[((l-1)*N_s + 1):(l*N_s)]
                    push!(all_raw, (coeff_slice, labels, dim, U_value, electrons, opt_loss, k_steps, l))
                end
            end
        else
            for (coefficients, labels, dim, U_value, opt_loss) in
                load_data_nn(electrons, system_size, u_indices; file_label)
                push!(all_raw, (coefficients, labels, dim, U_value, electrons, opt_loss, 1, 1))
            end
        end
    end
    return all_raw
end

function prepare_dataset_nn(folder_specs; U_max, include_dim=false,
    dim_max=nothing, include_electrons=false, use_scale_head=true,
    is_trotter=false, loss_type=:overlap, k_max=10)
    raw_data = load_all_data_nn(folder_specs; is_trotter, loss_type)
    return featurize_all(raw_data;
        U_max, include_dim, dim_max, include_electrons, use_scale_head,
        include_trotter=is_trotter, k_max)
end


# ─────────────────────────────────────────────────────────────
#  Two-stage MLP model
# ─────────────────────────────────────────────────────────────

"""
    build_two_stage_mlp(; feat_dim, base_hidden, embed_dim, context_hidden,
                          scale_hidden, n_context, use_scale_head)

Two-mode architecture:

**Original / unscaled mode** (`use_scale_head=false`):
- base MLP:    feat_dim*4 -> embed_dim
- context MLP: embed_dim + n_context -> 1
Returns `(base=..., context=...)`.

**Scaled mode** (`use_scale_head=true`):
- base MLP:    feat_dim*4 -> embed_dim
- shape MLP:   embed_dim + n_context -> 1  (antisymmetric shape)
- scale MLP:   n_context -> 1             (log10-scale, context only)
Returns `(base=..., context=..., scale=...)`.
"""
function build_two_stage_mlp(;
    feat_dim=3,
    base_hidden=[128, 128],
    embed_dim=64,
    context_hidden=[64, 32],
    scale_hidden=[32, 16],
    n_context=1,
    use_scale_head::Bool=true)

    function dense_stack(in_dim, hidden_dims, out_dim, act=tanh)
        layers = Any[]
        d = in_dim
        for h in hidden_dims
            push!(layers, Dense(d => h, act))
            d = h
        end
        push!(layers, Dense(d => out_dim))
        Flux.Chain(layers...)
    end

    base_mlp = dense_stack(feat_dim * 4, base_hidden, embed_dim)
    ctx_mlp = dense_stack(embed_dim + n_context, context_hidden, 1)
    if use_scale_head
        scale_mlp = dense_stack(n_context, scale_hidden, 1)
        return (base=base_mlp, context=ctx_mlp, scale=scale_mlp)
    else
        return (base=base_mlp, context=ctx_mlp)
    end
end

"""
    compute_F_batch(model, X1, X2, X3, X4, Ctx; use_scale_head) -> (output, nothing_or_log_scale)

Symmetrized forward pass over a batch.

**Original mode** (`use_scale_head=false`):
  Returns `(output, nothing)`.

**Scaled mode** (`use_scale_head=true`):
  shape = antisymmetrize(context_head)
  log_scale = scale_head(Ctx)
  Returns `(shape, log_scale)`.
"""
function compute_F_batch(model, X1, X2, X3, X4, Ctx; use_scale_head::Bool=true)
    B = size(X1, 2)

    all_scat = hcat(
        vcat(X1, X2, X3, X4), vcat(X2, X1, X3, X4),
        vcat(X1, X2, X4, X3), vcat(X2, X1, X4, X3),
        vcat(X3, X4, X1, X2), vcat(X4, X3, X1, X2),
        vcat(X3, X4, X2, X1), vcat(X4, X3, X2, X1),
    )

    emb = model.base(all_scat)            # (embed_dim, B*8)
    Ctx_8 = repeat(Ctx, 1, 8)
    raw_w = model.context(vcat(emb, Ctx_8)) # (1, B*8)
    raw_w = reshape(raw_w[1, :], B, 8)

    # Antisymmetric combination (no /8 — keeps initial output magnitude O(0.5)).
    antisym = (raw_w[:, 1] .- raw_w[:, 2] .- raw_w[:, 3] .+ raw_w[:, 4] .+
               raw_w[:, 5] .- raw_w[:, 6] .- raw_w[:, 7] .+ raw_w[:, 8]) ./ 8

    if use_scale_head
        log_scale = vec(model.scale(Ctx))   # (B,) — context-only scale head
        return antisym, log_scale
    else
        return antisym, nothing
    end
end

function to_device(x; use_gpu=true)
    if use_gpu && @isdefined(CUDA) && CUDA.functional()
        try
            return Flux.fmap(CUDA.cu, x)
        catch e
            @warn "GPU transfer failed ($(e)), falling back to CPU."
        end
    end
    return Flux.fmap(Flux.cpu, x)
end

"""
    train_mlp!(model, X1, X2, X3, X4, Ctx, Y, Y_log_scale; use_scale_head, ...)

**Original mode** (`use_scale_head=false`):
  loss = MSE(compute_F_batch(m,...), Y)
  Identical to the architecture that previously achieved ~1e-5 loss.

**Scaled mode** (`use_scale_head=true`):
  joint_loss = MSE(10^clamp(log_scale - Y_log_scale,-3,3) * shape, Y_normalized)
  aux_loss   = MSE(log_scale, Y_log_scale)
  loss       = (joint_loss + scale_loss_weight * aux_loss) / (1 + scale_loss_weight)

Returns per-epoch mean losses.
"""
function train_mlp!(model, X1, X2, X3, X4, Ctx, Y, Y_log_scale, Y_opt_loss;
    n_epochs=200, batch_size=256, lr=1e-3,
    scale_loss_weight=3f0, use_scale_head::Bool=true, verbose=true,
    weighting_scheme::AbstractString="low_u")
    N = size(X1, 2)
    opt_state = Flux.setup(Flux.Adam(lr), model)
    loss_history = Float64[]

    for epoch in 1:n_epochs
        epoch_loss = 0f0
        n_batches = 0
        idx_perm = randperm(N)
        for start in 1:batch_size:N
            idx = idx_perm[start:min(start + batch_size - 1, N)]
            bX1 = X1[:, idx]
            bX2 = X2[:, idx]
            bX3 = X3[:, idx]
            bX4 = X4[:, idx]
            bCtx = Ctx[:, idx]
            bY = Y[idx]
            bY_ls = Y_log_scale[idx]
            bY_ol = Y_opt_loss[idx]

            loss, grads = Flux.withgradient(model) do m
                output, log_scale = compute_F_batch(m, bX1, bX2, bX3, bX4, bCtx; use_scale_head)
                if use_scale_head
                    # 1. Low-U / Loss Importance Weighting:
                    #    - Reconstruct log10(U) from log-normalized context input: log10(U) = 2.5 * (x_U + 1) - 4
                    #    - Weight samples exponentially or via monotonic function of optimized loss
                    #    - Normalize weights over batch such that mean(w) = 1.0
                    log_norm_U = bCtx[1, :]
                    log10_U = (log_norm_U .+ 1.0f0) .* 2.5f0 .- 4.0f0
                    w = if weighting_scheme == "low_u_mild"
                        # Gently favour small U: exp(-0.25 * log10(U)) = U^(-0.25)
                        exp.( -0.25f0 .* log10_U )
                    elseif weighting_scheme == "low_u"
                        exp.( -0.5f0 .* log10_U )
                    elseif weighting_scheme == "low_u_strong"
                        # Strongly favour small U: exp(-1.5 * log10(U)) = U^(-1.5)
                        exp.( -1.5f0 .* log10_U )
                    elseif weighting_scheme == "low_u_very_strong"
                        # Very aggressively favour small U: exp(-3.0 * log10(U)) = U^(-3.0)
                        exp.( -3.0f0 .* log10_U )
                    elseif weighting_scheme == "uniform"
                        log10_U .* 0.0f0 .+ 1.0f0
                    elseif weighting_scheme == "high_u"
                        exp.( 0.5f0 .* log10_U )
                    elseif weighting_scheme == "high_u_strong"
                        # Stronger version: exp(1.5 * log10(U)) = U^1.5, heavily prioritises large U
                        exp.( 1.5f0 .* log10_U )
                    elseif weighting_scheme == "focus_u8"
                        # Gaussian centred at log10(8) ≈ 0.903, σ ≈ 0.4 in log10 space
                        exp.( -((log10_U .- 0.903f0) .^ 2) ./ (2f0 * 0.4f0^2) )
                    elseif weighting_scheme == "loss_mild"
                        (-log10.(max.(bY_ol, 1f-12))) .^ 1.15f0
                    elseif weighting_scheme == "loss_std"
                        (-log10.(max.(bY_ol, 1f-12))) .^ 2.0f0
                    elseif weighting_scheme == "loss_power_mild"
                        (max.(bY_ol, 1f-12)) .^ -0.07f0
                    elseif weighting_scheme == "loss_power_std"
                        (max.(bY_ol, 1f-12)) .^ -0.12f0
                    elseif weighting_scheme == "loss_power_neg03"
                        (max.(bY_ol, 1f-12)) .^ -0.3f0
                    elseif weighting_scheme == "loss_power_neg04"
                        (max.(bY_ol, 1f-12)) .^ -0.4f0
                    elseif weighting_scheme == "loss_power_neg05"
                        (max.(bY_ol, 1f-12)) .^ -0.5f0
                    elseif weighting_scheme == "one_minus_loss_power_03"
                        (1.0f0 .- clamp.(bY_ol, 0f0, 1f0)) .^ 0.3f0
                    elseif weighting_scheme == "one_minus_loss_power_05"
                        (1.0f0 .- clamp.(bY_ol, 0f0, 1f0)) .^ 0.5f0
                    elseif weighting_scheme == "one_minus_loss_power_07"
                        (1.0f0 .- clamp.(bY_ol, 0f0, 1f0)) .^ 0.7f0
                    elseif weighting_scheme == "one_minus_loss_power_15"
                        (1.0f0 .- clamp.(bY_ol, 0f0, 1f0)) .^ 1.5f0
                    elseif weighting_scheme == "one_minus_loss_power_20"
                        (1.0f0 .- clamp.(bY_ol, 0f0, 1f0)) .^ 2.0f0
                    elseif weighting_scheme == "one_minus_loss_power_25"
                        (1.0f0 .- clamp.(bY_ol, 0f0, 1f0)) .^ 2.5f0
                    elseif weighting_scheme == "one_minus_loss_power_10"
                        (1.0f0 .- clamp.(bY_ol, 0f0, 1f0)) .^ 1.0f0
                    else
                        error("Unknown weighting_scheme: $weighting_scheme")
                    end
                    w_mean = sum(w) / length(w)
                    w = w ./ w_mean
                    
                    # 2. Joint and Auxiliary Loss Formulation:
                    #    - log_ratio = log10(scale_pred / scale_true)
                    #    - pred = (scale_pred / scale_true) * y_norm_pred, matching y_norm_true
                    #    - joint_loss = weighted_MSE( pred, y_norm_true )
                    #    - aux_loss   = weighted_MSE( log_scale_pred, log_scale_true )
                    log_ratio = clamp.(log_scale .- bY_ls, -3f0, 3f0)
                    pred = (10f0 .^ log_ratio) .* output
                    joint_loss = sum(w .* (pred .- bY) .^ 2) / length(bY)
                    aux_loss = sum(w .* (log_scale .- bY_ls) .^ 2) / length(bY_ls)
                    
                    # Combined objective scaled by scale_loss_weight
                    (joint_loss + scale_loss_weight * aux_loss) / (1f0 + scale_loss_weight)
                else
                    Flux.mse(output, bY)   # original plain MSE
                end
            end
            Flux.update!(opt_state, model, grads[1])
            epoch_loss += loss
            n_batches += 1
        end

        mean_loss = epoch_loss / n_batches
        push!(loss_history, mean_loss)
        if verbose && (epoch == 1 || epoch % 10 == 0)
            @info "Epoch $epoch/$n_epochs  loss=$(round(mean_loss; sigdigits=5))"
        end
    end
    return loss_history
end


# ─────────────────────────────────────────────────────────────
#  Strategy 2: Neural network
# ─────────────────────────────────────────────────────────────

"""
    NeuralNetStrategy <: CoefficientStrategy

Supports two modes via `use_scale_head`:
- `false` (original): sigmoid-scaled 2-head network, plain MSE on raw Y.
- `true`  (scaled):   3-head network with context-only log-scale prediction.
"""
struct NeuralNetStrategy <: CoefficientStrategy
    model               # Flux NamedTuple — stored on CPU
    U_max::Float64
    include_dim::Bool
    include_electrons::Bool
    dim_max::Union{Nothing,Int}
    use_scale_head::Bool
end

"""
    NeuralNetStrategy(folder_specs; use_scale_head=false, U_max=10.0, kwargs...)

Train a `NeuralNetStrategy`.  Default `use_scale_head=false` restores the original
architecture that achieved ~1e-5 loss.  Set `use_scale_head=true` for the experimental
scaled architecture.

Keyword arguments: `include_dim`, `dim_max`, `include_electrons`, `n_epochs`, `batch_size`,
`lr`, `use_gpu`, `base_hidden`, `embed_dim`, `context_hidden`, `scale_hidden`.
"""
function NeuralNetStrategy(folder_specs;
    U_max=10.0,
    include_dim=false,
    include_electrons=false,
    dim_max=nothing,
    n_epochs=200,
    batch_size=512,
    lr=1e-3,
    use_gpu=true,
    base_hidden=[128, 128],
    embed_dim=64,
    context_hidden=[64, 32],
    scale_hidden=[32, 16],
    use_scale_head::Bool=false,
    weighting_scheme::AbstractString="low_u",
    scale_loss_weight::Float32=3.0f0,
    is_trotter::Bool=false,
    loss_type::Symbol=:overlap,
    k_max::Int=10)

    X1, X2, X3, X4, Ctx, Y, Y_log_scale, Y_opt_loss = prepare_dataset_nn(folder_specs;
        U_max, include_dim, dim_max, include_electrons, use_scale_head,
        is_trotter, loss_type, k_max)

    n_ctx = size(Ctx, 1)
    f_dim = size(X1, 1)   # d + 1

    model = build_two_stage_mlp(;
        feat_dim=f_dim, base_hidden, embed_dim,
        context_hidden, scale_hidden, n_context=n_ctx,
        use_scale_head)

    _dev(x) = to_device(x; use_gpu)
    model_d = _dev(model)
    X1d, X2d, X3d, X4d, Ctxd, Yd, Yd_ls, Yd_ol = (
        _dev(X1), _dev(X2), _dev(X3), _dev(X4), _dev(Ctx), _dev(Y), _dev(Y_log_scale), _dev(Y_opt_loss))

    train_mlp!(model_d, X1d, X2d, X3d, X4d, Ctxd, Yd, Yd_ls, Yd_ol;
        n_epochs, batch_size, lr, use_scale_head, weighting_scheme, scale_loss_weight)

    return NeuralNetStrategy(
        Flux.cpu(model_d),
        U_max, include_dim, include_electrons, dim_max,
        use_scale_head)
end

"""
    interpolate_coefficients(s::NeuralNetStrategy, ctx::NeuralNetContext, labels, dim)

Predict coefficients using the trained MLP.

- Original mode (`use_scale_head=false`): returns the sigmoid-scaled network output directly.
- Scaled mode  (`use_scale_head=true`):  returns 10^(log_scale) * shape_output.
"""
function interpolate_coefficients(s::NeuralNetStrategy, ctx::NeuralNetContext,
    labels, dim::Vector{Int})
    
    # Check if the model expects Trotter features
    n_ctx_expected = 1 + (s.include_dim ? 2 : 0) + (s.include_electrons ? 2 : 0)
    n_ctx_actual = s.use_scale_head ? size(s.model.scale[1].weight, 2) : (size(s.model.context[1].weight, 2) - size(s.model.base[end].weight, 1))
    include_trotter = (n_ctx_actual == n_ctx_expected + 2)

    return map(labels) do label
        k1, k2, k3, k4, spins = label_to_k(label, dim)
        x1 = reshape(nn_feature(k1, spins[1]), :, 1)
        x2 = reshape(nn_feature(k2, spins[2]), :, 1)
        x3 = reshape(nn_feature(k3, spins[3]), :, 1)
        x4 = reshape(nn_feature(k4, spins[4]), :, 1)

        feature_ctx = if s.use_scale_head
            val = Float32(log10(max(ctx.U_value, 1e-4)))
            log_min = -4.0f0
            log_max = Float32(log10(ctx.U_max))
            Float32[2f0 * (val - log_min) / (log_max - log_min) - 1f0]
        else
            Float32[normalize_U(ctx.U_value, ctx.U_max)]
        end
        s.include_dim && append!(feature_ctx, normalize_dim(dim, s.dim_max))
        s.include_electrons && append!(feature_ctx, normalize_electrons(ctx.electrons, prod(dim) ÷ 2))
        
        if include_trotter
            k = :k_steps in fieldnames(typeof(ctx)) ? ctx.k_steps : 1
            l = :l_step in fieldnames(typeof(ctx)) ? ctx.l_step : 1
            k_max_val = :k_max in fieldnames(typeof(ctx)) ? ctx.k_max : 10
            
            k_norm = Float32(2.0 * k / k_max_val - 1.0)
            l_norm = Float32(2.0 * l / k - 1.0)
            push!(feature_ctx, k_norm)
            push!(feature_ctx, l_norm)
        end

        ctx_m = reshape(feature_ctx, :, 1)

        output, log_scale = compute_F_batch(s.model, x1, x2, x3, x4, ctx_m;
            use_scale_head=s.use_scale_head)
        if s.use_scale_head
            (10.0f0^only(log_scale)) * only(output)
        else
            only(output)   # original: sigmoid scaling is already inside compute_F_batch
        end
    end
end

"""
    save_neural_network(filepath::AbstractString, strategy::NeuralNetStrategy)

Save the trained `NeuralNetStrategy` to the specified file using JLD2.
"""
function save_neural_network(filepath::AbstractString, strategy::NeuralNetStrategy)
    dir = dirname(filepath)
    if !isempty(dir) && !isdir(dir)
        mkpath(dir)
    end
    jldsave(filepath; strategy=strategy)
    println("Saved neural network strategy to: $filepath")
end

"""
    load_neural_network(filepath::AbstractString) -> NeuralNetStrategy

Load a `NeuralNetStrategy` from the specified JLD2 file.
"""
function load_neural_network(filepath::AbstractString)
    actual_path = filepath
    if !isfile(actual_path)
        resolved = joinpath("trained_neural_networks", "trained_neural_network_$(filepath).jld2")
        if isfile(resolved)
            actual_path = resolved
        else
            error("Neural network strategy file not found at: $filepath (also tried $resolved)")
        end
    end
    data = load(actual_path)
    if !haskey(data, "strategy")
        error("JLD2 file at $actual_path does not contain a saved 'strategy'.")
    end
    strategy = data["strategy"]
    println("Loaded neural network strategy from: $actual_path")
    return strategy
end

"""
    get_or_train_neural_strategy(nn_filepath::AbstractString, folder_specs; kwargs...) -> NeuralNetStrategy

Check if `nn_filepath` exists. If it does, load and return the saved `NeuralNetStrategy`.
Otherwise, train a new `NeuralNetStrategy` using `folder_specs` and `kwargs`, save it to
`nn_filepath`, and return it.
"""
function get_or_train_neural_strategy(nn_filepath::AbstractString, folder_specs;
    U_max=10.0,
    include_dim=false,
    include_electrons=false,
    dim_max=nothing,
    n_epochs=200,
    batch_size=512,
    lr=1e-3,
    use_gpu=true,
    base_hidden=[128, 128],
    embed_dim=64,
    context_hidden=[64, 32],
    scale_hidden=[32, 16],
    use_scale_head::Bool=false,
    weighting_scheme::AbstractString="low_u",
    scale_loss_weight::Float32=3.0f0,
    is_trotter::Bool=false,
    loss_type::Symbol=:overlap,
    k_max::Int=10)

    if isfile(nn_filepath)
        println("Found existing neural network at $nn_filepath. Loading...")
        return load_neural_network(nn_filepath)
    else
        println("No existing neural network found at $nn_filepath. Training a new model...")
        println("Training NN with architecture parameters: base_hidden=$base_hidden, embed_dim=$embed_dim, context_hidden=$context_hidden, scale_hidden=$scale_hidden")
        strategy = NeuralNetStrategy(folder_specs;
            U_max=U_max,
            include_dim=include_dim,
            include_electrons=include_electrons,
            dim_max=dim_max,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            use_gpu=use_gpu,
            base_hidden=base_hidden,
            embed_dim=embed_dim,
            context_hidden=context_hidden,
            scale_hidden=scale_hidden,
            use_scale_head=use_scale_head,
            weighting_scheme=weighting_scheme,
            scale_loss_weight=scale_loss_weight,
            is_trotter=is_trotter,
            loss_type=loss_type,
            k_max=k_max
        )
        save_neural_network(nn_filepath, strategy)
        return strategy
    end
end


