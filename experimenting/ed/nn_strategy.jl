using Flux

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
end

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

    interps = Dict{Int,CoefficientInterpolator}()
    for (u_idx, coeffs) in sort(collect(coefficients_by_u), by=first)
        c = filter_spin ? coeffs[valid] : coeffs
        println("Building Legendre interpolator for u_idx=$(u_idx) (U=$(U_values[u_idx]))...")
        interps[u_idx] = get_interpolator(c, labels_f, bd, dim; tol)
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
    featurize_all(raw_data; U_max, include_dim, dim_max, include_electrons)

Parallel featurization of pre-loaded raw data into normalized Float32 matrices.
Y values are normalized per-entry (per U-value) to unit RMS.  The log₁₀ of each
entry's RMS is stored in `Y_log_scale` so the scale head can be supervised directly.
Returns `(X1, X2, X3, X4, Ctx, Y, Y_log_scale)`.
"""
function featurize_all(raw_data;
    U_max,
    include_dim=false, dim_max=nothing,
    include_electrons=false)
    n_feat = length(raw_data[1][3]) + 1   # d + 1 (k components + spin)
    n_ctx = 1 + (include_dim ? 2 : 0) + (include_electrons ? 2 : 0)
    n_per = [length(d[1]) for d in raw_data]
    offsets = [0; cumsum(n_per)]
    N = offsets[end]

    X1 = Matrix{Float32}(undef, n_feat, N)
    X2 = Matrix{Float32}(undef, n_feat, N)
    X3 = Matrix{Float32}(undef, n_feat, N)
    X4 = Matrix{Float32}(undef, n_feat, N)
    Ctx = Matrix{Float32}(undef, n_ctx, N)
    Y = Vector{Float32}(undef, N)   # per-entry normalized coefficients
    Y_log_scale = Vector{Float32}(undef, N)   # log10(RMS) target for the scale head

    Threads.@threads for ti in eachindex(raw_data)
        (coefficients, labels, dim, U_value, electrons) = raw_data[ti]
        ctx_base = Float32[normalize_U(U_value, U_max)]
        include_dim && append!(ctx_base, normalize_dim(dim, dim_max))
        include_electrons && append!(ctx_base, normalize_electrons(electrons, prod(dim) ÷ 2))

        # Per-entry (per-U) RMS normalization.
        c32 = Float32.(coefficients)
        entry_rms = sqrt(sum(c32 .^ 2) / length(c32))
        entry_log_scale = log10(entry_rms)
        println(entry_rms)

        col_start = offsets[ti] + 1
        for j in eachindex(coefficients)
            col = col_start + j - 1
            k1, k2, k3, k4, spins = label_to_k(labels[j], dim)
            X1[:, col] = nn_feature(k1, spins[1])
            X2[:, col] = nn_feature(k2, spins[2])
            X3[:, col] = nn_feature(k3, spins[3])
            X4[:, col] = nn_feature(k4, spins[4])
            Ctx[:, col] = ctx_base
            Y[col] = c32[j] / entry_rms
            Y_log_scale[col] = entry_log_scale
        end
    end
    println("Y_log_scale range: [$(minimum(Y_log_scale)), $(maximum(Y_log_scale))]")
    return X1, X2, X3, X4, Ctx, Y, Y_log_scale
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
    return dic["coefficients"][2], U_value
end

function load_data_nn(electrons, system_size, u_indices; file_label="")
    folder, e_metadata, dim, labels = load_folder_header_nn(
        electrons, system_size; file_label)
    u_vec = collect(u_indices)
    result = Vector{Tuple}(undef, length(u_vec))
    Threads.@threads for i in eachindex(u_vec)
        coefficients, U_value = load_u_coefficients_nn(
            folder, electrons, u_vec[i], e_metadata)
        result[i] = (coefficients, labels, dim, U_value)
    end
    return result
end

function load_all_data_nn(folder_specs)
    all_raw = Any[]
    for (electrons, system_size, u_indices, file_label) in folder_specs
        for (coefficients, labels, dim, U_value) in
            load_data_nn(electrons, system_size, u_indices; file_label)
            push!(all_raw, (coefficients, labels, dim, U_value, electrons))
        end
    end
    return all_raw
end

function prepare_dataset_nn(folder_specs; U_max, include_dim=false,
    dim_max=nothing, include_electrons=false)
    return featurize_all(load_all_data_nn(folder_specs);
        U_max, include_dim, dim_max, include_electrons)
end


# ─────────────────────────────────────────────────────────────
#  Two-stage MLP model
# ─────────────────────────────────────────────────────────────

"""
    build_two_stage_mlp(; feat_dim, base_hidden, embed_dim, context_hidden,
                          scale_hidden, n_context, use_scale_head)

Two-mode architecture controlled by `use_scale_head`:

**Scaled mode** (`use_scale_head=true`, default):
- Stage 1 — base MLP:   ℝᶠ⁴  → ℝᵉ   (scattering embedding)
- Stage 2 — shape MLP:  ℝᵉ⁺ⁿᶜ → ℝ¹  (embedding + context → unit-normalized shape)
- Stage 3 — scale MLP:  ℝⁿᶜ   → ℝ¹  (context only → log₁₀-scale)
Returns `(base=..., context=..., scale=...)`.

**Unscaled mode** (`use_scale_head=false`):
- Stage 1 — base MLP:   ℝᶠ⁴  → ℝᵉ
- Stage 2 — context MLP: ℝᵉ⁺ⁿᶜ → ℝ¹  (single combined output; original architecture)
Returns `(base=..., context=...)`.
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
    compute_F_batch(model, X1, X2, X3, X4, Ctx; use_scale_head) -> (output, log_scale_or_nothing)

Symmetrized forward pass over a batch.

**Scaled mode** (`use_scale_head=true`):
- `output`    : antisymmetrized unit-normalized shape prediction.
- `log_scale` : permutation-invariant log₁₀-scale from the context-only scale MLP.
Full prediction = 10^(log_scale) .* output.

**Unscaled mode** (`use_scale_head=false`):
- `output`    : single antisymmetrized combined prediction (global-normalized).
- second return is `nothing`.
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
               raw_w[:, 5] .- raw_w[:, 6] .- raw_w[:, 7] .+ raw_w[:, 8])

    if use_scale_head
        log_scale = vec(model.scale(Ctx))   # (B,) — context-only scale head
        return antisym, log_scale
    else
        return antisym, nothing
    end
end

function to_device(x; use_gpu=true)
    if use_gpu && CUDA.functional()
        try
            return Flux.gpu(x)
        catch e
            @warn "GPU transfer failed ($(e)), falling back to CPU."
        end
    end
    return Flux.cpu(x)
end

"""
    train_mlp!(model, X1, X2, X3, X4, Ctx, Y, Y_log_scale; use_scale_head, ...)

**Scaled mode** (`use_scale_head=true`):
  joint_loss = MSE( 10^(clamp(log_scale − Y_log_scale, −3,3)) ⋅ unscaled,  Y_normalized )
  aux_loss   = MSE( log_scale,  Y_log_scale )
  total      = (joint_loss + scale_loss_weight ⋅ aux_loss) / (1 + scale_loss_weight)

**Unscaled mode** (`use_scale_head=false`):
  total = MSE( antisym_output,  Y_global_normalized )
  (identical to the original training loss; Y and Y_log_scale are ignored for the
   scale head, only Y is used)

Returns per-epoch mean losses.
"""
function train_mlp!(model, X1, X2, X3, X4, Ctx, Y, Y_log_scale;
    n_epochs=200, batch_size=256, lr=1e-3,
    scale_loss_weight=3f0, use_scale_head::Bool=true, verbose=true)
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

            loss, grads = Flux.withgradient(model) do m
                output, log_scale = compute_F_batch(m, bX1, bX2, bX3, bX4, bCtx;
                    use_scale_head)
                if use_scale_head
                    log_ratio = clamp.(log_scale .- bY_ls, -3f0, 3f0)
                    pred = (10f0 .^ log_ratio) .* output
                    joint_loss = Flux.mse(pred, bY)
                    aux_loss = Flux.mse(log_scale, bY_ls)
                    (joint_loss + scale_loss_weight * aux_loss) / (1f0 + scale_loss_weight)
                else
                    Flux.mse(output, bY)   # plain MSE on global-normalized targets
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

A trained MLP that predicts coefficients for any scattering label.
Supports two modes, selected at construction time via `use_scale_head`:

- **Scaled** (`use_scale_head=true`): 3-head model (base, shape, scale).  The scale
  head predicts `log₁₀(Y_scale)` from context alone; prediction = 10^(log_scale) × shape.
- **Unscaled** (`use_scale_head=false`): 2-head model (base, context).  Single combined
  output trained on Y
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
    NeuralNetStrategy(folder_specs; use_scale_head=true, U_max=10.0, kwargs...)

Train a `NeuralNetStrategy` on the datasets described by `folder_specs`.
Each entry in `folder_specs` is `(electrons, system_size, u_indices, file_label)`.

Set `use_scale_head=false` to use the original simpler 2-head (base + context) architecture
trained on globally-normalized Y with plain MSE loss.  Set `use_scale_head=true` (default)
for the 3-head scaled architecture.

Keyword arguments forwarded to `build_two_stage_mlp` and `train_mlp!`:
  - `include_dim`, `dim_max`, `include_electrons`
  - `n_epochs`, `batch_size`, `lr`, `use_gpu`
  - `base_hidden`, `embed_dim`, `context_hidden`, `scale_hidden`
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
    use_scale_head::Bool=true)

    X1, X2, X3, X4, Ctx, Y, Y_log_scale = prepare_dataset_nn(folder_specs;
        U_max, include_dim, dim_max, include_electrons)

    n_ctx = size(Ctx, 1)
    f_dim = size(X1, 1)   # d + 1

    model = build_two_stage_mlp(;
        feat_dim=f_dim, base_hidden, embed_dim,
        context_hidden, scale_hidden, n_context=n_ctx,
        use_scale_head)

    _dev(x) = to_device(x; use_gpu)
    model_d = _dev(model)
    X1d, X2d, X3d, X4d, Ctxd, Yd, Yd_ls = _dev(X1), _dev(X2), _dev(X3), _dev(X4),
    _dev(Ctx), _dev(Y), _dev(Y_log_scale)

    train_mlp!(model_d, X1d, X2d, X3d, X4d, Ctxd, Yd, Yd_ls;
        n_epochs, batch_size, lr, use_scale_head)

    return NeuralNetStrategy(
        Flux.cpu(model_d),
        U_max, include_dim, include_electrons, dim_max,
        use_scale_head)
end

"""
    interpolate_coefficients(s::NeuralNetStrategy, ctx::NeuralNetContext, labels, dim)

Predict coefficients for each entry in `labels` on lattice `dim` using the trained MLP.

- **Scaled mode**: returns `10^(log_scale) × shape_output`.
- **Unscaled mode**: returns `antisym_output`.
"""
function interpolate_coefficients(s::NeuralNetStrategy, ctx::NeuralNetContext,
    labels, dim::Vector{Int})
    return map(labels) do label
        k1, k2, k3, k4, spins = label_to_k(label, dim)
        x1 = reshape(nn_feature(k1, spins[1]), :, 1)
        x2 = reshape(nn_feature(k2, spins[2]), :, 1)
        x3 = reshape(nn_feature(k3, spins[3]), :, 1)
        x4 = reshape(nn_feature(k4, spins[4]), :, 1)

        feature_ctx = Float32[normalize_U(ctx.U_value, ctx.U_max)]
        s.include_dim && append!(feature_ctx, normalize_dim(dim, s.dim_max))
        s.include_electrons && append!(feature_ctx, normalize_electrons(ctx.electrons, prod(dim) ÷ 2))
        ctx_m = reshape(feature_ctx, :, 1)

        output, log_scale = compute_F_batch(s.model, x1, x2, x3, x4, ctx_m;
            use_scale_head=s.use_scale_head)
        if s.use_scale_head
            (10.0f0^only(log_scale)) * only(output)
        else
            only(output)
        end
    end
end
