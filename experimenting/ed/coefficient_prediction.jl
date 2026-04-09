ENV["GKSwstype"] = "nul"   # headless rendering (no display required)
using Lattices: AbstractLattice, Coordinate
using SparseArrays
using Plots
using Random
using JLD2
using Flux
using CUDA   # GPU support; if unavailable set use_gpu=false in to_device()

include("ed_objects.jl")
include("ed_functions.jl")
include("utility_functions.jl")


# ─────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────

"""
    load_folder_header(electrons, system_size; file_label="")

Load per-folder metadata and shared labels.
Returns `(folder, e_metadata, dim, labels)`.
"""
function load_folder_header(electrons, system_size; file_label="")
    folder = "data/N=$(electrons)_$(system_size[1])x$(system_size[2])$file_label"
    e_metadata = load_saved_dict(joinpath(folder, "meta_data_and_E.jld2"))
    dim = [parse(Int, x) for x in split(e_metadata["meta_data"]["sites"], "x")]
    shared = load_saved_dict(joinpath(folder,
        "unitary_map_energy_symmetry=false_N=$(electrons)_shared.jld2"))
    labels = shared["coefficient_labels"][2]
    return folder, e_metadata, dim, labels
end

"""
    load_u_coefficients(folder, electrons, u_idx, e_metadata)

Load coefficients and U value for a single unitary index.
Returns `(coefficients, U_value)`.
"""
function load_u_coefficients(folder, electrons, u_idx, e_metadata)
    U_value = e_metadata["meta_data"]["U_values"][u_idx]
    dic = load_saved_dict(joinpath(folder,
        "unitary_map_energy_symmetry=false_N=$(electrons)_u_$(u_idx).jld2"))
    return dic["coefficients"][2], U_value
end

"""
    load_data(electrons, system_size, u_indices; file_label="")

Load data for all `u_indices` in one folder.
Returns a `Vector` of `(coefficients, labels, dim, U_value)` tuples.
"""
function load_data(electrons, system_size, u_indices; file_label="")
    folder, e_metadata, dim, labels = load_folder_header(
        electrons, system_size; file_label)
    u_vec  = collect(u_indices)
    result = Vector{Tuple}(undef, length(u_vec))
    # Each iteration reads a distinct file → thread-safe
    Threads.@threads for i in eachindex(u_vec)
        coefficients, U_value = load_u_coefficients(folder, electrons, u_vec[i], e_metadata)
        result[i] = (coefficients, labels, dim, U_value)
    end
    return result
end


# ─────────────────────────────────────────────────────────────
#  Input normalization
# ─────────────────────────────────────────────────────────────

"""
    label_to_k(label, dim) -> (x1, x2, x3, x4)

Convert a scattering label to four normalized Float32 vectors [kx, ky, σ] ∈ [-1,1].
  - k = (n-1)*2π/dim  →  k/π - 1
  - spin {1,2}  →  {-1,+1}
"""
function label_to_k(label, dim::Vector{Int})
    function norm_x(term, d)
        k = 2π ./ d .* (collect(term[1].coordinates) .- 1)
        return Float32[k ./ π .- 1f0..., Float32(2*(term[2]-1) - 1)]
    end
    return norm_x(label[1], dim), norm_x(label[2], dim),
           norm_x(label[3], dim), norm_x(label[4], dim)
end

normalize_U(U, U_max)          = Float32(2 * U / U_max - 1)
normalize_dim(dim, dim_max)    = Float32.(2 .* dim ./ dim_max .- 1)

"""
    normalize_electrons(electrons, n_max) -> Vector{Float32}

Normalize (n_up, n_down) electron counts to [-1, 1] using `n_max` as the ceiling.
"""
normalize_electrons(el, n_max) = Float32[2*el[1]/n_max - 1, 2*el[2]/n_max - 1]


# ─────────────────────────────────────────────────────────────
#  Featurization
# ─────────────────────────────────────────────────────────────

"""
    featurize_entry(coefficients, labels, dim, U_value, electrons; kwargs...)

Convert one raw data entry to normalized input matrices and target vector.
Returns `(X1, X2, X3, X4, Ctx, Y)` (each with N columns, one per label).

Context vector always contains U; optionally appends:
  - lattice `dim`       (2 values) when `include_dim=true`
  - electron counts     (2 values: n_up, n_down) when `include_electrons=true`
"""
function featurize_entry(coefficients, labels, dim, U_value, electrons;
                         U_max,
                         include_dim       = false, dim_max         = nothing,
                         include_electrons = false)
    ctx_base = Float32[normalize_U(U_value, U_max)]

    if include_dim
        @assert !isnothing(dim_max) "dim_max required when include_dim=true"
        append!(ctx_base, normalize_dim(dim, dim_max))
    end
    if include_electrons
        append!(ctx_base, normalize_electrons(electrons, prod(dim) ÷ 2))
    end

    n_ctx = length(ctx_base)
    N     = length(coefficients)

    X1  = Matrix{Float32}(undef, 3, N)
    X2  = Matrix{Float32}(undef, 3, N)
    X3  = Matrix{Float32}(undef, 3, N)
    X4  = Matrix{Float32}(undef, 3, N)
    Ctx = repeat(reshape(ctx_base, n_ctx, 1), 1, N)
    Y   = Float32.(coefficients)

    for j in eachindex(labels)
        x1, x2, x3, x4 = label_to_k(labels[j], dim)
        X1[:, j] = x1;  X2[:, j] = x2
        X3[:, j] = x3;  X4[:, j] = x4
    end
    return X1, X2, X3, X4, Ctx, Y
end

"""
    load_all_data(folder_specs) -> Vector of (coefficients, labels, dim, U_value, electrons)

Load raw data from every folder spec by calling `load_data` for each spec.
`load_data` parallelises internally over u_indices with `Threads.@threads`.
Returns a flat vector of raw entries ready for featurization.
"""
function load_all_data(folder_specs)
    all_raw = Any[]
    for (electrons, system_size, u_indices, file_label) in folder_specs
        for (coefficients, labels, dim, U_value) in
                load_data(electrons, system_size, u_indices; file_label)
            push!(all_raw, (coefficients, labels, dim, U_value, electrons))
        end
    end
    return all_raw
end

"""
    featurize_all(raw_data; U_max, include_dim, dim_max, include_electrons)

Parallel featurization of pre-loaded raw data into normalized Float32 matrices.
Pre-computes column offsets so each thread writes to a non-overlapping slice
of the output arrays — no locks required.
Returns `(X1, X2, X3, X4, Ctx, Y)`.
"""
function featurize_all(raw_data;
                       U_max,
                       include_dim       = false, dim_max         = nothing,
                       include_electrons = false)
    n_ctx    = 1 + (include_dim ? 2 : 0) + (include_electrons ? 2 : 0)
    n_per    = [length(d[1]) for d in raw_data]
    offsets  = [0; cumsum(n_per)]
    N        = offsets[end]

    X1  = Matrix{Float32}(undef, 3,     N)
    X2  = Matrix{Float32}(undef, 3,     N)
    X3  = Matrix{Float32}(undef, 3,     N)
    X4  = Matrix{Float32}(undef, 3,     N)
    Ctx = Matrix{Float32}(undef, n_ctx, N)
    Y   = Vector{Float32}(undef, N)

    Threads.@threads for ti in eachindex(raw_data)
        (coefficients, labels, dim, U_value, electrons) = raw_data[ti]
        ctx_base = Float32[normalize_U(U_value, U_max)]
        include_dim       && append!(ctx_base, normalize_dim(dim, dim_max))
        include_electrons && append!(ctx_base, normalize_electrons(electrons, prod(dim) ÷ 2))

        col_start = offsets[ti] + 1
        for j in eachindex(coefficients)
            col = col_start + j - 1
            x1, x2, x3, x4 = label_to_k(labels[j], dim)
            X1[:,col]=x1; X2[:,col]=x2; X3[:,col]=x3; X4[:,col]=x4
            Ctx[:,col] = ctx_base
            Y[col] = Float32(coefficients[j])
        end
    end

    return X1, X2, X3, X4, Ctx, Y
end

"""
    prepare_dataset(folder_specs; U_max, include_dim, dim_max, include_electrons)

Convenience wrapper: loads all raw data then featurizes in parallel.
Equivalent to `featurize_all(load_all_data(folder_specs); kwargs...)`.
"""
function prepare_dataset(folder_specs;
                         U_max,
                         include_dim       = false, dim_max         = nothing,
                         include_electrons = false)
    return featurize_all(load_all_data(folder_specs);
                         U_max, include_dim, dim_max, include_electrons)
end



# ─────────────────────────────────────────────────────────────
#  Two-stage model
# ─────────────────────────────────────────────────────────────

"""
    build_two_stage_mlp(; base_hidden, embed_dim, context_hidden,
                          include_dim, include_electrons)

Stage 1 — base MLP:    ℝ¹²    → ℝᵉ   (scattering embedding)
Stage 2 — context MLP: ℝᵉ⁺ⁿᶜ → ℝ¹   (embedding + context → scalar)

Context size:  1 (U)  +  2 (dim, if include_dim)  +  2 (electrons, if include_electrons)
Returns a `NamedTuple (base, context)`.
"""
function build_two_stage_mlp(;
        base_hidden       = [128, 128],
        embed_dim         = 64,
        context_hidden    = [64, 32],
        include_dim       = false,
        include_electrons = false)

    function dense_stack(in_dim, hidden_dims, out_dim, act=tanh)
        layers = Any[]
        d = in_dim
        for h in hidden_dims
            push!(layers, Dense(d => h, act)); d = h
        end
        push!(layers, Dense(d => out_dim))
        Chain(layers...)
    end

    n_context = 1 + (include_dim ? 2 : 0) + (include_electrons ? 2 : 0)
    base_mlp  = dense_stack(12, base_hidden, embed_dim)
    ctx_mlp   = dense_stack(embed_dim + n_context, context_hidden, 1)
    return (base = base_mlp, context = ctx_mlp)
end


# ─────────────────────────────────────────────────────────────
#  Symmetrized forward pass
# ─────────────────────────────────────────────────────────────

"""
    compute_F_batch(model, X1, X2, X3, X4, Ctx) -> Vector

Evaluate the symmetrized F for a batch of B samples in one parallel forward pass.
Works on CPU or GPU arrays transparently.

F = [f(1234) - f(2134) - f(1243) + f(2143)
    +f(3412) - f(4312) - f(3421) + f(4321)] / 8
"""
function compute_F_batch(model, X1, X2, X3, X4, Ctx)
    B = size(X1, 2)

    all_scat = hcat(
        vcat(X1,X2,X3,X4), vcat(X2,X1,X3,X4),
        vcat(X1,X2,X4,X3), vcat(X2,X1,X4,X3),
        vcat(X3,X4,X1,X2), vcat(X4,X3,X1,X2),
        vcat(X3,X4,X2,X1), vcat(X4,X3,X2,X1),
    )  # 12 × (8B)

    emb     = model.base(all_scat)                          # embed_dim × (8B)
    Ctx_8   = repeat(Ctx, 1, 8)                             # n_ctx × (8B)
    outputs = vec(model.context(vcat(emb, Ctx_8)))          # length 8B

    o = reshape(outputs, B, 8)   # B × 8

    # Element-wise reduction (GPU-safe — no CPU-resident constant mixed in)
    return (o[:,1] .- o[:,2] .- o[:,3] .+ o[:,4] .+
            o[:,5] .- o[:,6] .- o[:,7] .+ o[:,8]) ./ 8f0
end


# ─────────────────────────────────────────────────────────────
#  GPU utilities
# ─────────────────────────────────────────────────────────────

"""
    to_device(x; use_gpu=true)

Move array or model to GPU if CUDA is functional and `use_gpu=true`, else CPU.
"""
to_device(x; use_gpu=true) =
    (use_gpu && CUDA.functional()) ? Flux.gpu(x) : Flux.cpu(x)


# ─────────────────────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────────────────────

"""
    train_mlp!(model, X1, X2, X3, X4, Ctx, Y; n_epochs, batch_size, lr, verbose)

Train the two-stage model using MSE loss.
Move model and data to GPU first via `to_device` for GPU acceleration.
Returns a vector of per-epoch mean losses.
"""
function train_mlp!(model, X1, X2, X3, X4, Ctx, Y;
                    n_epochs=200, batch_size=256, lr=1e-3, verbose=true)
    N = size(X1, 2)
    @assert N == size(X2,2) == size(X3,2) == size(X4,2) == size(Ctx,2) == length(Y)

    opt_state    = Flux.setup(Adam(lr), model)
    loss_history = Float64[]

    for epoch in 1:n_epochs
        epoch_loss = 0f0
        n_batches  = 0
        idx_perm   = randperm(N)
        for start in 1:batch_size:N
            idx  = idx_perm[start:min(start+batch_size-1, N)]
            bX1  = X1[:,idx]; bX2 = X2[:,idx]
            bX3  = X3[:,idx]; bX4 = X4[:,idx]
            bCtx = Ctx[:,idx]; bY = Y[idx]

            loss, grads = Flux.withgradient(model) do m
                Flux.mse(compute_F_batch(m, bX1, bX2, bX3, bX4, bCtx), bY)
            end
            Flux.update!(opt_state, model, grads[1])
            epoch_loss += loss;  n_batches += 1
        end

        mean_loss = epoch_loss / n_batches
        push!(loss_history, mean_loss)
        if verbose && (epoch==1 || epoch%10==0)
            @info "Epoch $epoch/$n_epochs  loss=$(round(mean_loss; sigdigits=5))"
        end
    end
    return loss_history
end


# ─────────────────────────────────────────────────────────────
#  Prediction
# ─────────────────────────────────────────────────────────────

"""
    predict_F(model, label, dim, U_value, electrons, U_max; kwargs...)

Evaluate F for a single label. Pass the same `include_*` flags used during training.
`model` should be on CPU (`Flux.cpu(model)`) before calling.
"""
function predict_F(model, label, dim, U_value, electrons, U_max;
                   include_dim=false,       dim_max=nothing,
                   include_electrons=false)
    x1, x2, x3, x4 = label_to_k(label, dim)
    ctx = Float32[normalize_U(U_value, U_max)]
    include_dim       && append!(ctx, normalize_dim(dim, dim_max))
    include_electrons && append!(ctx, normalize_electrons(electrons, prod(dim) ÷ 2))
    return only(compute_F_batch(model,
        reshape(x1,3,1), reshape(x2,3,1),
        reshape(x3,3,1), reshape(x4,3,1),
        reshape(ctx, length(ctx), 1)))
end


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────

function (@main)(ARGS)
    folder_specs = [
        ((3,3), [3,2], 2:52, "_2"),
        ((3,3), [3,2], 2:52, "_3"),
        ((4,4), [3,3], 2:52, "_2"),
        ((4,4), [4,2], 2:52, "_2"),
        ((4,5), [3,3], 2:52, ""),
        ((4,5), [3,3], 2:52, "_3"),
    ]
    U_max = 10.0
    epochs = 200
    X1, X2, X3, X4, Ctx, Y = prepare_dataset(folder_specs; U_max)
    println("Dataset: $(length(Y)) samples, $(Threads.nthreads()) threads, GPU=$(CUDA.functional())")

    model = build_two_stage_mlp(base_hidden=[128,128], embed_dim=64, context_hidden=[64,32])
    model, X1, X2, X3, X4, Ctx, Y = to_device.((model, X1, X2, X3, X4, Ctx, Y))
    loss_history = train_mlp!(model, X1, X2, X3, X4, Ctx, Y; n_epochs=epochs, batch_size=512, lr=1e-3)
    p = plot(loss_history; xlabel="Epoch", ylabel="MSE", yscale=:log10, label="Coefficient Prediction (U only)")

    X1, X2, X3, X4, Ctx, Y = prepare_dataset(folder_specs; U_max, include_electrons=true)
    model = build_two_stage_mlp(base_hidden=[128,128], embed_dim=64, context_hidden=[64,32], include_electrons=true)
    model, X1, X2, X3, X4, Ctx, Y = to_device.((model, X1, X2, X3, X4, Ctx, Y))
    loss_history = train_mlp!(model, X1, X2, X3, X4, Ctx, Y; n_epochs=epochs, batch_size=512, lr=1e-3)
    plot!(p, loss_history; xlabel="Epoch", ylabel="MSE", yscale=:log10, label="Coefficient Prediction (+electron density)")

    X1, X2, X3, X4, Ctx, Y = prepare_dataset(folder_specs; U_max, include_dim=true, include_electrons=true,dim_max=4)
    model = build_two_stage_mlp(base_hidden=[128,128], embed_dim=64, context_hidden=[64,32], include_electrons=true, include_dim=true)
    model, X1, X2, X3, X4, Ctx, Y = to_device.((model, X1, X2, X3, X4, Ctx, Y))
    loss_history = train_mlp!(model, X1, X2, X3, X4, Ctx, Y; n_epochs=epochs, batch_size=512, lr=1e-3)
    plot!(p, loss_history; xlabel="Epoch", ylabel="MSE", yscale=:log10, label="Coefficient Prediction (+electron density + dim)")

    savefig(p, "loss_curve_mlp.png")
    return 0
end