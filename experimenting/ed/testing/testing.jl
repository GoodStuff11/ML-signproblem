using LinearAlgebra
using SparseArrays


# The number of components in a_i and M_i

DIM = 2
N = 1000 # Order of truncation
Hs_dim = 4

# Make up some coeffcients
v_1, v_2 = ones(Hs_dim), ones(Hs_dim)
M = [sparse(0.01 * rand(Hs_dim, Hs_dim)) for _ in 1:DIM]

# The objective function.
objective = (a) -> 1 - abs2(v_1' * exp(Matrix(sum(i -> a[i] * M[i], 1:DIM))) * v_2)

# The analytic derivative, truncated to N = 1000.
# This is a slow implementation prioritizing clarity.

function d_objective_sparse(a)
    # Type assertions for globals to ensure performance
    local M_typed::Vector{SparseMatrixCSC{Float64,Int}} = M
    local v1_typed::Vector{Float64} = v_1
    local v2_typed::Vector{Float64} = v_2
    local N_typed::Int = N
    local Hs_dim_typed::Int = Hs_dim
    local DIM_typed::Int = DIM

    invN = 1.0 / N_typed

    # Forward Pass
    # Store r[k] = A^(k-1) * v_1
    # We allocate N+1 slots. r[k] corresponds to A^(k-1) v_1
    r = Vector{Vector{Float64}}(undef, N_typed + 1)

    # We can pre-allocate the vectors in r to be contiguous in memory?
    # For now, separate allocations is fine unless N is huge.

    r[1] = v1_typed

    # Pre-allocate buffer for matrix-vector product result
    # We will reuse this buffer logic or just allocate fresh. 
    # To be safe and simple: new allocation per step (or copy)

    for k in 1:N_typed
        # r[k+1] = A * r[k] = r[k] + 1/N * sum(a_j * M_j * r[k])
        rk = r[k]
        r_next = copy(rk) # Starts as Identity * rk

        # Add contribution from each M_j
        for j in 1:DIM_typed
            m = M_typed[j]
            aj_invN = a[j] * invN

            rows = rowvals(m)
            vals = nonzeros(m)

            # M_j * rk
            # Iterate columns of m
            for col in 1:size(m, 2)
                rj = rk[col]
                # If rj is small, we could skip, but check cost usually > mult cost
                for idx in nzrange(m, col)
                    row = rows[idx]
                    val = vals[idx]
                    r_next[row] += aj_invN * val * rj
                end
            end
        end
        r[k+1] = r_next
    end

    # Backward Pass
    grads = zeros(Float64, DIM_typed)
    l = copy(v2_typed)
    l_next = similar(l)

    for i in N_typed:-1:1
        r_current = r[i+1]

        # Gradient accumulation
        # val_j = l' * M_j * r_current
        for j in 1:DIM_typed
            m = M_typed[j]
            rows = rowvals(m)
            vals = nonzeros(m)
            val = 0.0
            for c in 1:size(m, 2)
                rc = r_current[c]
                for idx in nzrange(m, c)
                    val += l[rows[idx]] * vals[idx] * rc
                end
            end
            grads[j] += val
        end

        # Update l for next step: l = A' * l
        # l_new = l + 1/N * sum(a_j * M_j' * l)
        if i > 1
            copyto!(l_next, l)

            for j in 1:DIM_typed
                m = M_typed[j]
                aj_invN = a[j] * invN
                rows = rowvals(m)
                vals = nonzeros(m)

                # Compute M_j' * l
                # (M^T l)_col = sum_row M_row,col * l_row
                # Since we iterate cols of M (which are rows of M^T),
                # for a given col, we sum over rows of M.

                for col in 1:size(m, 2)
                    delta = 0.0
                    for idx in nzrange(m, col)
                        row = rows[idx]
                        val = vals[idx]
                        delta += val * l[row]
                    end
                    l_next[col] += aj_invN * delta
                end
            end
            copyto!(l, l_next)
        end
    end

    return grads ./ N_typed
end

function d_objective_dense(a)
    # Type assertions for globals to ensure performance
    local M_typed::Vector{SparseMatrixCSC{Float64,Int}} = M
    local v1_typed::Vector{Float64} = v_1
    local v2_typed::Vector{Float64} = v_2
    local N_typed::Int = N # determines the accuracy of the method
    local Hs_dim_typed::Int = Hs_dim
    local DIM_typed::Int = DIM

    # Reconstruct X (dense)
    X = zeros(eltype(M_typed[1]), Hs_dim_typed, Hs_dim_typed)
    for i in 1:DIM_typed
        # In Julia, adding sparse to dense is generally fast enough, 
        # but since we know they are disjoint, we are just filling vals.
        rows = rowvals(M_typed[i])
        vals = nonzeros(M_typed[i])
        v = a[i]
        for col in 1:size(M_typed[i], 2)
            for k in nzrange(M_typed[i], col)
                row = rows[k]
                val_k = vals[k]
                X[row, col] += v * val_k
            end
        end
    end

    # Pre-compute Operator A = I + X/N
    invN = 1.0 / N_typed
    A = Matrix{Float64}(I, Hs_dim_typed, Hs_dim_typed)
    @. A += X * invN

    # Forward Pass
    # Store r[k] = A^(k-1) * v_1
    # We need r corresponding to A^1 ... A^N.
    # Let's allocate N+1 slots, r[k] = A^(k-1) * v_1
    r = Vector{Vector{Float64}}(undef, N_typed + 1)
    r[1] = v1_typed
    for k in 1:N_typed
        r[k+1] = A * r[k]
    end

    # Backward Pass
    grads = zeros(Float64, DIM_typed)
    # l calculates v_2' * A^(N-i)
    # i goes N down to 1.
    l = copy(v2_typed) # Left vector

    # Temporary buffer for l update
    l_next = similar(l)

    for i in N_typed:-1:1
        # Current term: v_2' * A^(N-i) * M[j] * A^i * v_1
        # l holds (A^T)^(N-i) * v_2. So l' corresponds to v_2' A^(N-i).
        # r_vec should be A^i * v_1.
        # r index mapping: r[k] = A^(k-1) v_1. So we need r[i+1].

        r_current = r[i+1]

        # Accumulate gradients for each j
        for j in 1:DIM_typed
            # Compute scalar: l' * M[j] * r_current
            # Efficiently using sparse structure of M[j]
            m = M_typed[j]
            rows = rowvals(m)
            vals = nonzeros(m)
            val = 0.0

            # Iterate columns of sparse matrix
            for c in 1:size(m, 2)
                rc = r_current[c]
                # If rc is 0, we can skip? No, dense vector usually not 0.
                for k in nzrange(m, c)
                    # M[row, c] * r[c] * l[row]
                    val += l[rows[k]] * vals[k] * rc
                end
            end
            grads[j] += val
        end

        # Update l for next step (next i is smaller, so N-i is larger by 1 -> multiply by A')
        # l = A' * l
        if i > 1
            mul!(l_next, A', l)
            l .= l_next
        end
    end

    return grads ./ N_typed
end

# Compare with finite differece.
finite_difference_check = [
    objective([1, 1]) - objective([0.999, 1]),
    objective([1, 1]) - objective([1, 0.999])
] * 1000


isapprox(d_objective_dense([1, 1]), finite_difference_check, atol=1e-3)