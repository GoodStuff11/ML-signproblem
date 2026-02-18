
"""
    params_to_unitary(t_vals, rows, cols, signs, param_index_map, parameter_mapping, parity, computed_matrices, dim, antihermitian, use_symmetry)

Constructs the unitary matrix U from the parameters `t_vals` and any fixed `computed_matrices`.
U = exp(H_fixed + H(t_vals)).
"""
function params_to_unitary(t_vals, rows, cols, signs, param_index_map, parameter_mapping, parity, computed_matrices, dim, antihermitian, use_symmetry)
    # 1. Construct H(t_vals)
    vals = update_values(signs, param_index_map, t_vals, parameter_mapping, parity)
    H_new = sparse(rows, cols, vals, dim, dim)

    # 2. Add fixed H_fixed (if any)
    if !isempty(computed_matrices)
        H_fixed = sum(computed_matrices)
        # Note: computed_matrices are already stored as sparse matrices in the correct form (Hermitian/AntiHermitian)
        # However, they might be stored as "Matrix" or "SparseMatrixCSC".
        H_total = H_new + H_fixed
    else
        H_total = H_new
    end

    # 3. Exponentiate
    if !use_symmetry
        if antihermitian
            # If antihermitian, X is anti-hermitian. U = exp(X).
            # H_new is just sparse vals. We need to make it anti-hermitian?
            # update_values returns raw values. We need to enforce structure?
            # Yes, standard construction does `make_antihermitian`.
            H_total = make_antihermitian(H_total)
            return exp(Matrix(H_total))
        else
            # If hermitian, X is hermitian. U = exp(i X).
            H_total = make_hermitian(H_total)
            return exp(1im * Matrix(H_total))
        end
    else
        # With symmetry, we usually store block-diagonal or similar. 
        # But here `rows, cols` span the full dimension?
        # `update_values` gives values for the full matrix.
        if !antihermitian
            # Generator is i * H
            H_total = 1im * H_total
        end
        return exp(Matrix(H_total))
    end
end

"""
    unitary_to_params(U, ops, dim, antihermitian)

Extracts parameters `t_vals` from a unitary matrix `U` by projecting `log(U)` onto the generators `ops`.
Returns vector `t_vals`.
"""
function unitary_to_params(U, ops, dim, antihermitian)
    logU = log(U)
    n_ops = length(ops)
    t_vals = zeros(Float64, n_ops)

    for k in 1:n_ops
        op = ops[k]
        val = 0.0 + 0.0im
        rows_op = rowvals(op)
        vals_op = nonzeros(op)

        # Compute tr(op' * logU) efficiently
        for col in 1:size(op, 2)
            for idx in nzrange(op, col)
                row = rows_op[idx]
                v = vals_op[idx]
                val += conj(v) * logU[row, col]
                # Wait, tr(A B) = sum_ij A_ij B_ji.
                # A = op'. A_ij = conj(op_ji).
                # So sum_ji conj(op_ji) * logU_ji
                # Let's stick to: sum_rc (A[r,c] * B[c,r]).
                # A = op'.
                # A[c,r] = conj(op[r,c]).
                # So sum_rc conj(op[r,c]) * logU[c,r].
                # My previous code: v * logU[col, row] where v = op[row, col].
                # This computes sum_rc op[r,c] * logU[c,r] = tr(op * logU).
                # We want tr(op' * logU).
            end
        end

        # We calculated tr(op * logU).
        # if A_k = i op (Hermitian case, generator skew-Herm A_k).
        # Then A_k' = -i op' = -i op (if op is Hermitian).
        # If op is Hermitian, tr(op * logU) is fine.
        # But `op` from `ops` list might be just the basis matrix.
        # Let's assume `ops` are the basis matrices $O_k$.
        # Generator $A_k = i O_k$ (if !antihermitian) or $A_k = O_k$ (if antihermitian).

        # Projection: c_k = <A_k, logU> / <A_k, A_k>
        # <X, Y> = Re tr(X' Y).

        # Re-evaluating trace logic:
        # sum_{r,c} conj(op[r,c]) * logU[c, r] is correct for tr(op' * logU).

        # If !antihermitian: A_k = i op.
        # <A_k, logU> = Re tr( (i op)' logU ) = Re tr( -i op' logU ).
        # If op is Hermitian (usually true for Pauli strings): op' = op.
        # = Re( -i tr(op logU) ).
        # = Im( tr(op logU) ).

        # If antihermitian: A_k = op.
        # <A_k, logU> = Re tr( op' logU ).
        # If op is skew-hermitian (antihermitian=true): op' = -op.
        # = Re( - tr(op logU) ) = - Re( tr(op logU) ).

        # Let's compute tr(op logU) first.
        tr_op_logU = 0.0 + 0.0im
        for col in 1:size(op, 2)
            for idx in nzrange(op, col)
                row = rows_op[idx]
                v = vals_op[idx] # op[row, col]
                tr_op_logU += v * logU[col, row]
            end
        end

        if !antihermitian
            # A_k = i op. <A_k, A_k> = dim (approx).
            # c_k = Re( -i * tr_op_logU ) / dim
            t_vals[k] = real(-1im * tr_op_logU) / dim
        else
            # A_k = op.
            # If op is skew: op' = -op.
            # c_k = Re( - tr_op_logU ) / dim
            t_vals[k] = real(-tr_op_logU) / dim
        end
    end
    return t_vals
end
