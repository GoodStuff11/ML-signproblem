import os
import numpy as np
import h5py
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.optimize

def adjoint_loss_and_grad(t_vals, ops, v1, v2, p=None, antihermitian=False):
    """
    Computes the loss = 1 - |<v1 | exp((1 or 1j) * (A + p)) | v2>|^2
    And the gradient wrt t_vals using the continuous adjoint method and Simpson's rule.
    """
    if not ops:
        return 0.0, np.zeros_like(t_vals)
        
    A = t_vals[0] * ops[0]
    for t, op in zip(t_vals[1:], ops[1:]):
        A = A + t * op
    
    if p is not None:
        B = A + p
    else:
        B = A

    if antihermitian:
        scale = 1.0
    else:
        scale = 1.0j

    # Forward pass: evaluate at N_steps+1 points from 0 to 1
    N_steps = 50
    dt = 1.0 / N_steps

    phis = spla.expm_multiply(scale * B, v2, start=0.0, stop=1.0, num=N_steps+1, endpoint=True)
    
    psi_1 = phis[-1]
    overlap = np.vdot(v1, psi_1)
    loss = 1.0 - np.abs(overlap)**2

    # Backward pass: chis_tau[k] = exp(-scale * B * tau_k) * v1 for tau in [0, 1]
    chis_tau = spla.expm_multiply(-scale * B, v1, start=0.0, stop=1.0, num=N_steps+1, endpoint=True)
    
    # Reverse so chis[k] corresponds to time t_k = k/N_steps
    chis = chis_tau[::-1]

    # Simpson's rule weights
    weights = np.ones(N_steps + 1)
    weights[1:-1:2] = 4.0
    weights[2:-2:2] = 2.0
    weights *= (dt / 3.0)

    # dL/dt_i = -2 Re( conj(overlap) * dO_dt ) + 1e-3 * t_i
    conj_overlap = np.conj(overlap)
    
    grad = np.zeros(len(t_vals), dtype=float)

    for i, M_i in enumerate(ops):
        # M_i @ phis.T -> shape (dim, N_steps+1)
        phis_mapped = M_i.dot(phis.T) 

        integrand = np.sum(np.conj(chis) * phis_mapped.T, axis=1)
        val = np.sum(integrand * weights)

        if antihermitian:
            dO_dt = val
        else:
            dO_dt = val * 1.0j

        grad[i] = -2 * np.real(conj_overlap * dO_dt) + 1e-3 * t_vals[i]

    return float(loss), grad


def optimize_unitary(state1, state2, indexer, **kwargs):
    """
    Python implementation of optimize_unitary acting on exact states 
    (assumes ops and cache structures are supplied via operator_cache).
    """
    maxiters = kwargs.get('maxiters', 10)
    optimization_scheme = kwargs.get('optimization_scheme', [1, 2])
    optimizer_list = kwargs.get('optimizer', ['LBFGS'])
    if not isinstance(optimizer_list, list):
        optimizer_list = [optimizer_list]
        
    initial_coefficients = kwargs.get('initial_coefficients', [])
    operator_cache = kwargs.get('operator_cache', {})
    antihermitian = kwargs.get('antihermitian', False)

    max_order_scheme = max(optimization_scheme) if optimization_scheme else 0
    max_order_cache = max(operator_cache.keys()) if operator_cache else 0
    max_order = max(max_order_scheme, max_order_cache)

    computed_matrices = [None] * max_order
    computed_coefficients = [None] * max_order
    coefficient_labels = [None] * max_order
    parameter_mappings = [None] * max_order
    parities = [None] * max_order

    metrics = {
        "loss": [],
        "loss_std": [0.0],
        "other": []
    }

    initial_loss = 1.0 - np.abs(np.vdot(state1, state2))**2
    metrics["loss"].append(initial_loss)
    print(f"Initial loss: {initial_loss}")

    if initial_loss < 1e-15:
        print("States are already equal")
        return (computed_matrices, coefficient_labels, computed_coefficients,
                parameter_mappings, parities, metrics, operator_cache)

    for order in optimization_scheme:
        if order not in operator_cache:
            print(f"Warning: Order {order} structure not found in operator_cache. Skipping.")
            continue
            
        struct_data = operator_cache[order]
        ops = struct_data.get('ops', [])
        if isinstance(ops, list) and len(ops) > 0 and not sp.issparse(ops[0]):
            ops = [sp.csr_matrix(op) for op in ops]
            
        t_keys = struct_data.get('t_keys', None)
        coefficient_labels[order - 1] = t_keys
        parameter_mappings[order - 1] = struct_data.get('parameter_mapping', None)
        parities[order - 1] = struct_data.get('parity', None)

        if not ops:
            continue

        if len(initial_coefficients) >= order and initial_coefficients[order - 1] is not None:
            t_vals = np.array(initial_coefficients[order - 1], dtype=float)
        else:
            t_vals = np.random.randn(len(ops)) * (initial_loss * 100.0)

        def get_p_args():
            mats = [m for i, m in enumerate(computed_matrices) if i != (order - 1) and m is not None]
            return sum(mats) if mats else None

        def obj_and_grad(t):
            p = get_p_args()
            return adjoint_loss_and_grad(t, ops, state1, state2, p=p, antihermitian=antihermitian)

        print(f"Parameter count: {len(t_vals)}")

        loss = initial_loss
        for opt_sym in optimizer_list:
            print(f"Solving with {opt_sym}...")
            # Map Julia optimizer names to Scipy's
            method = 'L-BFGS-B' if opt_sym == 'LBFGS' else ('CG' if opt_sym == 'GradientDescent' else 'L-BFGS-B')

            res = scipy.optimize.minimize(
                obj_and_grad,
                t_vals,
                method=method,
                jac=True,
                options={'maxiter': maxiters, 'disp': True}
            )
            t_vals = res.x
            loss = res.fun
            print(f"loss={loss} avg_coef={np.mean(np.abs(t_vals))}")
            metrics["other"].append(str(res.message))

        computed_coefficients[order - 1] = t_vals.tolist()
        
        A = t_vals[0] * ops[0]
        for t, op in zip(t_vals[1:], ops[1:]):
            A = A + t * op
        computed_matrices[order - 1] = A

        print(f"Finished order {order}")
        metrics["loss"].append(loss)

    return (computed_matrices, coefficient_labels, computed_coefficients,
            parameter_mappings, parities, metrics, operator_cache)

def interaction_scan_map_to_state(target_vecs, instructions, indexer, spin_conserved=False, **kwargs):
    """
    Minimal Python equivalent of interaction_scan_map_to_state.
    """
    maxiters = kwargs.get('maxiters', 100)
    gradient = kwargs.get('gradient', 'gradient')
    optimizer = kwargs.get('optimizer', 'LBFGS')
    save_folder = kwargs.get('save_folder', None)
    save_name = kwargs.get('save_name', 'scan_data')
    initial_coefficients = kwargs.get('initial_coefficients', [])
    perturb_optimization = kwargs.get('perturb_optimization', 0.1)

    data_dict = {
        "norm1_metrics": [], "norm2_metrics": [], "loss_metrics": [],
        "labels": [], "loss_std_metrics": [], "all_matrices": [],
        "coefficients": [], "coefficient_labels": None, "param_mapping": None, "parities": None
    }

    shared_cache = {}
    current_coeffs = initial_coefficients

    u_indices = instructions.get("u_range", [])

    if save_folder and not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)

    shared_data_saved = False

    ref_u_idx = 0 
    ref_level = instructions.get("starting level", 1) - 1
    end_level = instructions.get("ending level", 1) - 1

    for u_idx in u_indices:
        print(f"\\n--- Scanning U index: {u_idx} ---")

        state1 = target_vecs[ref_u_idx][ref_level] if isinstance(target_vecs[ref_u_idx], (list, np.ndarray)) and len(target_vecs[ref_u_idx]) > ref_level else target_vecs[ref_u_idx]
        state2 = target_vecs[u_idx][end_level] if isinstance(target_vecs[u_idx], (list, np.ndarray)) and len(target_vecs[u_idx]) > end_level else target_vecs[u_idx]

        args = optimize_unitary(
            state1, state2, indexer,
            spin_conserved=spin_conserved,
            use_symmetry=instructions.get("use symmetry", False),
            maxiters=maxiters,
            optimization_scheme=instructions.get("optimization_scheme", [1, 2]),
            gradient=gradient,
            antihermitian=instructions.get("antihermitian", False),
            optimizer=optimizer,
            operator_cache=shared_cache,
            initial_coefficients=current_coeffs,
            perturb_optimization=perturb_optimization
        )

        computed_matrices, coefficient_labels, coefficient_values, param_mapping, parities, metrics, shared_cache = args
        current_coeffs = coefficient_values

        def _safe_norm(mat, ord):
            if mat is None: return 0.0
            try:
                import scipy.sparse.linalg as sla
                if sp.issparse(mat):
                    if ord == 1:
                        return float(abs(mat).sum(axis=0).max())
                    if ord == 2:
                        return sla.norm(mat)
                return float(np.linalg.norm(mat, ord=ord))
            except Exception:
                return 0.0

        data_dict["norm1_metrics"].append([_safe_norm(cm, 1) for cm in computed_matrices])
        data_dict["norm2_metrics"].append([_safe_norm(cm, 2) for cm in computed_matrices])
        data_dict["all_matrices"].append(computed_matrices)
        data_dict["coefficients"].append(coefficient_values)

        if data_dict["coefficient_labels"] is None:
            data_dict["coefficient_labels"] = coefficient_labels
            data_dict["param_mapping"] = param_mapping
            data_dict["parities"] = parities

        if save_folder and not shared_data_saved and coefficient_labels is not None:
            shared_dict = {
                "coefficient_labels": coefficient_labels,
                "param_mapping": param_mapping,
                "parities": parities,
                "instructions": str(instructions), 
                "u_range": list(u_indices)
            }
            with h5py.File(os.path.join(save_folder, f"{save_name}_shared.h5"), 'w') as f:
                for k, v in shared_dict.items():
                    try:
                        f.create_dataset(str(k), data=v)
                    except TypeError:
                        f.create_dataset(str(k), data=str(v))
            shared_data_saved = True

        if save_folder:
            iter_dict = {
                "u_idx": u_idx,
                "coefficient_values": coefficient_values,
                "norm1": [_safe_norm(cm, 1) for cm in computed_matrices],
                "norm2": [_safe_norm(cm, 2) for cm in computed_matrices]
            }
            with h5py.File(os.path.join(save_folder, f"{save_name}_u_{u_idx}.h5"), 'w') as f:
                for k, v in iter_dict.items():
                    try:
                        f.create_dataset(str(k), data=v)
                    except Exception:
                        f.create_dataset(str(k), data=str(v))

        for k, val in metrics.items():
            k_metric = k + "_metrics"
            if k_metric not in data_dict:
                data_dict[k_metric] = [val]
            else:
                data_dict[k_metric].append(val)

        data_dict["labels"].append({
            "starting state": {"level": instructions.get("starting level", 1), "U index": ref_u_idx},
            "ending state": {"level": instructions.get("ending level", 1), "U index": u_idx}
        })

    return data_dict
