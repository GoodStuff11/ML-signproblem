
import os
import time
import numpy as np
import tensorflow as tf
import scipy.sparse
from scipy.optimize import minimize

import openfermion
from openfermion.hamiltonians import fermi_hubbard
from openfermion.linalg import get_sparse_operator, get_ground_state
from openfermion.circuits import uccsd_generator
from openfermion.ops import FermionOperator

# Configure TensorFlow
tf.config.run_functions_eagerly(False)
# Ensure 64-bit precision for accuracy
tf.keras.backend.set_floatx('float64')

def main():
    print("--- OpenFermion + TensorFlow UCCSD for 2x3 Hubbard (PBC) ---")
    
    # --- Parameters ---
    x_dim = 2
    y_dim = 3
    n_sites = x_dim * y_dim
    n_electrons_up = 3
    n_electrons_down = 3
    n_electrons = n_electrons_up + n_electrons_down
    t = 1.0
    u = 4.0
    
    print(f"Lattice: {x_dim}x{y_dim}")
    print(f"Electrons: {n_electrons_up} up, {n_electrons_down} down")
    print(f"Parameters: t={t}, U={u}")

    # --- Hamiltonian ---
    # periodic=True for PBC
    hamiltonian = fermi_hubbard(
        x_dimension=x_dim,
        y_dimension=y_dim,
        tunneling=t,
        coulomb=u,
        chemical_potential=0.0, 
        magnetic_field=0.0,
        periodic=True,
        spinless=False
    )

    # Convert to sparse matrix
    H_sparse = get_sparse_operator(hamiltonian)
    dim = H_sparse.shape[0]
    print(f"Hilbert Space Dimension: {dim}")

    # --- Ground Truth (Exact Diagonalization) ---
    print("Calculating Exact Ground State...")
    t0 = time.time()
    exact_energy, exact_vec = get_ground_state(H_sparse)
    print(f"Exact Energy: {exact_energy:.8f}")
    print(f"Exact Calculation Time: {time.time() - t0:.2f}s")
    
    # Prepare H for TensorFlow
    # Convert Scipy Sparse to dense tensor (fits in memory for dim=4096)
    H_dense_tf = tf.constant(H_sparse.toarray(), dtype=tf.complex128)

    # --- UCCSD Ansatz Preparation ---
    print("Preparing UCCSD Generators...")
    
    # 1. Hartree-Fock State
    # Sites are indexed 0..11. Even=Up, Odd=Down? Or 0..N-1 Up, N..2N-1 Down?
    # OpenFermion fermi_hubbard uses standard ordering. Usually alternating spins?
    # Let's verify standard indexing: site i has spin up at 2i, spin down at 2i+1?
    # Or 0..N-1 and then ...
    # Wait, fermi_hubbard docs: "The indexing of qubits is as follows: site x,y has spin up at 2*(x*y_dim + y) and spin down at 2*(x*y_dim + y) + 1."
    # So indices are interleaved.
    
    # Fill lowest energy orbitals?
    # For Hubbard t=1, k-space is relevant. But here we work in site basis.
    # We need a reference state. A common choice is just filling indices 0..N_elec-1?
    # No, that's not necessarily the HF state for Hubbard.
    # However, for consistency with other scripts, let's use a simple filling and rely on UCCSD to find the rotation.
    # But usually one Diagonalizes the One-Body term to find MOs, fills them, and then maps back.
    # The provided 'openfermion_uccsd.py' filled explicit indices: [0, 3, 4, 7, 8, 11].
    # Let's try to replicate that or find the lowest non-interacting states.
    # 2x3 Lattice.
    # Let's check what 'openfermion_uccsd.py' did: `occupied_indices = [0, 3, 4, 7, 8, 11]`
    # 0 (site 0 up), 3 (site 1 down), 4 (site 2 up), 7 (site 3 down), 8 (site 4 up), 11 (site 5 down).
    # This looks like Antiferromagnetic ordering?
    # We will use the same reference state as the existing script to be consistent.
    
    occupied_indices = [0, 3, 4, 7, 8, 11]
    hf_vec = np.zeros(dim)
    hf_bits = 0
    for idx in occupied_indices:
        hf_bits |= (1 << idx)
    hf_vec[hf_bits] = 1.0
    
    hf_vec_tf = tf.constant(hf_vec, dtype=tf.complex128)
    
    hf_energy = hf_vec.T @ H_sparse @ hf_vec
    print(f"Reference Energy: {hf_energy.real:.8f}")

    # 2. Identify Excitations
    virtual_indices = [i for i in range(2 * n_sites) if i not in occupied_indices]
    
    # Collect generators indices and values for TF construction
    # A = sum theta_k * G_k
    # We will build A using scatter_nd.
    # We need:
    # all_indices: list of [row, col]
    # all_values_base: list of values from G_k
    # element_to_param_map: list of k (which param index scales this value)
    
    all_indices = []
    all_values_base = []
    element_to_param_map = []
    
    param_count = 0
    
    # Helper to process a FermionOperator from uccsd_generator
    def process_generator(generator_op, p_idx):
        # Convert to sparse matrix
        mat = get_sparse_operator(generator_op, n_qubits=2*n_sites)
        coo = mat.tocoo()
        
        # Append data
        current_indices = np.stack([coo.row, coo.col], axis=1)
        
        # We append to lists
        # Note: avoid append in loop for performance if huge, but here it's okay.
        return current_indices, coo.data

    indices_list = []
    values_list = []
    map_list = []

    # --- Singles ---
    # Spin conservation: i and a must have same spin (same parity)
    print("Generating Singles...")
    for i in occupied_indices:
        for a in virtual_indices:
            if (i % 2) == (a % 2):
                # Use uccsd_generator
                # It requires lists of [index, value]. We set value=1.0.
                single_amp = [[a, i], 1.0] # Excitation a^dag i
                # Note: uccsd_generator takes spatial or spin orbital indices depending on args?
                # Default is spin orbitals.
                # Arguments: calculate_op(single_amplitudes, double_amplitudes)
                # Amplitudes format: [[i, j], val] means t_{ij} a^\dagger_i a_j 
                # Wait, indices in amplitude list:
                # For singles: [i, j]. Is it i^dag j?
                # Docs say "t_{ij} (a^\dagger_i a_j - h.c.)" usually.
                # Let's pass [[a, i], 1.0]
                
                op = uccsd_generator(single_amplitudes=[single_amp], double_amplitudes=[])
                
                inds, vals = process_generator(op, param_count)
                if len(vals) > 0:
                    indices_list.append(inds)
                    values_list.append(vals)
                    map_list.append(np.full(len(vals), param_count, dtype=np.int32))
                    param_count += 1
    
    print(f"Num Singles: {param_count}")
    n_singles = param_count

    # --- Doubles ---
    print("Generating Doubles...")
    import itertools
    for i, j in itertools.combinations(occupied_indices, 2):
        for a, b in itertools.combinations(virtual_indices, 2):
            spin_i = i % 2
            spin_j = j % 2
            spin_a = a % 2
            spin_b = b % 2
            
            if (spin_i + spin_j) == (spin_a + spin_b):
                # Double amplitude [a, b, j, i] for a^dag b^dag j i - h.c.
                # Convention: t_{abji}
                double_amp = [[a, b, j, i], 1.0]
                
                op = uccsd_generator(single_amplitudes=[], double_amplitudes=[double_amp])
                
                inds, vals = process_generator(op, param_count)
                if len(vals) > 0:
                    indices_list.append(inds)
                    values_list.append(vals)
                    map_list.append(np.full(len(vals), param_count, dtype=np.int32))
                    param_count += 1

    print(f"Num Doubles: {param_count - n_singles}")
    print(f"Total Parameters: {param_count}")

    # Combine all data for TF
    if param_count == 0:
        print("No excitations found!")
        return

    all_indices = np.concatenate(indices_list, axis=0)
    all_values_base = np.concatenate(values_list, axis=0)
    element_to_param_map = np.concatenate(map_list, axis=0)
    
    # TF Constants
    tf_indices = tf.constant(all_indices, dtype=tf.int64)
    tf_vals_base = tf.constant(all_values_base, dtype=tf.complex128)
    tf_param_map = tf.constant(element_to_param_map, dtype=tf.int32)
    
    dense_shape = tf.constant([dim, dim], dtype=tf.int64)

    # --- Optimization Definition ---

    @tf.function
    def compute_energy(params):
        # params: (N_params,)
        
        # 1. Construct Generator Matrix A = sum theta_k G_k
        # Gather theta parameters for each non-zero element
        gathered_thetas = tf.gather(params, tf_param_map)
        
        # Scale the base values (which are from G_k with coeff 1.0)
        # Note: G_k are anti-hermitian, e.g. (a^dag a - a^dag a)
        # params are real.
        # values are complex.
        
        # Cast params to complex
        gathered_thetas_c = tf.cast(gathered_thetas, tf.complex128)
        
        updates = tf_vals_base * gathered_thetas_c
        
        # Create Dense Matrix A
        # scatter_nd sums duplicate indices, which is exactly what we want (summing G_k)
        A = tf.scatter_nd(tf_indices, updates, dense_shape)
        
        # 2. Matrix Exponentiation U = exp(A)
        U = tf.linalg.expm(A)
        
        # 3. Time Evolution / State Preparation
        psi = tf.linalg.matvec(U, hf_vec_tf)
        
        # 4. Expectation Value
        # E = <psi | H | psi>
        psi_conj = tf.math.conj(psi)
        H_psi = tf.linalg.matvec(H_dense_tf, psi)
        energy = tf.math.reduce_sum(psi_conj * H_psi)
        
        return tf.math.real(energy)

    # Gradient wrapper for Scipy
    def loss_and_grad(x):
        x_tf = tf.convert_to_tensor(x, dtype=tf.float64)
        with tf.GradientTape() as tape:
            tape.watch(x_tf)
            loss = compute_energy(x_tf)
        
        grad = tape.gradient(loss, x_tf)
        # Handle IndexedSlices if sparse gradient is returned
        if isinstance(grad, tf.IndexedSlices):
            grad = tf.convert_to_tensor(grad)
        
        return loss.numpy(), grad.numpy()

    # Initial guess
    np.random.seed(42)
    initial_params = np.zeros(param_count)
    # initial_params = np.random.normal(0, 0.01, size=param_count) # Small random noise

    print("Starting Optimization (L-BFGS-B)...")
    
    t_opt_start = time.time()
    res = minimize(loss_and_grad, initial_params, jac=True, method='L-BFGS-B', options={'disp': True})
    t_opt_end = time.time()
    
    print(f"Optimization Finished in {t_opt_end - t_opt_start:.2f}s")
    print(f"Final VQE Energy: {res.fun:.8f}")
    print(f"Exact Energy:     {exact_energy:.8f}")
    error = abs(res.fun - exact_energy)
    print(f"Error:            {error:.2e}")
    
    if error < 1e-4:
        print("SUCCESS: Chemical accuracy reached.")
    else:
        print("WARNING: Convergence might be insufficient.")

if __name__ == "__main__":
    main()
