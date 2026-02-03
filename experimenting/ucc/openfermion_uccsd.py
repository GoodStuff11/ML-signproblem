
import numpy as np
import scipy.sparse
from scipy.optimize import minimize
from openfermion.hamiltonians import fermi_hubbard
from openfermion.transforms import jordan_wigner
from openfermion.linalg import get_sparse_operator, get_ground_state
# import jax
import jax
import jax.numpy as jnp
from jax import grad, jit

# Enable 64-bit precision for JAX
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_enable_x64", False)

# --- Parameters ---
x_dim = 2
y_dim = 3
n_sites = x_dim * y_dim
n_electrons = 6
t = 1.0
u = 4.0

print(f"--- OpenFermion + JAX UCCSD for 2x3 Hubbard (PBC) ---")
print(f"Lattice: {x_dim}x{y_dim}")
print(f"U = {u}, t = {t}")
print(f"Electrons: {n_electrons}")

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
print(f"Hamiltonian Dimension: {dim}")

# --- Exact Diagonalization ---
print("Calculating Exact Ground State...")
exact_energy, exact_vec = get_ground_state(H_sparse)
print(f"Exact Energy: {exact_energy:.8f}")

# Convert H to dense JAX array for optimization (4096 is small enough)
H_dense_jax = jnp.array(H_sparse.toarray())

# --- UCCSD Ansatz ---
occupied_indices = [0, 3, 4, 7, 8, 11]
# Construct HF vector
hf_vec = np.zeros(dim)
hf_bits = 0
for idx in occupied_indices:
    hf_bits |= (1 << idx)
hf_vec[hf_bits] = 1.0

hf_vec_jax = jnp.array(hf_vec, dtype=jnp.complex128)

# Verify HF Energy
hf_energy = hf_vec.T @ H_sparse @ hf_vec
print(f"HF Reference Energy: {hf_energy.real:.8f}")

# Generate Excitations
from openfermion.ops import FermionOperator
from jax.experimental import sparse as jsparse

# List of generators (anti-hermitian operators G = T - T^dag)
# Store as sparse JAX matrices to save memory
# 117 * 4096*4096 dense is too big (~16GB).
generators_indices = []
generators_values = []

# Singles
virtual_indices = [i for i in range(2 * n_sites) if i not in occupied_indices]

print(f"Generators: Singles ({len(occupied_indices)} x {len(virtual_indices)})...")
count_S = 0
for i in occupied_indices:
    for a in virtual_indices:
        if (i % 2) == (a % 2):
            op = FermionOperator(((a, 1), (i, 0)), 1.0) - FermionOperator(((i, 1), (a, 0)), 1.0)
            mat = get_sparse_operator(op, n_qubits=2*n_sites)
            # COOk format: (data, (row, col))
            coo = mat.tocoo()
            indices = jnp.array([coo.row, coo.col]).T
            values = jnp.array(coo.data)
            
            generators_indices.append(indices)
            generators_values.append(values)
            count_S += 1

print(f"Singles: {count_S}")

# Doubles
print("Generators: Doubles (subset)...")
count_D = 0
import itertools
for i, j in itertools.combinations(occupied_indices, 2):
    for a, b in itertools.combinations(virtual_indices, 2):
        spin_i = i % 2
        spin_j = j % 2
        spin_a = a % 2
        spin_b = b % 2
        
        if (spin_i + spin_j) == (spin_a + spin_b):
             op = FermionOperator(((a, 1), (b, 1), (j, 0), (i, 0)), 1.0) - \
                  FermionOperator(((i, 1), (j, 1), (b, 0), (a, 0)), 1.0)
             mat = get_sparse_operator(op, n_qubits=2*n_sites)
             coo = mat.tocoo()
             indices = jnp.array([coo.row, coo.col]).T
             values = jnp.array(coo.data)
             
             generators_indices.append(indices)
             generators_values.append(values)
             count_D += 1

max_nnz = max(len(v) for v in generators_values)
print(f"Max NNZ (detected): {max_nnz}")

# Pad generators for batching
padded_indices = []
padded_values = []

for idx, val in zip(generators_indices, generators_values):
    curr_len = len(val)
    if curr_len < max_nnz:
        # Pad with (0,0) indices and 0.0 values
        # Adding 0.0 to (0,0) is safe (no-op)
        pad_len = max_nnz - curr_len
        idx_pad = jnp.pad(idx, ((0, pad_len), (0, 0)), constant_values=0)
        val_pad = jnp.pad(val, (0, pad_len), constant_values=0.0)
        padded_indices.append(idx_pad)
        padded_values.append(val_pad)
    else:
        padded_indices.append(idx)
        padded_values.append(val)

# Stack arrays
batch_indices = jnp.stack(padded_indices) # (N_params, max_nnz, 2)
batch_values = jnp.stack(padded_values)   # (N_params, max_nnz)

print(f"Batched Generators Shape: {batch_values.shape}")


# Function to compute A @ v where A = sum p_i G_i
# We use vmap over generators
def apply_A(params, vec):
    # vec shape (dim,)
    # Compute G_i @ vec for all i (batched)
    
    # helper for one generator
    def single_gen_matvec(indices, values, v):
        row = indices[:, 0]
        col = indices[:, 1]
        
        # gather v[col]
        v_gathered = v[col]
        
        # product
        prod = values * v_gathered
        
        # scatter add to row
        res = jnp.zeros_like(v).at[row].add(prod)
        return res

    # vmap over batch dimension
    # (N, max_nnz, 2), (N, max_nnz), (dim) -> (N, dim)
    # We broadcast vec over N
    all_Gv = jax.vmap(single_gen_matvec, in_axes=(0, 0, None))(batch_indices, batch_values, vec)
    
    # Weighted sum: sum(p_i * (G_i v))
    # params shape (N,)
    # all_Gv shape (N, dim)
    # dot product along axis 0
    res = jnp.dot(params, all_Gv)
    return res

# Taylor Series Exponential
def taylor_expm(params, v, order=20):
    # Taylor series: exp(A)v = v + Av + A(Av)/2! + ...
    
    def body(carry, _):
        term, total, k = carry
        
        # next_term = A @ term / k
        # Apply A
        Av = apply_A(params, term)
        next_term = Av / k
        
        return (next_term, total + next_term, k + 1.0), None

    # Initial state
    # term_0 = v
    # total_0 = v
    # start k = 1 (since we divide by k for next term)
    init = (v, v, 1.0)
    
    (last_term, final_total, _), _ = jax.lax.scan(body, init, None, length=order)
    return final_total

# Optimization
@jit
def energy_loss(params):
    # U @ hf
    # Use Taylor series order 30 for safety
    psi = taylor_expm(params, hf_vec_jax, order=30)
    
    # E = <psi | H | psi>
    E = jnp.vdot(psi, H_dense_jax @ psi)
    return E.real

# Gradient function
grad_fun = jit(grad(energy_loss))

# Optimization
# params0 = np.zeros(len(padded_values))
# Use random initialization to escape local minima/symmetry traps
np.random.seed(42)
params0 = np.random.normal(0, 0.01, size=len(padded_values))

print("Optimizing using BFGS with JAX gradients (Matrix-Free)...")
import time
start_time = time.time()
res = minimize(
    lambda x: float(energy_loss(x)), 
    params0, 
    method='BFGS', 
    jac=lambda x: np.array(grad_fun(x)),
    options={'maxiter': 50, 'disp': True}
)
print(f"Time: {time.time() - start_time:.2f} seconds")

print(f"VQE Energy (JAX Matrix-Free): {res.fun:.8f}")
print(f"Exact Energy:                 {exact_energy:.8f}")
print(f"Difference:                   {res.fun - exact_energy:.8f}")
