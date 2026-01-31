
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.optimize import minimize
from openfermion.hamiltonians import fermi_hubbard
from openfermion.transforms import jordan_wigner
from openfermion.linalg import get_sparse_operator, get_ground_state
# from openfermion.utils import uccsd_singlet_generator, uccsd_generator
# from openfermion.linalg import get_ground_state # Already imported above

# --- Parameters ---
x_dim = 2
y_dim = 3
n_sites = x_dim * y_dim
n_electrons = 6
t = 1.0
u = 4.0

print(f"--- OpenFermion UCCSD for 2x3 Hubbard (PBC) ---")
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
print(f"Hamiltonian Dimension: {H_sparse.shape[0]}")

# --- Exact Diagonalization ---
print("Calculating Exact Ground State...")
exact_energy, exact_vec = get_ground_state(H_sparse)
print(f"Exact Energy: {exact_energy:.8f}")

# --- UCCSD Ansatz ---
# uccsd_singlet_generator gives T operator (anti-hermitian usually? No, just T).
# Check docs: generates T = T1 + T2.
# We want U = exp(T - T^dag). 
# openfermion utils usually return the anti-hermitian generator if requested, 
# or just the cluster operator.
# uniform UCCSD usually implies we include all valid T1, T2.

print("Generating UCCSD Operators...")

occupied_indices = [0, 3, 4, 7, 8, 11]
# Construct HF vector
dim = 2**(2 * n_sites)
hf_vec = np.zeros(dim)
hf_bits = 0
for idx in occupied_indices:
    hf_bits |= (1 << idx)
hf_vec[hf_bits] = 1.0

# Verify HF Energy
hf_energy = hf_vec.T @ H_sparse @ hf_vec
print(f"HF Reference Energy: {hf_energy.real:.8f}")

# Generate Excitations
# This is tricky without a helper that respects the lattice symmetry or just basic SD.
# We can use the generic generator for all singles and doubles.

from openfermion.ops import FermionOperator

# List of generators (anti-hermitian operators G = T - T^dag)
generators_sparse = []

# Singles
# i in occupied, a in virtual
virtual_indices = [i for i in range(2 * n_sites) if i not in occupied_indices]

print(f"Generators: Singles ({len(occupied_indices)} x {len(virtual_indices)})...")
count_S = 0
for i in occupied_indices:
    for a in virtual_indices:
        # Check spin conservation? 
        # Site index = idx // 2. Spin = idx % 2.
        if (i % 2) == (a % 2):
            # G = a_a^dag a_i - a_i^dag a_a
            # Build sparse matrix
            op = FermionOperator(((a, 1), (i, 0)), 1.0) - FermionOperator(((i, 1), (a, 0)), 1.0)
            mat = get_sparse_operator(op, n_qubits=2*n_sites)
            generators_sparse.append(mat)
            count_S += 1

print(f"Singles: {count_S}")

# Doubles
print("Generators: Doubles (subset)...")
# Full doubles is too many (6*6 = 36 occupied pairs, 6*6=36 virtual pairs -> 1296 terms).
# 1296x4096 sparse matrices is expensive to store in list?
# 4096 dim is small. 1296 matrices is okay.
# We will filter to spin-conserving doubles.

count_D = 0
# Pairs of occupied
import itertools
for i, j in itertools.combinations(occupied_indices, 2):
    for a, b in itertools.combinations(virtual_indices, 2):
        # Spin conservation
        spin_i = i % 2
        spin_j = j % 2
        spin_a = a % 2
        spin_b = b % 2
        
        if (spin_i + spin_j) == (spin_a + spin_b):
             op = FermionOperator(((a, 1), (b, 1), (j, 0), (i, 0)), 1.0) - \
                  FermionOperator(((i, 1), (j, 1), (b, 0), (a, 0)), 1.0)
             mat = get_sparse_operator(op, n_qubits=2*n_sites)
             generators_sparse.append(mat)
             count_D += 1

print(f"Doubles: {count_D}")
print(f"Total Parameters: {len(generators_sparse)}")

# Optimization
def energy_func(params):
    # Construct A = sum params * G
    A = params[0] * generators_sparse[0]
    for k in range(1, len(params)):
        A = A + params[k] * generators_sparse[k]
        
    # U = exp(A)
    # Use scipy.sparse.linalg.expm_multiply to apply to vector directly?
    # Or just construct U. Sparse expm is efficient-ish.
    # U = scipy.sparse.linalg.expm(A) 
    # But A is 4096 x 4096. Dense expm is generally faster/more robust for this size than sparse Krylov if many steps?
    # Actually expm_multiply(A, vec) is best.
    
    psi = scipy.sparse.linalg.expm_multiply(A, hf_vec)
    
    # E = <psi | H | psi>
    E = psi.conj().T @ (H_sparse @ psi)
    return E.real

# Optimization
# Start with 0
params0 = np.zeros(len(generators_sparse))
# Use reduced set or few steps as verification
print("Optimizing (COBYLA for speed)...")
# Warning: Gradient-free optimization for ~200 params might be slow.
# Analytically we could do gradients.
# But for demonstration, create script that verifies it runs.
res = minimize(energy_func, params0, method='COBYLA', options={'maxiter': 50, 'disp': True})

print(f"VQE Energy: {res.fun:.8f}")
print(f"Exact Energy: {exact_energy:.8f}")

