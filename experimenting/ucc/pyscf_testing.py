import numpy as np
from pyscf import gto, scf, ao2mo, cc, fci

# --- Parameters ---
n_rows = 2
n_cols = 3
Ns = n_rows * n_cols  # 6 sites
Ne = 6  # 6 electrons (half-filling)
U = 4.0
t = 1.0

print(f"Lattice: {n_rows}x{n_cols} ({Ns} sites)")
print(f"Electrons: {Ne}")
print(f"U = {U}, t = {t}")

# --- Construct Hamiltonian Integrals ---

# 1-electron (hopping)
h1 = np.zeros((Ns, Ns))
for r in range(n_rows):
    for c in range(n_cols):
        i = r * n_cols + c
        
        # Right neighbor
        if c + 1 < n_cols:
            j = r * n_cols + (c + 1)
            h1[i, j] = h1[j, i] = -t
            
        # Down neighbor
        if r + 1 < n_rows:
            j = (r + 1) * n_cols + c
            h1[i, j] = h1[j, i] = -t

# 2-electron (interaction U)
# PySCF uses chemist notation (ij|kl) = \int i* j k* l
# Hubbard is U sum_i n_{i,up} n_{i,down}
# This corresponds to spatial integral (ii|ii) = U
eri = np.zeros((Ns, Ns, Ns, Ns))
for i in range(Ns):
    eri[i, i, i, i] = U

# --- Setup PySCF Molecule ---
# Create 6 Hydrogen atoms far apart to get 6 1s orbitals (orthogonal)
# Distance 100.0 ensures negligible overlap by default (though we overwrite it).
atom_list = []
for i in range(Ns):
    atom_list.append(f"H {i*10.0} 0 0")

mol = gto.M(
    atom=atom_list,
    basis='sto-3g',
    unit='Bohr',
    verbose=3 # useful for debugging
)
mol.nelectron = Ne
mol.spin = 0 # 3 up, 3 down
mol.incore_anyway = True # Ensure integrals are used in-core
# Remove nuclear repulsion
Enuc = mol.energy_nuc()
print(f"Nuclear Repulsion Energy (removed): {Enuc:.8f}")
mol.energy_nuc = lambda *args: 0.0

# --- Mean-Field (UHF) ---
# We use UHF to allow symmetry breaking (e.g. AFM) if needed,
# though standard init guess might land on RHF-like solution.
mf = scf.UHF(mol)

# Override matrix elements
mf.get_hcore = lambda *args: h1
mf.get_ovlp = lambda *args: np.eye(Ns)
mf._eri = ao2mo.restore(8, eri, Ns) # Use 8-fold symmetry for real integrals
# Or just keep it dense if small enough (6 sites is tiny)
# Let's use ao2mo.restore(1, eri, Ns) for no symmetry to be safe first.
mf._eri = ao2mo.restore(1, eri, Ns)

# Initial guess
# We can provide a custom density matrix to break spin symmetry (AFM)
# Or let it converge to whatever it finds.
# Let's try to break symmetry slightly.
dm_alpha = np.zeros((Ns, Ns))
dm_beta = np.zeros((Ns, Ns))
for i in range(Ns):
    if i % 2 == 0:
        dm_alpha[i, i] = 1.0 # Up
    else:
        dm_beta[i, i] = 1.0 # Down
# If Ne=6, this puts 3 up, 3 down. Perfect.
dm_init = np.array([dm_alpha, dm_beta])

print("Running UHF...")
mf.kernel(dm0=dm_init)
# mf.kernel()

print(f"UHF Energy: {mf.e_tot:.8f}")

# --- CCSD ---
print("Running CCSD...")
mycc = cc.CCSD(mf)
mycc.kernel()
print(f"CCSD Energy: {mycc.e_tot:.8f}")

# --- UCCSD ---
# pyscf.cc.CCSD automatically dispatches to UCCSD if mf is UHF.
# But we can explicitly check object type.
print(f"CC Object Type: {type(mycc)}")

# Solving lambda equations to get density matrices if needed (for properties)
# mycc.solve_lambda()

# Extract Amplitudes
t1 = mycc.t1
t2 = mycc.t2

print("\n--- UCCSD Amplitudes ---")
print("T1 is (t1a, t1b)")
print(f"t1a shape: {t1[0].shape}, Norm: {np.linalg.norm(t1[0]):.6f}")
print(f"t1b shape: {t1[1].shape}, Norm: {np.linalg.norm(t1[1]):.6f}")

print("\nT2 is (t2aa, t2ab, t2bb)")
print(f"t2aa shape: {t2[0].shape}, Norm: {np.linalg.norm(t2[0]):.6f}")
print(f"t2ab shape: {t2[1].shape}, Norm: {np.linalg.norm(t2[1]):.6f}")
print(f"t2bb shape: {t2[2].shape}, Norm: {np.linalg.norm(t2[2]):.6f}")

# Optional: Print significant likely amplitudes
# Since specific values might be requested, we can inspect max values
print(f"Max abs t1a: {np.max(np.abs(t1[0])):.6f}")
print(f"Max abs t2ab: {np.max(np.abs(t2[1])):.6f}")

# Comparison with FCI (Exact Diagonalization)
print("Running FCI for benchmark...")
# Create a fake object for FCI since it needs h1 and eri
# pyscf.fci.FCI needs a mean-field object or explicit integrals.
# direct_spin1 is for RHF, direct_uhf is for UHF.
# We can use solver generic.
fs = fci.FCI(mf)
fs.kernel()
print(f"FCI Energy: {fs.e_tot:.8f}")

diff = mycc.e_tot - fs.e_tot
print(f"CCSD - FCI: {diff:.8f}")

# --- 5. True Unitary CCSD (Variational) ---
print("\n--- True Unitary CCSD (Variational) ---")
print("Implementing explicit matrix exponential UCCSD (e^{T-T^dag}) in full Fock space...")

import scipy.sparse
from scipy.optimize import minimize

n_qubits = 2 * Ns # 12
dim = 2**n_qubits
print(f"Fock Space Dimension: {dim} (2^{n_qubits})")

# Helper to build sparse creation/annihilation operators
def get_ladder_ops(n_qubits):
    cre_ops = []
    
    for p in range(n_qubits):
        data = []
        row = []
        col = []
        
        # Iterate over all basis states |k>
        # Apply a_p^dag |k>
        for k in range(dim):
            if not (k & (1 << p)): # p-th bit is 0
                k_new = k | (1 << p)
                
                # Faster bit count for parity
                mask = (1 << p) - 1
                bits = k & mask
                # bin().count is acceptable for 4096
                pc = bin(bits).count('1')
                phase = 1.0 if pc % 2 == 0 else -1.0
                
                row.append(k_new)
                col.append(k)
                data.append(phase)
        
        # Create sparse matrix
        mat = scipy.sparse.csr_matrix((data, (row, col)), shape=(dim, dim))
        cre_ops.append(mat)
        
    return cre_ops

print("Constructing sparse ladder operators (this may take a second)...")
cre_ops = get_ladder_ops(n_qubits)
ann_ops = [m.T for m in cre_ops]

# Construct Hamiltonian in Fock Basis
print("Building Hamiltonian...")
H_mat = scipy.sparse.csr_matrix((dim, dim))

# One-body: sum h_{pq} a_p^dag a_q
# Map spin-orbitals: 0..5 Alpha, 6..11 Beta
# h1 is 6x6 spatial.
for p in range(Ns):
    for q in range(Ns):
        if abs(h1[p,q]) > 1e-10:
            # Alpha (0..5)
            term = cre_ops[p] @ ann_ops[q]
            H_mat += h1[p,q] * term
            # Beta (6..11)
            term = cre_ops[p+Ns] @ ann_ops[q+Ns]
            H_mat += h1[p,q] * term

# Two-body: U sum n_{i,up} n_{i,down}
for i in range(Ns):
    n_up = cre_ops[i] @ ann_ops[i]
    n_dn = cre_ops[i+Ns] @ ann_ops[i+Ns]
    H_mat += U * (n_up @ n_dn)

# Verify Exact Energy (in this space)
print("Diagonalizing Hamiltonian (Full Fock Space)...")
# H_mat is sparse, make dense for eigh
H_dense = H_mat.toarray()
evals_all, evecs_all = np.linalg.eigh(H_dense)
print(f"Lowest Eigenvalue (Global): {evals_all[0]:.8f}")

print(f"Comparison with PySCF FCI: {fs.e_tot:.8f}")

# Filter for N=6 Ground State
print("Searching for N=6 ground state...")
# Calculate N operator diagonal
N_diag = np.zeros(dim)
for k in range(dim):
    N_diag[k] = bin(k).count('1')

ground_state_N6_energy = None

for i in range(len(evals_all)):
    vec = evecs_all[:, i]
    # Check expectation value of N
    # Just check first non-zero component to check N (eigenstates have definite N)
    params_k = np.nonzero(vec)[0]
    if len(params_k) > 0:
        n_val = N_diag[params_k[0]]
        if abs(n_val - 6) < 0.1:
            ground_state_N6_energy = evals_all[i]
            print(f"Lowest N=6 Energy: {ground_state_N6_energy:.8f}")
            break

if ground_state_N6_energy is None:
    print("Could not find N=6 state?")
    ground_state_N6_energy = evals_all[0]

# Check sparse ladder ops
print(f"Number of sparse ladder ops: {len(cre_ops)}")
print(f"Shape of op[0]: {cre_ops[0].shape}, nnz: {cre_ops[0].nnz}")

# Construct Hamiltonian in Fock Basis
# ... (Same Hamiltonian construction) ...

# Reference State (HF)
print("Constructing HF reference state in Fock space...")
vac = np.zeros(dim)
vac[0] = 1.0
ref_vec = vac.copy()

mo_coeff = mf.mo_coeff # list [C_alpha, C_beta]

# Construct Creation Operators in MO Basis
cre_ops_mo_a = []
for p in range(Ns):
    # Alpha MO p
    # C_alpha is (Ns, Ns) matrix, columns are MOs
    coeffs = mo_coeff[0][:, p]
    op = scipy.sparse.csr_matrix((dim, dim))
    for mu in range(Ns):
        if abs(coeffs[mu]) > 1e-10:
            op += coeffs[mu] * cre_ops[mu]
    cre_ops_mo_a.append(op)

cre_ops_mo_b = []
for p in range(Ns):
    # Beta MO p
    coeffs = mo_coeff[1][:, p]
    op = scipy.sparse.csr_matrix((dim, dim))
    for mu in range(Ns):
        if abs(coeffs[mu]) > 1e-10:
            op += coeffs[mu] * cre_ops[mu + Ns]
    cre_ops_mo_b.append(op)
    
# Apply creation operators for occupied MOs
# First 3 Alpha MOs
for i in range(3):
    ref_vec = cre_ops_mo_a[i] @ ref_vec
    print(f"After Alpha MO {i}, norm: {np.linalg.norm(ref_vec):.6f}")

# First 3 Beta MOs
for i in range(3):
    ref_vec = cre_ops_mo_b[i] @ ref_vec
    print(f"After Beta MO {i}, norm: {np.linalg.norm(ref_vec):.6f}")

ref_vec = ref_vec / np.linalg.norm(ref_vec)
print(f"Final Reference Vector Norm: {np.linalg.norm(ref_vec):.6f}")

# Check Particle Number of ref_vec
N_op_diag = np.zeros(dim)
for k in range(dim):
    N_op_diag[k] = bin(k).count('1')
N_val = ref_vec.T @ (ref_vec * N_op_diag)
print(f"<N> of Reference State: {N_val:.6f}")

hf_energy = ref_vec.T @ H_mat @ ref_vec
print(f"HF Reference Energy (Fock Space): {hf_energy:.8f} (PySCF: {mf.e_tot:.8f})")

# Re-define lists for Generator construction
# Merge alpha/beta lists for indexing convience matching previous loops
cre_ops_mo = cre_ops_mo_a + cre_ops_mo_b
ann_ops_mo = [m.T for m in cre_ops_mo] # List of 12

# Define Generators
generators = []
amplitudes_init = []

# Singles (Alpha) i->a
occ_idx = range(3)
vir_idx = range(3, 6)
cc_t1 = mycc.t1
cc_t2 = mycc.t2

# Singles Alpha
for i_idx, i in enumerate(occ_idx):
    for a_idx, a in enumerate(vir_idx):
        G = cre_ops_mo[a] @ ann_ops_mo[i]
        tau = G - G.T
        generators.append(tau)
        amplitudes_init.append(cc_t1[0][i_idx, a_idx])

# Singles Beta
for i_idx, i in enumerate(occ_idx):
    for a_idx, a in enumerate(vir_idx):
        G = cre_ops_mo[a + Ns] @ ann_ops_mo[i + Ns]
        tau = G - G.T
        generators.append(tau)
        amplitudes_init.append(cc_t1[1][i_idx, a_idx])

# Doubles (Alpha-Alpha)
for i_idx, i in enumerate(occ_idx):
    for j_idx, j in enumerate(occ_idx):
        if i >= j: continue
        for a_idx, a in enumerate(vir_idx):
            for b_idx, b in enumerate(vir_idx):
                if a >= b: continue
                # a_dag b_dag j i
                G = cre_ops_mo[a] @ cre_ops_mo[b] @ ann_ops_mo[j] @ ann_ops_mo[i]
                tau = G - G.T
                generators.append(tau)
                amplitudes_init.append(cc_t2[0][i_idx, j_idx, a_idx, b_idx])

# Doubles (Alpha-Beta)
for i_idx, i in enumerate(occ_idx):
    for j_idx, j in enumerate(occ_idx):
        for a_idx, a in enumerate(vir_idx):
            for b_idx, b in enumerate(vir_idx):
                G = cre_ops_mo[a] @ cre_ops_mo[b + Ns] @ ann_ops_mo[j + Ns] @ ann_ops_mo[i]
                tau = G - G.T
                generators.append(tau)
                amplitudes_init.append(cc_t2[1][i_idx, j_idx, a_idx, b_idx])

# Doubles (Beta-Beta)
for i_idx, i in enumerate(occ_idx):
    for j_idx, j in enumerate(occ_idx):
        if i >= j: continue
        for a_idx, a in enumerate(vir_idx):
            for b_idx, b in enumerate(vir_idx):
                if a >= b: continue
                G = cre_ops_mo[a + Ns] @ cre_ops_mo[b + Ns] @ ann_ops_mo[j + Ns] @ ann_ops_mo[i + Ns]
                tau = G - G.T
                generators.append(tau)
                amplitudes_init.append(cc_t2[2][i_idx, j_idx, a_idx, b_idx])

amplitudes_init = np.array(amplitudes_init)
print(f"Number of UCCSD Parameters: {len(amplitudes_init)}")

# Optimize
def uccsd_energy(params):
    # A = sum p * G
    A = scipy.sparse.csr_matrix((dim, dim))
    for p, G in zip(params, generators):
        A += p * G
    
    # Use dense expm
    U_mat = scipy.linalg.expm(A.toarray())
    
    psi = U_mat @ ref_vec
    E = psi.T @ H_mat @ psi
    return E.real

print("Calculating energy with CCSD amplitudes...")
E_pert = uccsd_energy(amplitudes_init)
print(f"UCCSD (CCSD amps): {E_pert:.8f} (Diff: {E_pert - ground_state_N6_energy:.8f})")

print("Optimizing UCCSD variationally (BFGS)...")
res = minimize(uccsd_energy, amplitudes_init, method='BFGS', options={'disp': True, 'maxiter': 50})
print(f"Variational UCCSD Energy: {res.fun:.8f}")
print(f"Exact N=6 Ground State:   {ground_state_N6_energy:.8f}")
print(f"Error vs Exact:           {res.fun - ground_state_N6_energy:.8f}")
