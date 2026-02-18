
from __future__ import annotations
import ffsim
import ffsim.optimize
import numpy as np
import scipy.sparse.linalg
import scipy.sparse
import scipy.optimize
import time
from dataclasses import dataclass, InitVar
from typing import cast
import time
import itertools
from ffsim import contract, gates, linalg, protocols
from ffsim.linalg.util import unitary_from_parameters, unitary_to_parameters

# --- Implementation of UCCSDOpGeneralized (GSO Basis) ---

def _get_gso_indices(dim: int, norb: int, nelec: tuple[int, int]) -> np.ndarray:
    """Map spatial basis indices to GSO basis indices."""
    indices = np.arange(dim)
    strings = ffsim.addresses_to_strings(indices, norb, nelec)
    
    gso_norb = 2 * norb
    gso_nelec = (sum(nelec), 0)
    
    gso_indices = ffsim.strings_to_addresses(strings, gso_norb, gso_nelec)
    return gso_indices

def uccsd_generalized_linear_operator(
    params: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
) -> scipy.sparse.linalg.LinearOperator:
    """Return a linear operator for a generalized UCCSD operator generator in GSO basis.
    
    Params are assumed to be flattend [T1a, T1b, T2aa, T2bb, T2ab].
    """
    gso_norb = 2 * norb
    gso_nelec = (sum(nelec), 0)
    
    obs = np.zeros((gso_norb, gso_norb), dtype=complex)
    
    idx = 0
    
    # T1a: Indices 0..N-1
    # Iterate unique pairs (p, q) with p < q
    for p, q in itertools.combinations(range(norb), 2):
        val = params[idx]
        idx += 1
        obs[p, q] = val
        obs[q, p] = -val
        
    # T1b: Indices N..2N-1
    for p, q in itertools.combinations(range(norb), 2):
        val = params[idx]
        idx += 1
        P, Q = p + norb, q + norb
        obs[P, Q] = val
        obs[Q, P] = -val
        
    # Two body tensor
    tbs = np.zeros((gso_norb, gso_norb, gso_norb, gso_norb), dtype=complex)
    
    def add_term(p, q, r, s, val):
        tbs[p, q, r, s] += val
        tbs[q, p, r, s] -= val
        tbs[p, q, s, r] -= val
        tbs[q, p, s, r] += val
        
        tbs[s, r, p, q] -= val
        tbs[r, s, p, q] += val
        tbs[s, r, q, p] += val
        tbs[r, s, q, p] -= val

    # T2aa: unique pairs (p,q) and (r,s) from 0..N-1
    pairs_a = list(itertools.combinations(range(norb), 2))
    for (p, q), (r, s) in itertools.combinations(pairs_a, 2):
        val = params[idx]
        idx += 1
        add_term(p, q, r, s, val)
        
    # T2bb: unique pairs from N..2N-1
    pairs_b = list(itertools.combinations(range(norb), 2))
    pairs_b_idx = [(i + norb, j + norb) for i, j in pairs_b]
    for (p, q), (r, s) in itertools.combinations(pairs_b_idx, 2):
        val = params[idx]
        idx += 1
        add_term(p, q, r, s, val)

    # T2ab: mixed scale
    # (p,q) from alpha (0..N-1), (r,s) from beta (N..2N-1).
    for p in range(norb):
        for q in range(norb):
            for r_spatial in range(norb):
                for s_spatial in range(norb):
                    r = r_spatial + norb
                    s = s_spatial + norb
                    
                    if (p, q, r, s) <= (q, p, s, r):
                        continue
                        
                    val = params[idx]
                    idx += 1
                    
                    tbs[p, r, q, s] += val
                    tbs[r, p, q, s] -= val
                    tbs[p, r, s, q] -= val
                    tbs[r, p, s, q] += val
                    
                    tbs[q, s, p, r] -= val
                    tbs[s, q, p, r] += val
                    tbs[q, s, r, p] += val
                    tbs[s, q, r, p] -= val

    return contract.two_body_linop(
        tbs, norb=gso_norb, nelec=gso_nelec, one_body_tensor=obs
    )

def get_generalized_uccsd_generators(
    norb: int,
    nelec: tuple[int, int]
) -> list[scipy.sparse.coo_matrix]:
    """Pre-compute the sparse matrix representation of all Generalized UCCSD generators."""
    
    gso_norb = 2 * norb
    gso_nelec = (sum(nelec), 0)
    dim_gso = ffsim.dim(gso_norb, gso_nelec)
    
    print(f"Pre-computing {UCCSDOpGeneralized.n_params(norb)} generators for dim={dim_gso} (vectorized)...")
    t0 = time.time()
    
    # 1. Get all basis strings (integers)
    indices = np.arange(dim_gso, dtype=int)
    strings = ffsim.addresses_to_strings(indices, gso_norb, gso_nelec)
    strings = np.array(strings, dtype=int)
    
    generators = []
    
    def get_cre_ann_interaction(cre_ops, ann_ops):
        """Vectorized computation of T = cre * ann on all basis strings."""
        # Start with all strings
        current_strings = strings.copy()
        current_phases = np.ones(dim_gso, dtype=complex)
        valid_mask = np.ones(dim_gso, dtype=bool)
        
        # Apply annihilation operators
        # T = a^dag_p ... a_q ... a_s a_r
        # Apply right to left.
        # First apply a_r (last in list if list is [s, r]), then a_s.
        for orb in reversed(ann_ops):
            # Check if orbital is occupied
            occupied = (current_strings >> orb) & 1
            valid_mask &= (occupied == 1)
            
            # Phase: count bits < orb
            # Compute parity
            # count set bits below orb
            below_mask = (1 << orb) - 1
            n_below = np.zeros_like(current_strings)
            # Standard popcount is slow in numpy?
            # Creating a popcount lookup or simple method
            # For 12 qubits, iterating bits is minimal.
            # actually we can sum ((current_strings >> i) & 1) for i < orb
            for i in range(orb):
                n_below += (current_strings >> i) & 1
            
            phase_factors = (-1) ** n_below
            current_phases *= phase_factors
            
            # Remove electron
            current_strings ^= (1 << orb)
            
        # Apply creation operators (reversed order for phase?) 
        # T = a^dag_p ... a_q ...
        # Apply right to left. Annihilation already done.
        # Now creations.
        for orb in reversed(cre_ops):
            # Check if orbital is empty
            occupied = (current_strings >> orb) & 1
            valid_mask &= (occupied == 0)
            
            # Phase
            below_mask = (1 << orb) - 1
            n_below = np.zeros_like(current_strings)
            for i in range(orb):
                n_below += (current_strings >> i) & 1
                
            phase_factors = (-1) ** n_below
            current_phases *= phase_factors
            
            # Add electron
            current_strings ^= (1 << orb)
            
        return current_strings, current_phases, valid_mask

    def add_generator(cre_ops, ann_ops):
        # T term
        target_strings, phases, valid = get_cre_ann_interaction(cre_ops, ann_ops)
        
        if not np.any(valid):
            # Zero operator
            generators.append(scipy.sparse.coo_matrix((dim_gso, dim_gso), dtype=complex))
            return

        # Filter valid transitions
        src_indices = indices[valid]
        tgt_strs = target_strings[valid]
        data = phases[valid]
        
        # Convert target strings to indices
        tgt_indices = ffsim.strings_to_addresses(tgt_strs, gso_norb, gso_nelec)
        
        # Construct T matrix (sparse)
        # rows=tgt, cols=src
        row_T = tgt_indices
        col_T = src_indices
        data_T = data
        
        # Generator G = T - T^dag
        # G_ij = T_ij - T_ji.conj()
        # We can construct COO with (row, col, dat) and (col, row, -dat.conj)
        
        full_rows = np.concatenate([row_T, col_T])
        full_cols = np.concatenate([col_T, row_T])
        full_data = np.concatenate([data_T, -data_T.conj()])
        
        mat = scipy.sparse.coo_matrix((full_data, (full_rows, full_cols)), shape=(dim_gso, dim_gso))
        generators.append(mat)

    # Iterate
    # T1a
    for p, q in itertools.combinations(range(norb), 2):
        add_generator([p], [q])
    # T1b
    for p, q in itertools.combinations(range(norb), 2):
        P, Q = p + norb, q + norb
        add_generator([P], [Q])
        
    # T2aa
    pairs_a = list(itertools.combinations(range(norb), 2))
    for (p, q), (r, s) in itertools.combinations(pairs_a, 2):
        add_generator([p, q], [r, s]) # Swapped s,r to r,s for sign fix
        
    # T2bb
    pairs_b = list(itertools.combinations(range(norb), 2))
    pairs_b_idx = [(i + norb, j + norb) for i, j in pairs_b]
    for (p, q), (r, s) in itertools.combinations(pairs_b_idx, 2):
        add_generator([p, q], [r, s]) # Swapped s,r to r,s for sign fix

    # T2ab
    for p in range(norb):
        for q in range(norb):
            for r_spatial in range(norb):
                for s_spatial in range(norb):
                    r = r_spatial + norb
                    s = s_spatial + norb
                    if (p, q, r, s) <= (q, p, s, r): continue
                    add_generator([p, r], [s, q])
                    
    print(f"Generators computed in {time.time() - t0:.2f}s")
    return generators

@dataclass(frozen=True)
class UCCSDOpGeneralized(
    protocols.SupportsApplyUnitary, protocols.SupportsApproximateEquality
):
    """Generalized Unrestricted UCCSD operator (GSO basis)."""

    params: np.ndarray
    
    @property
    def norb(self):
        return 0 

    @staticmethod
    def n_params(norb: int) -> int:
        n_pairs = norb * (norb - 1) // 2
        n_t1 = 2 * n_pairs
        
        n_pairs_pairs = n_pairs * (n_pairs - 1) // 2
        n_t2_same = 2 * n_pairs_pairs
        
        n_t2_mixed = (norb**4 - norb**2) // 2
        
        return n_t1 + n_t2_same + n_t2_mixed

    def _apply_unitary_(self, vec: np.ndarray, norb: int, nelec: tuple[int, int], copy: bool) -> np.ndarray:
        if copy:
            vec = vec.copy()
        
        dim_orig = vec.shape[0]
        gso_indices = _get_gso_indices(dim_orig, norb, nelec)
        
        gso_norb = 2 * norb
        gso_nelec = (sum(nelec), 0)
        dim_gso = ffsim.dim(gso_norb, gso_nelec)
        
        vec_gso = np.zeros(dim_gso, dtype=vec.dtype)
        vec_gso[gso_indices] = vec
        
        linop = uccsd_generalized_linear_operator(self.params, norb, nelec)
        vec_gso = scipy.sparse.linalg.expm_multiply(linop, vec_gso, traceA=0.0)
             
        vec_new = vec_gso[gso_indices]
        if not copy:
             vec[:] = vec_new
             return vec
        return vec_new

    def _approx_eq_(self, other, rtol: float, atol: float) -> bool:
        return NotImplemented

# --- Main VQE Script ---

def main():
    # --- Hubbard Model Parameters ---
    nx, ny = 2, 3
    n_sites = nx * ny
    n_particles = (3, 3)  # 3 spin-up, 3 spin-down
    t = 1.0  # Tunneling
    u = 4.0  # Interaction

    print(f"Hubbard Model: {nx}x{ny} lattice, N_up={n_particles[0]}, N_down={n_particles[1]}")
    print(f"Parameters: t={t}, U={u}")

    # --- 1. Hamiltonian Construction (Site Basis) ---
    mat_1h = np.zeros((n_sites, n_sites), dtype=complex)
    for x in range(nx):
        for y in range(ny):
            i = x * ny + y
            nx_neighbor = ((x + 1) % nx) * ny + y
            mat_1h[i, nx_neighbor] -= t
            mat_1h[nx_neighbor, i] -= t
            ny_neighbor = x * ny + ((y + 1) % ny)
            mat_1h[i, ny_neighbor] -= t
            mat_1h[ny_neighbor, i] -= t

    mat_2h = np.zeros((n_sites, n_sites, n_sites, n_sites), dtype=complex)
    for i in range(n_sites):
        mat_2h[i, i, i, i] = u

    mol_ham = ffsim.MolecularHamiltonian(one_body_tensor=mat_1h, two_body_tensor=mat_2h)

    # --- 2. Exact Diagonalization ---
    print("Computing exact ground state energy...")
    linop = ffsim.linear_operator(mol_ham, n_sites, n_particles)
    
    # Use dense diagonalization to be absolutely sure
    # Dimension is small enough (924)
    # dim = ffsim.dim(n_sites, n_particles)
    # mat = linop @ np.eye(dim, dtype=complex)
    # vals = np.linalg.eigvalsh(mat)
    # ground_energy = vals[0]
    # print(f"Exact Ground Energy (Dense): {ground_energy:.6f}") # Should be lowest
    
    # --- 3. MO Basis Transformation & HF ---
    e_orbs, orbital_coeffs = np.linalg.eigh(mat_1h)
    
    C = orbital_coeffs
    one_body_mo = C.T.conj() @ mat_1h @ C
    two_body_mo = np.einsum('pqrs,pi,qj,rk,sl->ijkl', mat_2h, C, C, C, C, optimize=True)
    mol_ham_mo = ffsim.MolecularHamiltonian(one_body_tensor=one_body_mo, two_body_tensor=two_body_mo)
    
    hartree_fock_state = ffsim.hartree_fock_state(n_sites, n_particles)
    
    linop_mo = ffsim.linear_operator(mol_ham_mo, n_sites, n_particles)
    # e_min_mo, _ = scipy.sparse.linalg.eigsh(linop_mo, k=1, which="SA")
    # ground_energy_mo = e_min_mo[0]
    # print(f"Exact Ground Energy (MO Basis): {ground_energy_mo:.6f}")
    
    hf_energy = np.vdot(hartree_fock_state, linop_mo @ hartree_fock_state).real
    print(f"Hartree-Fock Energy: {hf_energy:.6f}")

    # --- 4. generalized UCCSD Ansatz ---
    print("Setting up Generalized UCCSD Ansatz...")

    # Calculate parameter count
    n_params = UCCSDOpGeneralized.n_params(n_sites)
    print(f"Number of parameters: {n_params}")

    generators = get_generalized_uccsd_generators(n_sites, n_particles)
    
    # Pre-compute GSO indices for mapping
    dim_spatial = ffsim.dim(n_sites, n_particles)
    gso_indices = _get_gso_indices(dim_spatial, n_sites, n_particles)
    
    gso_norb = 2 * n_sites
    gso_nelec = (sum(n_particles), 0)
    dim_gso = ffsim.dim(gso_norb, gso_nelec)
    
    # Define ground energy (from previous exact calc)
    ground_energy = -7.619260
    print(f"Exact Ground Energy: {ground_energy:.6f}")

    # Define cost function with gradient
    def cost_fn(theta):
        op = UCCSDOpGeneralized(theta)
        
        final_state = ffsim.apply_unitary(
            hartree_fock_state,
            op, 
            norb=n_sites,
            nelec=n_particles
        )
        
        # Normalize state
        norm = np.linalg.norm(final_state)
        final_state = final_state / norm
        
        # Energy
        h_psi = linop_mo @ final_state
        energy = np.vdot(final_state, h_psi).real
        
        # Gradient
        # Map vectors to GSO basis
        psi_gso = np.zeros(dim_gso, dtype=complex)
        psi_gso[gso_indices] = final_state
        
        h_psi_gso = np.zeros(dim_gso, dtype=complex)
        h_psi_gso[gso_indices] = h_psi
        
        grad = np.zeros(n_params)
        
        for k, G in enumerate(generators):
             # G is anti-hermitian. 
             # grad_k = 2 * Re(<psi| H G | psi>)
             #        = 2 * Re(<H psi | G psi>)
             G_psi = G.dot(psi_gso)
             grad[k] = 2 * np.vdot(h_psi_gso, G_psi).real
             
        return energy, grad

    print("Starting optimization using scipy.optimize (L-BFGS-B) with normalized analytic gradients...")
    t0 = time.time()
    
    # Initialize with slightly larger random noise to ensure gradient
    np.random.seed(42)
    initial_params = np.random.normal(0, 1e-2, n_params)
    
    # --- Gradient Check ---
    print("\n--- Performing Gradient Check ---")
    
    def get_energy(theta):
        op = UCCSDOpGeneralized(theta)
        final_state = ffsim.apply_unitary(hartree_fock_state, op, norb=n_sites, nelec=n_particles)
        norm = np.linalg.norm(final_state)
        final_state = final_state / norm
        h_psi = linop_mo @ final_state
        return np.vdot(final_state, h_psi).real

    # Compute analytic gradient
    e_anal, g_anal = cost_fn(initial_params)
    print(f"Energy at initial params: {e_anal}")
    
    # Compute numerical gradient for a few indices
    indices_to_check = [0, 10, 50, 100, 200, 400, 800] # Sample
    indices_to_check = [i for i in indices_to_check if i < n_params]
    
    epsilon = 1e-5
    print(f"{'Index':<10} {'Analytic':<15} {'Numeric':<15} {'Diff':<15} {'Ratio':<15}")
    for idx in indices_to_check:
        # Fwd
        p_plus = initial_params.copy()
        p_plus[idx] += epsilon
        e_plus = get_energy(p_plus)
        
        # Bwd
        p_minus = initial_params.copy()
        p_minus[idx] -= epsilon
        e_minus = get_energy(p_minus)
        
        g_num = (e_plus - e_minus) / (2 * epsilon)
        g_a = g_anal[idx]
        
        diff = abs(g_a - g_num)
        ratio = g_a / g_num if abs(g_num) > 1e-9 else 0.0
        
        print(f"{idx:<10} {g_a:<15.8f} {g_num:<15.8f} {diff:<15.8f} {ratio:<15.4f}")
        
    print("---------------------------------\n")

    res = scipy.optimize.minimize(cost_fn, initial_params, method="L-BFGS-B", jac=True)
    print(f"VQE Optimized Energy (Scipy): {res.fun:.6f}")

    error = abs(res.fun - ground_energy)
    print(f"Error: {error:.2e}")
    print(f"Time: {time.time() - t0:.2f} seconds")
    print(f"Number of parameters: {n_params}")

if __name__ == "__main__":
    main()
