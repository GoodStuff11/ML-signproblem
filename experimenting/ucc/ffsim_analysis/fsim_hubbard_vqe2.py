
import ffsim
import numpy as np
import scipy.sparse.linalg
import scipy.optimize
import time
from dataclasses import dataclass, InitVar
from ffsim import contract, gates, linalg, protocols
from ffsim.linalg.util import unitary_from_parameters, unitary_to_parameters
import itertools

# --- Implementation of Generalized UCCSD ---

def uccsd_generalized_linear_operator(
    t1: np.ndarray,
    t2: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
) -> scipy.sparse.linalg.LinearOperator:
    """Return a linear operator for a generalized UCCSD operator generator.
    
    The generator is T - T^dag.
    T1 = sum t1_{pq} a^dag_p a_q
    T2 = sum t2_{pqrs} a^dag_p a^dag_q a_r a_s

    ffsim.contract.two_body_linop implements:
    sum one_body_{pq} a^dag_p a_q + sum two_body_{pqrs} a^dag_p a_q a^dag_r a_s
    
    We must map T - T^dag into this form.
    
    1. One-body term:
       Generator A1 = T1 - T1^dag
       A1 = sum (t1_{pq} - t1_{qp}*) a^dag_p a_q
       Let mat1 = t1 - t1.conj().T
       So one_body_term = mat1.
       
    2. Two-body term:
       Generator A2 = T2 - T2^dag
       A2 = sum (t2_{pqrs} - t2_{srqp}*) a^dag_p a^dag_q a_r a_s
       (Using Hermitian conjugate property: (a^dag_s a^dag_r a_q a_p)^dag = a^dag_p a^dag_q a_r a_s)
       
       Let mat2_{pqrs} = t2_{pqrs} - t2_{srqp}*.
       Then A2 = sum mat2_{pqrs} a^dag_p a^dag_q a_r a_s.
       
       ffsim expects form: sum V_{prqs} a^dag_p a_r a^dag_q a_s
       (Note indices in ffsim doc/standard: usually chemist notation (pk|qm) -> a^p a^q a_k a_m ?? No, let's derive.)
       
       ffsim calls it `two_body_tensor`. Let's assume it implements:
       H2 = sum V_{pqrs} a^dag_p a_q a^dag_r a_s  (This is a common form for general 2-body)
       
       We want to equate:
       sum mat2_{pqrs} a^dag_p a^dag_q a_r a_s  = sum V_{ijkl} a^dag_i a_j a^dag_k a_l + Correction
       
       Commute a^dag_q a_r:
       a^dag_q a_r = - a_r a^dag_q + delta_{qr}
       So a^dag_p ( -a_r a^dag_q + delta_{qr} ) a_s
       = - a^dag_p a_r a^dag_q a_s + delta_{qr} a^dag_p a_s
       
       We want V such that V corresponds to mat2.
       So we need index variables to match a^dag_p a_r a^dag_q a_s.
       Let i=p, j=r, k=q, l=s.
       Then - a^dag_i a_j a^dag_k a_l has coefficient mat2_{pqrs} = mat2_{ikjl}.
       So V_{ikjl} = - mat2_{ikjl} => V_{pqrs} = - mat2_{prqs}.
       
       And there is a one-body correction term:
       sum_{pqrs} mat2_{pqrs} delta_{qr} a^dag_p a_s
       = sum_{pqs} mat2_{pqqs} a^dag_p a_s
       
       This correction must be added to the one_body_tensor.
    """
    
    # 1. Construct Generator Tensors (Anti-Hermitian)
    # Using 'complex' to ensure we don't drop phases if user passed real params
    t1 = t1.astype(complex)
    t2 = t2.astype(complex)
    
    # mat1 = t1 - t1^dag
    mat1 = t1 - t1.conj().T
    
    # mat2 = t2 - t2^dag
    # shape (norb, norb, norb, norb)
    # T^dag_{pqrs} corresponds to conjugate of T_{srqp}
    # transpose(3, 2, 1, 0)
    mat2 = t2 - t2.transpose(3, 2, 1, 0).conj()
    
    # 2. Map to ffsim tensors
    
    # Two-body tensor V_{pqrs} = - mat2_{prqs}
    # mat2 indices: p, q, r, s
    # We want indices p, r, q, s from mat2
    # So we take mat2, transpose (0, 2, 1, 3) -> (p, r, q, s)
    # And multiply by -1.
    two_body_tensor = -mat2.transpose(0, 2, 1, 3)
    
    # One-body correction
    # Correction_{ps} = sum_q mat2_{pqqs}
    # We can use einsum. mat2 indices (p,q,r,s). We want p=p, q=q, r=q, s=s.
    # einsum('pqqs->ps', mat2)
    correction = np.einsum('pqqs->ps', mat2)
    
    one_body_tensor = mat1 + correction
    
    return contract.two_body_linop(
        two_body_tensor, norb=norb, nelec=nelec, one_body_tensor=one_body_tensor
    )

@dataclass(frozen=True)
class UCCSDOpGeneralized:
    """Generalized UCCSD operator.
    
    Parameters are t1 and t2 amplitudes. 
    """
    t1: np.ndarray
    t2: np.ndarray

    @property
    def norb(self):
        return self.t1.shape[0]

    def _apply_unitary_(self, vec: np.ndarray, norb: int, nelec: tuple[int, int], copy: bool) -> np.ndarray:
        if copy:
            vec = vec.copy()
            
        linop = uccsd_generalized_linear_operator(self.t1, self.t2, norb, nelec)
        # Apply unitary exp(A)
        # scipy.sparse.linalg.expm_multiply computes exp(A) * v
        # Our operator A is anti-hermitian, so exp(A) is unitary.
        return scipy.sparse.linalg.expm_multiply(linop, vec, traceA=0.0)

# --- VQE Script ---

def main():
    # --- Hubbard Parameters ---
    nx, ny = 2, 3
    n_sites = nx * ny
    n_particles = (3, 3)
    t_val = 1.0
    u_val = 4.0

    print(f"Hubbard Model: {nx}x{ny} lattice, N_up={n_particles[0]}, N_down={n_particles[1]}")
    print(f"Parameters: t={t_val}, U={u_val}")

    # Hamiltonian
    mat_1h = np.zeros((n_sites, n_sites), dtype=complex)
    for x in range(nx):
        for y in range(ny):
            i = x * ny + y
            nx_neighbor = ((x + 1) % nx) * ny + y
            mat_1h[i, nx_neighbor] -= t_val
            mat_1h[nx_neighbor, i] -= t_val
            ny_neighbor = x * ny + ((y + 1) % ny)
            mat_1h[i, ny_neighbor] -= t_val
            mat_1h[ny_neighbor, i] -= t_val

    mat_2h = np.zeros((n_sites, n_sites, n_sites, n_sites), dtype=complex)
    for i in range(n_sites):
        mat_2h[i, i, i, i] = u_val

    # Exact Diagonalization
    mol_ham = ffsim.MolecularHamiltonian(one_body_tensor=mat_1h, two_body_tensor=mat_2h)
    linop = ffsim.linear_operator(mol_ham, n_sites, n_particles)
    e_min, _ = scipy.sparse.linalg.eigsh(linop, k=1, which="SA")
    ground_energy = e_min[0]
    print(f"Exact Ground Energy: {ground_energy:.6f}")
    
    # HF
    hartree_fock_state = ffsim.hartree_fock_state(n_sites, n_particles)
    e_hf = np.vdot(hartree_fock_state, linop @ hartree_fock_state).real
    print(f"Hartree-Fock Energy: {e_hf:.6f}")
    
    # MO basis NOT used for VQE ansatz in this script? 
    # Wait, the prompt's tmp.py used MO basis for Hamiltonian but HF state was site basis? 
    # "hartree_fock_state = ffsim.hartree_fock_state(n_sites, n_particles)" creates state |1...1 0...0>.
    # If the Hamiltonian is in site basis, this state is just occupation of first N sites.
    # In Hubbard model, HF state usually refers to momentum space or ground state of non-interacting Halmitonian.
    # The tmp.py did:
    # e_orbs, orbital_coeffs = np.linalg.eigh(mat_1h)
    # one_body_mo = ...
    # mol_ham_mo = ...
    # So tmp.py worked in MO basis.
    # I should do the same.
    
    # Transform Hamiltonian to MO basis
    e_orbs, C = np.linalg.eigh(mat_1h)
    one_body_mo = C.T.conj() @ mat_1h @ C
    two_body_mo = np.einsum('pqrs,pi,qj,rk,sl->ijkl', mat_2h, C, C, C, C, optimize=True)
    linop_mo = ffsim.linear_operator(ffsim.MolecularHamiltonian(one_body_mo, two_body_mo), n_sites, n_particles)

    # Initial State (HF in MO basis is just filling lowest energy MOs)
    # The 'hartree_fock_state' function returns |11...00>, which matches filling first N MOs if we work in MO basis.
    
    print("Setting up Generalized UCCSD...")
    
    # Parameterization
    # We parameterize the unique elements of the anti-hermitian generators.
    # T1: p > q (strictly lower triangle).
    # T2: P > R (strictly lower triangle of "pair matrix").
    
    # Index mapping
    pairs = list(itertools.combinations(range(n_sites), 2))
    
    # T1 indices: any p != q. Anti-hermitian implies we only need p > q.
    t1_indices = list(zip(*np.tril_indices(n_sites, k=-1)))
    n_t1 = len(t1_indices)
    
    # T2 indices
    # We consider unique quadruplets for anti-hermitian t2.
    # We can view t2 as matrix of pairs (pq) vs (rs).
    # Antisymmetry within pairs: t2_{pqrs} = -t2_{qprs} = -t2_{pqsr}.
    # So we only need pairs with p < q and r < s.
    # And Anti-hermiticity of generator: t2_{pqrs} = -t2_{rs pq}*.
    # So we only need "pair indices" P < R where P=(p,q), R=(r,s).
    
    t2_pair_indices = []
    # pairs is list of (p,q) with p < q
    for i, P in enumerate(pairs):
        for j, R in enumerate(pairs):
            if i > j: # Strictly lower triangle of pair-pair matrix
                t2_pair_indices.append((P, R))
                
    n_t2 = len(t2_pair_indices)
    
    n_params = n_t1 + n_t2
    print(f"Number of parameters: {n_params}")

    def get_operator(theta):
        t1 = np.zeros((n_sites, n_sites))
        t2 = np.zeros((n_sites, n_sites, n_sites, n_sites))
        
        idx = 0
        
        # Fill T1
        for (p, q) in t1_indices:
            val = theta[idx]
            idx += 1
            t1[p, q] = val
            t1[q, p] = -val
            
        # Fill T2
        for ((p, q), (r, s)) in t2_pair_indices:
            val = theta[idx]
            idx += 1
            # Current Term: t2_{pqrs}
            # Symmetries:
            # 1. Swap p,q -> -
            # 2. Swap r,s -> -
            # 3. Swap (pq),(rs) -> - (Anti-hermitian from our construction in A2) 
            #    Wait, we construct A2 = t2 - t2^dag.
            #    So if we set t2_{pqrs} = val, and t2_{rs pq} = 0,
            #    Then A2_{pqrs} = val. A2_{rs pq} = -val^*.
            #    So we just need to set the unique ones in t2.
            
            # We set t2_{pqrs} and its anti-symmetric partners within the indices
            t2[p, q, r, s] = val
            t2[q, p, r, s] = -val
            t2[p, q, s, r] = -val
            t2[q, p, s, r] = val
            
            # We do NOT set t2[r, s, p, q] etc, because the A2 construction handles the conjugate part.
            
        return UCCSDOpGeneralized(t1, t2)

    def cost_fn(theta):
        op = get_operator(theta)
        
        final_state = ffsim.apply_unitary(
            hartree_fock_state,
            op, 
            norb=n_sites, 
            nelec=n_particles
        )
        
        return np.vdot(final_state, linop_mo @ final_state).real

    # Optimize
    print("Starting optimization (COBYLA)...")
    np.random.seed(42)
    # Generalized CC usually benefits from starting near 0 but with noise to break symmetry
    initial_params = np.random.normal(0, 1e-1, n_params)
    
    res = scipy.optimize.minimize(cost_fn, initial_params, method="COBYLA", options={"maxiter": 2000})
    print(f"Final Energy: {res.fun:.6f}")
    print(f"Exact Energy: {ground_energy:.6f}")
    print(f"Error: {abs(res.fun - ground_energy):.2e}")

if __name__ == "__main__":
    main()
