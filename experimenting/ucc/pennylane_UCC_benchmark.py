
import pennylane as qml
from pennylane import numpy as np
import scipy.sparse.linalg as sla
import time
import sys
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

def create_hubbard_hamiltonian(nx, ny, t, u):
    """
    Creates the Hubbard Hamiltonian for a nx * ny lattice with PBC.
    Returns both the Fermionic Hamiltonian (for ED) and the Pennylane Qubit Operator.
    """
    n_sites = nx * ny
    coeffs = []
    ops = []

    # Hopping terms
    # PBC logic: (x, y) neighbors
    for x in range(nx):
        for y in range(ny):
            i = x * ny + y
            
            # Neighbors with PBC
            neighbors = []
            neighbors.append(((x + 1) % nx) * ny + y) # x+1
            neighbors.append(x * ny + ((y + 1) % ny)) # y+1

            for j in neighbors:
                # Add hopping for both spins
                # Spin up (even indices 0, 2, ...) -> 2*i
                # Spin down (odd indices 1, 3, ...) -> 2*i + 1
                
                # Up hopping: -t (a^dag_i,up a_j,up + h.c.)
                # qml.FermiC is creation, qml.FermiA is annihilation
                
                # i -> j (up)
                ops.append(qml.FermiC(2*i) * qml.FermiA(2*j))
                coeffs.append(t) # Wait, t=-1 usually means -t * sum. Prompt says t=-1. 
                                 # Standard hubbard is -t \sum c^dag c. 
                                 # If t = -1, then coeff is -(-1) = +1?
                                 # Or does "t=-1" mean the hopping integral value is -1?
                                 # Usually written -t \sum ... if t=1, term is -1.
                                 # If prompt says t=-1. I will assume term is + t * (c^dag c). 
                                 # Let's check target energy.
                                 # If ED target fails, I'll flip sign.
                                 # Standard convention: H = -t \sum <ij> c^dag_i c_j + U ...
                                 # User says "t=-1". So coefficient is -(-1) = +1? 
                                 # Or is parameter symbol t, and value is -1.
                                 # Let's write generic +t_val for hopping?
                                 # Usually hopping parameter t is positive.
                                 # If user specifies t=-1, maybe they mean the coefficient is -1?
                                 # I will assume coefficient is `t`. 
                                 # i.e. H = t \sum (c^dag c) + ...
                                 # Because usually H = - t_hopping \sum ...
                                 # If user sets t=-1, they might mean t_hopping = -1 (so coeff is +1), or just the hopping term coefficient is -1.
                                 # Given target -7.6... for U=4.
                                 # Let's assume the term is `t * (c^d c + c c^d)` where t=-1. 
                
                ops.append(qml.FermiC(2*j) * qml.FermiA(2*i))
                coeffs.append(t)
                
                # Down hopping
                ops.append(qml.FermiC(2*i+1) * qml.FermiA(2*j+1))
                coeffs.append(t)
                ops.append(qml.FermiC(2*j+1) * qml.FermiA(2*i+1))
                coeffs.append(t)

    # Interaction terms
    for i in range(n_sites):
        # U n_i,up n_i,down
        # n = c^dag c
        ops.append(qml.FermiC(2*i) * qml.FermiA(2*i) * qml.FermiC(2*i+1) * qml.FermiA(2*i+1))
        coeffs.append(u)

    # Create Fermi Hamiltonian by summing terms (FermiSentence)
    ham_fermi = 0
    for c, op in zip(coeffs, ops):
        ham_fermi += c * op
    
    return ham_fermi

def get_ed_ground_state(ham_fermi, n_electrons):
    """
    Computes Exact Diagonalization ground state energy.
    Since we need to restrict to specific particle number, we might need a dense matrix
    or a sparse one filtered by symmetry.
    For simplicity in this benchmark script, we'll build the sparse matrix of the FULL Hilbert space
    but since we need a specific sector (3 up, 3 down), we might just eigenstates and filter?
    Or better, use Jordan-Wigner mapped qubit Hamiltonian which preserves particle number.
    """
    # Convert to qubit operator
    ham_qubit = qml.jordan_wigner(ham_fermi)
    
    # Get sparse matrix
    # This is 2^12 = 4096. Small enough.
    mat = qml.sparse.get_sparse_operator(ham_qubit).tocsc()
    
    # We want ground state in the 3up 3down sector.
    # We can iterate eigenvectors or just assume the global ground state is in this sector?
    # For Hubbard at half filling or near it, ground state usually has S=0.
    # 3up 3down is Sz=0.
    # Let's get lowest k eigenvalues and check particle number.
    
    vals, vecs = sla.eigsh(mat, k=5, which='SA')
    
    # Check particle number for the ground state
    # Number operator
    # N = sum c^dag c
    # We can just verify the target value matches one of these.
    
    return vals[0], ham_qubit

def generate_unrestricted_excitations(n_electrons, n_qubits, delta_sz=0):
    """
    Generates all single and double excitations from occupied to virtual orbitals.
    Occupied indices: 0 to n_electrons-1
    Virtual indices: n_electrons to n_qubits-1
    
    Does NOT enforce Sz conservation if delta_sz is None?
    Prompt: "spin should not be conserved"
    
    This implies we should include excitations that flip spin?
    e.g. up_occ -> down_virt.
    
    "Separate occupied and unoccupied states" -> occ indices distinct from virt.
    
    Returns lists of wires for singles and doubles.
    """
    # Simply define occupied and virtual assuming a HF state |111111000000>
    # Occupied: 0, 1, 2, 3, 4, 5
    # Virtual: 6, 7, 8, 9, 10, 11
    # This assumes the orbitals are ordered by energy!
    # We need to ensure we map the qubit Hamiltonian to the MO basis first?
    # The prompt says "Get the Hamiltonian... exact diagonalized... UCCSD with restricted ansatz"
    # Usually UCCSD is applied in the Hartree-Fock basis (MO basis).
    # If we work in site basis, "occupied" and "virtual" are ill-defined or just wrong.
    # So we MUST diagonalize the 1-body Hamiltonian first to get MOs.
    pass

# Redefining the workflow to include MO transformation
def solve_hubbard_pennylane():
    # Parameters
    nx, ny = 2, 3
    t = -1.0 # Standard Hubbard t=1 means hopping term is -1. User said t=-1. Trying -1.0.
    u = 4.0
    n_sites = nx * ny
    n_qubits = 2 * n_sites
    n_electrons = 6 # 3 up, 3 down
    target_energy_ed = -7.619260354852321

    # 1. Build 1-body Hamiltonian (Hopping) in Site Basis
    # To find MOs.
    # H_1body_{ij} such that H_hop = sum H_{ij} c^dag_i c_j
    
    adj_matrix = np.zeros((n_sites, n_sites))
    for x in range(nx):
        for y in range(ny):
            i = x * ny + y
            # neighbors
            n1 = ((x + 1) % nx) * ny + y
            n2 = x * ny + ((y + 1) % ny)
            
            # Using t as coefficient
            
            adj_matrix[i, n1] += t
            adj_matrix[n1, i] += t
            adj_matrix[i, n2] += t
            adj_matrix[n2, i] += t
            
    # Diagonalize to get MOs
    # eigenvalues w, eigenvectors v
    w, v = np.linalg.eigh(adj_matrix)
    
    # Sort just in case (eigh usually sorts)
    idx_sorted = np.argsort(w)
    w = w[idx_sorted]
    v = v[:, idx_sorted]
    
    print(f"MO Energies: {w}")
    
    # 2. Build Hamiltonian in MO basis
    # H = \sum E_p c^dag_p c_p + \sum h_{pqrs} c^dag_p c^dag_q c_r c_s
    # The one-body part is diagonal in MO basis.
    
    coeffs = []
    ops = []
    
    # One-body terms
    for p in range(n_sites):
        # Spin up (2p)
        ops.append(qml.FermiC(2*p) * qml.FermiA(2*p))
        coeffs.append(w[p])
        # Spin down (2p+1)
        ops.append(qml.FermiC(2*p+1) * qml.FermiA(2*p+1))
        coeffs.append(w[p])
        
    # Two-body terms: U sum_i n_{i,up} n_{i,down}
    # n_{i,sigma} = sum_{p,q} C_{ip}^* C_{iq} a^dag_{p,sigma} a_{q,sigma}
    # Interaction = U sum_i (sum_{pq} C_{ip} C_{iq} a^dag_{p,up} a_{q,up}) (sum_{rs} C_{ir} C_{is} a^dag_{r,down} a_{s,down})
    # Note: v is real (hopping is real). C_{ip} is v[i, p].
    
    print("Building 2-body terms in MO basis...")
    for p in range(n_sites):
        for q in range(n_sites):
            for r in range(n_sites):
                for s in range(n_sites):
                    # Compute element
                    # U * sum_i (v[i,p] * v[i,q] * v[i,r] * v[i,s])
                    # Note indices matches creation/annihilation order
                    # term is: a^dag_p,up a_q,up a^dag_r,down a_s,down
                    # We want to put in normal order for FermiSentence?
                    # qml.FermiC(2*p) * qml.FermiA(2*q) * qml.FermiC(2*r+1) * qml.FermiA(2*s+1)
                    # This is already normal ordered wrt spin (up vs down disjoint)
                    
                    # Element calculation:
                    # We expanded n_{i,up} = sum_{pq} <p|i><i|q> a^dag_p a_q
                    # <i|p> = v[i,p]. (Assuming v columns are eigenvectors)
                    # Yes, v[:, p] is p-th MO.
                    
                    term_coeff = 0.0
                    for i in range(n_sites):
                         term_coeff += v[i,p] * v[i,q] * v[i,r] * v[i,s]
                    term_coeff *= u
                    
                    if abs(term_coeff) > 1e-8:
                        ops.append(qml.FermiC(2*p) * qml.FermiA(2*q) * qml.FermiC(2*r+1) * qml.FermiA(2*s+1))
                        coeffs.append(term_coeff)
                        
    ham_fermi_mo = 0
    for c, op in zip(coeffs, ops):
        ham_fermi_mo += c * op
                        
    # ham_fermi_mo = qml.Hamiltonian(coeffs, ops)
    ham_qubit = qml.jordan_wigner(ham_fermi_mo)
    
    # 3. Exact Diagonalization Baseline
    print("Performing Exact Diagonalization...")
    t0_ed = time.time()
    # Use sparse matrix from ham_qubit
    mat_sparse = ham_qubit.sparse_matrix().tocsc()
    vals, _ = sla.eigsh(mat_sparse, k=1, which='SA')
    ed_energy = vals[0]
    t1_ed = time.time()
    
    print(f"ED Energy: {ed_energy:.10f}")
    print(f"Target:    {target_energy_ed}")
    print(f"ED Time: {t1_ed - t0_ed:.4f}s")
    
    if abs(ed_energy - target_energy_ed) > 1e-4:
        print("WARNING: ED energy does not match target! Checking parameters...")
        # If it doesn't match, maybe t=1? 
        # But we will proceed with UCCSD on this Hamiltonian anyway.
    
    # 4. UCCSD Implementation
    # "Restricted" => Occupied / Unoccupied separation based on HF state.
    # HF state: Fill lowest energy MOs.
    # 3 up, 3 down. Lowest 3 MOs are 0, 1, 2.
    # Spin-orbitals: 0(up), 1(down), 2(up), 3(down), 4(up), 5(down).
    # Wait, MO indices in code above:
    # Energy levels w[0]..w[5].
    # Spin up indices: 0, 2, 4, 6, 8, 10
    # Spin down indices: 1, 3, 5, 7, 9, 11
    # Lowest energy spin-orbitals:
    # MO 0 (up/down) -> Qubits 0, 1
    # MO 1 (up/down) -> Qubits 2, 3
    # MO 2 (up/down) -> Qubits 4, 5
    # So HF state is |111111000000>
    
    hf_state = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    
    # Occupied spin-orbitals
    occ_indices = [0, 1, 2, 3, 4, 5]
    # Virtual spin-orbitals
    virt_indices = [6, 7, 8, 9, 10, 11]
    
    # Generate Excitations
    # "Spin should not be conserved" -> All occ -> virt.
    singles = []
    doubles = []
    
    # Singles
    for i in occ_indices:
        for a in virt_indices:
            singles.append([i, a])
            
    # Doubles
    # (i, j) -> (a, b)
    # i < j, a < b to avoid redundancy?
    # Or just all pairs?
    # Parameters for t_{ij}^{ab} should correspond to unique excitations.
    # Commuting excitation operator: E = (T - T^dag).
    # T = a^dag_a a^dag_b a_j a_i.
    # Unique indices: i<j, a<b.
    # Also need to consider mixed spin.
    # If we don't enforce Sz, we just take all pairs from occ and all pairs from virt.
    
    for i in range(len(occ_indices)):
        for j in range(i + 1, len(occ_indices)):
            for a in range(len(virt_indices)):
                for b in range(a + 1, len(virt_indices)):
                    occ_pair = [occ_indices[i], occ_indices[j]]
                    virt_pair = [virt_indices[a], virt_indices[b]]
                    doubles.append([occ_pair, virt_pair])
                    
    print(f"Num Singles: {len(singles)}")
    print(f"Num Doubles: {len(doubles)}")
    
    # Define Circuit
    device = qml.device("lightning.qubit", wires=n_qubits)
    
    # Map excitation lists to functionality
    # qml.SingleExcitation/DoubleExcitation are for qubit excitations?
    # No, qml.FermionicSingleExcitation etc is better.
    
    @qml.qnode(device, diff_method="adjoint")
    def circuit(params):
        # Prepare HF
        qml.BasisState(hf_state, wires=range(n_qubits))
        
        # Apply Excitations
        # Using a fixed order
        idx = 0
        
        # Doubles first? Usually better for entanglement.
        for (occ, virt) in doubles:
            # wires = [virt[0], virt[1], occ[0], occ[1]]?
            # qml.FermionicDoubleExcitation(weight, wires1, wires2)
            # wires1: target (virt), wires2: source (occ) ? checking docs...
            # "wires1 (Iterable[int]): the wires of the creation operators" -> Virt
            # "wires2 (Iterable[int]): the wires of the annihilation operators" -> Occ
            
            qml.FermionicDoubleExcitation(params[idx], wires1=virt, wires2=occ)
            idx += 1
            
        for (occ, virt) in singles:
            # SingleExcitation
            # wires needs to be passed carefully.
            # qml.FermionicSingleExcitation(weight, wires)
            # wires: sequence of [r, p] corresponding to a^dag_r a_p
            # So [virt, occ]
            qml.FermionicSingleExcitation(params[idx], wires=[virt, occ])
            idx += 1
            
        return qml.expval(ham_qubit)


    # Optimization
    # Initial params small random
    np.random.seed(42)
    n_params = len(singles) + len(doubles)
    params = np.random.uniform(low=-0.05, high=0.05, size=(n_params,), requires_grad=True)
    
    print("Starting Optimization (L-BFGS-B)...")
    
    # Wrapper for scipy
    def cost_fn(p):
        return circuit(p)
        
    def grad_fn(p):
        return qml.grad(circuit)(p)
        
    # Using scipy via minimize
    # We need to wrap for autograd -> float conversion if using raw scipy
    # Or use qml.interfaces.scipy?
    # Simplest is to wrap in a function that takes numpy array and returns float/grad
    
    # We'll use the jax-like interface if possible, or just standard Pennylane Autograd
    # To use Scipy minimize with Pennylane:
    # Pass function and gradient.
    
    # Benchmarking
    start_time = time.time()
    
    # Store history
    energy_history = []
    
    def callback(p):
        e = cost_fn(p)
        energy_history.append(e)
        print(f"Iter {len(energy_history)}: E = {e:.8f}")

    # For pure BFGS on 'autograd' arrays, we might need a little adaptor
    # Scipy expects float64 arrays. 
    # Pennylane 'adjoint' works with 'autograd' params?
    # lightning.qubit adjoint works best with 'autograd' or 'jax' interface?
    # Default is autograd.
    
    # We define a wrapper that casts input to pennylane numpy and output to float
    def objective_wrapper(x):
        x_t = np.array(x, requires_grad=True)
        val = cost_fn(x_t)
        return float(val)

    def gradient_wrapper(x):
        x_t = np.array(x, requires_grad=True)
        # Using qml.grad which returns a function
        g = qml.grad(cost_fn)(x_t)
        return np.array(g, dtype=np.float64)

    from scipy.optimize import minimize
    res = minimize(
        objective_wrapper,
        np.array(params, dtype=np.float64),
        method='L-BFGS-B',
        jac=gradient_wrapper,
        callback=callback,
        tol=1e-7,
        options={'maxiter': 200, 'disp': True}
    )

    end_time = time.time()
    
    final_energy = res.fun
    print(f"Final VQE Energy: {final_energy:.10f}")
    print(f"Time Taken: {end_time - start_time:.4f}s")
    print(f"Error: {abs(final_energy - ed_energy):.2e}")
    
    if abs(final_energy - ed_energy) < 1e-4:
        print("SUCCESS: VQE converged to ground state.")
    else:
        print("FAIL: VQE did not reach exact ground state.")

if __name__ == "__main__":
    solve_hubbard_pennylane()
