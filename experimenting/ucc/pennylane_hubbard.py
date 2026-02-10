
import pennylane as qml
from pennylane import numpy as np
from scipy.sparse import linalg as sla
import time

def generate_generalized_excitations(n_qubits):
    """
    Generates all unique generalized single and double excitations 
    that conserve spin z projection (sz), assuming interleaved spin ordering
    (even=up, odd=down).
    """
    s_wires = []
    d_wires = []

    # Singles: p -> q (a dagger_p a_q)
    # Unique pairs (p, q). Since a^dag_p a_q - a^dag_q a_p is anti-Hermitian,
    # (p, q) and (q, p) give same parameter space (just sign flip).
    # Enforce p < q.
    for p in range(n_qubits):
        for q in range(p + 1, n_qubits):
            # Spin conservation: parity must match
            if (p % 2) == (q % 2):
                s_wires.append([p, q])

    # Doubles: p, q -> r, s (a^dag_r a^dag_s a_q a_p)
    # Convention: wires_from = [q, p], wires_to = [s, r]
    
    # 1. Generate all unique pairs (i, j) with i < j
    pairs = []
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            pairs.append((i, j))
            
    # 2. Iterate all combinations of two pairs: pair1=(p, q), pair2=(r, s)
    # Enforce pair1 < pair2 to avoid double counting (Hermitian conjugate).
    
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            p, q = pairs[i] # p < q
            r, s = pairs[j] # r < s
            
            # Disjoint check
            if len({p, q, r, s}) != 4:
                continue
                
            # Spin conservation
            # (p%2 + q%2) must equal (r%2 + s%2)
            if (p % 2 + q % 2) == (r % 2 + s % 2):
                # Add to lists.
                # Structure as [[r, s], [p, q]] (matches qml.FermionicDoubleExcitation expected inputs)
                d_wires.append([[r, s], [p, q]])

    return s_wires, d_wires

def main():
    # --- Hubbard Model Parameters ---
    nx, ny = 2, 3
    n_sites = nx * ny
    n_electrons = 6
    t = 1.0
    u = 4.0

    print(f"Hubbard Model: {nx}x{ny} lattice, N_electrons={n_electrons}")
    print(f"Parameters: t={t}, U={u}")

    # --- 1. Hamiltonian Construction ---
    coeffs = []
    ops = []

    # Hopping terms
    for x in range(nx):
        for y in range(ny):
            i = x * ny + y
            neighbors = []
            neighbors.append(((x + 1) % nx) * ny + y) # x neighbor
            neighbors.append(x * ny + ((y + 1) % ny)) # y neighbor
            
            for j in neighbors:
                # Spin up
                ops.append(qml.FermiC(2*i) * qml.FermiA(2*j))
                coeffs.append(-t)
                ops.append(qml.FermiC(2*j) * qml.FermiA(2*i))
                coeffs.append(-t)
                # Spin down
                ops.append(qml.FermiC(2*i+1) * qml.FermiA(2*j+1))
                coeffs.append(-t)
                ops.append(qml.FermiC(2*j+1) * qml.FermiA(2*i+1))
                coeffs.append(-t)

    # Interaction terms
    for i in range(n_sites):
        # U n_up n_down
        ops.append(qml.FermiC(2*i) * qml.FermiA(2*i) * qml.FermiC(2*i+1) * qml.FermiA(2*i+1))
        coeffs.append(u)

    # Create Fermi Hamiltonian by summing terms (FermiSentence)
    ham_fermi = 0
    for c, op in zip(coeffs, ops):
        ham_fermi += c * op
        
    # --- MO Basis Transformation ---
    # 1. Construct 1-body matrix
    mat_1h = np.zeros((n_sites, n_sites))
    for x in range(nx):
        for y in range(ny):
            i = x * ny + y
            nx_neighbor = ((x + 1) % nx) * ny + y
            mat_1h[i, nx_neighbor] = -t
            mat_1h[nx_neighbor, i] = -t
            ny_neighbor = x * ny + ((y + 1) % ny)
            mat_1h[i, ny_neighbor] = -t
            mat_1h[ny_neighbor, i] = -t
            
    # Diagonalize
    e_orbs, C = np.linalg.eigh(mat_1h)
    print("Orbital calculation done.")
    
    # Now construct the Hamiltonian in MO basis
    coeffs_mo = []
    ops_mo = []
    
    # One-body
    for p in range(n_sites):
        # Spin up (index 2p)
        ops_mo.append(qml.FermiC(2*p) * qml.FermiA(2*p))
        coeffs_mo.append(float(e_orbs[p].real))
        # Spin down (index 2p+1)
        ops_mo.append(qml.FermiC(2*p+1) * qml.FermiA(2*p+1))
        coeffs_mo.append(float(e_orbs[p].real))
        
    # Two-body part: U sum_i n_i,up n_i,down transforms to MO
    print("Constructing 2-body terms in MO basis...")
    for p in range(n_sites):
        for q in range(n_sites):
            for r in range(n_sites):
                for s in range(n_sites):
                    val = 0.0
                    for i in range(n_sites):
                        val += C[i,p] * C[i,q] * C[i,r] * C[i,s]
                    val *= u
                    
                    if abs(val) > 1e-6:
                        ops_mo.append(qml.FermiC(2*p) * qml.FermiA(2*q) * qml.FermiC(2*r+1) * qml.FermiA(2*s+1))
                        coeffs_mo.append(float(val.real))
    
    ham_fermi_mo = 0
    for c, op in zip(coeffs_mo, ops_mo):
        ham_fermi_mo += c * op
    ham_qubit_mo_final = qml.jordan_wigner(ham_fermi_mo)
    print(f"Hamiltonian type: {type(ham_qubit_mo_final)}")
    
    # --- 2. Ansatz Setup (UCCSD) ---
    electrons = 6
    qubits = 12
    hf_state = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]) 
    hf_list = [int(x) for x in hf_state]
    
    # Generate Generalized Excitations for UCCSD
    # singles, doubles = qml.qchem.excitations(electrons, qubits)
    singles, doubles_formatted = generate_generalized_excitations(qubits)
    
    print(f"Number of generalized singles: {len(singles)}")
    print(f"Number of generalized doubles: {len(doubles_formatted)}")
    
    # Doubles are already formatted by our function as [[r,s], [p,q]]
    # doubles_formatted = [ [d[:2], d[2:]] for d in doubles ]
    
    # Manual UCCSD decomposition to bypass potential decoration issues
    def manual_uccsd(weights, wires, s_wires, d_wires, init_state):
        # 1. Prepare HF state
        qml.BasisState(init_state, wires=wires)
        
        # 2. Apply excitations
        # Doubles
        for i, (w1, w2) in enumerate(d_wires):
            w_idx = len(s_wires) + i
            qml.FermionicDoubleExcitation(weights[w_idx], wires1=w1, wires2=w2)
            
        # Singles
        for j, s_wires_ in enumerate(s_wires):
            w_idx = j
            qml.FermionicSingleExcitation(weights[w_idx], wires=s_wires_)

    # Define Device
    dev = qml.device("lightning.qubit", wires=qubits)
    
    @qml.qnode(dev)
    def circuit(weights):
        # Use manual decompositon
        manual_uccsd(weights, range(qubits), singles, doubles_formatted, hf_list)
        return qml.expval(ham_qubit_mo_final)

    def cost_fn(weights):
        return qml.math.real(circuit(weights))

    # --- 3. Optimization ---
    print("Starting optimization with AdamOptimizer...")
    opt = qml.AdamOptimizer(stepsize=0.05) 
    
    n_params = len(singles) + len(doubles_formatted)
    params = np.zeros(n_params, requires_grad=True)
    
    print(f"Initial Energy: {cost_fn(params):.6f}")
    
    max_steps = 300
    history = []
    
    t0 = time.time()
    for i in range(max_steps):
        params, prev_energy = opt.step_and_cost(cost_fn, params)
        history.append(prev_energy)
        
        if i % 2 == 0:
            print(f"Step {i}: Energy = {prev_energy:.6f}")
            
        # Basic covergence check
        if i > 5 and abs(history[-1] - history[-2]) < 1e-10:
            print("Converged.")
            break
            
    t1 = time.time()
    final_energy = circuit(params)
    print(f"Final VQE Energy: {final_energy:.6f}")
    print(f"Optimization time: {t1-t0:.2f}s")
    
    # Reference Value
    # Reference Value (Computed Exact Diagonalization)
    print("Computing exact ground state energy via sparse diagonalization...")
    H_sparse = ham_qubit_mo_final.sparse_matrix()
    # k=1 for ground state, which='SA' for Smallest Algebraic
    evals, _ = sla.eigsh(H_sparse, k=1, which='SA')
    exact_energy = evals[0]
    
    print(f"Exact Energy: {exact_energy:.6f}")
    print(f"Difference from Exact: {abs(final_energy - exact_energy):.6f}")

if __name__ == "__main__":
    main()
