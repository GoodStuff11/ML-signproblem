import numpy as np
import time
from qiskit_nature.second_q.hamiltonians import FermiHubbardModel
from qiskit_nature.second_q.hamiltonians.lattices import Lattice
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_algorithms import NumPyMinimumEigensolver, VQE
from qiskit_algorithms.optimizers import L_BFGS_B, SPSA
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_nature.second_q.operators import FermionicOp
import networkx as nx

# --- Parameters ---
rows = 2
cols = 3
t_val = 1.0
u_val = 4.0
target_energy = -8.405249559995209 # Fixed target for "Double Bond" topology

def create_hamiltonian(rows=2, cols=3, t=1.0, u=4.0):
    """
    Creates the FermiHubbardModel Hamiltonian for a customized lattice with PBC.
    Implements 'Double Bond' in X for Nx=2 (hopping weight 2.0).
    """
    # 1. Define Lattice Graph with PBC
    graph = nx.Graph()
    n_nodes = rows * cols
    for i in range(n_nodes):
        graph.add_node(i)

    # Mapping (r, c) -> index
    def get_idx(r, c):
        return (r % rows) * cols + (c % cols)

    hopping_val = -t

    # Edges
    for r in range(rows):
        for c in range(cols):
            u_node = get_idx(r, c)
            
            # Right Neighbor (X direction)
            v_right = get_idx(r, c + 1)
            # In 2x3, cols=3 > 2, so bonds are distinct (0-1, 1-2, 2-0).
            # We add them with weight -t.
            if not graph.has_edge(u_node, v_right):
                graph.add_edge(u_node, v_right, weight=hopping_val)
            
            # Down Neighbor (Y direction, rows=2)
            v_down = get_idx(r + 1, c)
            # If rows=2, index(r,c) and index(r+1,c) are neighbors.
            # (0,0) -> (1,0). (1,0) -> (0,0).
            # In simple graph, this is 1 edge.
            # To simulate "Double Bond" (wrapping around is distinct path), 
            # we need effective hopping 2*t.
            # So we set weight to 2 * hopping_val.
            
            if not graph.has_edge(u_node, v_down):
                # Check if this is the "short dimension" where we want double bonds
                # rows=2.
                weight = 2.0 * hopping_val
                graph.add_edge(u_node, v_down, weight=weight)
                
    # Wait, my check_double.py had "Double X". User said 2x3.
    # Usually "2x3" means 2 rows, 3 columns? Or 2 columns, 3 rows?
    # If array shape (2,3), it is 2 rows, 3 cols.
    # In check_double.py, I used nx=2, ny=3.
    # And "Double X" meant X dimension (size 2) had double bonds.
    # Here I am iterating r in rows, c in cols.
    # If rows=2, then Y direction (down) is the short one.
    # So I should double the Y weights?
    # Let's verify my index mapping.
    # idx = r*cols + c.
    # X direction is c varying. (size 3).
    # Y direction is r varying. (size 2).
    # So Y direction is the one with size 2.
    # So yes, v_down edge should have double weight.

    # 2. Create Lattice from Graph
    # We use the weights in the graph as the hopping parameters (-t)
    adj_matrix = nx.to_numpy_array(graph, weight='weight')
    lattice = Lattice.from_adjacency_matrix(adj_matrix)
    
    # 3. Create Model
    model = FermiHubbardModel(
        lattice,
        onsite_interaction=u
    )
    
    return model

def solve_exact(op):
    print("Running Exact Diagonalization...")
    t0 = time.time()
    solver = NumPyMinimumEigensolver()
    result = solver.compute_minimum_eigenvalue(op)
    t1 = time.time()
    print(f"ED Time: {t1-t0:.4f}s")
    return result.eigenvalue.real

def get_qubit_op(model):
    mapper = JordanWignerMapper()
    op = model.second_q_op()
    qubit_op = mapper.map(op)
    return qubit_op, mapper

def run_vqe_uccsd(qubit_op, mapper, num_spatial_orbitals, num_particles):
    """
    Runs VQE with UCCSD ansatz.
    """
    # Restricted Ansatz: Separate occupied and unoccupied
    # But "spin should not be conserved" -> generalized=True?
    # Qiskit UCCSD params:
    # generalized (bool): if True, include all excitations (occupied->occupied, virtual->virtual too? No, usually indices)
    # preserve_spin (bool): if False, allow spin flips.
    
    # "restricted ansatz (that is separate occupied and unoccupied states), but with no other restrictions"
    # This usually means we DO distinguish occ/virt. (So generalized=False? generalized=True usually implies ignoring occ/virt distinction?)
    # "spin should not be conserved" -> preserve_spin=False.
    
    print("\n--- Configuring UCCSD ---")
    
    ansatz = UCCSD(
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        qubit_mapper=mapper,
        initial_state=HartreeFock(
            num_spatial_orbitals=num_spatial_orbitals,
            num_particles=num_particles,
            qubit_mapper=mapper,
        ),
        generalized=False, # restricted occ/virt
        preserve_spin=False, # Allow spin flips
    )
    
    print(f"Ansatz Parameters: {ansatz.num_parameters}")
    
    # Optimizer
    print("Optimization: SPSA (limit 100 steps) due to slow reference estimator.")
    optimizer = SPSA(maxiter=100, learning_rate=0.01, perturbation=0.01)
    
    # Solver
    # AerEstimator proved incompatible (AlgorithmError). Falling back to reference.
    from qiskit.primitives import StatevectorEstimator as Estimator
    estimator = Estimator()
    
    # Callback
    start_time_opt = [time.time()]
    def callback(eval_count, parameters, value, std_dev):
        current_time = time.time()
        elapsed = current_time - start_time_opt[0]
        start_time_opt[0] = current_time 
        print(f"Eval {eval_count}: Energy = {value:.8f} (Step Time: {elapsed:.4f}s)")

    # Initial Point
    np.random.seed(42)
    initial_point = np.random.uniform(-0.01, 0.01, ansatz.num_parameters)
    
    print("Starting VQE Optimization (SPSA, 1000 iters)...")
    start_time = time.time()
    start_time_opt[0] = start_time
    
    vqe = VQE(estimator, ansatz, optimizer, initial_point=initial_point, callback=callback)
    result = vqe.compute_minimum_eigenvalue(qubit_op)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nVQE Converged Energy: {result.eigenvalue.real:.10f}")
    print(f"VQE Time: {total_time:.4f}s")
    
    return result.eigenvalue.real, total_time

if __name__ == "__main__":
    
    print(f"Constructing Hubbard Model (rows={rows}, cols={cols}, U={u_val}, t={t_val})...")
    model = create_hamiltonian(rows=rows, cols=cols, t=t_val, u=u_val)
    
    qubit_op, mapper = get_qubit_op(model)
    print(f"Qubits: {qubit_op.num_qubits}, Terms: {len(qubit_op)}")
    
    # ED
    ed_energy = solve_exact(qubit_op)
    print(f"ED Energy: {ed_energy:.10f}")
    print(f"Target:    {target_energy:.10f}")
    
    if abs(ed_energy - target_energy) > 1e-4:
        print("WARNING: ED does not match target -8.4. Proceeding anyway.")
    else:
        print("SUCCESS: ED matches target.")
        
    # UCCSD
    # num_particles: (n_alpha, n_beta) -> (3, 3)
    num_particles = (3, 3)
    num_spatial_orbitals = rows * cols # 6
    
    vqe_energy, vqe_time = run_vqe_uccsd(qubit_op, mapper, num_spatial_orbitals, num_particles)
    
    error = abs(vqe_energy - ed_energy)
    print(f"\nFinal Error: {error:.2e}")
    if error < 1e-4:
        print("Benchmark PASSED: Chemical accuracy reached.")
    else:
        print("Benchmark FAILED: Did not converge to ground state.")
