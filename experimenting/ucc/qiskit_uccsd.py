
import numpy as np
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA, SLSQP
from qiskit_algorithms.utils import algorithm_globals
from qiskit_nature.second_q.hamiltonians import FermiHubbardModel
from qiskit_nature.second_q.hamiltonians.lattices import Lattice
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.primitives import StatevectorEstimator as Estimator
import networkx as nx
import time

# --- Parameters ---
n_rows = 2
n_cols = 3
u = 4.0
t = 1.0
n_electrons = (3, 3) # 3 alpha, 3 beta

print(f"--- Qiskit Nature UCCSD for 2x3 Hubbard (PBC) ---")
print(f"U = {u}, t = {t}")
print(f"Electrons: {n_electrons}")

# --- Construct Lattice with PBC ---
# Explicitly define graph for 2x3 with Periodic Boundary Conditions
graph = nx.Graph()
# Nodes 0..5
for i in range(6):
    graph.add_node(i)

# Edges (t=1.0)
# Map (r, c) -> r*3 + c
def idx(r, c):
    return (r % n_rows) * n_cols + (c % n_cols)

edges = []
for r in range(n_rows):
    for c in range(n_cols):
        u_node = idx(r, c)
        
        # Right neighbor (PBC)
        v_right = idx(r, c+1)
        if not graph.has_edge(u_node, v_right):
            graph.add_edge(u_node, v_right, weight=1.0)
            
        # Down neighbor (PBC)
        v_down = idx(r+1, c)
        if not graph.has_edge(u_node, v_down):
            graph.add_edge(u_node, v_down, weight=1.0)

print("Lattice Edges (PBC):", graph.edges())

# Calculate Adjacency Matrix
adj_matrix = nx.to_numpy_array(graph)
lattice = Lattice.from_adjacency_matrix(adj_matrix)

# --- Hamiltonian ---
# disorder=None means uniform t and U
# But FermiHubbardModel usually takes onsite_potential argument?
# hopping_matrix is built from lattice.
model = FermiHubbardModel(
    lattice.uniform_parameters(
        uniform_interaction=u,
        uniform_onsite_potential=0.0,
    ),
    onsite_interaction=u
)


# uniform_parameters helper sets them.
lattice_with_params = lattice.uniform_parameters(
    uniform_interaction=u,
    uniform_onsite_potential=0.0,
)

for u_node, v, d in graph.edges(data=True):
    # Hopping usually -t
    d['weight'] = -t 

# Re-create lattice from graph with weights
adj_matrix = nx.to_numpy_array(graph, weight='weight')
# attributes? from_adjacency_matrix might loss them if not careful?
# It uses the matrix entries as weights.
lattice = Lattice.from_adjacency_matrix(adj_matrix)

model = FermiHubbardModel(
    lattice,
    onsite_interaction=u
)

for u_node, v, d in graph.edges(data=True):
    d['weight'] = t # Hopping parameter t

# Now uniform parameters utility might overwrite this?
# Construct explicitly.
op = model.second_q_op()

# --- Mapper ---
mapper = JordanWignerMapper()
qubit_op = mapper.map(op)
print(f"Hamiltonian Qubit Operator Terms: {len(qubit_op)}")

# --- Ansatz ---
num_spatial_orbitals = 6
num_particles = n_electrons
mapper_for_ansatz = mapper

ansatz = UCCSD(
    num_spatial_orbitals=num_spatial_orbitals,
    num_particles=num_particles,
    qubit_mapper=mapper_for_ansatz,
    initial_state=HartreeFock(
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        qubit_mapper=mapper_for_ansatz,
    ),
)

print("UCCSD Ansatz Created")
print(f"Number of Parameters: {ansatz.num_parameters}")

# --- Solver ---
algorithm_globals.random_seed = 42
estimator = Estimator()
from qiskit_algorithms.optimizers import COBYLA, SLSQP, SPSA

# ...

# --- Solver ---
algorithm_globals.random_seed = 42
estimator = Estimator()
# COBYLA / SLSQP are slow for high param count without fast gradients (adjoint)
# SPSA is much faster per step (2 evaluations) though requires more steps.
optimizer = SPSA(maxiter=50) # Reduced for demonstration speed
# SPSA is stochastic, so 50 might not converge fully but will show speedup.

initial_point = np.zeros(ansatz.num_parameters) # Start from HF

# Skip transpilation as requested
print("Running VQE with SPSA...")
t = time.time()
vqe = VQE(estimator, ansatz, optimizer, initial_point=initial_point)
result = vqe.compute_minimum_eigenvalue(qubit_op)
print(f"Time: {time.time() - t:.2f} seconds")
print(f"VQE Result Energy: {result.eigenvalue.real:.8f}")



# --- Verification (Exact Diagonalization) ---
from qiskit_algorithms import NumPyMinimumEigensolver

exact_solver = NumPyMinimumEigensolver()
exact_result = exact_solver.compute_minimum_eigenvalue(qubit_op)
print(f"Exact Energy:      {exact_result.eigenvalue.real:.8f}")
# print(f"Error:             {result.eigenvalue.real - exact_result.eigenvalue.real:.8f}")
