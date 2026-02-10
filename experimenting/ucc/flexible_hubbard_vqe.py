
"""
Flexible Qiskit VQE Script for Fermi-Hubbard Model
Allows variable system dimensions and efficient simulation.
"""

import argparse
import time
import numpy as np
import networkx as nx
from qiskit_algorithms import VQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import L_BFGS_B, SLSQP, COBYLA, SPSA
from qiskit_algorithms.gradients import ReverseEstimatorGradient
from qiskit_algorithms.utils import algorithm_globals
from qiskit_nature.second_q.hamiltonians import FermiHubbardModel
from qiskit_nature.second_q.hamiltonians.lattices import Lattice
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
from qiskit.primitives import StatevectorEstimator as Estimator

def create_lattice(rows, cols, t=1.0, pbc=False):
    """
    Creates a lattice for the Fermi-Hubbard model.
    """
    graph = nx.grid_2d_graph(rows, cols)
    
    # Add PBC edges if requested
    if pbc:
        for r in range(rows):
            src = (r, cols-1)
            dst = (r, 0)
            if not graph.has_edge(src, dst):
                 graph.add_edge(src, dst)
        
        for c in range(cols):
            src = (rows-1, c)
            dst = (0, c)
            if not graph.has_edge(src, dst):
                graph.add_edge(src, dst)
                
    # Assign weights (hopping parameter t)
    for u, v, d in graph.edges(data=True):
        d['weight'] = t
        
    adj_matrix = nx.to_numpy_array(graph)
    return Lattice.from_adjacency_matrix(adj_matrix)

def main():
    parser = argparse.ArgumentParser(description="Run VQE on Fermi-Hubbard Model")
    parser.add_argument("--rows", type=int, default=2, help="Number of rows in lattice")
    parser.add_argument("--cols", type=int, default=2, help="Number of columns in lattice")
    parser.add_argument("--u", type=float, default=4.0, help="On-site interaction U")
    parser.add_argument("--t", type=float, default=1.0, help="Hopping parameter t")
    parser.add_argument("--pbc", action="store_true", help="Use Periodic Boundary Conditions")
    parser.add_argument("--mapper", type=str, default="jw", choices=["jw", "parity"], help="Fermion to Qubit Mapper (jw=Jordan-Wigner, parity=Parity with reduction)")
    parser.add_argument("--optimizer", type=str, default="lbfgs", choices=["lbfgs", "slsqp", "cobyla", "spsa"], help="Optimizer (default: cobyla)")
    parser.add_argument("--maxiter", type=int, default=50, help="Max iterations for optimizer")
    
    args = parser.parse_args()
    
    print(f"--- F-Sim VQE Setup ---")
    print(f"Lattice: {args.rows}x{args.cols} (PBC={args.pbc})")
    print(f"Parameters: U={args.u}, t={args.t}")
    
    # 1. Define Model
    lattice = create_lattice(args.rows, args.cols, args.t, args.pbc)
    model = FermiHubbardModel(lattice, onsite_interaction=args.u)
    hamiltonian = model.second_q_op()
    
    # 2. Define Mapper
    n_sites = args.rows * args.cols
    # Assume Half-Filling for simplicity, or allow config? Let's assume half-filling.
    n_particles = (n_sites // 2, n_sites // 2) if n_sites % 2 == 0 else ((n_sites+1)//2, n_sites//2)
    print(f"Particles (alpha, beta): {n_particles}")

    if args.mapper == "parity":
        print("Using Parity Mapper with TwoQubitReduction")
        mapper = ParityMapper(num_particles=n_particles)
        qubit_op = mapper.map(hamiltonian)
    else:
        print("Using Jordan-Wigner Mapper")
        mapper = JordanWignerMapper()
        qubit_op = mapper.map(hamiltonian)
        
    print(f"Number of Qubits: {qubit_op.num_qubits}")
    
    # 3. Setup Ansatz
    ansatz = UCCSD(
        num_spatial_orbitals=n_sites,
        num_particles=n_particles,
        qubit_mapper=mapper,
        initial_state=HartreeFock(
            num_spatial_orbitals=n_sites,
            num_particles=n_particles,
            qubit_mapper=mapper,
        ),
    )
    
    print(f"Ansatz Parameters: {ansatz.num_parameters}")
    
    # 4. Optimizer
    print(f"Optimizer: {args.optimizer.upper()}")
    if args.optimizer == "lbfgs":
        optimizer = L_BFGS_B(maxiter=args.maxiter)
    elif args.optimizer == "slsqp":
        optimizer = SLSQP(maxiter=args.maxiter)
    elif args.optimizer == "spsa":
        optimizer = SPSA(maxiter=args.maxiter)
    else:
        optimizer = COBYLA(maxiter=args.maxiter)
        
    # 5. Solver
    algorithm_globals.random_seed = 42
    estimator = Estimator()
    
    # Use ReverseEstimatorGradient for faster gradients
    # Note: Requires statevector simulator
    gradient = ReverseEstimatorGradient(estimator)

    # Callback to show progress
    counts, values = [], []
    def callback(eval_count, parameters, mean, std):
        counts.append(eval_count)
        values.append(mean)
        print(f"Eval {eval_count}: Energy = {mean:.6f}")

    print(f"Starting VQE...")
    # Warning removed as we are using analytical gradients now
    
    start_time = time.time()
    
    # Initial point from HF (all zeros for UCCSD usually starts at HF, but explicit zeros helps)
    initial_point = np.zeros(ansatz.num_parameters)
    
    vqe = VQE(estimator, ansatz, optimizer, gradient=gradient, initial_point=initial_point, callback=callback)
    result = vqe.compute_minimum_eigenvalue(qubit_op)
    
    elapsed = time.time() - start_time
    print(f"VQE Finished in {elapsed:.2f}s")
    print(f"VQE Energy: {result.eigenvalue.real:.6f}")
    
    # 6. Exact Check (for small systems)
    if n_sites <= 6: # 12 qubits max for quick check
        print("Running Exact Diagonalization for comparison...")
        exact_solver = NumPyMinimumEigensolver()
        exact_result = exact_solver.compute_minimum_eigenvalue(qubit_op)
        exact_energy = exact_result.eigenvalue.real
        print(f"Exact Energy: {exact_energy:.6f}")
        print(f"Error: {abs(result.eigenvalue.real - exact_energy):.2e}")
        
if __name__ == "__main__":
    main()
