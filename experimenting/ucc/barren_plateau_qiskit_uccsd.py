r"""
barren_plateau_qiskit_uccsd.py

Demonstrate barren plateaus in quantum neural networks (QNNs) using Qiskit's UCCSD ansatz.
Computes the variance of the gradient of the energy expectation value over random parameter samples
for a UCCSD ansatz of variable spatial orbitals (sites) and electron count.

Usage:
  python barren_plateau_qiskit_uccsd.py [options]

Options:
  --norb=<n> (optional): Number of spatial orbitals. Default: 4.
  --n_alpha=<a> (optional): Number of spin-up electrons. Default: 2.
  --n_beta=<b> (optional): Number of spin-down electrons. Default: 2.
  --num_samples=<s> (optional): Number of random parameter samples to estimate variance. Default: 30.
  --hamiltonian_type=<type> (optional): Type of Hamiltonian. Default: "hubbard".
                        Valid options:
                        - "hubbard": 1D Fermi-Hubbard chain with parameters t and U.
                        - "local": Number operator on the first orbital spin-up (n_{0, \uparrow}).
  --t=<t> (optional): Tunneling parameter t for Hubbard Hamiltonian. Default: 1.0.
  --u=<u> (optional): Interaction parameter U for Hubbard Hamiltonian. Default: 4.0.
  --pbc (optional): Use periodic boundary conditions for Hubbard model. Default: False.
  --parameter_index=<i> (optional): Index of the parameter to compute the gradient variance for.
                         If set to -1, computes and reports variance for all parameters. Default: 0.
  --num_processes=<p> (optional): Number of parallel processes to use. Default: None (use all available CPUs).

Examples:
  python barren_plateau_qiskit_uccsd.py --norb=4 --n_alpha=2 --n_beta=2 --num_samples=30 --hamiltonian_type=hubbard
"""

import os
import sys
import argparse
import time
import numpy as np
import warnings
import multiprocessing
from scipy.sparse import SparseEfficiencyWarning
warnings.filterwarnings('ignore', category=SparseEfficiencyWarning)

from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.hamiltonians import FermiHubbardModel
from qiskit_nature.second_q.hamiltonians.lattices import Lattice
from qiskit_nature.second_q.operators import FermionicOp
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms.gradients import ReverseEstimatorGradient

def get_default_num_processes():
    """
    Returns the default number of processes to use.
    Tries to detect CPU affinity (e.g. SLURM allocation limit) first,
    falling back to os.cpu_count().
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def generate_hamiltonian(norb, nelec, hamiltonian_type='hubbard', t=1.0, u=4.0, pbc=False):
    """
    Generates the Hamiltonian mapped to qubit space.

    Parameters:
        norb (int): Number of spatial orbitals.
        nelec (tuple[int, int]): Number of alpha and beta electrons.
        hamiltonian_type (str): 'hubbard' or 'local'.
        t (float): Tunneling parameter for Hubbard model.
        u (float): Interaction parameter for Hubbard model.
        pbc (bool): Use periodic boundary conditions for Hubbard model.

    Returns:
        tuple[SparsePauliOp, JordanWignerMapper]: The mapped Hamiltonian and mapper.
    """
    mapper = JordanWignerMapper()
    if hamiltonian_type == 'hubbard':
        # 1D Fermi-Hubbard chain
        mat_1h = np.zeros((norb, norb), dtype=complex)
        for i in range(norb - 1):
            mat_1h[i, i + 1] = -t
            mat_1h[i + 1, i] = -t
        if pbc and norb > 2:
            mat_1h[norb - 1, 0] = -t
            mat_1h[0, norb - 1] = -t
            
        lattice = Lattice.from_adjacency_matrix(mat_1h)
        model = FermiHubbardModel(lattice, onsite_interaction=u)
        op = model.second_q_op()
        return mapper.map(op), mapper
    elif hamiltonian_type == 'local':
        # Local operator n_{0, \uparrow}
        op = FermionicOp({"+_0 -_0": 1.0}, num_spin_orbitals=2 * norb)
        return mapper.map(op), mapper
    else:
        raise ValueError(f"Unknown Hamiltonian type: {hamiltonian_type}. Use 'hubbard' or 'local'.")

def generate_ansatz(norb, nelec, mapper):
    """
    Generates the UCCSD ansatz circuit.

    Parameters:
        norb (int): Number of spatial orbitals.
        nelec (tuple[int, int]): Number of alpha and beta electrons.
        mapper (JordanWignerMapper): The qubit mapper.

    Returns:
        QuantumCircuit: The parameterized UCCSD circuit.
    """
    return UCCSD(
        num_spatial_orbitals=norb,
        num_particles=nelec,
        qubit_mapper=mapper,
        initial_state=HartreeFock(
            num_spatial_orbitals=norb,
            num_particles=nelec,
            qubit_mapper=mapper,
        )
    )

def worker_evaluate_batch(args):
    """
    Worker function to compute gradients and energies for a chunk of parameter samples.

    Parameters:
        args (tuple): A tuple containing (ansatz, hamiltonian, parameter_samples).

    Returns:
        tuple[np.ndarray, np.ndarray]: (gradients, energies) for the chunk.
    """
    ansatz, hamiltonian, parameter_samples = args
    if len(parameter_samples) == 0:
        return np.empty((0, ansatz.num_parameters)), np.empty((0,))
    
    grad_estimator = ReverseEstimatorGradient()
    num_samples = len(parameter_samples)
    
    circuits = [ansatz] * num_samples
    observables = [hamiltonian] * num_samples
    
    job_g = grad_estimator.run(circuits, observables, parameter_samples)
    gradients = np.array(job_g.result().gradients)
    
    estimator = StatevectorEstimator()
    pub = (ansatz, hamiltonian, parameter_samples)
    job_e = estimator.run([pub])
    energies = np.array(job_e.result()[0].data.evs)
    
    return gradients, energies

def compute_gradients_batch(ansatz, hamiltonian, parameter_samples):
    """
    Computes the gradients of the expectation value of the Hamiltonian with respect to the
    parameters of the ansatz for a batch of random parameter samples using ReverseEstimatorGradient.

    Parameters:
        ansatz (QuantumCircuit): The parameterized ansatz circuit.
        hamiltonian (SparsePauliOp): The Hamiltonian operator.
        parameter_samples (np.ndarray): 2D array of shape (num_samples, num_parameters).

    Returns:
        np.ndarray: 2D array of shape (num_samples, num_parameters) containing the gradients.
    """
    num_samples = len(parameter_samples)
    grad_estimator = ReverseEstimatorGradient()
    
    circuits = [ansatz] * num_samples
    observables = [hamiltonian] * num_samples
    
    job = grad_estimator.run(circuits, observables, parameter_samples)
    result = job.result()
    
    return np.array(result.gradients)

def compute_energies_batch(ansatz, hamiltonian, parameter_samples):
    """
    Computes the expectation values of the Hamiltonian for a batch of parameter samples.

    Parameters:
        ansatz (QuantumCircuit): The parameterized ansatz circuit.
        hamiltonian (SparsePauliOp): The Hamiltonian operator.
        parameter_samples (np.ndarray): 2D array of shape (num_samples, num_parameters).

    Returns:
        np.ndarray: 1D array of shape (num_samples,) containing the expectation values.
    """
    estimator = StatevectorEstimator()
    pub = (ansatz, hamiltonian, parameter_samples)
    job = estimator.run([pub])
    result = job.result()
    return result[0].data.evs

def run_gradient_variance_study(norb, nelec, num_samples, hamiltonian_type='hubbard', 
                                t=1.0, u=4.0, pbc=False, num_processes=1):
    """
    Runs the study to compute the variance of the gradients of the energy expectation value.

    Parameters:
        norb (int): Number of spatial orbitals.
        nelec (tuple[int, int]): Number of alpha and beta electrons.
        num_samples (int): Number of random parameter samples to draw.
        hamiltonian_type (str): 'hubbard' or 'local'.
        t (float): Tunneling parameter for Hubbard model.
        u (float): Interaction parameter for Hubbard model.
        pbc (bool): Use periodic boundary conditions for Hubbard model.
        num_processes (int): Number of parallel processes to use.

    Returns:
        dict: A dictionary containing statistics (variances, means, parameters, etc.).
    """
    # 1. Generate Hamiltonian and mapper
    hamiltonian, mapper = generate_hamiltonian(norb, nelec, hamiltonian_type, t, u, pbc)
    
    # 2. Generate Ansatz
    ansatz = generate_ansatz(norb, nelec, mapper)
    num_params = ansatz.num_parameters
    
    # 3. Sample random parameters uniformly from [-pi, pi]
    parameter_samples = np.random.uniform(-np.pi, np.pi, (num_samples, num_params))
    
    print(f"Total parameterized coefficients: {num_params}")
    print(f"Starting evaluation for {num_samples} samples...")
    t0 = time.time()
    
    # 4. Compute gradients and energies
    if num_processes > 1:
        # Split samples across workers, removing any empty arrays if num_processes > num_samples
        chunks = np.array_split(parameter_samples, num_processes)
        pool_args = [(ansatz, hamiltonian, chunk) for chunk in chunks if len(chunk) > 0]
        
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(worker_evaluate_batch, pool_args)
            
        gradients = np.concatenate([res[0] for res in results], axis=0)
        energies = np.concatenate([res[1] for res in results], axis=0)
    else:
        gradients = compute_gradients_batch(ansatz, hamiltonian, parameter_samples)
        energies = compute_energies_batch(ansatz, hamiltonian, parameter_samples)
    
    t1 = time.time()
    print(f"Evaluation completed in {t1 - t0:.4f} seconds.")
    
    # 5. Compute statistics
    means = np.mean(gradients, axis=0)
    variances = np.var(gradients, axis=0)
    mean_energy = np.mean(energies)
    var_energy = np.var(energies)
    
    return {
        "norb": norb,
        "nelec": nelec,
        "num_samples": num_samples,
        "hamiltonian_type": hamiltonian_type,
        "num_parameters": num_params,
        "means": means,
        "variances": variances,
        "gradients": gradients,
        "mean_energy": mean_energy,
        "var_energy": var_energy,
        "energies": energies
    }

def parse_arguments(args):
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Demonstrate barren plateaus using Qiskit's UCCSD ansatz.")
    
    parser.add_argument("--norb", type=int, default=4, help="Number of spatial orbitals. Default: 4.")
    parser.add_argument("--n_alpha", type=int, default=2, help="Number of spin-up electrons. Default: 2.")
    parser.add_argument("--n_beta", type=int, default=2, help="Number of spin-down electrons. Default: 2.")
    parser.add_argument("--num_samples", type=int, default=30, help="Number of random parameter samples. Default: 30.")
    
    parser.add_argument("--hamiltonian_type", type=str, default="hubbard", choices=["hubbard", "local"],
                        help=r"Type of Hamiltonian. 'hubbard' = 1D Hubbard model, 'local' = Local number operator n_{0, \uparrow}. Default: 'hubbard'.")
    
    parser.add_argument("--t", type=float, default=1.0, help="Tunneling parameter t for Hubbard Hamiltonian. Default: 1.0.")
    parser.add_argument("--u", type=float, default=4.0, help="Interaction parameter U for Hubbard Hamiltonian. Default: 4.0.")
    parser.add_argument("--pbc", action="store_true", help="Use periodic boundary conditions for Hubbard model. Default: False.")
    
    parser.add_argument("--parameter_index", type=int, default=0,
                        help="Index of parameter to report variance for. If -1, reports all. Default: 0.")
    parser.add_argument("--num_processes", type=int, default=None,
                        help="Number of parallel processes to use. Default: None (use all available CPUs).")
    
    return parser.parse_args(args)
 
def main(args):
    """
    Main function to execute the barren plateau study using Qiskit UCCSD.
    """
    parsed_args = parse_arguments(args)
    
    nelec = (parsed_args.n_alpha, parsed_args.n_beta)
    
    num_proc = parsed_args.num_processes
    if num_proc is None:
        num_proc = get_default_num_processes()
    
    print("="*60)
    print("      Qiskit UCCSD Barren Plateau Study      ")
    print("="*60)
    print(f"Number of spatial orbitals (norb): {parsed_args.norb}")
    print(f"Electrons (alpha, beta):           {nelec}")
    print(f"Hamiltonian type:                  {parsed_args.hamiltonian_type}")
    if parsed_args.hamiltonian_type == "hubbard":
        print(f"Hubbard parameters:                t = {parsed_args.t}, U = {parsed_args.u} (PBC = {parsed_args.pbc})")
    print(f"Number of samples:                 {parsed_args.num_samples}")
    print(f"Number of processes:               {num_proc}")
    print("-"*60)
    
    results = run_gradient_variance_study(
        norb=parsed_args.norb,
        nelec=nelec,
        num_samples=parsed_args.num_samples,
        hamiltonian_type=parsed_args.hamiltonian_type,
        t=parsed_args.t,
        u=parsed_args.u,
        pbc=parsed_args.pbc,
        num_processes=num_proc
    )
    
    print("-"*60)
    print("Study Results:")
    print(f"Total parameterized coefficients: {results['num_parameters']}")
    print(f"Mean energy (loss):               {results['mean_energy']:.6e}")
    print(f"Variance of loss (energy):        {results['var_energy']:.6e}")
    
    param_idx = parsed_args.parameter_index
    if param_idx == -1:
        print("\nGradient statistics for all parameters:")
        for idx in range(results['num_parameters']):
            print(f"Parameter θ[{idx}]: Mean = {results['means'][idx]:.6e}, Variance = {results['variances'][idx]:.6e}")
    else:
        if 0 <= param_idx < results['num_parameters']:
            print(f"Target Parameter θ[{param_idx}]:")
            print(f"  Mean of gradient:     {results['means'][param_idx]:.6e}")
            print(f"  Variance of gradient: {results['variances'][param_idx]:.6e}")
        else:
            print(f"Error: Parameter index {param_idx} is out of range (0 to {results['num_parameters']-1}).")
            
    print("="*60)

if __name__ == "__main__":
    main(sys.argv[1:])
