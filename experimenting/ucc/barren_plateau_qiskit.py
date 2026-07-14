"""
barren_plateau_qiskit.py

Demonstrate barren plateaus in quantum neural networks (QNNs) using Qiskit.
Computes the variance of the gradient of the energy expectation value over random parameter samples
for a hardware-efficient ansatz of variable width (qubits) and depth.

Usage:
  python barren_plateau_qiskit.py [options]

Options:
  --num_qubits=<n> (optional): Number of qubits. Default: 4.
  --depth=<d> (optional): Depth of the hardware-efficient ansatz. Default: 3.
  --num_samples=<s> (optional): Number of random parameter samples to estimate variance. Default: 50.
  --hamiltonian_type=<type> (optional): Type of Hamiltonian. Default: "local".
                        Valid options:
                        - "local": Local Z operator on the first qubit (Z_0).
                        - "global": Global Z operator on all qubits (Z_0 Z_1 ... Z_n-1).
                        - "tfim": 1D Transverse Field Ising Model (TFIM) H = -J \sum Z_i Z_i+1 - g \sum X_i.
  --tfim_j=<J> (optional): Coupling constant J for TFIM. Default: 1.0.
  --tfim_g=<g> (optional): Transverse field strength g for TFIM. Default: 1.0.
  --ansatz_type=<type> (optional): Type of hardware-efficient ansatz. Default: "real_amplitudes".
                        Valid options:
                        - "real_amplitudes": RealAmplitudes circuit (Ry rotations and CNOTs).
                        - "efficient_su2": EfficientSU2 circuit (Ry, Rz rotations and CNOTs).
  --entanglement=<type> (optional): Entanglement strategy for ansatz. Default: "linear".
                        Valid options:
                        - "linear": Linear entanglement.
                        - "full": Full (all-to-all) entanglement.
  --parameter_index=<i> (optional): Index of the parameter to compute the gradient variance for.
                         If set to -1, computes and reports variance for all parameters. Default: 0.

Examples:
  python barren_plateau_qiskit.py --num_qubits=4 --depth=3 --num_samples=50 --hamiltonian_type=local
"""

import sys
import argparse
import time
import numpy as np

from qiskit.circuit.library import real_amplitudes, efficient_su2
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.gradients import ReverseEstimatorGradient
from qiskit.primitives import StatevectorEstimator

def generate_ansatz(num_qubits, depth, ansatz_type='real_amplitudes', entanglement='linear'):
    """
    Generates a hardware-efficient ansatz circuit.

    Parameters:
        num_qubits (int): Number of qubits.
        depth (int): Repetitions (depth) of the ansatz layers.
        ansatz_type (str): Either 'real_amplitudes' or 'efficient_su2'.
        entanglement (str): Entanglement pattern, e.g. 'linear' or 'full'.

    Returns:
        QuantumCircuit: The parameterized ansatz circuit.
    """
    if ansatz_type == 'real_amplitudes':
        return real_amplitudes(num_qubits=num_qubits, reps=depth, entanglement=entanglement)
    elif ansatz_type == 'efficient_su2':
        return efficient_su2(num_qubits=num_qubits, reps=depth, entanglement=entanglement)
    else:
        raise ValueError(f"Unknown ansatz type: {ansatz_type}. Use 'real_amplitudes' or 'efficient_su2'.")

def generate_hamiltonian(num_qubits, hamiltonian_type='local', tfim_j=1.0, tfim_g=1.0):
    """
    Generates the Hamiltonian operator.

    Parameters:
        num_qubits (int): Number of qubits.
        hamiltonian_type (str): Either 'local', 'global', or 'tfim'.
        tfim_j (float): Coupling constant J for TFIM.
        tfim_g (float): Transverse field strength g for TFIM.

    Returns:
        SparsePauliOp: The Hamiltonian operator.
    """
    if hamiltonian_type == 'local':
        # Local Hamiltonian: Z on the first qubit (Z_0), identity elsewhere
        return SparsePauliOp.from_sparse_list([("Z", [0], 1.0)], num_qubits=num_qubits)
    elif hamiltonian_type == 'global':
        # Global Hamiltonian: Z on all qubits (Z_0 Z_1 ... Z_n-1)
        return SparsePauliOp.from_list([("Z" * num_qubits, 1.0)])
    elif hamiltonian_type == 'tfim':
        # 1D Transverse Field Ising Model (TFIM) with open boundaries:
        # H = - J * \sum Z_i Z_{i+1} - g * \sum X_i
        terms = []
        for i in range(num_qubits - 1):
            terms.append(("ZZ", [i, i+1], -tfim_j))
        for i in range(num_qubits):
            terms.append(("X", [i], -tfim_g))
        return SparsePauliOp.from_sparse_list(terms, num_qubits=num_qubits)
    else:
        raise ValueError(f"Unknown Hamiltonian type: {hamiltonian_type}. Use 'local', 'global', or 'tfim'.")

def compute_gradients_batch(ansatz, hamiltonian, parameter_samples):
    """
    Computes the gradients of the expectation value of the Hamiltonian with respect to the
    parameters of the ansatz for a batch of random parameter samples.

    Parameters:
        ansatz (QuantumCircuit): The parameterized ansatz circuit.
        hamiltonian (SparsePauliOp): The Hamiltonian operator.
        parameter_samples (np.ndarray): 2D array of shape (num_samples, num_parameters).

    Returns:
        np.ndarray: 2D array of shape (num_samples, num_parameters) containing the gradients.
    """
    num_samples = len(parameter_samples)
    grad_estimator = ReverseEstimatorGradient()
    
    # We batch the execution by passing a list of circuits and operators
    # corresponding to the parameter samples.
    circuits = [ansatz] * num_samples
    observables = [hamiltonian] * num_samples
    
    job = grad_estimator.run(circuits, observables, parameter_samples)
    result = job.result()
    
    # Convert list of gradients into a 2D numpy array
    return np.array(result.gradients)

def run_gradient_variance_study(num_qubits, depth, num_samples, hamiltonian_type='local', 
                                ansatz_type='real_amplitudes', entanglement='linear',
                                tfim_j=1.0, tfim_g=1.0):
    """
    Runs the study to compute the variance of the gradients of the energy expectation value.

    Parameters:
        num_qubits (int): Number of qubits.
        depth (int): Depth of the ansatz.
        num_samples (int): Number of random parameter samples to draw.
        hamiltonian_type (str): 'local', 'global', or 'tfim'.
        ansatz_type (str): 'real_amplitudes' or 'efficient_su2'.
        entanglement (str): 'linear' or 'full'.
        tfim_j (float): Coupling constant J for TFIM.
        tfim_g (float): Transverse field strength g for TFIM.

    Returns:
        dict: A dictionary containing statistics (variances, means, parameters, etc.).
    """
    # 1. Generate Ansatz
    ansatz = generate_ansatz(num_qubits, depth, ansatz_type, entanglement)
    num_params = ansatz.num_parameters
    
    # 2. Generate Hamiltonian
    hamiltonian = generate_hamiltonian(num_qubits, hamiltonian_type, tfim_j, tfim_g)
    
    # 3. Sample random parameters uniformly from [-pi, pi]
    parameter_samples = np.random.uniform(-np.pi, np.pi, (num_samples, num_params))
    
    print(f"Starting gradient evaluation for {num_samples} samples...")
    t0 = time.time()
    
    # 4. Compute gradients
    gradients = compute_gradients_batch(ansatz, hamiltonian, parameter_samples)
    
    t1 = time.time()
    print(f"Gradient evaluation completed in {t1 - t0:.4f} seconds.")

    print(f"Starting energy evaluation for {num_samples} samples...")
    t2 = time.time()
    estimator = StatevectorEstimator()
    pub = (ansatz, hamiltonian, parameter_samples)
    job = estimator.run([pub])
    energies = np.array(job.result()[0].data.evs)
    t3 = time.time()
    print(f"Energy evaluation completed in {t3 - t2:.4f} seconds.")
    
    # 5. Compute statistics
    means = np.mean(gradients, axis=0)
    variances = np.var(gradients, axis=0)
    
    return {
        "num_qubits": num_qubits,
        "depth": depth,
        "num_samples": num_samples,
        "hamiltonian_type": hamiltonian_type,
        "ansatz_type": ansatz_type,
        "entanglement": entanglement,
        "tfim_j": tfim_j,
        "tfim_g": tfim_g,
        "num_parameters": num_params,
        "means": means,
        "variances": variances,
        "gradients": gradients,
        "mean_energy": np.mean(energies),
        "var_energy": np.var(energies),
        "energies": energies
    }

def parse_arguments(args):
    """
    Parses command-line arguments.

    Parameters:
        args (list): List of command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Demonstrate barren plateaus using Qiskit.")
    
    parser.add_argument("--num_qubits", type=int, default=4, help="Number of qubits. Default: 4.")
    parser.add_argument("--depth", type=int, default=3, help="Depth of the hardware-efficient ansatz. Default: 3.")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of random parameter samples. Default: 50.")
    
    parser.add_argument("--hamiltonian_type", type=str, default="local", choices=["local", "global", "tfim"],
                        help="Type of Hamiltonian. 'local' = Z_0, 'global' = Z_0 Z_1 ... Z_n-1, 'tfim' = 1D Transverse Field Ising Model. Default: 'local'.")
    parser.add_argument("--tfim_j", type=float, default=1.0, help="Coupling constant J for TFIM. Default: 1.0.")
    parser.add_argument("--tfim_g", type=float, default=1.0, help="Transverse field strength g for TFIM. Default: 1.0.")
    
    parser.add_argument("--ansatz_type", type=str, default="real_amplitudes", choices=["real_amplitudes", "efficient_su2"],
                        help="Type of hardware-efficient ansatz. 'real_amplitudes' or 'efficient_su2'. Default: 'real_amplitudes'.")
    
    parser.add_argument("--entanglement", type=str, default="linear", choices=["linear", "full"],
                        help="Entanglement pattern for ansatz. 'linear' or 'full'. Default: 'linear'.")
    
    parser.add_argument("--parameter_index", type=int, default=0,
                        help="Index of parameter to report variance for. If -1, reports all. Default: 0.")
    
    return parser.parse_args(args)

def main(args):
    """
    Main function to execute the barren plateau demonstration.
    """
    parsed_args = parse_arguments(args)
    
    print("="*60)
    print("      Qiskit Barren Plateau Demonstration      ")
    print("="*60)
    print(f"Number of qubits:     {parsed_args.num_qubits}")
    print(f"Ansatz type:          {parsed_args.ansatz_type}")
    print(f"Ansatz depth:         {parsed_args.depth}")
    print(f"Entanglement:         {parsed_args.entanglement}")
    print(f"Hamiltonian type:     {parsed_args.hamiltonian_type}")
    print(f"Number of samples:    {parsed_args.num_samples}")
    print("-"*60)
    
    results = run_gradient_variance_study(
        num_qubits=parsed_args.num_qubits,
        depth=parsed_args.depth,
        num_samples=parsed_args.num_samples,
        hamiltonian_type=parsed_args.hamiltonian_type,
        ansatz_type=parsed_args.ansatz_type,
        entanglement=parsed_args.entanglement,
        tfim_j=parsed_args.tfim_j,
        tfim_g=parsed_args.tfim_g
    )
    
    print("-"*60)
    print("Study Results:")
    print(f"Total parameterized coefficients: {results['num_parameters']}")
    print(f"Mean of energy (loss):           {results['mean_energy']:.6e}")
    print(f"Variance of energy (loss):       {results['var_energy']:.6e}")
    
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
