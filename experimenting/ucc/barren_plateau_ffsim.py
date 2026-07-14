"""
barren_plateau_ffsim.py

Demonstrate barren plateaus using the real-valued restricted UCCSD ansatz.
Computes the variance of the loss (energy expectation value) over random parameter samples
for a real restricted UCCSD ansatz of variable spatial orbitals (sites) and electron count.

Usage:
  python barren_plateau_ffsim.py [options]

Options:
  --norb=<n> (optional): Number of spatial orbitals. Default: 4.
  --nocc=<o> (optional): Number of occupied spatial orbitals in the reference state. Default: 2.
  --n_alpha=<a> (optional): Number of spin-up electrons. Default: 2.
  --n_beta=<b> (optional): Number of spin-down electrons. Default: 2.
  --num_samples=<s> (optional): Number of random parameter samples to estimate variance. Default: 30.
  --hamiltonian_type=<type> (optional): Type of Hamiltonian. Default: "hubbard".
                        Valid options:
                        - "hubbard": 1D Fermi-Hubbard chain with parameters t and U.
                        - "random": Random Hermitian one-body tensor and zero two-body tensor.
  --t=<t> (optional): Tunneling parameter t for Hubbard Hamiltonian. Default: 1.0.
  --u=<u> (optional): Interaction parameter U for Hubbard Hamiltonian. Default: 4.0.
  --pbc (optional): Use periodic boundary conditions for Hubbard model. Default: False.
  --num_processes=<p> (optional): Number of processes for parallel execution. Default: None (use all available CPUs).

Examples:
  python barren_plateau_ffsim.py --norb=4 --nocc=2 --num_samples=30 --hamiltonian_type=hubbard
"""

import os
import sys
import argparse
import time
import numpy as np
import concurrent.futures

import ffsim

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
    Generates the Hamiltonian linear operator in site basis.

    Parameters:
        norb (int): Number of spatial orbitals.
        nelec (tuple[int, int]): Number of alpha and beta electrons.
        hamiltonian_type (str): 'hubbard' or 'random'.
        t (float): Tunneling parameter for Hubbard model.
        u (float): Interaction parameter for Hubbard model.
        pbc (bool): Use periodic boundary conditions for Hubbard model.

    Returns:
        scipy.sparse.linalg.LinearOperator: The Hamiltonian operator.
    """
    if hamiltonian_type == 'hubbard':
        mat_1h = np.zeros((norb, norb), dtype=complex)
        for i in range(norb - 1):
            mat_1h[i, i + 1] = -t
            mat_1h[i + 1, i] = -t
        if pbc and norb > 2:
            mat_1h[norb - 1, 0] = -t
            mat_1h[0, norb - 1] = -t
            
        mat_2h = np.zeros((norb, norb, norb, norb), dtype=complex)
        for i in range(norb):
            mat_2h[i, i, i, i] = u
            
        mol_ham = ffsim.MolecularHamiltonian(one_body_tensor=mat_1h, two_body_tensor=mat_2h)
        return ffsim.linear_operator(mol_ham, norb, nelec)
    elif hamiltonian_type == 'random':
        mat_1h = np.random.randn(norb, norb) + 1j * np.random.randn(norb, norb)
        mat_1h = (mat_1h + mat_1h.conj().T) / 2.0
        mat_2h = np.zeros((norb, norb, norb, norb), dtype=complex)
        mol_ham = ffsim.MolecularHamiltonian(one_body_tensor=mat_1h, two_body_tensor=mat_2h)
        return ffsim.linear_operator(mol_ham, norb, nelec)
    else:
        raise ValueError(f"Unknown Hamiltonian type: {hamiltonian_type}. Use 'hubbard' or 'random'.")

def compute_energy(params, norb, nocc, nelec, hf_state, h_op):
    """
    Computes the expectation value of the Hamiltonian for a given set of parameters.
    """
    op = ffsim.UCCSDOpRestrictedReal.from_parameters(params, norb=norb, nocc=nocc)
    state = ffsim.apply_unitary(hf_state, op, norb=norb, nelec=nelec)
    return np.real(np.vdot(state, h_op @ state))

# Globals to share state vectors across worker processes without pickling
_global_hf_state = None
_global_h_op = None

def _init_worker(norb, nelec, hamiltonian_type, t, u, pbc):
    """
    Initializes worker process global state to avoid recreating/pickling heavy objects.
    """
    global _global_hf_state, _global_h_op
    _global_hf_state = ffsim.hartree_fock_state(norb, nelec)
    _global_h_op = generate_hamiltonian(norb, nelec, hamiltonian_type, t, u, pbc)

def _worker_compute_energy(params, norb, nocc, nelec):
    """
    Computes energy expectation value for a worker process using its local global state.
    """
    global _global_hf_state, _global_h_op
    op = ffsim.UCCSDOpRestrictedReal.from_parameters(params, norb=norb, nocc=nocc)
    state = ffsim.apply_unitary(_global_hf_state, op, norb=norb, nelec=nelec)
    return np.real(np.vdot(state, _global_h_op @ state))

def run_loss_variance_study(norb, nocc, nelec, num_samples, hamiltonian_type='hubbard', 
                            t=1.0, u=4.0, pbc=False, num_processes=None):
    """
    Runs the study to compute the variance of the loss (energy expectation value) over random parameter samples.
    """
    # 1. Calculate number of parameters
    num_params = ffsim.UCCSDOpRestrictedReal.n_params(norb, nocc)
    
    print(f"Total parameterized coefficients: {num_params}")
    print(f"Starting parallel energy evaluation for {num_samples} samples...")
    t0 = time.time()
    
    # 2. Sample parameters
    parameter_samples = [np.random.uniform(-np.pi, np.pi, num_params) for _ in range(num_samples)]
    
    # 3. Compute energies in parallel
    energies = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_processes,
        initializer=_init_worker,
        initargs=(norb, nelec, hamiltonian_type, t, u, pbc)
    ) as executor:
        futures = [
            executor.submit(_worker_compute_energy, params, norb, nocc, nelec)
            for params in parameter_samples
        ]
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            energies.append(future.result())
            completed += 1
            if completed % 5 == 0 or completed == num_samples:
                print(f"Evaluated {completed}/{num_samples} samples...")
            
    energies = np.array(energies)
    t1 = time.time()
    print(f"Energy evaluation completed in {t1 - t0:.4f} seconds.")
    
    # 5. Compute statistics
    mean_energy = np.mean(energies)
    var_energy = np.var(energies)
    
    return {
        "norb": norb,
        "nocc": nocc,
        "nelec": nelec,
        "num_samples": num_samples,
        "hamiltonian_type": hamiltonian_type,
        "num_parameters": num_params,
        "mean_energy": mean_energy,
        "var_energy": var_energy,
        "energies": energies
    }

def parse_arguments(args):
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Demonstrate barren plateaus using ffsim real-valued restricted UCCSD ansatz.")
    
    parser.add_argument("--norb", type=int, default=4, help="Number of spatial orbitals. Default: 4.")
    parser.add_argument("--nocc", type=int, default=None, help="Number of occupied spatial orbitals in reference state. Default: 2.")
    parser.add_argument("--n_alpha", type=int, default=None, help="Number of spin-up electrons. Default: 2.")
    parser.add_argument("--n_beta", type=int, default=None, help="Number of spin-down electrons. Default: 2.")
    parser.add_argument("--num_samples", type=int, default=30, help="Number of random parameter samples. Default: 30.")
    
    parser.add_argument("--hamiltonian_type", type=str, default="hubbard", choices=["hubbard", "random"],
                        help="Type of Hamiltonian. 'hubbard' = 1D Hubbard model, 'random' = Random Hermitian one-body tensor. Default: 'hubbard'.")
    
    parser.add_argument("--t", type=float, default=1.0, help="Tunneling parameter t for Hubbard Hamiltonian. Default: 1.0.")
    parser.add_argument("--u", type=float, default=4.0, help="Interaction parameter U for Hubbard Hamiltonian. Default: 4.0.")
    parser.add_argument("--pbc", action="store_true", help="Use periodic boundary conditions for Hubbard model. Default: False.")
    parser.add_argument("--num_processes", type=int, default=None,
                        help="Number of processes for parallel execution. Default: None (use all available CPUs).")
    
    parsed_args = parser.parse_args(args)
    
    # Resolve optional inputs consistently
    nocc = parsed_args.nocc
    n_alpha = parsed_args.n_alpha
    n_beta = parsed_args.n_beta
    
    if nocc is not None:
        if n_alpha is None:
            n_alpha = nocc
        if n_beta is None:
            n_beta = nocc
    else:
        if n_alpha is not None and n_beta is not None:
            nocc = min(n_alpha, n_beta)
        elif n_alpha is not None:
            n_beta = n_alpha
            nocc = n_alpha
        elif n_beta is not None:
            n_alpha = n_beta
            nocc = n_beta
        else:
            # Fallback default when none of the optional inputs are specified
            nocc = 2
            n_alpha = 2
            n_beta = 2
            
    # Set resolved values back to parsed_args
    parsed_args.nocc = nocc
    parsed_args.n_alpha = n_alpha
    parsed_args.n_beta = n_beta
    
    # Validation
    if parsed_args.n_alpha > parsed_args.norb or parsed_args.n_beta > parsed_args.norb:
        parser.error(f"Number of electrons (n_alpha={parsed_args.n_alpha}, n_beta={parsed_args.n_beta}) cannot exceed number of spatial orbitals (norb={parsed_args.norb}).")
    if parsed_args.nocc > parsed_args.norb:
        parser.error(f"Number of occupied spatial orbitals (nocc={parsed_args.nocc}) cannot exceed number of spatial orbitals (norb={parsed_args.norb}).")
    if parsed_args.nocc > min(parsed_args.n_alpha, parsed_args.n_beta):
        parser.error(f"Number of occupied spatial orbitals (nocc={parsed_args.nocc}) cannot exceed the number of spin-up (n_alpha={parsed_args.n_alpha}) or spin-down (n_beta={parsed_args.n_beta}) electrons.")
        
    return parsed_args

def main(args):
    """
    Main function to execute the barren plateau study using ffsim.
    """
    parsed_args = parse_arguments(args)
    
    nelec = (parsed_args.n_alpha, parsed_args.n_beta)
    
    num_proc = parsed_args.num_processes
    if num_proc is None:
        num_proc = get_default_num_processes()
    
    print("="*60)
    print("      ffsim UCCSD Restricted Real Barren Plateau Study      ")
    print("="*60)
    print(f"Number of spatial orbitals (norb): {parsed_args.norb}")
    print(f"Number of occupied (nocc):         {parsed_args.nocc}")
    print(f"Electrons (alpha, beta):           {nelec}")
    print(f"Hamiltonian type:                  {parsed_args.hamiltonian_type}")
    if parsed_args.hamiltonian_type == "hubbard":
        print(f"Hubbard parameters:                t = {parsed_args.t}, U = {parsed_args.u} (PBC = {parsed_args.pbc})")
    print(f"Number of samples:                 {parsed_args.num_samples}")
    print(f"Number of processes:               {num_proc}")
    print("-"*60)
    
    results = run_loss_variance_study(
        norb=parsed_args.norb,
        nocc=parsed_args.nocc,
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
    print(f"Mean of energy (loss):           {results['mean_energy']:.6e}")
    print(f"Variance of energy (loss):       {results['var_energy']:.6e}")
    print("="*60)

if __name__ == "__main__":
    main(sys.argv[1:])
