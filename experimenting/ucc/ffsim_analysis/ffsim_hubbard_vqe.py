
import ffsim
import ffsim.optimize
import numpy as np
import scipy.sparse.linalg
import scipy.optimize
import time

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
    e_min, _ = scipy.sparse.linalg.eigsh(linop, k=1, which="SA")
    ground_energy = e_min[0]
    print(f"Exact Ground Energy: {ground_energy:.6f}")

    # --- 3. MO Basis Transformation & HF ---
    e_orbs, orbital_coeffs = np.linalg.eigh(mat_1h)
    
    # Manual transformation
    C = orbital_coeffs
    one_body_mo = C.T.conj() @ mat_1h @ C
    two_body_mo = np.einsum('pqrs,pi,qj,rk,sl->ijkl', mat_2h, C, C, C, C, optimize=True)
    mol_ham_mo = ffsim.MolecularHamiltonian(one_body_tensor=one_body_mo, two_body_tensor=two_body_mo)
    
    hartree_fock_state = ffsim.hartree_fock_state(n_sites, n_particles)
    
    linop_mo = ffsim.linear_operator(mol_ham_mo, n_sites, n_particles)
    hf_energy = np.vdot(hartree_fock_state, linop_mo @ hartree_fock_state).real
    print(f"Hartree-Fock Energy: {hf_energy:.6f}")

    # --- 4. UCCSD Ansatz ---
    print("Setting up UCCSD Ansatz (Unrestricted)...")
    
    try:
        from uccsd_op import UCCSDOp
        # Calculate parameter count
        n_params = UCCSDOp.n_params(n_sites, n_particles)
        print(f"Number of parameters: {n_params}")
    except ImportError as e:
        print(f"Error importing UCCSDOp: {e}")
        return

    # Define cost function using UCCSDOp
    def cost_fn(theta):
        # Construct operator with current params
        op = UCCSDOp.from_parameters(theta, norb=n_sites, nelec=n_particles)
        
        # Apply ansatz
        final_state = ffsim.apply_unitary(
            hartree_fock_state,
            op, 
            norb=n_sites,
            nelec=n_particles
        )
        
        ret = np.vdot(final_state, linop_mo @ final_state).real
        return ret

    print("Starting optimization using Linear Method...")
    
    from linear_method import minimize_linear_method
    from uccsd_op import jacobian_uccsd_unrestricted
    
    def params_to_vec(theta):
        op = UCCSDOp.from_parameters(theta, norb=n_sites, nelec=n_particles)
        return ffsim.apply_unitary(hartree_fock_state, op, norb=n_sites, nelec=n_particles)
        
    def jacobian_func(theta, vec):
        return jacobian_uccsd_unrestricted(theta, vec, norb=n_sites, nelec=n_particles)

    t0 = time.time()
    # Use zeros as initial guess
    res = minimize_linear_method(
        params_to_vec,
        linop_mo,
        np.zeros(n_params),
        jacobian_func=jacobian_func,
        maxiter=30,
        regularization=0.01,
        optimize_regularization=True,
        callback=lambda res: print(f"Iter {res.nit}: Energy = {res.fun:.6f}" + (f", Norm Grad = {np.linalg.norm(res.jac):.2e}" if hasattr(res, "jac") else ""))
    )
    
    print(f"VQE Optimized Energy (Linear Method): {res.fun:.6f}")
    
    error = abs(res.fun - ground_energy)
    print(f"Error: {error:.2e}")
    print(f"Nfev: {res.nfev}, Nlinop: {res.nlinop}")
    print(f"Time: {time.time() - t0:.2f} seconds")

if __name__ == "__main__":
    main()
