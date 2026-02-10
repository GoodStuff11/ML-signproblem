
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
    print("Setting up UCCSD Ansatz...")
    
    # Use UCCSDOpRestricted if available, as it handles parameters for us
    try:
        from ffsim import UCCSDOpRestricted
        # UCCSDOpRestricted expects t1 and t2 tensors (amplitudes)
        n_occ = n_particles[0]
        n_virt = n_sites - n_occ
        t1 = np.zeros((n_occ, n_virt))
        t2 = np.zeros((n_occ, n_occ, n_virt, n_virt))
        ansatz_op = UCCSDOpRestricted(t1=t1, t2=t2)
    except ImportError:
        # Fallback or error
        print("Error: UCCSDOpRestricted not found.")
        return


    # Define fallback cost function
    def cost_fn(theta):
        t = time.time()
        # Unpack parameters
        # t1: (n_occ, n_virt) -> (3, 3) size 9
        # t2: (n_occ, n_occ, n_virt, n_virt) -> (3, 3, 3, 3) size 81
        n_occ = n_particles[0]
        n_virt = n_sites - n_occ
        
        size_t1 = n_occ * n_virt
        t1 = theta[:size_t1].reshape(n_occ, n_virt)
        t2 = theta[size_t1:].reshape(n_occ, n_occ, n_virt, n_virt)
        
        # Construct operator with current params
        op = UCCSDOpRestricted(t1=t1, t2=t2)
        
        # Apply ansatz
        # apply_unitary(vec, op, norb, nelec)
        final_state = ffsim.apply_unitary(
            hartree_fock_state,
            op, 
            norb=n_sites,
            nelec=n_particles
        )
        
        ret = np.vdot(final_state, linop_mo @ final_state).real
        # print(time.time() - t)
        return ret

    # Calculate parameter count
    n_occ = n_particles[0]
    n_virt = n_sites - n_occ
    size_t1 = n_occ * n_virt
    size_t2 = n_occ * n_occ * n_virt * n_virt
    n_params = size_t1 + size_t2
    
    print("Starting optimization using scipy.optimize...")
    t0 = time.time()
    res = scipy.optimize.minimize(cost_fn, np.zeros(n_params), method="L-BFGS-B")
    print(f"VQE Optimized Energy (Scipy): {res.fun:.6f}")

    error = abs(res.fun - ground_energy)
    print(f"Error: {error:.2e}")
    print(f"Time: {time.time() - t0:.2f} seconds")
    print(f"Number of parameters: {n_params}")
    

if __name__ == "__main__":
    main()
