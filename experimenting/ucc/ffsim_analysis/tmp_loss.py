
import ffsim
import ffsim.optimize
import numpy as np
import scipy.sparse.linalg
import scipy.optimize
import time
from ffsim import UCCSDOpRestricted

def get_hamiltonian(t, u, geometry):
    nx, ny = geometry
    n_sites = nx * ny
    
    # One-body tensor (Hopping)
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

    # Two-body tensor (Interaction)
    mat_2h = np.zeros((n_sites, n_sites, n_sites, n_sites), dtype=complex)
    for i in range(n_sites):
        mat_2h[i, i, i, i] = u

    mol_ham = ffsim.MolecularHamiltonian(one_body_tensor=mat_1h, two_body_tensor=mat_2h)
    return mol_ham, mat_1h, mat_2h

def main():
    # --- Hubbard Model Parameters ---
    nx, ny = 2, 3
    geometry = (nx, ny)
    n_sites = nx * ny
    n_particles = (3, 3)  # 3 spin-up, 3 spin-down
    t_val = 1.0
    u_val = 4.0
    
    print(f"Hubbard Model: {nx}x{ny} lattice, N_up={n_particles[0]}, N_down={n_particles[1]}")
    
    # --- 1. Get Hamiltonians & Diagonalize ---
    
    # Non-interacting (U=0)
    print("Constructing non-interacting Hamiltonian (U=0)...")
    mol_ham_0, mat_1h_0, _ = get_hamiltonian(t_val, 0.0, geometry)
    linop_0 = ffsim.linear_operator(mol_ham_0, n_sites, n_particles)
    print("Diagonalizing H(U=0)...")
    e0, v1_states = scipy.sparse.linalg.eigsh(linop_0, k=1, which="SA")
    v1 = v1_states[:, 0]
    print(f"Ground Energy (U=0): {e0[0]:.6f}")

    # Interacting (U=4)
    print(f"Constructing interacting Hamiltonian (U={u_val})...")
    mol_ham_int, mat_1h_int, mat_2h_int = get_hamiltonian(t_val, u_val, geometry)
    linop_int = ffsim.linear_operator(mol_ham_int, n_sites, n_particles)
    print(f"Diagonalizing H(U={u_val})...")
    e_int, v2_states = scipy.sparse.linalg.eigsh(linop_int, k=1, which="SA")
    v2 = v2_states[:, 0]
    print(f"Ground Energy (U={u_val}): {e_int[0]:.6f}")

    # --- 2. Setup UCCSD Ansatz ---
    print("Setting up UCCSD Ansatz...")
    
    # To check overlap, we need to apply the unitary to v1 and measure overlap with v2.
    # Note: v1 is NOT necessarily a Hartree-Fock state, it is the exact ground state of H(U=0).
    
    n_occ = n_particles[0]
    n_virt = n_sites - n_occ
    size_t1 = n_occ * n_virt
    size_t2 = n_occ * n_occ * n_virt * n_virt
    n_params = size_t1 + size_t2
    
    print(f"Number of parameters: {n_params}")

    # Define cost function: 1 - |<v2| U(theta) |v1>|^2
    def cost_fn(theta):
        # Unpack parameters
        t1 = theta[:size_t1].reshape(n_occ, n_virt)
        t2 = theta[size_t1:].reshape(n_occ, n_occ, n_virt, n_virt)
        
        # Construct operator
        op = UCCSDOpRestricted(t1=t1, t2=t2)
        
        v_final = ffsim.apply_unitary(
            v1,
            op, 
            norb=n_sites,
            nelec=n_particles
        )
        
        overlap = np.vdot(v2, v_final)
        loss = 1.0 - (abs(overlap)**2)
        # print(loss)
        return loss

    print("Starting optimization...")
    t0 = time.time()
    # Initialize with small random noise or zeros? Zeros implies identity.
    # If the states are different, we need some non-zero parameters.
    # Let's try zeros first.
    initial_guess = np.zeros(n_params)
    
    res = scipy.optimize.minimize(cost_fn, initial_guess, method="L-BFGS-B")
    
    print(f"Optimization finished.")
    print(f"Final Loss: {res.fun:.6e}")
    print(f"Overlap |<v2|U|v1>|^2: {1.0 - res.fun:.6f}")
    print(f"Time: {time.time() - t0:.2f} seconds")

if __name__ == "__main__":
    main()
