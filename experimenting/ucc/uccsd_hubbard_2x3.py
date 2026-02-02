import pennylane as qml
from pennylane import numpy as np
from scipy.sparse import linalg as sla
import warnings

# Suppress the warning about determining differentiability of Hermitian observable
warnings.filterwarnings("ignore", message="Differentiating with respect to the input parameters of Hermitian")

# --- 1. System and Hamiltonian Setup ---

n_rows = 2
n_cols = 3
n_qubits = 2 * n_rows * n_cols  # 12 qubits
n_electrons = 6  # Half-filling

u_coulomb = 4.0
hopping_t = 1.0

print(f"--- System Setup ---")
print(f"Lattice: {n_rows}x{n_cols}")
print(f"Qubits: {n_qubits}")
print(f"Electrons: {n_electrons} (Half-filling)")
print(f"U = {u_coulomb}, t = {hopping_t}")
print("-" * 20)

# Generate Hamiltonian
# Geometry: Square 2x3
H_obj = qml.spin.fermi_hubbard("square", [n_rows, n_cols], hopping=hopping_t, coulomb=u_coulomb)
H_sparse = H_obj.sparse_matrix()

# --- 2. Exact Diagonalization (Benchmark) ---

print("Calculating exact ground state (lanczos)...")
# We start with a random vector in the target sector (N_up=3, N_down=3) to guide the solver
# assuming interleaved ordering: up, down, up, down...
# Actually, let's just use a general sector filter if needed, but for now generic sparse eigsh
# with enough roots should find it.
# To be safe and fast, let's just run it.
evals, evecs = sla.eigsh(H_sparse, k=5, which="SA")
exact_energy = evals[0]
print(f"Exact Ground State Energy: {exact_energy:.6f}")
print("-" * 20)


# --- 3. VQE Setup ---

dev = qml.device("lightning.qubit", wires=n_qubits, shots=None)

# Define Hartree-Fock State
# For half-filling, we just fill the first N_electrons spin-orbitals
# This corresponds to filling the lowest energy levels in non-interacting limit if ordered by energy
# But here we just pick a computational basis state that has correct particle number.
# A logical choice for 2x3 AFM ground state seed is Néel state, but HF simply filling first N indices is standard start.
hf_state = qml.qchem.hf_state(n_electrons, n_qubits)
print(f"Hartree-Fock State: {hf_state}")

# Generate Excitations
# This can be expensive for large systems.
singles, doubles = qml.qchem.excitations(n_electrons, n_qubits)
s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)
print(f"Number of Single Excitations: {len(singles)}")
print(f"Number of Double Excitations: {len(doubles)}")
n_coeffs = len(singles) + len(doubles)

@qml.qnode(dev, interface="autograd", diff_method="adjoint")
def circuit(weights):
    qml.BasisState(hf_state, wires=range(n_qubits))
    
    # Apply UCCSD cluster operator
    # Note: UCCSD typically implies exp(T - T_dag). PennyLane Fermionic Excitation gates implement this.
    
    # Singles
    for i in range(len(s_wires)):
        qml.FermionicSingleExcitation(weights[i], wires=s_wires[i])

    # Doubles
    for i in range(len(d_wires)):
        wires_from = d_wires[i][0]
        wires_to = d_wires[i][1]
        qml.FermionicDoubleExcitation(
            weights[len(s_wires) + i], wires1=wires_to, wires2=wires_from
        )
        
    return qml.expval(H_obj)

# --- 4. Optimization ---

# Initialize parameters
# Start effectively from HF state (params=0) implies result is HF energy.
# VQE usually requires small random noise to break symmetries if HF is stationary point.
np.random.seed(42)
params = np.array(np.random.normal(0, 0.01, n_coeffs), requires_grad=True)

optimizer = qml.AdamOptimizer(stepsize=0.05)
max_steps = 100

print(f"--- Starting VQE Optimization ---")
print(f"Initial Energy: {circuit(params):.6f}")

for n in range(max_steps):
    params, energy = optimizer.step_and_cost(circuit, params)
    
    print(f"Step {n+1:3d}: Energy = {energy:.8f} (Error: {energy - exact_energy:.8f})")

print("-" * 20)
print(f"Final VQE Energy: {energy:.8f}")
print(f"Exact Energy:     {exact_energy:.6f}")
print(f"Energy Error:     {energy - exact_energy:.8f}")
