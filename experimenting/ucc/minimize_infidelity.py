import pennylane as qml
from pennylane import numpy as np
from scipy.sparse import linalg as sla
import warnings

# Suppress the warning about determining differentiability of Hermitian observable
warnings.filterwarnings("ignore", message="Differentiating with respect to the input parameters of Hermitian")

# --- 1. System and Hamiltonian Setup ---

# We will model a 2x3 Hubbard model, which maps to 12 qubits (6 sites x 2 spin-orbitals)
# with 6 electrons (half-filling).

n_qubits = 12
n_electrons = 6

# Option to specify spin sectors
use_spin_sector = True
target_n_up = 3
target_n_down = 3

def get_sector_indices(n_qubits, n_electrons=None, n_up=None, n_down=None):
    """
    Returns the indices of basis states belonging to the specified sector.
    
    If n_up and n_down are specified, filters by spin components.
    Otherwise, if n_electrons is specified, filters by total particle number.
    
    Assumes INTERLEAVED qubit ordering:
    Site 0: q0 (up), q1 (down)
    Site 1: q2 (up), q3 (down)
    ...
    """
    indices = []
    for i in range(2**n_qubits):
        
        s = f"{i:0{n_qubits}b}"
        
        # s[0] -> qubit 0 (Site 0 Up)
        # s[1] -> qubit 1 (Site 0 Down)
        # s[2] -> qubit 2 (Site 1 Up)
        # s[3] -> qubit 3 (Site 1 Down)
        # ...
        
        # Extract up/down bits assuming interleaved ordering
        up_bits = s[0::2]
        down_bits = s[1::2]
        
        c_up = up_bits.count('1')
        c_down = down_bits.count('1')
        total = c_up + c_down
        
        match = True
        if n_up is not None and c_up != n_up:
            match = False
        if n_down is not None and c_down != n_down:
            match = False
        if n_electrons is not None and total != n_electrons:
            match = False
            
        if match:
            indices.append(i)
            
    return indices

if use_spin_sector:
    sector_indices = get_sector_indices(n_qubits, n_up=target_n_up, n_down=target_n_down)
    print(f"Targeting sector: N_up={target_n_up}, N_down={target_n_down} (Indices count: {len(sector_indices)})")
else:
    sector_indices = get_sector_indices(n_qubits, n_electrons=n_electrons)
    print(f"Targeting sector: N_total={n_electrons} (Indices count: {len(sector_indices)})")


# --- Define Target Hamiltonian (for |v1>) ---

u_target = 4.0
H_target_obj = qml.spin.fermi_hubbard("square", [2,3], hopping=1.0, coulomb=u_target)
H_target_sparse = H_target_obj.sparse_matrix()


# Create a random initial vector in the sector to help eigsh find states in this sector
np.random.seed(123)
v0_sector = np.zeros(2**n_qubits)
# Randomly populate the sector indices
v0_sector[sector_indices] = np.random.rand(len(sector_indices))
v0_sector /= np.linalg.norm(v0_sector)

# Find the ground state of the sparse target Hamiltonian *within the sector* 
# We use v0 to bias the search towards our sector.
target_energies, target_evecs = sla.eigsh(H_target_sparse, k=40, which="SA", v0=v0_sector)

min_energy_in_sector_v1 = float("inf")
ground_state_in_sector_v1 = None

for i, energy in enumerate(target_energies):
    eigenvector = target_evecs[:, i]
    # Check if the eigenvector lives in the desired subspace
    norm_in_sector = np.sum(np.abs(eigenvector[sector_indices]) ** 2)
    if np.isclose(norm_in_sector, 1.0):
        if energy < min_energy_in_sector_v1:
            min_energy_in_sector_v1 = energy
            ground_state_in_sector_v1 = eigenvector

if ground_state_in_sector_v1 is None:
    # Fallback
    raise RuntimeError(
        "Could not find a ground state in the specified sector for the target Hamiltonian. Try increasing k."
    )

v1 = ground_state_in_sector_v1
print(f"Found |v1> with Energy: {min_energy_in_sector_v1:.6f}")

# --- Define Initial Hamiltonian (for |v2>) ---

u_initial = 1.0
# Ensure geometry matches!
H_initial_obj = qml.spin.fermi_hubbard("square", [2,3], hopping=1.0, coulomb=u_initial)
H_initial_sparse = H_initial_obj.sparse_matrix()

# Find the ground state of the initial Hamiltonian
all_initial_energies, all_initial_evecs = sla.eigsh(
    H_initial_sparse, k=40, which="SA", v0=v0_sector
)

min_energy_in_sector_v2 = float("inf")
ground_state_in_sector_v2 = None

for i, energy in enumerate(all_initial_energies):
    eigenvector = all_initial_evecs[:, i]
    norm_in_sector = np.sum(np.abs(eigenvector[sector_indices]) ** 2)
    if np.isclose(norm_in_sector, 1.0):
        if energy < min_energy_in_sector_v2:
            min_energy_in_sector_v2 = energy
            ground_state_in_sector_v2 = eigenvector

if ground_state_in_sector_v2 is None:
    raise RuntimeError(
        "Could not find a ground state in the specified sector for the initial Hamiltonian."
    )

v2 = ground_state_in_sector_v2
print(f"Found |v2> with Energy: {min_energy_in_sector_v2:.6f}")

print("--- System Setup ---")
print(f"Number of qubits: {n_qubits}")
print(f"Target U = {u_target}, Initial U = {u_initial}")
print(f"Shape of |v1>: {v1.shape}")
print(f"Shape of |v2>: {v2.shape}")
print("-" * 20)

# --- 2. UCCSD Circuit and Loss Function ---

# Use adjoint differentiation
# Use shots=None for exact expectation values
dev = qml.device("lightning.qubit", wires=n_qubits, shots=None)

# Generalized Excitations
def generate_generalized_excitations(n_qubits):
    """
    Generates all unique generalized single and double excitations 
    that conserve spin z projection (sz), assuming interleaved spin ordering
    (even=up, odd=down).
    """
    s_wires = []
    d_wires = []
    
    for p in range(n_qubits):
        for q in range(p + 1, n_qubits):
            # Spin conservation: parity must match
            if (p % 2) == (q % 2):
                s_wires.append([p, q])

    # 1. Generate all unique pairs (i, j) with i < j
    pairs = []
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            pairs.append((i, j))
    
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            # pair_to = (s, r) = pairs[j] (indices are larger usually? Doesn't matter)
            # pair_from = (q, p) = pairs[i]
            
            p, q = pairs[i] # p < q
            r, s = pairs[j] # r < s
            
            # Disjoint check
            if len({p, q, r, s}) != 4:
                continue
                
            # Spin conservation
            # (p%2 + q%2) must equal (r%2 + s%2)
            if (p % 2 + q % 2) == (r % 2 + s % 2):
                d_wires.append([[r, s], [p, q]])

    return s_wires, d_wires

s_wires, d_wires = generate_generalized_excitations(n_qubits)
print(f"Generalized Excitations: {len(s_wires)} singles, {len(d_wires)} doubles")
n_coeffs = len(s_wires) + len(d_wires)

# Construct the Observable P = |v1><v1|
# This allows us to compute Overlap^2 = <psi| P |psi>
P_mat = np.outer(v1, v1.conj())
H_obs = qml.Hermitian(P_mat, wires=range(n_qubits))

@qml.qnode(dev, interface="autograd", diff_method="adjoint")
def ucc_overlap_circuit(weights):
    """
    Prepares the initial state |v2> and applies the UCCSD unitary U(a).
    Returns the expectation value of |v1><v1|.
    """
    qml.StatePrep(v2, wires=range(n_qubits))  # Prepare |v2>

    for i in range(len(s_wires)):
        qml.FermionicSingleExcitation(weights[i], wires=s_wires[i])

    for i in range(len(d_wires)):
        wires_from = d_wires[i][0]
        wires_to = d_wires[i][1]
        qml.FermionicDoubleExcitation(
            weights[len(s_wires) + i], wires1=wires_to, wires2=wires_from
        )

    # We want to maximize <psi|v1><v1|psi> = |<v1|psi>|^2
    return qml.expval(H_obs)

def loss_function(weights):
    # ucc_overlap_circuit returns Fidelity F
    fidelity = ucc_overlap_circuit(weights)
    return 1 - fidelity

# --- 3. Optimization Loop ---

initial_overlap = np.abs(v1.conj() @ v2)**2
print(f"Initial Fidelity |<v1|v2>|^2: {initial_overlap:.6f}")
print("-" * 20)

np.random.seed(42)
params = np.array(np.random.normal(0, 0.01, n_coeffs), requires_grad=True)

optimizer = qml.AdamOptimizer(stepsize=0.05)
steps = 200

print("--- Starting Optimization (Adjoint) ---")
for i in range(steps):
    params, cost = optimizer.step_and_cost(loss_function, params)

    if (i + 1) % 10 == 0:
        print(f"Step {i+1:3d}: Cost (Infidelity) = {cost:.8f}")

print("-" * 20)
print("Optimization finished.")
print(f"Final Infidelity: {cost:.8f}")