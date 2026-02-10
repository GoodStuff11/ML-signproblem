from openfermion.hamiltonians import fermi_hubbard
from openfermion.ops import FermionOperator as FOp
from openfermion.ops import QubitOperator
from openfermion.transforms import jordan_wigner, normal_ordered
from openfermion.linalg import get_sparse_operator, get_ground_state, eigenspectrum
from openfermion.utils import hermitian_conjugated, commutator, count_qubits
from qcor import *
import numpy as np
from types import MethodType

#Define openFermion model
Nlat = 2 #number of lattice sites
x_dimension = 2 #two sites like this *--*
y_dimension = 3
tunneling = 1.0 #t
coulomb = 3 
nfill = 1.0
chemical_potential = 0 #nfill*coulomb/2.0
periodic = 1
spinless = 0 #spinfull case
hubbard_r = fermi_hubbard(x_dimension, y_dimension, tunneling, coulomb, chemical_potential, spinless = 0)
hubbard_r.compress(abs_tol=1e-12)
print(f'  {Nlat:d} site Hubbard model: {len(hubbard_r.terms):d} terms in the Hamiltonian')
#perform jordan wigner transform
hubbard_rjw = jordan_wigner(hubbard_r)

sparse_operator = get_sparse_operator(hubbard_rjw, n_qubits = Nlat*2)
gs = get_ground_state(sparse_operator)
E0 = gs[0]
#define the quantum circuit (kernel) for your "ansatz" or initial guess wavefunction, this is factorized UCC ansatz
#exp_i_theta does implicit first order trotterization
@qjit
def ansatz(q: qreg, x: List[float], exp_args: List[FermionOperator]):
    X(q[0])
    X(q[1])
    for i, exp_arg in enumerate(exp_args):
        exp_i_theta(q, x[i], exp_args[i])
 
# Create OpenFermion operators for our quantum kernel...
exp_args_openfermion = [FOp('2^ 3^ 1 0') - FOp('0^ 1^ 3 2'),
                        FOp('2^ 0') - FOp('0^ 2'), 
                        FOp('3^ 1') - FOp('1^ 3')]
print(exp_args_openfermion)
print(type(exp_args_openfermion[0]))
# We need to translate OpenFermion ops into qcor Operators to use with kernels...
exp_args_qcor = [createOperator('fermion', fop) for fop in exp_args_openfermion]

# translates arguments between quantum kernel and optimizer
def ansatz_translate(self, q: qreg, x: List[float]):
    ret_dict = {}    
    ret_dict["q"] = q
    ret_dict["x"] = x
    ret_dict["exp_args"] = exp_args_qcor
    return ret_dict
ansatz.translate = MethodType(ansatz_translate, qjit)
n_params = len(exp_args_qcor)
x_init  = np.random.rand(n_params).tolist()
ansatz.print_kernel(qalloc(4), [1.0, 1.0, 1.0], exp_args_qcor)
u_mat = ansatz.as_unitary_matrix(qalloc(4), [1.0, 1.0, 1.0], exp_args_qcor)
print("unitary mat: ", u_mat)
#VQE is an example of an "objectiveFunction", where we seek to minimize <U(x)|H|U(x)>
obj = createObjectiveFunction(ansatz, hubbard_rjw, n_params, {'gradient-strategy': 'parameter-shift'})
optimizer = createOptimizer('nlopt', {'algorithm': 'l-bfgs', 'initial-parameters':x_init})
results = optimizer.optimize(obj)
print(results)