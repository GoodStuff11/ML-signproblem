# Overview
The overall goal of this calculation is to minimize the loss function by tuning the real-valued parameters $A_j$ (expressed in terms of physics language and pure linear algebra language)
```math
\mathcal{L}(A) = 1-|\langle \psi_f | \mathcal{U} | \psi_i \rangle|^2=1-|\mathbf{v}_f\cdot \mathcal{U}\mathbf{v}_i|^2\in [0, 1]
```
where
```math
\mathcal{U} = \exp(i[\sum_{k_1, k_2} A^{(1)}_{k_1k_2}c^\dagger_{k_1} c_{k_2}+\sum_{k_1, k_2,k_3,k_4} A^{(2)}_{k_1k_2k_3k_4}c^\dagger_{k_1} c^\dagger_{k_2} c_{k_3} c_{k_4}])=\exp(i\sum_{j=1}^N A_j H_j)
```

<!-- Another alternative implementation of the unitary that isn't implemented here, but could be implemented is as follows
```math
\mathcal{U} = \prod_{k_1, k_2} \exp(i[A^{(1)}_{k_1k_2}c^\dagger_{k_1} c_{k_2}])\prod_{k_1, k_2,k_3,k_4} \exp(i[A^{(2)}_{k_1k_2k_3k_4}c^\dagger_{k_1} c^\dagger_{k_2} c_{k_3} c_{k_4}])=\prod_j \exp(iA_j H_j)
``` -->

Here $H_j$ are sparse Hermitian matrices and $A_j$ are real-valued parameters. $\mathbf{v}_f$ ($|\psi_f\rangle$) and $\mathbf{v}_i$ ($|\psi_i\rangle$) are the normalized final and initial states, respectively, represented as real vectors. The unitary is represented in the momentum basis, in a subspace corresponding to a momentum eigenstate with the lowest Hubbard model energy. The $k_i$ notation present in the unitary equation denotes not just a 2d momentum vector but also what the corresponding spin is, $k_i=(\vec{k}_i,\sigma)$, where $\sigma \in \{\uparrow, \downarrow\}=\{1,2\}$. The $A$ values are non-zero only when the fermionic operators which it is a coefficient are lattice momentum conserving, and spin conserving.


This optimization is performed for a selected initial state (the ground state of the non-interacting Hubbard model) and a set of final states which are each perturbations of previous final states (the ground state of the interacting Hubbard model with the same symmetries as the initial state). These states are stored in a file. To make the optimization perform better, when we optimize over a single final state, we can re-use those tuned parameters for the next optimization, which will generally give a good initial guess. 

For the optimization, the gradient is computed with adjoint gradients to make it as efficient as possible. It is also implemented by exactly computing the exponential as supposed to trotterizing it in any way. 

# Hard Constraints (MUST NOT BREAK)
The accuracy of the final result is verified by running `sanity_check()` on a small system. This function runs a series of checks to ensure that the optimization is working as intended. The following are the checks that are performed:

## 1. Hermiticity
Every matrix `Hⱼ` produced (initially made by `build_nth_order_sparse` and wrapped by `make_hermitian`) must satisfy `Hⱼ == Hⱼ'` (up to 1e-10). This is mandatory for `exp(i·ΣAⱼHⱼ)` to be unitary.

## 2. Unitarity of the propagator
For any normalised vector `v` and Hermitian matrix `H` constructed from the operators.


## 3. Loss-matrix consistency (end-to-end)
After `optimize_unitary` returns, for each `order`:
```
all_matrices[order]  ≈  Σ_k  coefficients[order][k] * ops[k]
```
This catches any mismatch between the matrix returned and the coefficients stored, verifying that the form of the unitary is correct.

---

## Performance goal
The score (printed by `score.jl`) is (in attempt to get as small of a loss as possible while making sure it doesn't take too long):
```
loss × 1e5 + wall_time_seconds
```

# Hints
Bottlenecks
* `build_nth_order_operator` is O(L^{2n}) in the number of sites L. For a 4×3 lattice with n=2, this is ~10⁴ operators. `build_nth_order_sparse` then iterates over all basis states × all operators. Both are targets for optimisation.
* Code may benefit from using GPU acceleration, though it isn't beneficial for smaller systems. The code currently uses multi-threading for acceleration
* `adjoint_loss` is generally expensive.

