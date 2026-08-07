# Core Conventions

## Vector Ordering
*   **Lattice Ordering (ColSnake):** The `.h5` data files for Exact Diagonalization (ED) ground states use a `ColSnake` lattice ordering. This means the $y$-coordinate changes faster than the $x$-coordinate (column-major order). 
    *   **Example (3x2 Lattice):** The lattice sites are traversed and mapped to bit indices in the exact following order: 
        1. (0,0) → bit 0
        2. (0,1) → bit 1
        3. (1,0) → bit 2
        4. (1,1) → bit 3
        5. (2,0) → bit 4
        6. (2,1) → bit 5
    *   This ordering is natively supported by and intrinsically tied to the `:spin_first` sign convention.

## Sign Conventions
*   **Generators:** When reading or saving circuit generator matrices, assume an anti-Hermitian sign convention by default.
*   **Jordan-Wigner Signs:** When loading basis elements from `.h5` or `.jld2` files via `load_ED_data`, Jordan-Wigner signs are automatically applied by computing the permutation parity relative to the selected sign convention. 
    *   **Mapping:** The `:spin_first` convention maps exactly to `ColSnake` ordering. The `:coordinate_first` convention maps to `RowSnake` ordering. You must ensure the selected convention matches the underlying lattice ordering of your dataset to preserve the correct relative signs of the basis states.

## Matrix Properties & Hamiltonian Construction
*   **`HubbardMomentumBasis` Usage:** When building Hamiltonians via `Trotter.TamFermion.HubbardMomentumBasis(t, u, Lvec, nvec; ...)`, it is a strict requirement to pass a non-zero value for the `u` parameter (e.g., `u=1.0`) if you want the interaction matrix `H_int` to be generated.
    *   **Failure Mode:** Passing `u=0.0` causes the function to completely skip the $H_{int}$ calculation. It will return an entirely empty (all-zeros) sparse matrix for `H_int`. Consequently, any interaction expectation value `state' * H_int * state` will incorrectly compute to exactly `0.0`. If you intend to use `H_int` separately from `H_hop`, you must supply a non-zero `u` when calling the function.
