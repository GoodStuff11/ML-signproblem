# Data Pipelines & File I/O

## Exact Diagonalization (ED) HDF5 Data (`.h5`)
*   **Purpose:** Stores the full physical state for an Exact Diagonalization simulation (e.g. `HubbardED_Slater_3x2_...h5`).
*   **Target Functions:** `load_ED_data()`, `load_h5_ED_data()`, `load_jld2_ED_data()` in `ed_functions.jl`.
*   **Internal File Structure:** The `.h5` file uses a nested group directory structure.
    *   `/metadata/`:
        *   `Lvec`: 1D array representing the spatial dimensions of the lattice (e.g. `[3, 2]`).
        *   `kvecs`: 2D array of shape `(N_sites, 2)` representing the discrete momentum vectors for each orbital.
        *   `nup` & `ndown`: Scalars representing the particle number for spin-up and spin-down respectively.
        *   `slater_labels/k_sec`: A group containing the basis state electron configurations for a given momentum sector `k_sec`.
            *   `slater_labels/k_sec/up`: Matrix of shape `(H_dim, nup)` where rows are basis elements and columns represent individual up-electrons. The values (e.g. `0` to `5`) map directly to the zero-indexed orbital the electron occupies.
            *   `slater_labels/k_sec/dn`: Matrix of shape `(H_dim, ndown)` containing the zero-indexed occupied orbitals for down-electrons.
    *   `/data/`:
        *   `uvec`: 1D array of the interaction parameter values $U$ at which ground states were computed.
        *   `energies/k_sec`: Matrix of ground state and excited state energies. Typically shape `(n_U, num_states)`.
        *   `evecs/k_sec`: 3D tensor of the raw eigenvectors, typically with shape `(n_U, num_states, H_dim)`.

*   **State Representation (`ints`):** 
    *   When processed in Julia, basis states are converted into an integer array (`ints`), which strictly represents the occupation basis in binary format.
    *   For a $3\times2$ lattice (6 sites total), the lowest 6 bits (bits 0 to 5) are exclusively the spin-up electron occupations. The next 6 bits (bits 6 to 11) are exclusively the spin-down electron occupations.

## Exact Diagonalization JLD2 Data (`.jld2`)
*   **Purpose:** Stores pre-processed datasets, cached observables, or checkpointed dictionary structures resulting from ED runs (e.g. `meta_data_and_E.jld2` or `trotter_...jld2`).
*   **Target Function:** `load_saved_dict(file_path)` from `utility_functions.jl` (which calls `JLD2.load`).
*   **Dictionary Structure:** The loaded dictionary generally contains the following primary keys:
    *   `"meta_data"`: A nested dictionary storing configuration values such as the required `"sign_convention"` (which defaults to `:coordinate_first` if missing) and `"U_values"`.
    *   `"E"`: A nested array or matrix storing the energy sectors evaluated. This is used dynamically for ground-state sector selection.
    *   `"all_full_eig_vecs"`: The array holding the processed eigenvectors for the chosen ground-state sector.
    *   `"indexer"`: A pre-built `CombinationIndexer` object that maps states strictly to their matrix indices, avoiding the need to rebuild it from scratch.
    *   `"precomputed_structures"`: A dictionary caching dense matrices or frequently accessed components.

## Target Vectors Arrays (`target_vecs`)
*   **Purpose:** Holds the computed eigenvectors (states) extracted from the ED `.h5` or `.jld2` datasets.
*   **Array Construction:** 
    *   When calling parsing functions like `load_ED_data`, the `use_slater_reference` keyword argument dictates how the `target_vecs` matrix is constructed.
    *   **Default Behavior:** By default, when loading `.h5` files, `use_slater_reference = true`.
*   **Target Vectors Array Shape (`target_vecs`):**
    *   With `use_slater_reference = true`, the returned `target_vecs` matrix is of shape **`((n_U + 1), basis_size)`** (where `n_U` is the number of $U$ parameter values in the dataset).
    *   With `use_slater_reference = false`, the shape is exactly **`(n_U, basis_size)`**.
*   **Row Indexing (CRITICAL RULE):** The matrix is transposed such that states are indexed by rows. You must **always take slices along the rows** (e.g., `state = target_vecs[2, :]`), NEVER along columns.
    *   **Row 1 (`target_vecs[1, :]`):** If `use_slater_reference = true`, this row exclusively contains the Slater determinant reference state $|D_{i^*}\rangle$ (a single basis vector, `reference_slater_index`): $i^* = \arg\max_i |\langle D_i|\phi_{weak}\rangle|^2$, where $\phi_{weak}$ is a stored eigenstate at the smallest U in the file, chosen by `reference_target` (`weak_reference_state`): `:weak_ground_state` (default) = the lowest stored state of the selected sector at that U; `:weak_connected` = the target state's own member at that U. (Before 2026-11-16 the reference was instead the minimum-kinetic-energy determinant among those with $|c_i|^2 > 0.1\max|c|^2$; `testing/test_excited_state_targets.jl` part A lists for every data_h5_fixed file whether the two rules agree.) An integer `use_slater_reference` is used directly as $i^*$.
    *   **Row 2 (`target_vecs[2, :]`):** This contains the Exact Diagonalization ground state computed at the first interaction value $U_1$. 
    *   **Row i+1 (`target_vecs[i+1, :]`):** This contains the target eigenstate at the parameter value $U_i$ (the ground state with the default options).
*   **Default sector choice (`find_best_energy_sector`):** at every U each sector within `atol = 1e-6` of the lowest energy gets one tally; the sector with the most tallies wins; ties go to the smallest sector index (smallest $(q_x,q_y)$). For one-file-per-sector folders the tie-break key is the sector index stored in each file. Until 2026-11-17 ties were broken by `Dict` hash order; the sector changes for N=(3,2)_3x2 (4→2), (3,2)_3x3, (4,3)_3x3, (5,4)_3x3 (3→1), (5,4)_4x3 (2→1), (4,4)_4x4 (8→0), (5,4)_4x4 (12→1) (`testing/test_sector_tiebreak.jl`). For the five of these with saved results the trained sector is recorded in the folder's `momentum_sector.toml` (see [[ed_file_formats]]), which `load_h5_ED_data` uses as the default, so existing results load and resume in their own sector (`testing/test_sector_tiebreak.jl` reproduces their saved losses through a default load).
*   **Selecting excited targets (`load_h5_ED_data` keywords):** `sector` = flat key `<k>` of the momentum sector (default: automatic, the ground-state sector); `target_state` = `:lowest` (default; the sector's lowest state at the smallest U, followed in U by maximal overlap, `select_continuous_state_indices`) or `:min_abs_energy` (per U the stored state with the smallest $|E-\sigma|$ above `zero_mode_tol`, only for `eig_target = "closest_to_sigma"` files); `reference_target` as above; `zero_mode_tol` (default `1e-8`). A non-ground-state `sector` with `target_state=:lowest` gives a slightly excited target; a tracked file (UCC_Hubbard `TrackHubbardED_XDiag.jl`, e.g. `N=(5, 4)_4x3_excited_level4`) with `target_state=:tracked, reference_target=:weak_connected` gives one excited eigenstate followed continuously in U (see [[ed_file_formats]]). JLD2 data supports only `sector` (1-based list index) and `target_state=:lowest`. `run_trotter_scan_optimization.jl` exposes these as `--sector`, `--target_state`, `--reference_target`, `--zero_mode_tol`, and adds `sector=<k>`, `target=<mode>`, `zmtol=<v>`, `reftarget=<mode>` to the saved-file prefix for non-default values.
