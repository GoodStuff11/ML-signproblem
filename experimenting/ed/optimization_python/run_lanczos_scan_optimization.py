import sys
import os
import h5py
import numpy as np

# Import the exact things needed for minimal implementation
from ed_optimization import interaction_scan_map_to_state

def load_jld2_dict(filepath):
    """
    Extract dict references from JLD2 directly.
    JLD2 often stores dicts via linear arrays of keys and values.
    """
    res = {}
    try:
        with h5py.File(filepath, 'r') as f:
            def _deref(val):
                if isinstance(val, h5py.h5r.Reference):
                    return _deref(f[val][()])
                elif isinstance(val, np.ndarray) and val.dtype == object:
                    # Array of references
                    return [_deref(x) for x in val]
                elif isinstance(val, np.ndarray) and val.dtype.names and 're' in val.dtype.names:
                    return val['re'] + 1j * val['im']
                return val

            for k in f.keys():
                if k == '_types': continue
                res[k] = _deref(f[k][()])
    except Exception as e:
        print(f"Failed extracting JLD2: {e}")
    return res

def main():
    folder = "../data/N=(3, 3)_3x2"
    file_path = os.path.join(folder, "meta_data_python_export.jld2")

    # Only load what we need for the actual scan
    dic = load_jld2_dict(file_path)

    meta_data = dic.get("meta_data", {})
    U_values = dic.get("U_values", [])
    N = dic.get("N", None)
    spin_conserved = bool(dic.get("spin_conserved", False))
    
    all_full_eig_vecs = dic.get("all_full_eig_vecs", [])
    all_E = dic.get("E", [])
    indexer = None # Not exported explicitly to python flat file, but can be built inside ed_optimization

    print("Meta data:")
    print(meta_data)

    # Extracted scalar N
    
    use_symmetry = len(sys.argv) > 1 and sys.argv[1].lower() == "true"

    min_E = float('inf')
    k_min = 0 # Python uses 0-based indexing
    
    for k, E_vec in enumerate(all_E):
        if hasattr(E_vec, '__len__') and len(E_vec) > 0:
            E_ground = E_vec[0]
            if E_ground < min_E:
                min_E = E_ground
                k_min = k

    print(f"Selected lowest energy symmetry sector: {k_min} with Energy {min_E}")

    # Select the eigenvectors for this sector
    target_vecs = []
    if hasattr(all_full_eig_vecs, '__len__') and len(all_full_eig_vecs) > k_min:
        target_vecs = all_full_eig_vecs[k_min]
        
    if isinstance(indexer, (list, tuple, np.ndarray)) and len(indexer) > k_min:
        indexer = indexer[k_min]

    # Julia uses 1-based indexing and range `25:length(U_values)` implies from the 25th element inclusive,
    # which in 0-based python is index 24 up to len(U_values) - 1.
    list_len = len(U_values) if hasattr(U_values, '__len__') else 0
    start_u_idx = min(24, max(0, list_len - 1)) if list_len > 0 else 24
    u_range = range(start_u_idx, list_len) if list_len > 0 else []

    scan_instructions = {
        "starting level": 1, # logical level idx, left as 1 like Julia
        "ending level": 1,
        "u_range": list(u_range),
        "optimization_scheme": [2],
        "use symmetry": use_symmetry
    }

    interaction_scan_map_to_state(
        target_vecs,
        scan_instructions,
        indexer,
        spin_conserved=spin_conserved,
        maxiters=200,
        gradient="adjoint_gradient",
        optimizer=["GradientDescent", "LBFGS", "GradientDescent", "LBFGS"],
        save_folder=folder,
        save_name=f"unitary_map_energy_symmetry={use_symmetry}_N={N}"
    )

    return 0

if __name__ == "__main__":
    sys.exit(main())
