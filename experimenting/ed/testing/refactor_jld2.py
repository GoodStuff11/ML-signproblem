import sys
import re

def rewrite_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    pattern_jld2 = re.compile(r'function load_jld2_ED_data\(file_path::String; verbose=false, kwargs\.\.\.\).*?return U_values, target_vecs, indexer, precomputed_structures, N, spin_conserved, use_symmetry, requested_sign_convention\nend', re.DOTALL)
    
    new_jld2 = """function load_jld2_ED_data(file_path::String; verbose=false, kwargs...)
    dic = load_saved_dict(file_path)

    meta_data = dic["meta_data"]
    file_sign_convention = get(meta_data, "sign_convention", :coordinate_first)
    if file_sign_convention isa String
        file_sign_convention = Symbol(file_sign_convention)
    end
    use_slater_reference = get(kwargs, :use_slater_reference, false)
    su2_symmetry = get(kwargs, :su2_symmetry, false)
    omit_indexer = get(kwargs, :omit_indexer, false)

    U_values = meta_data["U_values"]
    all_full_eig_vecs = dic["all_full_eig_vecs"]
    all_E = dic["E"] # Needed for energy selection

    indexer = dic["indexer"]
    precomputed_structures = get(dic, "precomputed_structures", Dict())

    if verbose
        println("Meta data:")
        display(meta_data)
    end

    # Extract N for saving
    N = meta_data["electron count"]
    spin_conserved = !isa(meta_data["electron count"], Number) # True if tuple (N_up, N_down)
    use_symmetry = false

    # Find lowest energy sector 
    k_min = find_best_energy_sector(all_E, U_values; verbose=verbose, data=dic, su2_symmetry=su2_symmetry)

    # Select the eigenvectors for this sector
    target_vecs = all_full_eig_vecs[k_min]
    if indexer isa Vector
        indexer = indexer[k_min]
    end

    if indexer !== nothing
        Lvec_native = [indexer.lattice.L...]
        order_native = indexer.order
        H_dim = length(indexer.inv_comb_dict)
    else
        Lvec_native = get(meta_data, "Lvec", [1, 1]) # Fallback
        order_native = RowSnake()
        H_dim = size(target_vecs, 1)
    end

    if size(target_vecs, 1) == H_dim
        target_vecs = Matrix(transpose(target_vecs))
    end

    use_slater_ref = (use_slater_reference !== false && use_slater_reference !== nothing)
    if use_slater_ref
        slater_index = get_slater_ground_state(dic, k_min; custom_ref=use_slater_reference)
        if slater_index == -1
            error("No Slater ground state could be found in sector $k_min.")
        end
        reference_state = zeros(ComplexF64, H_dim)
        reference_state[slater_index] = 1.0
        target_vecs = vcat(transpose(reference_state), target_vecs) # shape (n_U + 1, H_dim)
    end

    if omit_indexer
        indexer = nothing
    end

    return U_values, target_vecs, indexer, precomputed_structures, N, spin_conserved, use_symmetry, file_sign_convention, Lvec_native, order_native
end"""

    content = pattern_jld2.sub(new_jld2, content)
    
    with open(filepath, 'w') as f:
        f.write(content)

if __name__ == "__main__":
    rewrite_file(sys.argv[1])
