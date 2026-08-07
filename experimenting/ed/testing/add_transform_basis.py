import sys
import re

def rewrite_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    new_functions = """
function transform_basis(target_vecs::Matrix{ComplexF64}, 
                         native_indexer::CombinationIndexer, 
                         target_indexer::CombinationIndexer, 
                         native_sign::Symbol, 
                         target_sign::Symbol,
                         coord_map=nothing)
    # If indexers are identical and signs are identical, do nothing
    if native_indexer === target_indexer && native_sign == target_sign
        return target_vecs
    end
    
    sorted_native = sort(native_indexer.a, order=native_indexer.order)
    sorted_target = sort(target_indexer.a, order=target_indexer.order)
    
    native_coord_to_idx = Dict(c => i for (i, c) in enumerate(sorted_native))
    target_coord_to_idx = Dict(c => i for (i, c) in enumerate(sorted_target))
    
    N_sites = length(native_indexer.a)
    
    new_vecs = zeros(eltype(target_vecs), size(target_vecs, 1), size(target_vecs, 2))
    
    for (native_idx, conf) in enumerate(native_indexer.inv_comb_dict)
        native_up, native_dn = conf
        
        # Map coordinates if a mapping function is provided (e.g., swapping x and y)
        if coord_map !== nothing
            target_up = Set([coord_map(c) for c in native_up])
            target_dn = Set([coord_map(c) for c in native_dn])
        else
            target_up, target_dn = native_up, native_dn
        end
        
        if !haskey(target_indexer.comb_dict, (target_up, target_dn))
            error("Target indexer does not contain combination for native index $native_idx")
        end
        target_idx = target_indexer.comb_dict[(target_up, target_dn)]
        
        # Compute JW parity for native state
        native_up_sorted = sort(collect(native_up), order=native_indexer.order)
        native_dn_sorted = sort(collect(native_dn), order=native_indexer.order)
        native_jw = Vector{Int}(undef, length(native_up_sorted) + length(native_dn_sorted))
        for (i, c) in enumerate(native_up_sorted)
            native_jw[i] = (native_sign == :spin_first) ? native_coord_to_idx[c] : 2 * native_coord_to_idx[c] - 1
        end
        for (i, c) in enumerate(native_dn_sorted)
            native_jw[length(native_up_sorted) + i] = (native_sign == :spin_first) ? N_sites + native_coord_to_idx[c] : 2 * native_coord_to_idx[c]
        end
        sgn_native = 1 - 2 * permutation_parity(native_jw)
        
        # Compute JW parity for target state
        target_up_sorted = sort(collect(target_up), order=target_indexer.order)
        target_dn_sorted = sort(collect(target_dn), order=target_indexer.order)
        target_jw = Vector{Int}(undef, length(target_up_sorted) + length(target_dn_sorted))
        for (i, c) in enumerate(target_up_sorted)
            target_jw[i] = (target_sign == :spin_first) ? target_coord_to_idx[c] : 2 * target_coord_to_idx[c] - 1
        end
        for (i, c) in enumerate(target_dn_sorted)
            target_jw[length(target_up_sorted) + i] = (target_sign == :spin_first) ? N_sites + target_coord_to_idx[c] : 2 * target_coord_to_idx[c]
        end
        sgn_target = 1 - 2 * permutation_parity(target_jw)
        
        # The vector coefficient picks up (sgn_native * sgn_target)
        # Because |state_native> = sgn_native * c^+...c^+ |0>
        # |state_target> = sgn_target * c^+...c^+ |0>
        # So |state_native> = (sgn_native * sgn_target) |state_target>
        new_vecs[:, target_idx] .= target_vecs[:, native_idx] .* (sgn_native * sgn_target)
    end
    
    return new_vecs
end

function load_ED_data(folder; verbose=false, kwargs...)
    jld2_path = joinpath(folder, "meta_data_and_E.jld2")
    
    if !isfile(jld2_path)
        U_values, native_vecs, native_indexer, precomputed_structures, N, spin_conserved, use_symmetry, native_sign, native_Lvec, native_order = load_h5_ED_data(folder; verbose=verbose, kwargs...)
    else
        U_values, native_vecs, native_indexer, precomputed_structures, N, spin_conserved, use_symmetry, native_sign, native_Lvec, native_order = load_jld2_ED_data(jld2_path; verbose=verbose, kwargs...)
    end
    
    # Read requested target conventions from kwargs
    target_sign = get(kwargs, :sign_convention, native_sign)
    target_order = get(kwargs, :order, native_order)
    target_Lvec = get(kwargs, :Lvec, native_Lvec)
    
    omit_indexer = get(kwargs, :omit_indexer, false)
    
    if omit_indexer || native_indexer === nothing
        return U_values, native_vecs, native_indexer, precomputed_structures, N, spin_conserved, use_symmetry, native_sign
    end
    
    # If no transformation is needed, return native
    if target_sign == native_sign && target_order == native_order && target_Lvec == native_Lvec
        return U_values, native_vecs, native_indexer, precomputed_structures, N, spin_conserved, use_symmetry, native_sign
    end
    
    if verbose
        println("Transforming basis from native (sign: $native_sign, order: $native_order, Lvec: $native_Lvec)")
        println("                  to target (sign: $target_sign, order: $target_order, Lvec: $target_Lvec)")
    end
    
    # Determine if coordinate mapping is needed (e.g., Lvec was reversed)
    coord_map = nothing
    if target_Lvec != native_Lvec
        if target_Lvec == reverse(native_Lvec)
            coord_map = c -> Coordinate(reverse(c.coordinates))
        else
            error("Target Lvec $target_Lvec does not match native Lvec $native_Lvec and is not its reverse. Cannot automatically map.")
        end
    end
    
    # Construct target indexer
    lattice = Square(tuple(target_Lvec...), Periodic())
    
    # Map the k_sector to the target lattice if Lvec changed
    if coord_map !== nothing
        # If the coordinates are transposed, the k_vector is also transposed
        # Wait, if x -> y, then phase kx*x + ky*y = ky*x + kx*y
        # so k_x_target = k_y_native
        k_target = reverse(native_indexer.k)
    else
        k_target = native_indexer.k
    end
    
    subspace = HubbardSubspace(N..., lattice; k=k_target)
    target_indexer = CombinationIndexer(subspace; order=target_order)
    
    target_vecs = transform_basis(native_vecs, native_indexer, target_indexer, native_sign, target_sign, coord_map)
    
    return U_values, target_vecs, target_indexer, precomputed_structures, N, spin_conserved, use_symmetry, target_sign
end
"""

    # We need to replace the old load_ED_data with the new one and insert transform_basis.
    pattern = re.compile(r'function load_ED_data\(folder; verbose=false, kwargs\.\.\.\).*?end\n', re.DOTALL)
    
    content = pattern.sub(new_functions, content)
    
    with open(filepath, 'w') as f:
        f.write(content)

if __name__ == "__main__":
    rewrite_file(sys.argv[1])
