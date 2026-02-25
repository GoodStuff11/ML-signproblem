function make_hermitian(A::SparseMatrixCSC)
    # acts similar to Hermitian(A) but is when only one of A[i,j] and A[j,i] are non-zero
    # This function doesn't override non-zero values with zero values like Hermitian(A) can
    I, J, V = findnz(A)
    return sparse(
        vcat(I, J),
        vcat(J, I),
        vcat(V, conj.(V)),
        size(A, 1), size(A, 2)
    )
end
function make_antihermitian(A::SparseMatrixCSC)
    # acts similar to Hermitian(A) but is when only one of A[i,j] and A[j,i] are non-zero
    # This function doesn't override non-zero values with zero values like Hermitian(A) can

    I, J, V = findnz(A)
    return sparse(
        vcat(I, J),
        vcat(J, I),
        vcat(V, -conj.(V)),
        size(A, 1), size(A, 2)
    )
end


function compute_jw_sign(
    conf::Tuple{Set{T},Set{T}},
    sorted_sites::Vector{T},
    ops::Vector{Tuple{T,Int,Symbol}}
) where T
    # computes the sign for the term given by ops (in second quantized), associated with the 
    # configuration conf.

    # Full JW order over sites and spins
    jw_order = [(s, σ) for s in sorted_sites for σ in (1, 2)]

    # Map each mode to its index in JW order
    jw_index = Dict{Tuple{T,Int},Int}((sσ, i) for (i, sσ) in enumerate(jw_order))

    # Initial occupation vector (ordered)
    occupied_modes = Set{Tuple{T,Int}}()
    for (s, σ) in jw_order
        if s ∈ conf[σ]
            push!(occupied_modes, (s, σ))
        end
    end

    # Compute sign by counting how many occupied modes come before each operator
    sign = 1

    # Process from right to left (annihilate first)
    for (site, spin, op) in reverse(ops)
        mode = (site, spin)
        idx = jw_index[mode]

        # Count how many occupied modes come *before* this mode
        n_occupied_before = count(m -> jw_index[m] < idx, occupied_modes)
        sign *= (-1)^n_occupied_before

        # Update occupation based on op
        if op == :annihilate
            # Fermion is removed — no longer present
            delete!(occupied_modes, mode)
        elseif op == :create
            # Fermion is added — affects future operators
            push!(occupied_modes, mode)
        else
            error("Unknown operator: $op")
        end
    end

    return sign
end

function build_n_body_structure(
    t::Dict{Vector{Tuple{T,Int,Symbol}},U},
    indexer::CombinationIndexer;
    skip_lower_triangular::Bool=false
) where {T,U<:Number}
    build_n_body_structure_from_keys(collect(keys(t)), indexer, U; skip_lower_triangular)
end

function build_n_body_structure_from_keys(
    t_keys::AbstractVector,
    indexer::CombinationIndexer{T},
    ::Type{U}=Float64;
    skip_lower_triangular::Bool=false
) where {T,U<:Number}
    sorted_sites = sort(indexer.a)
    rows = Int[]
    cols = Int[]
    signs = U[]
    ops_list = Vector{Vector{Tuple{T,Int,Symbol}}}()

    for (i1, conf) in enumerate(indexer.inv_comb_dict)
        for ops in t_keys
            # Clone the config
            conf_new = [copy(conf[1]), copy(conf[2])]
            valid = true

            # Apply operators
            for (site, spin, op) in reverse(ops)
                if op == :annihilate
                    if site ∉ conf_new[spin]
                        valid = false
                        break
                    end
                    delete!(conf_new[spin], site)
                elseif op == :create
                    if site ∈ conf_new[spin]
                        valid = false
                        break
                    end
                    push!(conf_new[spin], site)
                else
                    error("Invalid operator symbol: $op")
                end
            end

            if !valid
                continue
            end

            i2 = index(indexer, conf_new[1], conf_new[2])
            if skip_lower_triangular && i1 > i2 #only considering upper diagonal so ensure hermiticity
                continue
            end
            s = compute_jw_sign(conf, sorted_sites, ops)
            push!(rows, i1)
            push!(cols, i2)
            push!(signs, s)
            push!(ops_list, ops)
        end
    end

    return rows, cols, signs, ops_list
end

""" used to optimize update_values """
function build_param_index_map(
    ops_list::Vector{Vector{Tuple{T,Int,Symbol}}},
    t_keys::Vector{Vector{Tuple{T,Int,Symbol}}}
) where {T}
    # Build reverse lookup: key -> index in t_keys
    key_to_idx = Dict(t_keys[i] => i for i in eachindex(t_keys))
    # For each element in ops_list, find which t_key index it refers to
    return [key_to_idx[ops_list[i]] for i in eachindex(ops_list)]
end

function update_values(
    signs::Vector{U},
    param_index_map::Vector{Int},
    t_vals::Vector{V},
    parameter_mapping::Union{Vector{Int},Nothing}=nothing,
    parity::Union{Vector{Int},Nothing}=nothing
) where {U<:Number,V<:Number}
    # it's allowed for length(t_vals) < length(t_keys), but a parameter_mapping to make the difference is required.
    if isnothing(parameter_mapping)
        return [t_vals[param_index_map[i]] * signs[i] for i in eachindex(signs)]
    else
        type_UV = promote_type(U, V)
        return [parameter_mapping[param_index_map[i]] == 0 ? zero(type_UV) : t_vals[parameter_mapping[param_index_map[i]]] * parity[param_index_map[i]] * signs[i]
                for i in eachindex(signs)]
    end
end

function create_randomized_nth_order_operator(n::Int, indexer::CombinationIndexer;
    magnitude::T=1e-3 + 0im, omit_H_conj::Bool=false, conserve_spin::Bool=false, normalize_coefficients::Bool=false, conserve_momentum::Bool=false) where T
    # function creates a dictionary of free parameters in the form of a dictionary. 
    # when spin is conserved, the Hilbert space is smaller, so a restricted number of coefficients are possible. The rest aren't filled in
    # When hermiticity is forced, we only need to worry about upper diagonal elements. The rest can be filled in afterward

    t_dict = Dict{Vector{Tuple{Coordinate{2,Int64},Int,Symbol}},T}()
    site_list = sort(indexer.a) #ensuring normal ordering
    all_ops(label) = combinations([(s, σ, label) for s in site_list for σ in 1:2], n)
    equal_spin(create, annihilate) = sum((σ * 2 - 3) for (s, σ, _) in create) == sum((σ * 2 - 3) for (s, σ, _) in annihilate)
    geq_ops(create, annihilate) = [(s.coordinates..., σ) for (s, σ, _) in create] <= [(s.coordinates..., σ) for (s, σ, _) in annihilate]

    for (ops_create, ops_annihilate) in Iterators.product(all_ops(:create), all_ops(:annihilate))
        key = [ops_create; ops_annihilate]

        # We must conserve momentum if the user specified conserve_momentum=true
        # OR if the indexer is explicitly restricted to a momentum sector (which means non-conserving ops will jump out of the Hilbert space)
        must_conserve_momentum = conserve_momentum || (!isnothing(indexer.k) && !isnothing(indexer.lattice_dims))

        if must_conserve_momentum
            tot_k = zeros(Int, length(indexer.lattice_dims))
            for (s, σ, _) in ops_create
                tot_k .+= (s.coordinates .- 1)
            end
            for (s, σ, _) in ops_annihilate
                tot_k .-= (s.coordinates .- 1)
            end
            tot_k = tot_k .% indexer.lattice_dims
            is_momentum_conserved = all(tot_k .== 0)
        else
            is_momentum_conserved = true
        end

        if (!omit_H_conj || geq_ops(ops_create, ops_annihilate)) && (!conserve_spin || equal_spin(ops_create, ops_annihilate)) && is_momentum_conserved
            if key ∉ keys(t_dict)
                t_dict[key] = (2 * rand() - 1) / 2 * magnitude
            else
                t_dict[key] += (2 * rand() - 1) / 2 * magnitude
            end
        end
    end
    if normalize_coefficients
        normalization_coefficient = length(values(t_dict))
        for key in keys(t_dict)
            t_dict[key] /= normalization_coefficient
        end
    end
    return t_dict
end


"""
    find_symmetry_groups(X, Lx, Ly; trans_x=false, trans_y=false, refl_x=false, refl_y=false, spin_symmetry=false, hermitian=false, antihermitian=false)

Find groups of indices in X that are related by the specified symmetries.

# Arguments
- `X`: Vector of configurations, where each element is a vector of tuples `((x, y), s, op)`
      where op is either :create or :annihilate
- `Lx`: Lattice size in x direction
- `Ly`: Lattice size in y direction
- `trans_x`: Include translational symmetry in x direction
- `trans_y`: Include translational symmetry in y direction
- `refl_x`: Include reflection symmetry about x axis (y → Ly + 1 - y) (NOTE: [σx, Tx] ≠ 0)
- `refl_y`: Include reflection symmetry about y axis (x → Lx + 1 - x)
- `spin_symmetry`: Include spin flip symmetry (s: 1 ↔ 2) (NOTE: this doesn't force symmetry with Sx, only e^{iπSx}.
     It's an additional symmetry that the Hubbard model has, but it's a bit weaker than Sx. This always applies to 
     fixed N systems but only applies on polarized systems when N↑=N↓)
- `hermitian`: Include hermitian conjugation (:create ↔ :annihilate)
- `antihermitian`: Include hermitian conjugation with a sign flip (anti-hermitian symmetry A† = -A)

# Returns
- `groups`: Vector of vectors of integers, where each sub-vector contains indices that are symmetry-equivalent.
- `inverse_map`: Vector of integers where inverse_map[i] gives the group index containing element i.
- `parity`: Vector of integers (±1) where parity[i] gives the parity of swaps needed to map X[i] to the representative.
"""
function find_symmetry_groups(X, Lx, Ly; trans_x=false, trans_y=false, refl_x=false, refl_y=false, spin_symmetry=false, hermitian=false, antihermitian=false)
    n = length(X)
    visited = falses(n)
    groups = Vector{Vector{Int}}()
    inverse_map = zeros(Int, n)
    parity = ones(Int, n)

    # Pre-compute hash for fast lookup
    config_hashes = [hash(sort(config)) for config in X]
    hash_to_indices = Dict{UInt64,Vector{Int}}()
    for (i, h) in enumerate(config_hashes)
        if haskey(hash_to_indices, h)
            push!(hash_to_indices[h], i)
        else
            hash_to_indices[h] = [i]
        end
    end

    for i in 1:n
        if visited[i]
            continue
        end

        group = Int[i]
        visited[i] = true
        group_index = length(groups) + 1
        inverse_map[i] = group_index
        parity[i] = 1

        # Generate all symmetry-related configurations
        equivalent_configs = generate_symmetric_configs(X[i], Lx, Ly,
            trans_x, trans_y, refl_x, refl_y,
            spin_symmetry, hermitian, antihermitian)

        # For each equivalent config, compute hash and check only candidates
        should_skip_group = false
        for (config, sign) in equivalent_configs
            h = hash(sort(config))
            if !haskey(hash_to_indices, h)
                continue
            end

            # Check self-consistency for antihermitian symmetry
            # If a config maps to itself with a sign flip (e.g. diagonal term), it must be zero.
            if sign == -1
                is_equal, swap_parity = configs_equal_with_parity(X[i], config)
                if is_equal
                    should_skip_group = true
                    break
                end
            end

            # Only check indices with matching hash
            for j in hash_to_indices[h]
                if j <= i || visited[j]
                    continue
                end

                is_equal, swap_parity = configs_equal_with_parity(X[j], config)
                if is_equal
                    push!(group, j)
                    visited[j] = true
                    inverse_map[j] = group_index
                    parity[j] = swap_parity * sign
                end
            end
        end

        if should_skip_group
            inverse_map[i] = 0 # Mark as zeroed
            parity[i] = 0
            for jdx in group
                if jdx != i
                    inverse_map[jdx] = 0
                    parity[jdx] = 0
                end
            end
        else
            push!(groups, group)
        end
    end

    return groups, inverse_map, parity
end

"""
    generate_symmetric_configs(config, Lx, Ly, trans_x, trans_y, refl_x, refl_y, spin_symmetry, hermitian, antihermitian)

Generate all configurations related by the specified symmetries.
Returns a vector of tuples `(config, sign)`, where sign is -1 if generated by anti-hermitian conjugation, 1 otherwise.
"""
function generate_symmetric_configs(config, Lx, Ly, trans_x, trans_y, refl_x, refl_y, spin_symmetry, hermitian, antihermitian)
    configs = [(config, 1)]

    # Apply translational symmetries - generate all translations
    if trans_x
        new_configs = Vector{eltype(configs)}()
        for (c, s) in configs
            for dx in 1:(Lx-1)
                push!(new_configs, (translate_x(c, dx, Lx), s))
            end
        end
        append!(configs, new_configs)
    end

    if trans_y
        new_configs = Vector{eltype(configs)}()
        for (c, s) in configs
            for dy in 1:(Ly-1)
                push!(new_configs, (translate_y(c, dy, Ly), s))
            end
        end
        append!(configs, new_configs)
    end

    # Apply reflection symmetries
    if refl_x
        reflected = [(reflect_x(c, Ly), s) for (c, s) in configs]
        append!(configs, reflected)
    end

    if refl_y
        reflected = [(reflect_y(c, Lx), s) for (c, s) in configs]
        append!(configs, reflected)
    end

    # Apply spin flip symmetry
    if spin_symmetry
        spin_flipped = [(flip_spin(c), s) for (c, s) in configs]
        append!(configs, spin_flipped)
    end

    # Apply hermitian conjugation
    if hermitian
        conjugated = [(hermitian_conjugate(c), s) for (c, s) in configs]
        append!(configs, conjugated)
    elseif antihermitian
        conjugated = [(hermitian_conjugate(c), -s) for (c, s) in configs]
        append!(configs, conjugated)
    end

    # Remove duplicates
    return unique(configs)
end

"""
    configs_equal_with_parity(c1, c2)

Check if two configurations are equal (considering order) and compute parity.
Returns (is_equal, parity) where parity is 1 for even number of swaps, -1 for odd.
Uses optimized permutation finding with early termination.
"""
function configs_equal_with_parity(c1, c2)
    n = length(c1)
    if n != length(c2)
        return false, 1
    end

    # Build a lookup dict for c2 for O(1) access
    c2_map = Dict{typeof(c2[1]),Vector{Int}}()
    for (j, elem) in enumerate(c2)
        if haskey(c2_map, elem)
            push!(c2_map[elem], j)
        else
            c2_map[elem] = [j]
        end
    end

    # Find the permutation
    permutation = zeros(Int, n)
    used = falses(n)

    for i in 1:n
        if !haskey(c2_map, c1[i])
            return false, 1
        end

        # Find first unused index in c2 that matches c1[i]
        found = false
        for j in c2_map[c1[i]]
            if !used[j]
                permutation[i] = j
                used[j] = true
                found = true
                break
            end
        end

        if !found
            return false, 1
        end
    end

    # Count inversions efficiently
    swap_count = count_inversions(permutation)
    parity = iseven(swap_count) ? 1 : -1

    return true, parity
end

"""
    count_inversions(perm)

Count the number of inversions in a permutation using merge sort approach.
This is O(n log n) instead of O(n²).
"""
function count_inversions(perm)
    n = length(perm)
    if n <= 1
        return 0
    end

    # Use merge sort to count inversions
    temp = similar(perm)
    return merge_sort_count!(copy(perm), temp, 1, n)
end

function merge_sort_count!(arr, temp, left, right)
    inv_count = 0
    if left < right
        mid = div(left + right, 2)
        inv_count += merge_sort_count!(arr, temp, left, mid)
        inv_count += merge_sort_count!(arr, temp, mid + 1, right)
        inv_count += merge_count!(arr, temp, left, mid, right)
    end
    return inv_count
end

function merge_count!(arr, temp, left, mid, right)
    i = left
    j = mid + 1
    k = left
    inv_count = 0

    while i <= mid && j <= right
        if arr[i] <= arr[j]
            temp[k] = arr[i]
            i += 1
        else
            temp[k] = arr[j]
            inv_count += (mid - i + 1)
            j += 1
        end
        k += 1
    end

    while i <= mid
        temp[k] = arr[i]
        i += 1
        k += 1
    end

    while j <= right
        temp[k] = arr[j]
        j += 1
        k += 1
    end

    for idx in left:right
        arr[idx] = temp[idx]
    end

    return inv_count
end


"""
    flip_spin(config)

Flip all spin values in config (1 ↔ 2).
The coordinates and op field are preserved.
"""
function flip_spin(config)
    return [(c, 3 - s, op) for (c, s, op) in config]
end

"""
    hermitian_conjugate(config)

Apply hermitian conjugation: swap :create ↔ :annihilate for all operators.
The coordinates and spin values are preserved.
"""
function hermitian_conjugate(config)
    return [(c, s, op == :create ? :annihilate : :create) for (c, s, op) in config[end:-1:1]]
end
"""
    translate_x(config, dx, Lx)

Translate all coordinates in config by dx in x direction with periodic boundaries.
The op field (:create or :annihilate) is preserved.
"""
function translate_x(config, dx, Lx)
    if dx == 0
        return config
    end
    return [(Coordinate(mod1(c.coordinates[1] + dx, Lx), c.coordinates[2]), s, op) for (c, s, op) in config]
end

"""
    translate_y(config, dy, Ly)

Translate all coordinates in config by dy in y direction with periodic boundaries.
The op field (:create or :annihilate) is preserved.
"""
function translate_y(config, dy, Ly)
    if dy == 0
        return config
    end
    return [(Coordinate(c.coordinates[1], mod1(c.coordinates[2] + dy, Ly)), s, op) for (c, s, op) in config]
end

"""
    reflect_x(config, Ly)

Reflect all coordinates in config about x axis (y → Ly + 1 - y).
The op field (:create or :annihilate) is preserved.
"""
function reflect_x(config, Ly)
    return [(Coordinate(c.coordinates[1], Ly + 1 - c.coordinates[2]), s, op) for (c, s, op) in config]
end

"""
    reflect_y(config, Lx)

Reflect all coordinates in config about y axis (x → Lx + 1 - x).
The op field (:create or :annihilate) is preserved.
"""
function reflect_y(config, Lx)
    return [(Coordinate(Lx + 1 - c.coordinates[1], c.coordinates[2]), s, op) for (c, s, op) in config]
end

function compute_conf_differences(s1::Tuple{Set,Set}, s2::Tuple{Set,Set})
    """
    The weight is the number of differences between two sets. Also,
    this is twice the number of swaps required to turn one set into the other
    """
    creation = Tuple([setdiff(s2[i], s1[i]) for i = 1:2])
    annihilation = Tuple([setdiff(s1[i], s2[i]) for i = 1:2])
    return creation, annihilation, sum([length(k) for k in creation])
end

function collect_all_conf_differences(indexer::CombinationIndexer)
    """
    returns a dictionary weights and dictionary weight_inv. weights maps 
    a tuple of indices (sorted) to their corresponding weight. weight_inv
    maps a weight to an array of all pairs with that weight.
    """
    difference_dict = Dict()
    for (i, s1) in enumerate(indexer.inv_comb_dict)
        for (j, s2) in enumerate(indexer.inv_comb_dict[i+1:end])
            j += i
            creation, annihilation, N = compute_conf_differences(s1, s2)
            if haskey(difference_dict, N)
                if haskey(difference_dict[N], (creation, annihilation))
                    push!(difference_dict[N][(creation, annihilation)], (i, j))
                else
                    difference_dict[N][(creation, annihilation)] = [(i, j)]
                end
            else
                difference_dict[N] = Dict((creation, annihilation) => [(i, j)])
            end
        end
    end
    return difference_dict
end
function find_N_body_interactions(U::AbstractArray, indexer::CombinationIndexer)
    H = -1im * log(U)
    difference_dict = collect_all_conf_differences(indexer)

    n_electrons = sum([length(indexer.inv_comb_dict[1][j]) for j = 1:2])
    second_quantized_order_labels = Dict()
    n_electrons = sum([length(indexer.inv_comb_dict[1][j]) for j = 1:2])
    # defining range of indices in the second quantized representation
    d = length(indexer.a) * 2
    m = 1
    for order ∈ 1:n_electrons
        if order == 1
            second_quantized_order_labels[order] = 1:d^(2*order)
            m = (length(indexer.a) * 2)^(2 * order)
        else
            second_quantized_order_labels[order] = (m+1):(m+d^(2*order))
            m += d^(2 * order)
        end
    end
    second_quantized_dimension = sum(length(s) for s in values(second_quantized_order_labels))
    second_quantized_solution = spzeros(ComplexF64, second_quantized_dimension)
    second_quantized_nullspace = []
    site_indexer = merge(Dict((s, :up) => k for (k, s) in enumerate(indexer.a)),
        Dict((s, :down) => k + length(indexer.a) for (k, s) in enumerate(indexer.a)))
    # inv_site_indexer = [[(s,:up) for s in indexer.a]; [(s,:down) for s in indexer.a]]

    # difference_dict = collect_all_conf_differences(indexer)
    for (swaps, N_diff_dict) in difference_dict
        # println(swaps)
        for (site_diff, index_pairs) in N_diff_dict
            creation, annihilation = site_diff

            params = binomial(2 * length(indexer.a) - 2 * swaps, n_electrons - swaps)
            variables = cumsum([binomial(2 * length(indexer.a) - 2 * swaps, k) for k in 0:n_electrons-swaps])
            min_order = argmax(variables .- params .>= 0)

            # this maps an index to a combination of sites (not including the hopping ones) which
            # have n_i applied on them
            variable_mapping = []
            inverse_variable_mapping = Dict()

            # defining variable_mapping and inverse_variable_mapping
            var_index = 1
            sites_available = [setdiff(setdiff(indexer.a, creation[σ]), annihilation[σ]) for σ in 1:2]
            for n_operators in 0:min_order-1
                for n_up in 0:n_operators
                    n_down = n_operators - n_up
                    up_site_combs = [Set(s) for s in combinations(sites_available[1], n_up)]
                    down_site_combs = [Set(s) for s in combinations(sites_available[2], n_down)]
                    for filled_sites in Iterators.product(up_site_combs, down_site_combs)
                        push!(variable_mapping, filled_sites)
                        inverse_variable_mapping[filled_sites] = var_index
                        var_index += 1
                    end
                end
            end


            matrix = zeros(ComplexF64, (params, variables[min_order]))
            vector = zeros(ComplexF64, params)
            # break
            row_index = 1
            for (i, j) in index_pairs
                common_sites = [intersect(indexer.inv_comb_dict[i][k], indexer.inv_comb_dict[j][k]) for k = 1:2]
                for (col_index, s) in enumerate(variable_mapping)
                    # col_index = inverse_variable_mapping[Tuple(common_sites)]
                    if issubset(s[1], common_sites[1]) && issubset(s[2], common_sites[2])
                        matrix[row_index, col_index] = 1
                    end
                end
                vector[row_index] = H[i, j]
                row_index += 1
            end

            nullspace_solution = nullspace(matrix)
            particular_solution = matrix \ vector
            if length(nullspace_solution) > 0
                push!(second_quantized_nullspace, spzeros(ComplexF64, second_quantized_dimension))
            end

            # put solution into a sparse matrix form in second quantized
            #figure out the indices for the sites, map it to an index and assign it to the sparse vector
            creation_index_list = []
            annihilation_index_list = []
            for (σ_i, σ) ∈ enumerate([:up, :down])
                for create_site in creation[σ_i]
                    push!(creation_index_list, site_indexer[(create_site, σ)])
                end
                for annihilate_site in annihilation[σ_i]
                    push!(annihilation_index_list, site_indexer[(annihilate_site, σ)])
                end
            end
            indices = [sort(creation_index_list) sort(annihilation_index_list)]'[:]

            for (k, s) in enumerate(variable_mapping)
                _indices = copy(indices)
                for (σ_i, σ) in enumerate([:up, :down])
                    for site in s[σ_i]
                        push!(_indices, site_indexer[(site, σ)])
                        push!(_indices, site_indexer[(site, σ)])
                    end
                end
                order = swaps + length(s[1]) + length(s[2])
                # println("order: $order swaps: $swaps indices: $_indices k: $k")
                starting_index = minimum(second_quantized_order_labels[order])
                i = sum((_indices[n] - 1) * d^(n - 1) for n in eachindex(_indices))
                second_quantized_solution[starting_index+i] = particular_solution[k]
                if length(nullspace_solution) > 0
                    second_quantized_nullspace[end][starting_index+i] = nullspace_solution[k]
                end
            end
        end
    end
    return second_quantized_solution, second_quantized_nullspace, second_quantized_order_labels

end


truncate(x, threshold) = ifelse(abs(x) < threshold, 0.0, x)
