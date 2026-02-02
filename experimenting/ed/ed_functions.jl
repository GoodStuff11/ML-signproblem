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

function permutation_parity(a::Vector)
    # computes the number of swaps required to sort and returns its parity (0,1)
    # vector can be anything so long as it can be sorted.
    p = sortperm(a, alg=MergeSort)
    visited = falses(length(p))
    parity = 0
    for i in eachindex(p)
        if !visited[i]
            len = 0
            j = i
            while !visited[j]
                visited[j] = true
                j = p[j]
                len += 1
            end
            parity += len - 1  # Each cycle of length `len` contributes (len - 1) swaps
        end
    end
    return parity % 2
end

function reordered_electron_parity(conf1::Vector, conf2::Vector, mapping)
    # given configurations of spin up and spin down electrons,
    # first map them to their new locations and find the number of 
    # permutations requires to reorder them. 1 if even, -1 if odd.

    # sorting is required since the configuration is unordered. Only when applying the mapping
    # does it become ordered.
    arr = sort(vcat(conf1, conf2)) # sorting ensures initial parity is 0.
    for i in eachindex(arr)
        arr[i] = mapping[arr[i]]
    end
    parity = permutation_parity(arr)
    return 1 - 2 * parity
end
function electron_parity(conf1_up::Vector, conf1_down::Vector, conf2_up::Vector, conf2_down::Vector)
    # given configurations of spin up and spin down electrons,
    # first map them to their new locations and find the number of 
    # permutations requires to reorder them. 1 if even, -1 if odd.

    # NOTE: it's important that the configurations are the right order when they're being used.
    arr1 = vcat(conf1_up, conf1_down)
    arr2 = vcat(conf2_up, conf2_down)
    parity = (permutation_parity(arr1) + permutation_parity(arr2)) % 2

    return 1 - 2 * parity
end

function degenerate_subspaces(E)
    # assumes that the energy eigenstates are sorted
    Ediff = diff(E)
    Ediff[abs.(Ediff).<1e-10] .= 0
    subspaces = []
    starting_index = 1
    for i ∈ eachindex(Ediff)
        if Ediff[i] != 0
            push!(subspaces, starting_index:i)
            starting_index = i + 1
        end
    end
    push!(subspaces, starting_index:length(Ediff)+1)

    return subspaces
end
function degeneracy_count(E)
    Ediff = diff(E)
    Ediff[abs.(Ediff).<1e-10] .= 0
    degen_tally = Dict()
    count = 0
    for d ∈ Ediff
        if d == 0
            count += 1
        elseif count > 0
            if haskey(degen_tally, count + 1)
                degen_tally[count+1] += 1
            else
                degen_tally[count+1] = 1
            end
            count = 0
        end
    end
    if count > 0
        if haskey(degen_tally, count + 1)
            degen_tally[count+1] += 1
        else
            degen_tally[count+1] = 1
        end
    end
    return degen_tally
end
function eigenvalue_of_qn(vec::Vector; atol::Real=1e-8)
    unique_vals = []
    for x in vec
        if all(y -> abs(x - y) > atol, unique_vals)
            push!(unique_vals, x)
        end
    end
    return unique_vals
end
function eigenvalue_mask(v::AbstractVector, qn::Int; atol=1e-8)
    target = eigenvalue_of_qn(v)[qn]
    idx = findall(x -> isapprox(x, target; atol=atol), v)

    return idx, target
end

# block diagonalization 
function build_block_graph(A; tol=1e-12)
    n = size(A, 1)
    G = Graphs.SimpleGraph(n)
    for i in 1:n, j in 1:n
        if abs(A[i, j]) > tol && i != j
            Graphs.add_edge!(G, i, j)
        end
    end
    return G
end

# Find connected components (blocks)
function find_nonadjacent_blocks(A; tol=1e-12)
    G = build_block_graph(A; tol=tol)
    comps = Graphs.connected_components(G)
    a = filter(x -> length(x) != 1, comps)
    b = filter(x -> length(x) == 1, comps)
    if length(b) > 0
        return a, reduce(vcat, b)
    else
        return a, []
    end
end
# end of block diagonalization

function filter_subspace(op_list::Vector, qn_list::Vector{Int}; atol=1e-8)
    # Ensure inputs are complex
    op_list_tmp = [ComplexF64.(copy(op)) for op in op_list]
    n = size(op_list[1], 1)

    # Initialize total basis transform as identity
    V_total = Matrix{ComplexF64}(LinearAlgebra.I, n, n)
    indices = collect(1:n)
    eigenvalues = []

    for (i, op) in enumerate(op_list_tmp)
        # Restrict operator and Hamiltonian to current subspace
        V_total = V_total[:, indices]
        op_sub = V_total' * op * V_total
        # restrict total basis transform too

        blocks, others = find_nonadjacent_blocks(op_sub)
        all_eigenvalues = zeros(ComplexF64, length(indices))
        if length(others) > 0
            all_eigenvalues[others] = diag(op_sub)[others]
        end
        indices = []
        for block in blocks
            # Diagonalize current operator
            # println(sum(abs.(op_sub' - op_sub)))
            _, V, E = schur(op_sub[block, block]) # diagonalizes normal matrices
            # println(sum(abs.(V'*V - I)))
            # println()
            all_eigenvalues[block] = E
            V_total[:, block] += V_total[:, block] * (V - LinearAlgebra.I)
        end
        sort!(indices)
        idx_mask, selected_eigs = eigenvalue_mask(all_eigenvalues, qn_list[i]; atol=atol)
        push!(eigenvalues, selected_eigs)
        append!(indices, idx_mask)
        # Apply current transform
    end
    # this matrix is composed of a unitary and a projector. It's unitary property is ensured
    V_total = V_total[:, indices]
    # println(sum(abs.(V_total'*V_total - I)))

    return V_total, eigenvalues
end
function filter_degenerate_subspace(H_unpert, H_pert, unperturbed_eigenstates)
    unpert_E = real.(diag(unperturbed_eigenstates' * H_unpert * unperturbed_eigenstates))
    subspaces = degenerate_subspaces(unpert_E)
    H_pert_eff = unperturbed_eigenstates' * H_pert * unperturbed_eigenstates
    for subspace in subspaces
        println(H_pert_eff[subspace, subspace])
    end
end
function count_degeneracies_per_subspace(H, ops)
    # returns a dictionary where the input is the quantum number, which maps to
    # a tuple containing the a dict containing the number of degeneracies and 
    # the dimension of the subspace
    degen = Dict()
    n = length(ops)
    indices = ones(Int64, n)
    while true
        try
            V, _ = filter_subspace(ops, indices)
            H_sub = V' * H * V
            degen[copy(indices)] = [degeneracy_count(real.(eigvals(H_sub))), size(H_sub)[1]]
            indices[end] += 1
        catch e
            if !(e isa BoundsError)
                rethrow(e)
            end
            if all(indices[2:end] .== 1)
                break
            end
            for j in n:-1:2
                if indices[j] > 1
                    indices[j] = 1
                    indices[j-1] += 1
                    break
                end
            end
        end
    end
    return degen
end
# function compute_sign(conf, sorted_sites::Vector{T}, creation_operators::Vector{Tuple{T,Int}}, annihilation_operators::Vector{Tuple{T,Int}}) where T
#     # c_{up,j} = F_{1} F_{2} ... F_{j-1} a_{up,j}
#     # c_{down,j} = F_{1} F_{2} ... F_{j} a_{down,j}
#     # F_i = (-1)^{n_i}, n_i = n_{up,i} + n_{down,i}
#     # where c^dagger_{i2,σ2} c_{i1,σ1} and assuming i2 >= i1 (if not, add negative sign)

#     # we assume that the sites in creation_operators/annihilation_operators are in sorted order (normal order)
#     creation_upper_site_bounds = Int[]
#     annihilation_upper_site_bounds = Int[]
#     for (op,list) in zip([creation_operators,annihilation_operators],[creation_upper_site_bounds,annihilation_upper_site_bounds])
#         for (s,σ) in op
#             i = findfirst(==(s),sorted_sites)
#             push!(list, i - (σ==1))
#         end
#     end
#     # find list of sites to count the electrons at
#     electron_count_sites = nothing
#     for (i, j) in zip(creation_upper_site_bounds, annihilation_upper_site_bounds)
#         (i,j) = sort([i,j])
#         if isnothing(electron_count_sites)
#             electron_count_sites = i:j
#         else
#             electron_count_sites = symdiff(electron_count_sites, i:j)
#         end
#     end

#     # count the number of electrons at the sites
#     swap_count = 0
#     for s in electron_count_sites
#         for c in conf
#             for occupation in c
#                 if sorted_sites[s] == occupation
#                     swap_count += 1
#                     break
#                 end
#             end
#         end
#     end

#     # adjust swap count to account for the creation and annihilation operators


#     return (-1)^swap_count
# end
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
function count_in_range(s::Set{T}, a::T, b::T; lower_eq::Bool=true, upper_eq::Bool=true) where T
    ### given a set of numbers s, counts the number of elements that are between
    ### a and b (where a could be larger or less than b). lower_eq and upper_eq specify
    ### whether the upper/lower bound is an equality or an inequality
    count = 0
    upper_bound = max(a, b)
    lower_bound = min(a, b)
    for elem in s
        if lower_eq
            cond1 = lower_bound <= elem
        else
            cond1 = lower_bound < elem
        end
        if upper_eq
            cond2 = elem <= upper_bound
        else
            cond2 = elem < upper_bound
        end
        if cond1 && cond2
            count += 1
        end
    end
    return count
end
function create_Sx!(rows::Vector{Int}, cols::Vector{Int}, vals::Vector{Float64}, magnitude::Float64, indexer::CombinationIndexer)
    for (i1, conf) in enumerate(indexer.inv_comb_dict)
        for σ ∈ [1, 2]
            for site_index ∈ setdiff(conf[σ], conf[3-σ])
                site_index = Set([site_index])
                # println(conf[1])
                # println(site_index)
                # println(union(conf[1], site_index))
                # println(indexer)
                if σ == 1
                    i2 = index(indexer, setdiff(conf[1], site_index), union(conf[2], site_index))
                else
                    i2 = index(indexer, union(conf[1], site_index), setdiff(conf[2], site_index))
                end
                push!(rows, i1)
                push!(cols, i2)
                push!(vals, magnitude / 2)
            end
        end
    end
end
function create_SziSzj!(rows::Vector{Int}, cols::Vector{Int}, vals::Vector{Float64}, magnitude::Float64, indexer::CombinationIndexer; iequalsj::Bool=false, NN::Union{Missing,AbstractLattice}=missing)
    if iequalsj
        create_chemical_potential!(rows, cols, vals, 1 / 4 * magnitude, indexer)
        create_hubbard_interaction!(rows, cols, vals, -1 / 2 * magnitude, false, indexer)
    end
    # this is for i != j
    # (n_{i,up} - n_{i,down}) (n_{j,up} - n_{j,down})
    for (i, conf) in enumerate(indexer.inv_comb_dict)
        total = 0
        for σ ∈ [1, 2]
            for σp ∈ [1, 2]
                for site_index1 ∈ setdiff(conf[σ], conf[3-σ])
                    for site_index2 ∈ setdiff(conf[σp], conf[3-σp])
                        if (site_index1 != site_index2) && (ismissing(NN) || (site_index2 in neighbors(NN, site_index1)))
                            total += (-1)^(σ != σp)
                        end
                    end
                end
            end
        end
        push!(rows, i)
        push!(cols, i)
        push!(vals, total / 4 * magnitude)
    end
end
function create_SiSj!(rows::Vector{Int}, cols::Vector{Int}, vals::Vector{Float64}, magnitude::Float64, indexer::CombinationIndexer; NN::Union{Missing,AbstractLattice}=missing)
    # This is for i!=j
    # c+_{i,up}c_{i,down}c+_{j,down}c_{j,up}
    for (i1, conf) in enumerate(indexer.inv_comb_dict)
        for σ ∈ [1, 2]
            for site_index1 ∈ setdiff(conf[σ], conf[3-σ])
                for site_index2 ∈ setdiff(conf[3-σ], conf[σ])
                    if !ismissing(NN) && !(site_index2 in neighbors(NN, site_index1))
                        continue
                    end
                    if σ == 1
                        i2 = index(indexer, replace(conf[1], site_index1 => site_index2), replace(conf[2], site_index2 => site_index1))
                    else
                        i2 = index(indexer, replace(conf[1], site_index2 => site_index1), replace(conf[2], site_index1 => site_index2))
                    end
                    push!(rows, i1)
                    push!(cols, i2)
                    push!(vals, magnitude / 2)
                end
            end
        end
    end
    create_SziSzj!(rows, cols, vals, magnitude, indexer; iequalsj=false, NN=NN)
end
function create_S2!(rows::Vector{Int}, cols::Vector{Int}, vals::Vector{Float64}, magnitude::Float64, indexer::CombinationIndexer)
    # sum Si*Si
    create_chemical_potential!(rows, cols, vals, 3 / 4 * magnitude, indexer)
    create_hubbard_interaction!(rows, cols, vals, -3 / 2 * magnitude, false, indexer)
    create_SiSj!(rows, cols, vals, magnitude, indexer)

end
function general_single_body!(
    rows::Vector{Int},
    cols::Vector{Int},
    vals::Vector{Float64},
    t::Dict,
    indexer::CombinationIndexer
)
    sorted_sites = sort(indexer.a)
    for (i1, conf) in enumerate(indexer.inv_comb_dict)
        for (σ1, σ2) ∈ Iterators.product(1:2, 1:2) # 1=up 2=down
            for site_index1 ∈ conf[σ1]
                possible_sites = setdiff(indexer.a, conf[σ2])
                if σ1 == σ2
                    possible_sites = union(possible_sites, [site_index1])
                end
                for site_index2 ∈ possible_sites
                    if Set([(site_index1, σ1), (site_index2, σ2)]) ∉ keys(t)
                        continue
                    end

                    # annihilate site_index1 and create site_index2
                    new_conf = [Set(), Set()]
                    if σ1 == σ2
                        new_conf[σ1] = replace(conf[σ1], site_index1 => site_index2)
                        new_conf[3-σ1] = conf[3-σ1]
                    else
                        new_conf[σ1] = setdiff(conf[σ1], [site_index1])
                        new_conf[σ2] = union(conf[σ2], [site_index2])
                    end
                    i2 = index(indexer, new_conf[1], new_conf[2])

                    sign = compute_jw_sign(conf, sorted_sites, [(site_index2, σ2, :create), (site_index1, σ1, :annihilate)])
                    push!(rows, i1)
                    push!(cols, i2)
                    push!(vals, t[Set([(site_index1, σ1), (site_index2, σ2)])] * sign)
                end
            end
        end
    end
end

function build_n_body_structure(
    t::Dict{Vector{Tuple{T,Int,Symbol}},U},
    indexer::CombinationIndexer;
    skip_lower_triangular::Bool=false
) where {T,U<:Number}
    build_n_body_structure(collect(keys(t)), indexer, U; skip_lower_triangular)
end

function build_n_body_structure(
    t_keys::Vector{Vector{Tuple{T,Int,Symbol}}},
    indexer::CombinationIndexer,
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
    t_vals::Vector{U},
    parameter_mapping::Union{Vector{Int},Nothing}=nothing,
    parity::Union{Vector{Int},Nothing}=nothing
) where {U<:Number}
    # it's allowed for length(t_vals) < length(t_keys), but a parameter_mapping to make the difference is required.
    if isnothing(parameter_mapping)
        return [t_vals[param_index_map[i]] * signs[i] for i in eachindex(signs)]
    else
        return [parameter_mapping[param_index_map[i]] == 0 ? U(0) : t_vals[parameter_mapping[param_index_map[i]]] * parity[param_index_map[i]] * signs[i]
                for i in eachindex(signs)]
    end
end
function general_n_body!(
    rows::Vector{Int},
    cols::Vector{Int},
    vals::Vector{U},
    t::Dict{Vector{Tuple{T,Int,Symbol}},U},
    indexer::CombinationIndexer
) where {T,U<:Number}
    # requires applying Hermitian to the resulting sparse matrix
    _rows, _cols, signs, ops_list = build_n_body_structure(t, indexer; skip_lower_triangular=false)
    _vals = update_values(signs, ops_list, collect(keys(t)), collect(values(t)))
    append!(rows, _rows)
    append!(cols, _cols)
    append!(vals, _vals)
end
function compute_correlation(state::Vector, order::Int, indexer::CombinationIndexer; correlation_args=nothing)
    # computes γ_ij = ⟨ψ|c†_i c_j|ψ⟩ (generalized to more than single body)
    if isnothing(correlation_args)
        mats, unique_sites = correlation_matrix(order, indexer)
    else
        mats, unique_sites = correlation_args
    end

    correlation = Matrix{ComplexF64}(undef, length(unique_sites), length(unique_sites))
    for j in axes(mats, 2)
        for i in axes(mats, 1)
            correlation[i, j] = state' * mats[i, j] * state
        end
    end
    return correlation
end
function correlation_matrix(order::Int, indexer::CombinationIndexer)
    # computes the matrix of operators c†_i c_j
    t_dict = create_randomized_nth_order_operator(order, indexer)
    dim = length(indexer.inv_comb_dict)
    rows, cols, signs, ops_list = build_n_body_structure(t_dict, indexer; skip_lower_triangular=false)
    unique_sites = unique([[o[1:2] for o in op][1:length(op)÷2] for op in ops_list])
    unique_ops = unique(ops_list)
    site_to_index = Dict(s => i for (i, s) in enumerate(unique_sites))

    mats = Matrix{AbstractArray}(undef, length(unique_sites), length(unique_sites))

    for op in unique_ops
        indices = findall(x -> x == op, ops_list)
        s1 = [o[1:2] for o in op][1:length(op)÷2]
        s2 = [o[1:2] for o in op][length(op)÷2+1:end]
        # println(s1)
        mats[site_to_index[s1], site_to_index[s2]] = sparse(rows[indices], cols[indices], signs[indices], dim, dim)
    end
    return mats, unique_sites
end
function create_nearest_neighbor_operator(t::Float64, subspace::HubbardSubspace, indexer::CombinationIndexer)
    t_dict = Dict{Vector{Tuple{Coordinate{2,Int64},Int,Symbol}},Float64}()

    for σ in 1:2
        for s1 in indexer.a
            for s2 in neighbors(subspace.lattice, s1)
                if [(s1, σ, :create), (s2, σ, :annihilate)] ∉ keys(t_dict)
                    t_dict[[(s1, σ, :create), (s2, σ, :annihilate)]] = 0.5 * t
                else
                    t_dict[[(s1, σ, :create), (s2, σ, :annihilate)]] += 0.5 * t
                end
            end
        end
    end
    return t_dict
end
function is_slater_determinant(state::Vector, indexer::CombinationIndexer; get_value::Bool=false, correlation_args=nothing)
    γ = compute_correlation(state, 1, indexer; correlation_args=correlation_args)
    val = sum(abs.(γ^2 - γ))
    if get_value
        return val
    end
    return val < 1e-10
end
function create_randomized_nth_order_operator(n::Int, indexer::CombinationIndexer;
    magnitude::T=1e-3 + 0im, omit_H_conj::Bool=false, conserve_spin::Bool=false) where T
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
        if (!omit_H_conj || geq_ops(ops_create, ops_annihilate)) && (!conserve_spin || equal_spin(ops_create, ops_annihilate))
            if key ∉ keys(t_dict)
                t_dict[key] = (2 * rand() - 1) / 2 * magnitude
            else
                t_dict[key] += (2 * rand() - 1) / 2 * magnitude
            end
        end
    end
    return t_dict
end


# output is a minimal number of parameters that are a subset of t_dict values (t_dict should be random, so exactly which shouldn't matter)
# as a vector reducedparameters, and also a parameter mapping vector parameter_mapping. 
# parameter_mapping is defined such that values(t_dict)[i] = reduced_parameters[parameter_mapping[i]]
# NOTE: this function only measures equivalences based on the symmetries, it doesn't care if the transformed states are actually in t_dict
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
     It's an additional symmetry that the Hubbard model has, but it's a bit weaker than Sx. However, this does apply
     to systems in which N↑=N↓)
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
            # If the group is invalid (self-cancelling), we effectively discard it.
            # We can mark it as visited but set parity to 0 to indicate it should be dropped?
            # Or just set its inverse_map to 0?
            inverse_map[i] = 0 # Mark as zeroed
            parity[i] = 0
            for jdx in group
                if jdx != i
                    inverse_map[jdx] = 0
                    parity[jdx] = 0
                end
            end
            # Remove the group from the list (or replace with empty/dummy)
            # Since we appended to groups at the end, we just don't append it.
            # But we incremented group_index.
            # Let's clean up group_index usage.
            # Wait, we incremented inverse_map with group_index.
            # If we skip, we should handle that.
            # Actually, reusing group_index logic:
            # If we don't push(groups, group), indices shift? 
            # inverse_map stores index into groups.
            # So if we skip, we don't add to groups.
            # But we already assigned group_index = length(groups) + 1.
            # So if we don't push, subsequent groups will have wrong index?
            # No, if we skip, group_index will be reused for next i.
            # Wait, `group_index` variable is local.
            # BUT we assigned `inverse_map[i] = group_index`.
            # We should revert that.
        else
            push!(groups, group)
        end
    end

    # Clean up inverse_map (optional, but good for safety)
    # 0 values indicate terms that vanish due to symmetry constraints

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

    for i in left:right
        arr[i] = temp[i]
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


function create_nn_hopping!(rows::Vector{Int}, cols::Vector{Int}, vals::Vector{Float64}, t::Union{Float64,AbstractArray{Float64}}, lattice::AbstractLattice, indexer::CombinationIndexer)
    if isa(t, Number)
        t = [t]
    end
    for (i1, conf) in enumerate(indexer.inv_comb_dict)
        for σ ∈ [1, 2]
            for site_index1 ∈ conf[σ]
                for order in eachindex(t)
                    for site_index2 ∈ neighbors(lattice, site_index1, order)
                        if site_index2 ∉ conf[σ]
                            new_conf = replace(conf[σ], site_index1 => site_index2)
                            if σ == 1
                                i2 = index(indexer, new_conf, conf[2])
                            else
                                i2 = index(indexer, conf[1], new_conf)
                            end
                            sign = (-1)^(count_in_range(conf[1], site_index1, site_index2; lower_eq=true, upper_eq=false) +
                                         count_in_range(if (σ == 2)
                                                 new_conf
                                             else
                                                 conf[2]
                                             end, site_index1, site_index2; lower_eq=false, upper_eq=true) +
                                         (site_index1 > site_index2))
                            push!(rows, i1)
                            push!(cols, i2)
                            push!(vals, -0.5 * t[order] * sign)# 0.5 due to double counting from neighbors for some reason
                        end
                    end
                end
            end
        end
    end
end

function create_hubbard_interaction!(rows::Vector{Int}, cols::Vector{Int}, vals::Vector{Float64}, U::Float64, half_filling::Bool, indexer::CombinationIndexer)
    # this corresponds to only n_{i up} n_{i down}
    if half_filling
        for (i, conf) in enumerate(indexer.inv_comb_dict)
            num_negative = length(setdiff(union(conf[1], conf[2]), intersect(conf[1], conf[2])))
            num_positive = length(indexer.a) - num_negative
            push!(rows, i)
            push!(cols, i)
            push!(vals, U * (num_positive - num_negative) / 4)
        end
    else
        for (i, conf) in enumerate(indexer.inv_comb_dict)
            push!(rows, i)
            push!(cols, i)
            push!(vals, U * length(intersect(conf[1], conf[2])))
        end
    end
end
function create_chemical_potential!(rows::Vector{Int}, cols::Vector{Int}, vals::Vector{Float64}, μ::Float64, indexer::CombinationIndexer)
    for (i, conf) in enumerate(indexer.inv_comb_dict)
        push!(rows, i)
        push!(cols, i)
        # last part breaks degeneracy ensuring that ED orders them consistently
        push!(vals, μ * (length(conf[1]) + length(conf[2]))) #+ 1e-7*sum(conf[1]) + 43e-7*sum(conf[2]) 
    end
end
function create_∏σx!(rows::Vector{Int}, cols::Vector{Int}, vals::Vector{Float64}, magnitude::Float64, indexer::CombinationIndexer)
    for (i1, conf) in enumerate(indexer.inv_comb_dict)
        i2 = index(indexer, conf[2], conf[1])
        sign = electron_parity(collect(conf[1]), collect(conf[2]), collect(conf[2]), collect(conf[1]))
        push!(rows, i1)
        push!(cols, i2)
        push!(vals, magnitude * sign)
    end
end
function create_Sz!(rows::Vector{Int}, cols::Vector{Int}, vals::Vector{Float64}, magnitude::Float64, indexer::CombinationIndexer)
    for (i, conf) in enumerate(indexer.inv_comb_dict)
        push!(rows, i)
        push!(cols, i)
        push!(vals, magnitude * (length(conf[1]) - length(conf[2])))
    end
end
function create_transform!(rows::Vector{Int}, cols::Vector{Int}, vals::Vector{Float64}, magnitude::Float64, mapping::Dict, indexer::CombinationIndexer)
    # hermitian and unitary matrix that reflects across the x axis
    for (i, conf) in enumerate(indexer.inv_comb_dict)
        push!(rows, i)
        # println(conf[1])
        # println(mapping)
        new_conf1 = replace(conf[1], mapping...)
        new_conf2 = replace(conf[2], mapping...)
        sign = reordered_electron_parity(collect(conf[1]), collect(conf[2]), mapping)
        # this doesn't work I think because replace on sets messes up the ordering.
        # sign = electron_parity(collect(conf[1]), collect(conf[2]), collect(new_conf1), collect(new_conf2)) 
        push!(cols, index(indexer, new_conf1, new_conf2))
        push!(vals, magnitude * sign)
    end
end
function create_L2!(rows::Vector{Int}, cols::Vector{Int}, vals::Vector{Float64}, magnitude::Float64, indexer::CombinationIndexer)
    # sum Sz^2 operator
    create_hubbard_interaction!(rows, cols, vals, magnitude, false, indexer)
    for (i1, conf) in enumerate(indexer.inv_comb_dict)
        Nup = length(conf[1])
        Ndown = length(conf[2])
        n = length(indexer.a)
        push!(rows, i1)
        push!(cols, i1)
        push!(vals, magnitude * ((Nup + Ndown - n)^2 / 4 - (Nup + Ndown - n)))

        # L+L-  +  L-L+
        for site_index1 ∈ intersect(conf[1], conf[2])
            for site_index2 ∈ setdiff(indexer.a, union(conf[1], conf[2]))
                i2 = index(indexer, replace(conf[1], site_index1 => site_index2), replace(conf[2], site_index1 => site_index2))
                push!(rows, i1)
                push!(cols, i2)
                push!(vals, magnitude * (-1)^(sum(site_index1.coordinates) + sum(site_index2.coordinates)))
            end
        end
    end
end

function create_operator(Hs::HubbardSubspace, op; kind=1)
    dim = get_subspace_dimension(Hs)
    indexer = CombinationIndexer(reduce(vcat, collect(sites(Hs.lattice))), get_subspace_info(Hs)...)
    rows = Int[]
    cols = Int[]
    vals = Float64[]

    #insert stuff here
    if op == :σ
        mapping = reflection_mapping(Hs.lattice, kind)
        create_transform!(rows, cols, vals, 1.0, mapping, indexer)
    elseif op == :Sx
        create_Sx!(rows, cols, vals, 1.0, indexer)
    elseif op == :∏σx
        create_∏σx!(rows, cols, vals, 1.0, indexer)
    elseif op == :S2
        create_S2!(rows, cols, vals, 1.0, indexer)
    elseif op == :L2
        create_L2!(rows, cols, vals, 1.0, indexer)
    elseif op == :T
        mapping = translation_mapping(Hs.lattice, kind)
        create_transform!(rows, cols, vals, 1.0, mapping, indexer)
    end

    H = sparse(rows, cols, vals, dim, dim)

    return H
end
function create_Hubbard(Hm::HubbardModel, Hs::HubbardSubspace; get_indexer::Bool=false, indexer::Union{CombinationIndexer,Nothing}=nothing)
    # specify the subspace
    dim = get_subspace_dimension(Hs)
    if isnothing(indexer)
        indexer = CombinationIndexer(reduce(vcat, collect(sites(Hs.lattice))), get_subspace_info(Hs)...)
    end
    rows = Int[]
    cols = Int[]
    vals = Float64[]

    #Constructs the sparse hopping Hamiltonian matrix \sum_{<i,j>} c^\dagger_i c_j.
    if Hm.t > 0
        create_nn_hopping!(rows, cols, vals, Hm.t, Hs.lattice, indexer)
    end
    if Hm.U > 0
        create_hubbard_interaction!(rows, cols, vals, Hm.U, Hm.half_filling, indexer)
    end
    if Hm.μ > 0
        create_chemical_potential!(rows, cols, vals, Hm.μ, indexer)
    end

    # create_SziSzj!(rows, cols, vals, 0.021, indexer; iequalsj=true)
    # create_operator!(rows, cols, vals, 1e-2, indexer)

    # constuct Hamiltonian
    H = sparse(rows, cols, vals, dim, dim)
    if get_indexer
        return H, indexer
    end
    return H
end


function create_Heisenberg(t, J, Hs::HubbardSubspace)
    # specify the subspace
    dim = get_subspace_dimension(Hs)
    indexer = CombinationIndexer(collect(1:prod(size(Hs.lattice))), get_subspace_info(Hs)...)

    rows = Int[]
    cols = Int[]
    vals = Float64[]

    #Constructs the sparse hopping Hamiltonian matrix \sum_{<i,j>} c^\dagger_i c_j.
    create_nn_hopping!(rows, cols, vals, t, Hs.lattice, indexer)
    create_SiSj!(rows, cols, vals, J, indexer; NN=Hs.lattice)

    # constuct Hamiltonian
    H = sparse(rows, cols, vals, dim, dim)

    return H, indexer
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
function full_unitary_analysis(degen_rm_U::Vector, difference_dict::Dict, U_values::Vector)
    # data = Dict(order=>Dict() for order in 1:max(keys(difference_dict)))
    norders = maximum(collect(keys(difference_dict)))
    data = Dict(order => Dict() for order in 1:norders)
    for (u_index, u) in enumerate(U_values)
        hopping = log(degen_rm_U[1]' * degen_rm_U[u_index])
        for (order, creation_annihiation) in difference_dict
            if length(data[order]) == 0
                data[order] = Dict(u => [])
            elseif u ∉ keys(data[order])
                data[order][u] = []
            end

            for index_list in values(creation_annihiation)
                for (i, j) in index_list
                    push!(data[order][u], hopping[i, j])
                end
            end
        end
    end

    labels = ["norm1", "norm2", "total_count", "count_nonzero"]
    summarized_data = Dict{String,Any}(label => Dict("orders" => Vector{Vector{Any}}(undef, norders)) for label ∈ labels)
    for order in 1:norders
        for key in keys(summarized_data)
            summarized_data[key]["orders"][order] = []
        end
        for u in U_values
            push!(summarized_data["norm1"]["orders"][order], norm(data[order][u], 1))
            push!(summarized_data["norm2"]["orders"][order], norm(data[order][u], 2))
            push!(summarized_data["count_nonzero"]["orders"][order], sum(abs.(data[order][u]) .> 0))
            push!(summarized_data["total_count"]["orders"][order], length(data[order][u]))
        end
    end
    return summarized_data
end

function greedy_col_permutation_for_diag(A::AbstractMatrix)
    @assert size(A, 1) == size(A, 2) "Matrix must be square"

    n = size(A, 1)
    assigned_cols = falses(n)
    permutation = zeros(Int, n)

    for row in 1:n
        best_col = 0
        best_val = -Inf

        for col in 1:n
            if !assigned_cols[col] && abs(A[row, col]) > best_val
                best_val = abs(A[row, col])
                best_col = col
            end
        end

        if best_col == 0
            error("Failed to assign col for row $row — no unassigned cols left.")
        end

        permutation[row] = best_col
        assigned_cols[best_col] = true
    end

    A_permuted = A[:, permutation]
    return permutation, A_permuted
end
function create_consistent_basis(H::Vector, ops::Vector; reference_index::Int64=1)
    degen = count_degeneracies_per_subspace(H[reference_index], ops)
    return create_consistent_basis(H, ops, degen)
end
function create_consistent_basis(H::Vector, ops::Vector, degen::Dict)
    """
    H is a list of hamiltonians (matrices) where adjacent elements are
    sufficiently close to each other so that energy eigenstates with adjacent
    Hamiltonians should have high overlap.

    degen comes from the output of count_degeneracies_per_subspace()

    Returns a vector of unitary operators which diagonalize the Hamiltonian,
    with the property that V[i]'*V[i+1] approx diagonal.

    """
    degen_rm_U = []
    transforms = Dict()
    for indices in keys(degen)
        prev_perm = nothing
        prev_phases = nothing
        for (h_i, h) in enumerate(H) # assumes these hamiltonians are sorted
            # println(indices)
            if indices in keys(transforms)
                basis_transform = transforms[indices]
            else
                basis_transform, _ = filter_subspace(ops, indices)
                transforms[indices] = basis_transform
            end
            # algorithm for uncrossing eigenstates
            _, V1, _ = schur(basis_transform' * h * basis_transform)
            if isnothing(prev_perm)
                prev_perm = 1:size(V1, 2)
                prev_phases = ones(size(V1, 2))
            end
            if h_i == 1
                perm = 1:size(V1, 2)
                phases = LinearAlgebra.I
            else
                _, V0, _ = schur(basis_transform' * H[h_i-1] * basis_transform)
                perm, mat = greedy_col_permutation_for_diag((V0[:, prev_perm] * prev_phases)' * V1)
                phases = diagm(abs.(diag(mat)) ./ diag(mat))
                # println(diag(phases*mat))
            end


            if length(degen_rm_U) < h_i
                push!(degen_rm_U, basis_transform * V1[:, perm] * phases)
            else
                degen_rm_U[h_i] = hcat(degen_rm_U[h_i], basis_transform * V1[:, perm] * phases)
            end
            prev_perm = perm
            prev_phases = phases
        end
    end


    return degen_rm_U
end

"""
Using only information saved to files, reconstruct sparse matrix for unitary. This can reconstruct
the matrices stored in all_matrices

ex.
data_dict_tmp = load_saved_dict(joinpath(folder,"unitary_map_energy_N=4_200.jld2"))
energy_dict = load_saved_dict(joinpath(folder,"meta_data_and_E.jld2"))

mat = construct_sparse(energy_dict["indexer"], 
    identity.(data_dict_tmp["coefficient_labels"][1]),
    data_dict_tmp["coefficients"][1][1])
"""
function construct_sparse(
    indexer::CombinationIndexer, coefficient_labels::Vector{Vector{Tuple{T,Int64,Symbol}}},
    coefficients; parameter_mapping=nothing, parities=nothing) where {T}
    dim = length(indexer.inv_comb_dict)
    rows, cols, signs, ops_list = build_n_body_structure(coefficient_labels, indexer)
    param_index_map = build_param_index_map(ops_list, collect(keys(coefficient_labels)))
    vals = update_values(signs, param_index_map, coefficients, parameter_mapping, parities)
    return make_hermitian(sparse(rows, cols, vals, dim, dim))
end

truncate(x, threshold) = ifelse(abs(x) < threshold, 0.0, x)

function project_hermitian(H, v::AbstractVector, target_eig_idx::Int, all_eigs::Vector{<:Real}; safety_factor=1.1)
    target_eig = all_eigs[target_eig_idx]

    # 1. Analyze Spectral Properties
    E_min, E_max = minimum(all_eigs), maximum(all_eigs)
    spectral_width = E_max - E_min

    # Calculate the smallest gap between target and any other eigenvalue
    # Filter out the target itself to avoid finding a gap of 0
    other_eigs = filter(e -> !isapprox(e, target_eig, atol=1e-9), all_eigs)

    if isempty(other_eigs)
        return v # If there's only 1 eigenvalue, v is already in the eigenspace
    end

    min_gap = minimum(abs.(other_eigs .- target_eig))

    # 2. Determine Sampling Parameters

    # Time step dt: Nyquist limit to avoid aliasing the spectrum
    # We add a small buffer (+1e-5) to spectral_width to ensure we are strictly below Nyquist
    dt = 2 * π / (spectral_width + 1.0)

    # Total time T: Required to resolve the smallest energy gap
    # Uncertainty principle: Time * Energy ~ 2π
    T_required = (2 * π / min_gap) * safety_factor

    # Number of steps K
    K = ceil(Int, T_required / dt)

    # 3. Perform the Projection Sum
    # P = (1/K) * sum_{k=0}^{K-1} exp(-i * t_k * target) * exp(i * t_k * H)

    v_proj = zero(v)

    # We use a rectangular window here. 
    # Note: For very crowded spectra, a Gaussian window (tapering coefficients) 
    # would suppress spectral leakage better, but rectangular is the standard "sum".
    for k in 1:K-1
        t = k * dt

        # Coefficient: cancel the phase of the target eigenvalue
        # c_k = exp(-i * \lambda_{target} * t)
        coeff = exp(-im * target_eig * t)

        # Evolution: exp(i * H * t) * v
        v_evolved = expv(im * t, H, v)

        v_proj .+= coeff .* v_evolved
    end

    return normalize(v_proj)
end


function find_representatives(dim::Int, eig_indices::Vector{Int}, n_eigs::Vector{Int},
    mapping::Vector, mapping_sign::Vector)
    # EXAMPLE:
    # n_eigs = [2,4]
    # eig_indices = [2,1]
    # mapping = []
    # s_mapping = [] 
    # begin for kind in 1:2
    #     op = create_operator(subspace,:T, kind=kind)
    #     r, c, v = findnz(op)
    #     push!(mapping, r)
    #     push!(mapping_sign, v)
    # end        
    # 1<= eig_idx <= n_eigs
    checked_indices = Array{Any}(undef, dim)
    representative_indices = []
    associated_representative = zeros(Int, dim)
    magnitude = []
    for i = 1:dim
        if isassigned(checked_indices, i)
            continue
        end
        # println(i)

        period = ones(Int, length(eig_indices))
        num_stabilizers = 1 # identity is a stabilizer
        stabilizers = []
        stabilizer_signs = []
        period_signs = ones(Int, length(eig_indices))

        # finding periods
        for (l, (map_l, sign_l)) in enumerate(zip(mapping, mapping_sign))
            j = map_l[i]
            period_signs[l] *= sign_l[j]
            while j != i
                period[l] += 1
                j = map_l[j]
                period_signs[l] *= sign_l[j]
            end
        end

        # searching through remaining states corresponding to representative
        index_matrix = zeros(Int, period...)
        sign_matrix = ones(Int, period...)
        index_matrix[ones(Int, period...)...] = i
        checked_indices[i] = (Tuple(ones(Int, length(period))), 1)
        # println(checked_indices[1:20])
        for indices in Iterators.product([1:k for k in period]...)
            if index_matrix[indices...] != 0
                continue
            end

            prev_indices = collect(indices)
            op_k = 0
            for k in eachindex(indices)
                if indices[k] > 1
                    prev_indices[k] -= 1
                    op_k = k
                    break
                end
            end

            prev_index = index_matrix[prev_indices...]
            j = mapping[op_k][prev_index]
            applied_sign = Int(mapping_sign[op_k][prev_index])
            index_matrix[indices...] = j
            sign_matrix[indices...] = applied_sign * sign_matrix[prev_indices...]

            if !isassigned(checked_indices, j) && prev_indices != checked_indices[prev_index][1]
                checked_indices[j] = (indices, sign_matrix[indices...])
                associated_representative[j] = i
            end
            if j == i && isassigned(checked_indices, j)
                # we have a stabilizer, since a state corresponding to a representative is
                # associated with the representative in >1 ways
                num_stabilizers += 1
                push!(stabilizers, collect(indices))
                push!(stabilizer_signs, sign_matrix[indices...])
            end
        end

        # computing representative weight
        # mR/N + 1/2 = Z if overall_sign == -1
        # mR + N/2 = Z*N
        # mR/N = Z if overall_sign == 1
        mag = num_stabilizers
        for (s, p, n_eig, eig_i) in zip(period_signs, period, n_eigs, eig_indices)
            if p == n_eig
                continue
            elseif s == -1 && n_eig % (2 * p) != 0
                mag *= 0 #sum(exp(1im*l*2π*((eig_i-1)*p/n_eig + 0.5)) for l=0:(n_eig ÷ p-1))
                println("THIS SHOULDNT HAPPEN")
            elseif ((eig_i - 1) * p + (s == -1) * n_eig / 2) % n_eig ≈ 0
                mag *= n_eig / p
            else
                mag *= 0
                break
            end
        end
        # stabilizers impose an additional constraint on making the magnitude non-zero, 2π k⋅g = 2π(Z + 1/2) (1/2 is omited when s==1)
        all_n_eigs = prod(n_eigs)
        for (coord, s) in zip(stabilizers, stabilizer_signs)
            if !((dot(eig_indices .- 1, (coord .- 1) .* all_n_eigs .÷ n_eigs) + (s == -1) * all_n_eigs ÷ 2) % all_n_eigs ≈ 0)
                mag = 0
                break
            end

        end
        # println(mag)
        if !(mag ≈ 0)
            push!(representative_indices, i)
            associated_representative[i] = -length(representative_indices)
            push!(magnitude, mag)
        end

        # if i > 50
        #     break
        # end

    end


    return checked_indices, representative_indices, associated_representative, magnitude
end

function construct_hamiltonian(
    r, c, v;
    checked_indices, representative_indices, associated_representative,
    magnitude, n_eigs, eig_indices
)
    h = spzeros(ComplexF64, length(representative_indices), length(representative_indices))

    i = 1
    for (in_rep_idx, rep_idx) in enumerate(representative_indices)
        while i <= length(c) && c[i] < rep_idx
            i += 1
            # if i > 3000
            #     error("hey")
            # end
        end
        # println(in_rep_idx)
        while i <= length(c) && c[i] == rep_idx
            output_idx = r[i]
            h_val = v[i]
            if associated_representative[output_idx] < 0
                out_rep_idx = abs(associated_representative[output_idx])
                phase = 1
                relative_sign = 1
            elseif associated_representative[output_idx] > 0 && associated_representative[associated_representative[output_idx]] < 0
                out_rep_idx = abs(associated_representative[associated_representative[output_idx]])
                (unitary_distance, relative_sign) = checked_indices[output_idx]
                exp_val = 2π * sum((l - 1) * (m - 1) / N for (l, N, m) in zip(unitary_distance, n_eigs, eig_indices))
                phase = cis(-exp_val)
            else
                i += 1
                continue
            end
            # if (out_rep_idx == 1 && in_rep_idx == 45)|| (out_rep_idx == 45 && in_rep_idx == 1)
            #     println(i)
            #     println("($out_rep_idx,$in_rep_idx) - ($(r[i]),$(c[i])):  $h_val, $relative_sign, $phase, $(magnitude[out_rep_idx]), $(magnitude[in_rep_idx])")
            # end
            h[out_rep_idx, in_rep_idx] += h_val * relative_sign * phase * sqrt(abs(magnitude[out_rep_idx] / magnitude[in_rep_idx]))
            i += 1
        end
    end
    return h
end

function reconstruct_full_vector(
    vec::AbstractVector,
    mapping::Vector,
    s_mapping::Vector,
    representative_indices::Vector,
    magnitude::AbstractVector,
    eig_indices::Vector,
    n_eigs::Vector)

    reconstruct_full_vector(
        reshape(vec, 1, length(vec)),
        mapping, s_mapping, representative_indices, magnitude, eig_indices, n_eigs
    )[1, :]
end

function reconstruct_full_vector(
    vec::AbstractMatrix,
    mapping::Vector,
    s_mapping::Vector,
    representative_indices::Vector,
    magnitude::AbstractVector,
    eig_indices::Vector,
    n_eigs::Vector,
)
    n_ops = length(mapping)
    full_dim = length(mapping[1])

    T = promote_type(eltype(vec), ComplexF64)
    full_vec = zeros(T, size(vec)[1], full_dim)

    eigvals = cis.(2π .* (eig_indices .- 1) ./ n_eigs)
    group_size = prod(n_eigs)

    function apply_ops!(α, idx, phase, op)
        if op > n_ops
            full_vec[:, idx] += vec[:, α] * phase / sqrt(magnitude[α] * group_size)
            return
        end

        cur_idx = idx
        cur_phase = phase

        for _ in 1:n_eigs[op]
            apply_ops!(α, cur_idx, cur_phase, op + 1)
            cur_phase *= eigvals[op] * s_mapping[op][cur_idx]
            cur_idx = mapping[op][cur_idx]
        end
    end

    for α in 1:size(vec)[2]
        apply_ops!(α, representative_indices[α], one(T), 1)
    end

    return full_vec
end