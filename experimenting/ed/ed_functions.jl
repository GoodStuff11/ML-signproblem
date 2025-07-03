function permutation_parity(a::Vector)
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
    arr = sort(vcat(conf1, conf2))
    for i in eachindex(arr)
        arr[i] = mapping[arr[i]]
    end
    parity = permutation_parity(arr)
    return 1-2*parity

end
function degenerate_subspaces(E)
    # assumes that the energy eigenstates are sorted
    Ediff = diff(E)
    Ediff[abs.(Ediff) .< 1e-10] .= 0
    subspaces = []
    starting_index = 1
    for i ∈ eachindex(Ediff)
        if Ediff[i] != 0 
            push!(subspaces, starting_index:i)
            starting_index = i+1
        end
    end
    push!(subspaces, starting_index:length(Ediff)+1)

    return subspaces
end
function degeneracy_count(E)
    Ediff = diff(E)
    Ediff[abs.(Ediff) .< 1e-10] .= 0
    degen_tally = Dict()
    count = 0
    for d ∈ Ediff
        if d == 0
            count += 1
        elseif count > 0
            if haskey(degen_tally, count+1)
                degen_tally[count+1] += 1
            else
                degen_tally[count+1] = 1
            end
            count = 0
        end
    end
    if count > 0
        if haskey(degen_tally, count+1)
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
        if abs(A[i,j]) > tol && i != j
            Graphs.add_edge!(G, i, j)
        end
    end
    return G
end

# Find connected components (blocks)
function find_nonadjacent_blocks(A; tol=1e-12)
    G = build_block_graph(A; tol=tol)
    comps = Graphs.connected_components(G)
    a = filter(x->length(x)!=1,comps)
    b = filter(x->length(x)==1,comps)
    if length(b) > 0
        return a,reduce(vcat, b)
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
    V_total = Matrix{ComplexF64}(I, n, n)
    indices = collect(1:n)
    eigenvalues = []

    for (i, op) in enumerate(op_list_tmp)
        # Restrict operator and Hamiltonian to current subspace
        V_total = V_total[:, indices]
        op_sub = V_total'*op*V_total
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
            V_total[:, block] += V_total[:, block] * (V-I)
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
    unpert_E = real.(diag(unperturbed_eigenstates'*H_unpert*unperturbed_eigenstates))
    subspaces = degenerate_subspaces(unpert_E)
    H_pert_eff = unperturbed_eigenstates'*H_pert*unperturbed_eigenstates
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
            H_sub = V'*H*V
            degen[copy(indices)] = [degeneracy_count(real.(eigvals(H_sub))), size(H_sub)[1]]
            indices[end] += 1
        catch BoundsError
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
    conf::Tuple{Set{T}, Set{T}}, 
    sorted_sites::Vector{T}, 
    ops::Vector{Tuple{T,Int,Symbol}}
) where T
    # Precompute occupation dictionary: (site, spin) => occupied (Bool)
    occ = Dict{Tuple{T, Int}, Bool}()
    for s in sorted_sites
        occ[(s, 1)] = s in conf[1]  # spin up
        occ[(s, 2)] = s in conf[2] # spin down
    end

    # Flatten total sorted basis: (site, spin) pairs in order
    full_basis = [(s, σ) for s in sorted_sites for σ in (1, 2)]

    # Build occupation vector: vector of 0s and 1s in JW order
    occ_vec = [occ[(s, σ)] for (s, σ) in full_basis]

    # For each operator in `ops`, determine how many fermions it "passes"
    sign = -1
    for n in 1:length(ops)
        site_n, spin_n, _ = ops[n]
        idx_n = findfirst(==((site_n, spin_n)), full_basis)
        
        # Count number of occupied fermions before idx_n not in the operator itself
        for m in 1:idx_n-1
            # site_m, spin_m = full_basis[m]
            # Exclude if it's the same operator acting (i.e., operator site)
            if occ_vec[m] == 1
                sign *= -1
            end
        end

        # Toggle occupancy as if the op has acted (important for consistency when ops repeat)
        # But only update if not the last operator
        if n < length(ops)
            occ_vec[idx_n] ⊻= 1
        end
    end

    return sign
end
function count_in_range(s::Set{T}, a::T, b::T; lower_eq::Bool=true, upper_eq::Bool=true) where T
    ### given a set of numbers s, counts the number of elements that are between
    ### a and b (where a could be larger or less than b). lower_eq and upper_eq specify
    ### whether the upper/lower bound is an equality or an inequality
    count = 0
    upper_bound = max(a,b) 
    lower_bound = min(a,b)
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
function create_cicj(Hs::HubbardSubspace)
    # c^dagger_i c_j 

    dim = get_subspace_dimension(Hs)
    indexer = CombinationIndexer(collect(1:prod(size((Hs.lattice)))), get_subspace_info(Hs)...)
    cicj = [spzeros(Float64, dim, dim) for _ in 1:length(indexer.a), _ in 1:length(indexer.a), _ in 1:2]
    for (i1, conf1) in enumerate(indexer.inv_comb_dict)
        for σ ∈ [1, 2]
            for site_index1 ∈ conf1[σ]
                for site_index2 ∈ [setdiff(indexer.a, conf1[σ]); site_index1]
                    new_conf = copy(conf1[σ])
                    delete!(new_conf, site_index1)
                    push!(new_conf, site_index2)
                    if σ == 1
                        i2 = index(indexer, new_conf, conf1[2])
                    else
                        i2 = index(indexer, conf1[1], new_conf)
                    end
                    # sign from jordan-wigner string. assuming i<j, c+_i c_j gives a positive sign times (-1)^(# electrons between sites i and j)
                    # if j < i, then there's an extra negative sign
                    sign = (-1)^(count_in_range(conf1[1], site_index1, site_index2; lower_eq=true, upper_eq=false) + 
                        count_in_range(if (σ == 2) new_conf else conf1[2] end, site_index1, site_index2; lower_eq=false, upper_eq=true) +
                        (site_index1 > site_index2))
                    cicj[site_index1, site_index2, σ][i1, i2] += sign
                end
            end
        end
    end
    return cicj
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
                push!(vals, magnitude/2)
            end
        end
    end
end
function create_SziSzj!(rows::Vector{Int}, cols::Vector{Int}, vals::Vector{Float64}, magnitude::Float64, indexer::CombinationIndexer; iequalsj::Bool=false, NN::Union{Missing, AbstractLattice}=missing)
    if iequalsj
        create_chemical_potential!(rows, cols, vals, 1/4*magnitude, indexer)
        create_hubbard_interaction!(rows, cols, vals, -1/2*magnitude, false, indexer)
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
        push!(vals, total/4*magnitude)
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
                        i2 = index(indexer, replace(conf[1], site_index1=>site_index2), replace(conf[2], site_index2=>site_index1))
                    else
                        i2 = index(indexer, replace(conf[1], site_index2=>site_index1), replace(conf[2], site_index1=>site_index2))
                    end
                    push!(rows, i1)
                    push!(cols, i2)
                    push!(vals, magnitude/2)
                end
            end
        end
    end
    create_SziSzj!(rows, cols, vals, magnitude, indexer; iequalsj=false, NN=NN)
end
function create_S2!(rows::Vector{Int}, cols::Vector{Int}, vals::Vector{Float64}, magnitude::Float64, indexer::CombinationIndexer)
    # sum Si*Si
    create_chemical_potential!(rows, cols, vals, 3/4*magnitude, indexer)
    create_hubbard_interaction!(rows, cols, vals, -3/2*magnitude, false, indexer)
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
        for (σ1,σ2) ∈ Iterators.product(1:2,1:2) # 1=up 2=down
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
                        new_conf[σ1] = replace(conf[σ1], site_index1=>site_index2)
                        new_conf[3-σ1] = conf[3-σ1]
                    else
                        new_conf[σ1] = setdiff(conf[σ1], [site_index1])
                        new_conf[σ2] = union(conf[σ2], [site_index2])
                    end
                    i2 = index(indexer, new_conf[1], new_conf[2])

                    sign = compute_jw_sign(conf, sorted_sites, [(site_index2,σ2,:create), (site_index1,σ1,:annihilate)])
                    push!(rows, i1)
                    push!(cols, i2)
                    push!(vals, t[Set([(site_index1, σ1), (site_index2, σ2)])]*sign)
                end
            end
        end
    end
end
function build_n_body_structure(
    t::Dict{Vector{Tuple{T,Int,Symbol}}, U},
    indexer::CombinationIndexer
) where {T, U<:Number}
    sorted_sites = sort(indexer.a)
    rows = Int[]
    cols = Int[]
    signs = U[]
    ops_list = Vector{Vector{Tuple{T,Int,Symbol}}}()

    for (i1, conf) in enumerate(indexer.inv_comb_dict)
        for ops in keys(t)
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
            if i1 > i2 #only considering upper diagonal so ensure hermiticity
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
function update_values(
    signs::Vector{U},
    ops_list::Vector{Vector{Tuple{T,Int,Symbol}}}, 
    t::Dict{Vector{Tuple{T,Int,Symbol}}, U}, 
) where {T, U<:Number}
    return [t[ops_list[i]]*signs[i] for i in eachindex(signs)]
end
function general_n_body!(
    rows::Vector{Int}, 
    cols::Vector{Int}, 
    vals::Vector{U}, 
    t::Dict{Vector{Tuple{T,Int,Symbol}}, U}, 
    indexer::CombinationIndexer
) where {T, U<:Number}
    # requires applying Hermitian to the resulting sparse matrix
    _rows, _cols, signs, ops_list = build_n_body_structure(t, indexer)
    println(U)
    _vals = update_values(signs, ops_list, t)
    append!(rows, _rows)
    append!(cols, _cols)
    append!(vals, _vals)
end
function create_randomized_nth_order_operator(n::Int, indexer::CombinationIndexer; magnitude::Float64=1e-3)
    t_dict = Dict{Vector{Tuple{Coordinate{2,Int64},Int,Symbol}}, ComplexF64}()
    site_list = sort(indexer.a) #ensuring normal ordering
    all_ops(label) = combinations([(s, σ,label) for s in site_list for σ in 1:2],n)
    for (ops_create, ops_annihilate) in Iterators.product(all_ops(:create), all_ops(:annihilate))
        key = [ops_create; ops_annihilate]
        if key ∉ keys(t_dict)
            t_dict[key] = rand()*magnitude
        else
            t_dict[key] += rand()*magnitude
        end
    end
    return t_dict
end

function create_nn_hopping!(rows::Vector{Int}, cols::Vector{Int}, vals::Vector{Float64}, t::Union{Float64, AbstractArray{Float64}}, lattice::AbstractLattice, indexer::CombinationIndexer)
    if isa(t, Number) 
        t = [t]
    end
    for (i1, conf) in enumerate(indexer.inv_comb_dict)
        for σ ∈ [1, 2]
            for site_index1 ∈ conf[σ]
                for order in eachindex(t)
                    for site_index2 ∈ neighbors(lattice, site_index1,order)
                        if site_index2 ∉ conf[σ]
                            new_conf = replace(conf[σ], site_index1=>site_index2)
                            if σ == 1
                                i2 = index(indexer, new_conf, conf[2])
                            else
                                i2 = index(indexer, conf[1], new_conf)
                            end
                            sign = (-1)^(count_in_range(conf[1], site_index1, site_index2; lower_eq=true, upper_eq=false) + 
                                        count_in_range(if (σ == 2) new_conf else conf[2] end, site_index1, site_index2; lower_eq=false, upper_eq=true) +
                                        (site_index1 > site_index2))
                            push!(rows, i1)
                            push!(cols, i2)
                            push!(vals, -0.5*t[order]*sign)# 0.5 due to double counting from neighbors for some reason
                        end
                    end
                end
            end
        end
    end
end

function create_hubbard_interaction!(rows::Vector{Int}, cols::Vector{Int}, vals::Vector{Float64}, U::Float64, half_filling::Bool, indexer::CombinationIndexer)
    if half_filling
        for (i, conf) in enumerate(indexer.inv_comb_dict)
            num_negative = length(setdiff(union(conf[1], conf[2]), intersect(conf[1], conf[2])))
            num_positive = length(indexer.a) - num_negative
            push!(rows, i)
            push!(cols, i)
            push!(vals, U*(num_positive - num_negative)/4)
        end
    else
        for (i, conf) in enumerate(indexer.inv_comb_dict)
            push!(rows, i)
            push!(cols, i)
            push!(vals, U*length(intersect(conf[1], conf[2])))
        end
    end
end
function create_chemical_potential!(rows::Vector{Int}, cols::Vector{Int}, vals::Vector{Float64}, μ::Float64, indexer::CombinationIndexer)
    for (i, conf) in enumerate(indexer.inv_comb_dict)
        push!(rows, i)
        push!(cols, i)
        # last part breaks degeneracy ensuring that ED orders them consistently
        push!(vals, μ*(length(conf[1]) + length(conf[2])) ) #+ 1e-7*sum(conf[1]) + 43e-7*sum(conf[2]) 
    end
end
function create_Sz!(rows::Vector{Int}, cols::Vector{Int}, vals::Vector{Float64}, magnitude::Float64, indexer::CombinationIndexer)
    for (i, conf) in enumerate(indexer.inv_comb_dict)
        push!(rows, i)
        push!(cols, i)
        push!(vals, magnitude*(length(conf[1]) - length(conf[2])) )
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
        push!(cols, index(indexer, new_conf1, new_conf2))
        push!(vals, magnitude*sign)
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
        push!(vals, magnitude*((Nup+Ndown-n)^2/4-(Nup+Ndown-n)))

        # L+L-  +  L-L+
        for site_index1 ∈ intersect(conf[1], conf[2])
            for site_index2 ∈ setdiff(indexer.a, union(conf[1], conf[2]))
                i2 = index(indexer, replace(conf[1], site_index1=>site_index2), replace(conf[2], site_index1=>site_index2))
                push!(rows, i1)
                push!(cols, i2)
                push!(vals, magnitude*(-1)^(sum(site_index1.coordinates) + sum(site_index2.coordinates)))
            end
        end
    end
end

function create_operator(Hs::HubbardSubspace, op; kind=1)
    dim = get_subspace_dimension(Hs)
    indexer = CombinationIndexer(reduce(vcat,collect(sites(Hs.lattice))), get_subspace_info(Hs)...)
    rows = Int[]
    cols = Int[]
    vals = Float64[]

    #insert stuff here
    if op == :σ
        mapping = reflection_mapping(Hs.lattice, kind)
        create_transform!(rows, cols, vals, 1.0, mapping, indexer)
    elseif op == :Sx
        create_Sx!(rows, cols, vals, 1.0, indexer)
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
function create_Hubbard(Hm::HubbardModel, Hs::HubbardSubspace; perturbations::Bool=false)
    # specify the subspace
    dim = get_subspace_dimension(Hs)
    indexer = CombinationIndexer(reduce(vcat,collect(sites(Hs.lattice))), get_subspace_info(Hs)...)

    rows = Int[]
    cols = Int[]
    vals = Float64[]

    #Constructs the sparse hopping Hamiltonian matrix \sum_{<i,j>} c^\dagger_i c_j.
    create_nn_hopping!(rows, cols, vals, Hm.t, Hs.lattice, indexer)
    create_hubbard_interaction!(rows, cols, vals, Hm.U, Hm.half_filling, indexer)
    create_chemical_potential!(rows, cols, vals, Hm.μ, indexer)
    if perturbations
        # create_Sx!(rows, cols, vals, sqrt(2)*1e-5, indexer)
        # create_S2!(rows, cols, vals, sqrt(3)*1e-5, indexer)
        create_L2!(rows, cols, vals, sqrt(5)*1e-5, indexer)
        # for dim ∈ [1,2]
        #     mapping = reflection_mapping(Hs.lattice, dim)
        #     create_transform!(rows, cols, vals, sqrt(1+sqrt(dim+1))*1e-5, mapping, indexer)
        # end
    end
    # create_SziSzj!(rows, cols, vals, 0.021, indexer; iequalsj=true)
    # create_operator!(rows, cols, vals, 1e-2, indexer)

    # constuct Hamiltonian
    H = sparse(rows, cols, vals, dim, dim)
    
    return H
end


function create_Heisenberg(t,J, Hs::HubbardSubspace)
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

function compute_conf_differences(s1::Tuple{Set, Set}, s2::Tuple{Set, Set})
    """
    The weight is the number of differences between two sets. Also,
    this is twice the number of swaps required to turn one set into the other
    """
    creation = Tuple([setdiff(s2[i], s1[i]) for i=1:2])
    annihilation = Tuple([setdiff(s1[i], s2[i]) for i=1:2])
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
                    push!(difference_dict[N][(creation, annihilation)], (i,j))
                else
                    difference_dict[N][(creation, annihilation)] = [(i,j)]
                end
            else
                difference_dict[N] = Dict((creation, annihilation)=>[(i,j)])
            end
        end
    end
    return difference_dict
end
function find_N_body_interactions(U::AbstractArray, indexer::CombinationIndexer)
    H = -1im*log(U)
    difference_dict = collect_all_conf_differences(indexer)

    n_electrons = sum([length(indexer.inv_comb_dict[1][j]) for j=1:2])
    second_quantized_order_labels = Dict()
    n_electrons = sum([length(indexer.inv_comb_dict[1][j]) for j=1:2])
    # defining range of indices in the second quantized representation
    d = length(indexer.a)*2
    m = 1
    for order ∈ 1:n_electrons
        if order == 1
            second_quantized_order_labels[order] = 1:d^(2*order)
            m = (length(indexer.a)*2)^(2*order)
        else
            second_quantized_order_labels[order] = (m+1):(m+d^(2*order))
            m += d^(2*order)
        end
    end
    second_quantized_dimension = sum(length(s) for s in values(second_quantized_order_labels))
    second_quantized_solution = spzeros(ComplexF64, second_quantized_dimension)
    second_quantized_nullspace = []
    site_indexer = merge(Dict((s,:up)=>k for (k,s) in enumerate(indexer.a)), 
                        Dict((s,:down)=>k+length(indexer.a) for (k,s) in enumerate(indexer.a)))
    # inv_site_indexer = [[(s,:up) for s in indexer.a]; [(s,:down) for s in indexer.a]]

    # difference_dict = collect_all_conf_differences(indexer)
    for (swaps, N_diff_dict) in difference_dict
        # println(swaps)
        for (site_diff, index_pairs) in N_diff_dict
            creation, annihilation = site_diff

            params = binomial(2*length(indexer.a) - 2*swaps, n_electrons-swaps)
            variables = cumsum([binomial(2*length(indexer.a) - 2*swaps, k) for k in 0:n_electrons-swaps])
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


            matrix = zeros(ComplexF64, (params,variables[min_order]))
            vector = zeros(ComplexF64, params)
            # break
            row_index = 1
            for (i,j) in index_pairs
                common_sites = [intersect(indexer.inv_comb_dict[i][k], indexer.inv_comb_dict[j][k]) for k=1:2]
                for (col_index, s) in enumerate(variable_mapping)
                    # col_index = inverse_variable_mapping[Tuple(common_sites)]
                    if issubset(s[1], common_sites[1]) && issubset(s[2], common_sites[2])
                        matrix[row_index, col_index] = 1
                    end
                end
                vector[row_index] = H[i,j]
                row_index += 1
            end

            nullspace_solution  = nullspace(matrix)
            particular_solution = matrix \ vector
            if length(nullspace_solution) >0 
                push!(second_quantized_nullspace,spzeros(ComplexF64,second_quantized_dimension))
            end

            # put solution into a sparse matrix form in second quantized
            #figure out the indices for the sites, map it to an index and assign it to the sparse vector
            creation_index_list = []
            annihilation_index_list = []
            for (σ_i, σ) ∈ enumerate([:up, :down])
                for create_site in creation[σ_i]
                    push!(creation_index_list,site_indexer[(create_site, σ)])
                end
                for annihilate_site in annihilation[σ_i]
                    push!(annihilation_index_list,site_indexer[(annihilate_site, σ)])
                end
            end
            indices = [sort(creation_index_list) sort(annihilation_index_list)]'[:]

            for (k,s) in enumerate(variable_mapping)
                _indices = copy(indices)
                for (σ_i, σ) in enumerate([:up, :down])
                    for site in s[σ_i]
                        push!(_indices,site_indexer[(site, σ)])
                        push!(_indices,site_indexer[(site, σ)])
                    end
                end
                order = swaps + length(s[1]) + length(s[2])
                # println("order: $order swaps: $swaps indices: $_indices k: $k")
                starting_index = minimum(second_quantized_order_labels[order])
                i = sum((_indices[n] - 1)*d^(n-1) for n in eachindex(_indices))
                second_quantized_solution[starting_index + i] = particular_solution[k]
                if length(nullspace_solution) > 0 
                    second_quantized_nullspace[end][starting_index + i] = nullspace_solution[k]
                end
            end
        end

    end
    return second_quantized_solution, second_quantized_nullspace, second_quantized_order_labels

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
    return create_consistent_basis(H, degen)
end
function create_consistent_basis(H::Vector, degen::Dict)
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
            _, V1, _ = schur(basis_transform'*h*basis_transform)
            if isnothing(prev_perm)
                prev_perm = 1:size(V1, 2)
                prev_phases = ones(size(V1,2))
            end
            if h_i == 1
                perm = 1:size(V1, 2)
                phases = I
            else
                _, V0, _ = schur(basis_transform'*H[h_i-1]*basis_transform)
                perm, mat = greedy_col_permutation_for_diag((V0[:,prev_perm]*prev_phases)'*V1)
                phases = diagm(abs.(diag(mat)) ./diag(mat))
                # println(diag(phases*mat))
            end
            

            if length(degen_rm_U) < h_i
                push!(degen_rm_U,basis_transform*V1[:,perm]*phases)
            else
                degen_rm_U[h_i] = hcat(degen_rm_U[h_i],basis_transform*V1[:,perm]*phases)
            end
            prev_perm = perm
            prev_phases = phases
        end
    end


    return degen_rm_U
end