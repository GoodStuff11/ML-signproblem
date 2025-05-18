
function count_in_range(s::Set{T}, a::T, b::T; lower_eq::Bool=true, upper_eq::Bool=true) where T
    ### given a set of numbers s, counts the number of elements that are between
    ### a and b (where a could be larger or less than b). lower_eq and upper_eq specify
    ### whether the upper/lower bound is an equality or an inequality
    count = 0
    upper_bound = maximum([a,b]) 
    lower_bound = minimum([a,b])
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
    indexer = CombinationIndexer(collect(1:nv(Hs.lattice)), get_subspace_info(Hs)...)
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
                if σ == 1
                    i2 = index(indexer, setdiff(conf[1], site_index), union(conf[2], site_index))
                else
                    i2 = index(indexer, union(conf[1], site_index), setdiff(conf[2], site_index))
                end
                push!(rows, i1)
                push!(cols, i2)
                push!(vals, magnitude)
            end
        end
    end
end
function create_SziSzj!(rows::Vector{Int}, cols::Vector{Int}, vals::Vector{Float64}, magnitude::Float64, indexer::CombinationIndexer; iequalsj::Bool=false, NN::Union{Missing, AbstractGraph}=missing)
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
function create_SiSj!(rows::Vector{Int}, cols::Vector{Int}, vals::Vector{Float64}, magnitude::Float64, indexer::CombinationIndexer; NN::Union{Missing,AbstractGraph}=missing)
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
function create_nn_hopping!(rows::Vector{Int}, cols::Vector{Int}, vals::Vector{Float64}, t::Float64, lattice::Lattice, indexer::CombinationIndexer)
    for (i1, conf1) in enumerate(indexer.inv_comb_dict)
        # if length(intersect(conf1[1], conf1[2])) > 0 
        #     continue
        # end
        for σ ∈ [1, 2]
            for site_index1 ∈ conf1[σ]
                for site_index2 ∈ neighbors(lattice, site_index1)
                    if site_index2 ∉ conf1[σ]
                        new_conf = replace(conf1[σ], site_index1=>site_index2)
                        # if length(intersect(new_conf, conf1[3-σ])) > 0 
                        #     continue
                        # end
                        if σ == 1
                            i2 = index(indexer, new_conf, conf1[2])
                        else
                            i2 = index(indexer, conf1[1], new_conf)
                        end
                        # evaluating this sign is likely the source of any error
                        # sign from jordan-wigner string. assuming i<j, c+_i c_j gives a positive sign times (-1)^(# electrons between sites i and j)
                        # if j < i, then there's an extra negative sign
                        # sign = (-1)^count_in_range(new_conf, site_index1, site_index2)
                        sign = (-1)^(count_in_range(conf1[1], site_index1, site_index2; lower_eq=true, upper_eq=false) + 
                                    count_in_range(if (σ == 2) new_conf else conf1[2] end, site_index1, site_index2; lower_eq=false, upper_eq=true) +
                                    (site_index1 > site_index2))
                        # println(sign)
                        push!(rows, i1)
                        push!(cols, i2)
                        push!(vals, -t*sign)
                    end
                end
            end
        end
    end
end

function create_hubbard_interaction!(rows::Vector{Int}, cols::Vector{Int}, vals::Vector{Float64}, U::Float64, half_filling::Bool, indexer::CombinationIndexer)
    if half_filling
        for (i, conf) in enumerate(indexer.inv_comb_dict)
            total_U = 0
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
        new_conf1 = replace(conf[1], mapping...)
        new_conf2 = replace(conf[2], mapping...)
        push!(cols, index(indexer, new_conf1, new_conf2))
        println()
        push!(vals, magnitude)
    end
end

function create_operator(Hs::HubbardSubspace)
    dim = get_subspace_dimension(Hs)
    indexer = CombinationIndexer(collect(1:nv(Hs.lattice)), get_subspace_info(Hs)...)

    rows = Int[]
    cols = Int[]
    vals = Float64[]

    #insert stuff here
    mapping = reflection_mapping(Hs.lattice, 1)
    create_transform!(rows, cols, vals, 1.0, mapping, indexer)

    H = sparse(rows, cols, vals, dim, dim)
    
    return H
end
function create_Hubbard(Hm::HubbardModel, Hs::HubbardSubspace; perturbations::Bool=false)
    # specify the subspace
    dim = get_subspace_dimension(Hs)
    indexer = CombinationIndexer(collect(1:nv(Hs.lattice)), get_subspace_info(Hs)...)

    rows = Int[]
    cols = Int[]
    vals = Float64[]

    #Constructs the sparse hopping Hamiltonian matrix \sum_{<i,j>} c^\dagger_i c_j.
    create_nn_hopping!(rows, cols, vals, Hm.t, Hs.lattice, indexer)
    create_hubbard_interaction!(rows, cols, vals, Hm.U, Hm.half_filling, indexer)
    create_chemical_potential!(rows, cols, vals, Hm.μ, indexer)
    if perturbations
        create_Sz!(rows, cols, vals, sqrt(2)*1e-5, indexer)
        create_S2!(rows, cols, vals, sqrt(3)*1e-5, indexer)
        mapping = reflection_mapping(Hs.lattice, 1)
        create_transform!(rows, cols, vals, sqrt(5)*1e-5, mapping, indexer)
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
    indexer = CombinationIndexer(collect(1:nv(Hs.lattice)), get_subspace_info(Hs)...)

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


