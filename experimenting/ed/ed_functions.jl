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

function find_best_energy_sector(all_E::Vector, U_values::Vector; labels=nothing)
    if length(all_E) == 1
        if labels === nothing
            return 1
        else
            return labels[1]
        end
    end
    # todo verify which states I'm looking at 887
    println("Finding best energy sector")
    counts = Dict()
    for u_idx in eachindex(all_E[1])
        energies = [real.(all_E[k])[u_idx] for k in eachindex(all_E)]
        indices = sortperm(energies)
        print_indices = min(2, length(energies))
        println("U=$(U_values[u_idx]) k=$(indices[1:print_indices]) $(energies[1:print_indices])")
        counts[indices[1]] = get(counts, indices[1], 0) + 1
    end

    # pick the k value which is most frequently the ground state
    k_min = argmax(counts)
    println("Selected ground state symmetry sector: $k_min")
    # error("")
    if labels === nothing
        return k_min
    else
        return labels[k_min]
    end
end

"""
Expects open hdf5 file and a number from 0 to the number of momentum sectors - 1. 

Returns the index corresponding to a ground state of the tight-binding model for the given 
momentum sector.
"""
function get_slater_ground_state(data, sector::Int)
    println("Computing slater ground state for sector $sector")

    # expects data to be an h5 open file object
    # get momentum sector, a vector of two elements [kx, ky] (starting at zero)
    mom_sector = read(data, "metadata/kvecs")[:, sector+1]
    L = read(data, "metadata/Lvec")
    Ne = read(data, "metadata/nup"), read(data, "metadata/ndown")

    separate_spins_stored = (read(data, "metadata/slater_labels/$sector") isa Dict)
    if !separate_spins_stored
        # slater labels contain indices which can be obtained from coordinates via ind2sub
        slater_labels = read(data, "metadata/slater_labels/$sector") # dim (Ne_up,H_dim,2)
        H_dim = size(slater_labels, 2)
    else
        slater_labels_up = read(data, "metadata/slater_labels/$sector/up") # dim (Ne_up,H_dim)
        slater_labels_down = read(data, "metadata/slater_labels/$sector/dn") # dim (Ne_down,H_dim)
        H_dim = size(slater_labels_up, 2)
    end

    # find occupation of the tight-binding model ground state in momentum coordinates
    L = read(data, "metadata/Lvec")
    single_spin_energies = zeros(Float64, L[1]*L[2])
    momenta = [[i,j] for j in 0:L[2]-1 for i in 0:L[1]-1 ]
    for (k_idx, k) in enumerate(momenta)
        i = k[1]
        j = k[2]
        single_spin_energies[k_idx] = -2*(cos(2*pi*i/L[1]) + cos(2*pi*j/L[2]))
    end

    best_idx = -1
    min_E = Inf
    
    # computes the energy of each slater determinant state, keeping track of the first occurrings lowest energy one
    for idx in 1:H_dim
        if !separate_spins_stored
            # println(slater_labels[:, idx, 1])
            # println(slater_labels[:, idx, 2])
            if slater_labels[1] isa UInt
                # convert slater_labels[spin, idx] to binary
                E_up = sum(Float64.(digits(slater_labels[1, idx], base=2, pad=prod(L))).*single_spin_energies)
                E_dn = sum(Float64.(digits(slater_labels[2, idx], base=2, pad=prod(L))).*single_spin_energies)
                # println(digits(slater_labels[2, idx], base=2, pad=prod(L)))
                # println(E_dn)
                # error("")
            else
                E_up = sum(single_spin_energies[k+1] for k in slater_labels[:, idx, 1])
                E_dn = sum(single_spin_energies[k+1] for k in slater_labels[:, idx, 2])
            end
        else
            E_up = sum(single_spin_energies[k+1] for k in slater_labels_up[:, idx])
            E_dn = sum(single_spin_energies[k+1] for k in slater_labels_down[:, idx])
        end
        E = E_up + E_dn
        if E < min_E
            min_E = E
            best_idx = idx
        end
    end

    return best_idx
end

function load_h5_ED_data(folder)
    valid_files = [f for f in readdir(folder) if occursin("HubbardED", f) && occursin("(0)", f)] # remove (0) requirement

    file_path = joinpath(folder, valid_files[1])
    sign_convention = :spin_first
    println("Loading hdf5 data: $file_path")

    h5open(file_path, "r") do data
        N = (read(data, "metadata/nup"),read(data, "metadata/ndown"))
        spin_conserved = true
        use_symmetry = false

        Lvec = read(data, "metadata/Lvec")
        U_values = read(data, "data/uvec")
        kvecs = read(data, "metadata/kvecs")

        key_labels = [parse(Int, k) for k in keys(data["data/energies"])]
        all_E = [real.(read(data, "data/energies/$(k)"))[:, 1] for k in key_labels] # Needed for energy selection
        k_min = find_best_energy_sector(all_E, U_values; labels=key_labels)
        println(all_E)
        target_vecs = read(data, "data/evecs/$(k_min)")[:, 1, :] # shape (length(U_values), dim)

        # using the highest overlap slater determinant state as reference. 
        slater_index = get_slater_ground_state(data, k_min)
        reference_state = zeros(ComplexF64, size(target_vecs,1))
        reference_state[slater_index] = 1.0
        target_vecs = transpose(hcat(reference_state, target_vecs))

        lattice = Square(tuple(Lvec...), Periodic())
        subspace = HubbardSubspace(N..., lattice; k=tuple((kvecs[:,k_min+1] .+ 1)...))
        println("Computing indexer")
        indexer = CombinationIndexer(subspace; order=ColSnake())

        precomputed_structures = Dict()

        return U_values, target_vecs, indexer, precomputed_structures, N, spin_conserved, use_symmetry, sign_convention
    end

end

function load_jld2_ED_data(file_path::String)
    dic = load_saved_dict(file_path)

    sign_convention = :coordinate_first

    meta_data = dic["meta_data"]
    U_values = meta_data["U_values"]
    all_full_eig_vecs = dic["all_full_eig_vecs"]
    all_E = dic["E"] # Needed for energy selection

    indexer = dic["indexer"]
    precomputed_structures = get(dic, "precomputed_structures", Dict())

    println("Meta data:")
    display(meta_data)

    # Extract N for saving
    N = meta_data["electron count"]
    spin_conserved = !isa(meta_data["electron count"], Number) # True if tuple (N_up, N_down)
    use_symmetry = false

    # Find lowest energy sector 
    k_min = find_best_energy_sector(all_E, U_values)

    # Select the eigenvectors for this sector
    # all_full_eig_vecs is a list of sectors. each sector is a list of vectors (per U).
    target_vecs = all_full_eig_vecs[k_min]
    if indexer isa Vector
        indexer = indexer[k_min]
    end

    return U_values, target_vecs, indexer, precomputed_structures, N, spin_conserved, use_symmetry, sign_convention
end

"""
Load ED data from folder

Args:
    folder: Path to folder containing ED data

Returns:
    Tuple of (U_values, target_vecs, indexer, precomputed_structures, N, spin_conserved, use_symmetry)

"""
function load_ED_data(folder)
    alternate_sign_convention = false
    jld2_path = joinpath(folder, "meta_data_and_E.jld2")
    if !isfile(jld2_path)
        args = load_h5_ED_data(folder)
    else
        args = load_jld2_ED_data(jld2_path)
    end
    return args
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


function compute_jw_sign(
    conf::Tuple{Set{T},Set{T}},
    sorted_sites::Vector{T},
    ops::Vector{Tuple{T,Int,Symbol}};
    sign_convention::Symbol=:spin_first
) where T
    # computes the sign for the term given by ops (in second quantized), associated with the 
    # configuration conf.
    if sign_convention == :spin_first
        jw_order = [(s, σ) for σ in (1, 2) for s in sorted_sites] # tamras sign convention
    else
        jw_order = [(s, σ)  for s in sorted_sites for σ in (1, 2)] # my sign convention
    end
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

function create_Sx!(rows::Vector{Int}, cols::Vector{Int}, vals::Vector{Float64}, magnitude::Float64, indexer::CombinationIndexer; momentum_basis::Bool=false, sign_convention::Symbol=:spin_first)
    # create_Sx! in momentum basis is exactly the same as position space,
    # since S^x = \frac{1}{2} \sum_k (c^\dagger_{k\uparrow} c_{k\downarrow} + c^\dagger_{k\downarrow} c_{k\uparrow})
    sorted_sites = sort(indexer.a, order=sign_convention==:spin_first ? ColSnake() : RowSnake())
    for (i1, conf) in enumerate(indexer.inv_comb_dict)
        for σ ∈ [1, 2]
            for site_index ∈ setdiff(conf[σ], conf[3-σ])
                # Flip σ to 3-σ
                # Annihilate σ, Create 3-σ
                ops = [(site_index, σ, :annihilate), (site_index, 3 - σ, :create)]
                sign = compute_jw_sign(conf, sorted_sites, ops; sign_convention=sign_convention)

                if σ == 1
                    i2 = index(indexer, setdiff(conf[1], [site_index]), union(conf[2], [site_index]))
                else
                    i2 = index(indexer, union(conf[1], [site_index]), setdiff(conf[2], [site_index]))
                end

                push!(rows, i1)
                push!(cols, i2)
                push!(vals, magnitude / 2 * sign)
            end
        end
    end
end
function create_SziSzj!(rows::Vector{Int}, cols::Vector{Int}, vals::Vector{Float64}, magnitude::Float64, indexer::CombinationIndexer; iequalsj::Bool=false, NN::Union{Missing,AbstractLattice}=missing, momentum_basis::Bool=false)
    if momentum_basis
        @warn "dense S_i^z S_j^z in momentum basis is not fully implemented for all J_ij patterns"
        return
    end

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
function create_SiSj!(rows::Vector{Int}, cols::Vector{Int}, vals::Vector{Float64}, magnitude::Float64, indexer::CombinationIndexer; NN::Union{Missing,AbstractLattice}=missing, momentum_basis::Bool=false, sign_convention::Symbol=:spin_first)
    if momentum_basis
        @warn "SiSj in momentum basis not fully implemented yet"
        return
    end
    # This is for i!=j
    # We want 0.5 * (S_i^+ S_j^- + S_i^- S_j^+)
    # This swaps spins: i_up j_down -> i_down j_up   OR   i_down j_up -> i_up j_down
    sorted_sites = sort(indexer.a, order=sign_convention==:spin_first ? ColSnake() : RowSnake())
    for (i1, conf) in enumerate(indexer.inv_comb_dict)
        for σ ∈ [1, 2] # σ is the spin starting at site_index1
            for site_index1 ∈ setdiff(conf[σ], conf[3-σ])
                for site_index2 ∈ setdiff(conf[3-σ], conf[σ])
                    if !ismissing(NN) && !(site_index2 in neighbors(NN, site_index1))
                        continue
                    end

                    ops = [
                        (site_index2, σ, :create),
                        (site_index2, 3 - σ, :annihilate),
                        (site_index1, 3 - σ, :create),
                        (site_index1, σ, :annihilate)
                    ]

                    sign = compute_jw_sign(conf, sorted_sites, ops; sign_convention=sign_convention)

                    if σ == 1
                        i2 = index(indexer, replace(conf[1], site_index1 => site_index2), replace(conf[2], site_index2 => site_index1))
                    else
                        i2 = index(indexer, replace(conf[1], site_index2 => site_index1), replace(conf[2], site_index1 => site_index2))
                    end
                    push!(rows, i1)
                    push!(cols, i2)
                    push!(vals, (magnitude / 2) * sign)
                end
            end
        end
    end
    create_SziSzj!(rows, cols, vals, magnitude, indexer; iequalsj=false, NN=NN, momentum_basis=momentum_basis)
end
function create_S2!(rows::Vector{Int}, cols::Vector{Int}, vals::Vector{Float64}, magnitude::Float64, indexer::CombinationIndexer; momentum_basis::Bool=false, sign_convention::Symbol=:spin_first)
    # The algebraic form of total S^2 is invariant under any unitary single-particle basis transformation 
    # (such as Fourier transform to momentum space) because it is a global SU(2) Casimir invariant. 
    # Thus, the exact same configuration loop works perfectly for both position and momentum bases!
    sorted_sites = sort(indexer.a, order=sign_convention==:spin_first ? ColSnake() : RowSnake())
    for (i1, conf) in enumerate(indexer.inv_comb_dict)
            sz = (length(conf[1]) - length(conf[2])) / 2.0
            diagonal_val = sz * (sz + 1.0)
            
            # Add \sum_k n_{k \downarrow} (1 - n_{k \uparrow})
            for k_down in conf[2]
                if k_down ∉ conf[1]
                    diagonal_val += 1.0
                end
            end
            
            # Now we add the S^- S^+ terms
            for k_up_annihilate ∈ conf[1]
                for k_down_annihilate ∈ conf[2]
                    # Note S^+ annihilates down and creates up: c^\dagger_{k'\uparrow} c_{k'\downarrow}
                    # So k' = k_down_annihilate. And we create up at k'.
                    # S^- annihilates up and creates down: c^\dagger_{k\downarrow} c_{k\uparrow}
                    # So k = k_up_annihilate. And we create down at k.
                    k_up_create = k_down_annihilate
                    k_down_create = k_up_annihilate
                    
                    if k_up_create != k_up_annihilate && k_up_create ∉ conf[1] && k_down_create ∉ conf[2]
                        # creating k_up_create, k_down_create and annihilating k_up_annihilate, k_down_annihilate
                        new_up = union(setdiff(conf[1], [k_up_annihilate]), [k_up_create])
                        new_down = union(setdiff(conf[2], [k_down_annihilate]), [k_down_create])
                        i2 = index(indexer, new_up, new_down)
                        
                        # operator: c^\dagger_{k\downarrow} c_{k\uparrow} c^\dagger_{k'\uparrow} c_{k'\downarrow}
                        ops = [
                            (k_down_create, 2, :create),
                            (k_up_annihilate, 1, :annihilate),
                            (k_up_create, 1, :create),
                            (k_down_annihilate, 2, :annihilate)
                        ]
                        sign = compute_jw_sign(conf, sorted_sites, ops; sign_convention=sign_convention)
                        push!(rows, i1)
                        push!(cols, i2)
                        push!(vals, magnitude * sign)
                    end
                end
            end
            
        if abs(diagonal_val) > 1e-12
            push!(rows, i1)
            push!(cols, i1)
            push!(vals, magnitude * diagonal_val)
        end
    end
end
function general_single_body!(
    rows::Vector{Int},
    cols::Vector{Int},
    vals::Vector{Float64},
    t::Dict,
    indexer::CombinationIndexer; 
    sign_convention::Symbol=:spin_firstS
)
    sorted_sites = sort(indexer.a, order=sign_convention==:spin_first ? ColSnake() : RowSnake())
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

                    sign = compute_jw_sign(conf, sorted_sites, [(site_index2, σ2, :create), (site_index1, σ1, :annihilate)]; sign_convention=sign_convention)
                    push!(rows, i1)
                    push!(cols, i2)
                    push!(vals, t[Set([(site_index1, σ1), (site_index2, σ2)])] * sign)
                end
            end
        end
    end
end


function build_n_body_structure_from_keys(
    t_keys::AbstractVector,
    indexer::CombinationIndexer{T},
    ::Type{U}=Float64;
    skip_lower_triangular::Bool=false,
    sign_convention::Symbol=:spin_first
) where {T,U<:Number}
    sorted_sites = sort(indexer.a, order=sign_convention==:spin_first ? ColSnake() : RowSnake())
    println(sorted_sites)
    inv_comb_dict = indexer.inv_comb_dict
    n_states = length(inv_comb_dict)
    num_chunks = Threads.nthreads()
    chunk_size = cld(n_states, num_chunks)
    
    results = Vector{Tuple{Vector{Int}, Vector{Int}, Vector{U}, Vector{Vector{Tuple{T,Int,Symbol}}}}}(undef, num_chunks)
    
    Threads.@threads for chunk in 1:num_chunks
        start_idx = (chunk - 1) * chunk_size + 1
        end_idx = min(chunk * chunk_size, n_states)
        
        local_rows = Int[]
        local_cols = Int[]
        local_signs = U[]
        local_ops_list = Vector{Vector{Tuple{T,Int,Symbol}}}()
        
        for i1 in start_idx:end_idx
            if chunk == 1 && i1 % 500 == 0
                println("complete: $(round((i1/(end_idx - start_idx))*100, digits=2))%")
            end
            conf = inv_comb_dict[i1]
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
                s = compute_jw_sign(conf, sorted_sites, ops; sign_convention=sign_convention)
                push!(local_rows, i1)
                push!(local_cols, i2)
                push!(local_signs, s)
                push!(local_ops_list, ops)
            end
        end
        results[chunk] = (local_rows, local_cols, local_signs, local_ops_list)
    end
    
    rows = vcat([r[1] for r in results]...)
    cols = vcat([r[2] for r in results]...)
    signs = vcat([r[3] for r in results]...)
    ops_list = vcat([r[4] for r in results]...)

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

"""
    precompute_n_body_structures(indexer, max_order=2; spin_conserved::Bool=false, momentum_basis::Bool=false, sign_convention::Symbol=:spin_first)

Precompute and cache `n_body_structure` for optimization, avoiding expensive generation at runtime.
Computes for both `use_symmetry` branches (true/false).
"""
function precompute_n_body_structures(indexer::CombinationIndexer, max_order::Int=2; spin_conserved::Bool=false, momentum_basis::Bool=false, sign_convention::Symbol=:spin_first)
    precomputed_structures = Dict()
    for order in 1:max_order
        for use_sym in [true, false]
            t_dict, t_keys = create_randomized_nth_order_operator(order, indexer, true; magnitude=1.0+0im, omit_H_conj=!use_sym, conserve_spin=spin_conserved, normalize_coefficients=false, conserve_momentum=momentum_basis, sign_convention=sign_convention)
            rows, cols, signs, ops_list = build_n_body_structure_from_keys(t_keys, indexer, typeof(t_dict[t_keys[1]]); sign_convention=sign_convention)
            param_index_map = build_param_index_map(ops_list, t_keys)
            precomputed_structures[(order, use_sym)] = Dict(
                :rows => rows, :cols => cols, :signs => signs, 
                :ops_list => ops_list, :t_keys => t_keys, :param_index_map => param_index_map
            )
        end
    end
    return precomputed_structures
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
function general_n_body!(
    rows::Vector{Int},
    cols::Vector{Int},
    vals::Vector{U},
    t::Dict{Vector{Tuple{T,Int,Symbol}},U},
    indexer::CombinationIndexer;
    sign_convention::Symbol=:spin_first
) where {T,U<:Number}
    # requires applying Hermitian to the resulting sparse matrix
    _rows, _cols, signs, ops_list = build_n_body_structure(t, indexer; skip_lower_triangular=false, sign_convention=sign_convention)
    t_keys = sort!(collect(keys(t)), order=sign_convention==:spin_first ? ColSnake() : RowSnake())
    _vals = update_values(signs, ops_list, t_keys, [t[k] for k in t_keys])
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
function correlation_matrix(order::Int, indexer::CombinationIndexer; sign_convention::Symbol=:spin_first)
    # computes the matrix of operators c†_i c_j
    t_dict, t_keys = create_randomized_nth_order_operator(order, indexer, true; sign_convention=sign_convention)
    dim = length(indexer.inv_comb_dict)
    rows, cols, signs, ops_list = build_n_body_structure_from_keys(t_keys, indexer, typeof(t_dict[t_keys[1]]); skip_lower_triangular=false, sign_convention=sign_convention)
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
function create_randomized_nth_order_operator(n::Int, indexer::CombinationIndexer, return_keys::Bool=false;
    magnitude::T=1e-3 + 0im, omit_H_conj::Bool=false, conserve_spin::Bool=false, normalize_coefficients::Bool=false, conserve_momentum::Bool=false, sign_convention::Symbol=:spin_first) where T
    # function creates a dictionary of free parameters in the form of a dictionary. 
    # when spin is conserved, the Hilbert space is smaller, so a restricted number of coefficients are possible. The rest aren't filled in
    # When hermiticity is forced, we only need to worry about upper diagonal elements. The rest can be filled in afterward

    t_dict = Dict{Vector{Tuple{Coordinate{2,Int64},Int,Symbol}},T}()
    site_list = sort(indexer.a, order=sign_convention==:spin_first ? ColSnake() : RowSnake()) #ensuring normal ordering
    all_ops(label) = combinations([(s, σ, label) for s in site_list for σ in 1:2], n)
    equal_spin(create, annihilate) = sum((σ * 2 - 3) for (s, σ, _) in create) == sum((σ * 2 - 3) for (s, σ, _) in annihilate)
    geq_ops(create, annihilate) = [(s.coordinates..., σ) for (s, σ, _) in create] <= [(s.coordinates..., σ) for (s, σ, _) in annihilate]
    
    all_pairs = collect(Iterators.product(all_ops(:create), all_ops(:annihilate)))
    
    num_chunks = Threads.nthreads()
    chunk_size = cld(length(all_pairs), num_chunks)
    
    results = Vector{Dict{Vector{Tuple{Coordinate{2,Int64},Int,Symbol}},T}}(undef, num_chunks)
    
    Threads.@threads for chunk in 1:num_chunks
        local_dict = Dict{Vector{Tuple{Coordinate{2,Int64},Int,Symbol}},T}()
        start_idx = (chunk - 1) * chunk_size + 1
        end_idx = min(chunk * chunk_size, length(all_pairs))
        
        for i in start_idx:end_idx
            (ops_create, ops_annihilate) = all_pairs[i]
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
                if key ∉ keys(local_dict)
                    local_dict[key] = (2 * rand() - 1) / 2 * magnitude
                else
                    local_dict[key] += (2 * rand() - 1) / 2 * magnitude
                end
            end
        end
        results[chunk] = local_dict
    end
    
    # Merge dictionaries
    for res in results
        for (k, v) in res
            if k ∉ keys(t_dict)
                t_dict[k] = v
            else
                t_dict[k] += v
            end
        end
    end
    if normalize_coefficients
        normalization_coefficient = length(values(t_dict))
        for key in keys(t_dict)
            t_dict[key] /= normalization_coefficient
        end
    end
    if return_keys
        sorted_keys = sort!(collect(keys(t_dict)), order=sign_convention==:spin_first ? ColSnake() : RowSnake())
        return t_dict, sorted_keys
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


function create_nn_hopping!(rows::Vector{Int}, cols::Vector{Int}, vals::Vector{Float64}, t::Union{Float64,AbstractArray{Float64}}, lattice::AbstractLattice, indexer::CombinationIndexer; momentum_basis::Bool=false, sign_convention::Symbol=:spin_first)
    if isa(t, Number)
        t = [t]
    end
    
    if momentum_basis
        # In momentum basis, the hopping H_t is diagonal!
        # \epsilon_k = -2t (\sum_d \cos(2\pi k_d / L_d))
        # Note: assuming simple cubic/square lattice where order 1 is nearest neighbor, order 2 is NNN etc.
        # This requires knowing the lattice dimensions. 
        # For simplicity, if t is a number, we assume nearest neighbor.
        dims = indexer.lattice_dims
        for (i, conf) in enumerate(indexer.inv_comb_dict)
            energy = 0.0
            for σ ∈ [1, 2]
                for k_coord ∈ conf[σ]
                    # k_coord.coordinates are 1-indexed. k_i \in \{1, ..., L\}
                    # The actual momentum is p_j = 2\pi (k_j - 1) / L_j
                    for d in 1:length(dims)
                        p_d = 2 * π * (k_coord.coordinates[d] - 1) / dims[d]
                        # For order=1 (nearest neighbor), the band structure is -2t \sum \cos(p_d)
                        if length(t) >= 1
                            energy += -2 * t[1] * cos(p_d)
                        end
                        if length(t) >= 2
                            # For order=2 (next nearest neighbor on 2D square), assuming diagonal hops
                            # -4t' \cos(p_x) \cos(p_y) etc... but this might depend on the lattice.
                            # We will just do order 1 for now if momentum_basis is used, or print a warning.
                            if d < length(dims)
                                p_d_next = 2 * π * (k_coord.coordinates[d+1] - 1) / dims[d+1]
                                energy += -4 * t[2] * cos(p_d) * cos(p_d_next)
                            end
                        end
                    end
                end
            end
            if abs(energy) > 1e-12
                push!(rows, i)
                push!(cols, i)
                push!(vals, energy)
            end
        end
        return
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
                            # sign = (-1)^(count_in_range(conf[1], site_index1, site_index2; lower_eq=true, upper_eq=false) +
                            #              count_in_range(if (σ == 2)
                            #                      new_conf
                            #                  else
                            #                      conf[2]
                            #                  end, site_index1, site_index2; lower_eq=false, upper_eq=true) +
                            #              (site_index1 > site_index2))
                            sign = compute_jw_sign(conf, sort(indexer.a, order=sign_convention==:spin_first ? ColSnake() : RowSnake()), 
                                    [(site_index2, σ, :create), (site_index1, σ, :annihilate)]; sign_convention=sign_convention)
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

function create_hubbard_interaction!(rows::Vector{Int}, cols::Vector{Int}, vals::Vector{Float64}, U::Float64, half_filling::Bool, indexer::CombinationIndexer; momentum_basis::Bool=false, sign_convention::Symbol=:spin_first)
    if momentum_basis
        # H_U = (U/N) \sum_{k,k',q} c^\dagger_{k-q,\uparrow} c_{k,\uparrow} c^\dagger_{k'+q,\downarrow} c_{k',\downarrow}
        # N is the number of sites.
        N = isnothing(indexer.lattice_dims) ? length(indexer.a) : prod(indexer.lattice_dims)
        V = U / N
        sorted_sites = sort(indexer.a, order=sign_convention==:spin_first ? ColSnake() : RowSnake())
        
        for (i1, conf) in enumerate(indexer.inv_comb_dict)
            for k_up_annihilate ∈ conf[1]
                for k_down_annihilate ∈ conf[2]
                    for k_up_create ∈ indexer.a
                        # momentum conservation: k_up_create + k_down_create = k_up_annihilation + k_down_annihilation
                        # so k_down_create = k_up_annihilate + k_down_annihilate - k_up_create (mod dims)
                        k_down_create_coords = mod.(k_up_annihilate.coordinates .+ k_down_annihilate.coordinates .- k_up_create.coordinates .- 1, indexer.lattice_dims) .+ 1
                        k_down_create = Coordinate(k_down_create_coords...)

                        # ensure k_up_create and k_down_create are valid
                        if k_up_create == k_up_annihilate && k_down_create == k_down_annihilate
                            # diagonal term
                            push!(rows, i1)
                            push!(cols, i1)
                            push!(vals, V)
                        elseif k_up_create ∉ conf[1] && k_down_create ∉ conf[2]
                            # off-diagonal
                            # creating k_up_create, k_down_create and annihilating k_up_annihilate, k_down_annihilate
                            new_up = union(setdiff(conf[1], [k_up_annihilate]), [k_up_create])
                            new_down = union(setdiff(conf[2], [k_down_annihilate]), [k_down_create])
                            i2 = index(indexer, new_up, new_down)
                            
                            # operator ordering: c^\dagger_{up} c_{up} c^\dagger_{down} c_{down}
                            ops = [
                                (k_up_create, 1, :create),
                                (k_up_annihilate, 1, :annihilate),
                                (k_down_create, 2, :create),
                                (k_down_annihilate, 2, :annihilate)
                            ]
                            sign = compute_jw_sign(conf, sorted_sites, ops; sign_convention=sign_convention)
                            push!(rows, i1)
                            push!(cols, i2)
                            push!(vals, V * sign)
                        end
                    end
                end
            end
        end
        return
    end

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
function create_∏σx!(rows::Vector{Int}, cols::Vector{Int}, vals::Vector{Float64}, magnitude::Float64, indexer::CombinationIndexer; momentum_basis::Bool=false, sign_convention::Symbol=:spin_first)
    sorted_sites = sort(indexer.a, order=sign_convention==:spin_first ? ColSnake() : RowSnake())
    for (i1, conf) in enumerate(indexer.inv_comb_dict)
        i2 = index(indexer, conf[2], conf[1])

        ops = Tuple{eltype(sorted_sites),Int,Symbol}[]
        for site in sorted_sites
            in_up = site in conf[1]
            in_down = site in conf[2]
            if in_up && in_down
                push!(ops, (site, 1, :annihilate))
                push!(ops, (site, 2, :annihilate))
                push!(ops, (site, 2, :create))
                push!(ops, (site, 1, :create))
            elseif in_up
                # u -> d
                push!(ops, (site, 1, :annihilate))
                push!(ops, (site, 2, :create))
            elseif in_down
                # d -> u
                push!(ops, (site, 2, :annihilate))
                push!(ops, (site, 1, :create))
            end
        end

        sign = compute_jw_sign(conf, sorted_sites, ops; sign_convention=sign_convention)
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
function create_Sx!(rows::Vector{Int}, cols::Vector{Int}, vals::Vector{Float64}, magnitude::Float64, indexer::CombinationIndexer; sign_convention::Symbol=:spin_first)
    sorted_sites = sort(indexer.a, order=sign_convention==:spin_first ? ColSnake() : RowSnake())
    for (i1, conf) in enumerate(indexer.inv_comb_dict)
        for σ ∈ [1, 2]
            for site_index ∈ setdiff(conf[σ], conf[3-σ])
                # Flip σ to 3-σ
                # Annihilate σ, Create 3-σ
                ops = [(site_index, σ, :annihilate), (site_index, 3 - σ, :create)]
                sign = compute_jw_sign(conf, sorted_sites, ops; sign_convention=sign_convention)

                if σ == 1
                    i2 = index(indexer, setdiff(conf[1], [site_index]), union(conf[2], [site_index]))
                else
                    i2 = index(indexer, union(conf[1], [site_index]), setdiff(conf[2], [site_index]))
                end

                push!(rows, i1)
                push!(cols, i2)
                push!(vals, magnitude / 2 * sign)
            end
        end
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

function create_operator(Hs::HubbardSubspace, op; kind=1, momentum_basis::Bool=false, sign_convention::Symbol=:spin_first)
    dim = get_subspace_dimension(Hs)
    indexer = CombinationIndexer(Hs; order=sign_convention == :spin_first ? ColSnake() : RowSnake())
    rows = Int[]
    cols = Int[]
    vals = Float64[]

    #insert stuff here
    if op == :Sx
        create_Sx!(rows, cols, vals, 1.0, indexer; momentum_basis=momentum_basis, sign_convention=sign_convention)
    elseif op == :∏σx
        create_∏σx!(rows, cols, vals, 1.0, indexer; momentum_basis=momentum_basis, sign_convention=sign_convention)
    elseif op == :S2
        create_S2!(rows, cols, vals, 1.0, indexer; momentum_basis=momentum_basis, sign_convention=sign_convention)
    elseif op == :T
        if momentum_basis
            # Translation operator is exactly diagonal in momentum basis.
            # T(\Delta r) |k> = exp(-i k \cdot \Delta r) |k>
            # kind should specify the direction of translation, e.g. 1 for x, 2 for y.
            shift = zeros(Int, length(size(Hs.lattice)))
            shift[kind] = 1 # shift by one site in 'kind' direction
            
            for (i, conf) in enumerate(indexer.inv_comb_dict)
                total_phase = 0.0
                for σ ∈ [1, 2]
                    for k_coord ∈ conf[σ]
                        p_val = 2 * π * (k_coord.coordinates[kind] - 1) / size(Hs.lattice)[kind]
                        total_phase += p_val
                    end
                end
                push!(rows, i)
                push!(cols, i)
                push!(vals, exp(1im * total_phase)) # Note: values will be complex now, should convert sparse matrix type if so, but sparse() handles it below
            end
        else
            mapping = translation_mapping(Hs.lattice, kind)
            create_transform!(rows, cols, vals, 1.0, mapping, indexer)
        end
    end

    H = sparse(rows, cols, promote_type(eltype(vals), Float64)[v for v in vals], dim, dim)
    return H
end

function create_Hubbard(Hm::HubbardModel, Hs::HubbardSubspace; get_indexer::Bool=false, indexer::Union{CombinationIndexer,Nothing}=nothing, momentum_basis::Bool=false, sign_convention::Symbol=:spin_first)
    # specify the subspace
    dim = get_subspace_dimension(Hs)
    if isnothing(indexer)
        indexer = CombinationIndexer(Hs; order=sign_convention == :spin_first ? ColSnake() : RowSnake())
    end
    rows = Int[]
    cols = Int[]
    vals = Float64[]

    if !(Hs.k === nothing)
        momentum_basis = true
    end

    #Constructs the sparse hopping Hamiltonian matrix \sum_{<i,j>} c^\dagger_i c_j.
    if Hm.t > 0 || (Hm.t isa AbstractArray)
        create_nn_hopping!(rows, cols, vals, Hm.t, Hs.lattice, indexer; momentum_basis=momentum_basis,sign_convention=sign_convention)
    end
    if Hm.U > 0
        create_hubbard_interaction!(rows, cols, vals, Hm.U, Hm.half_filling, indexer; momentum_basis=momentum_basis,sign_convention=sign_convention)
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


function create_Heisenberg(t, J, Hs::HubbardSubspace; sign_convention::Symbol=:spin_first)
    # specify the subspace
    dim = get_subspace_dimension(Hs)
    indexer = CombinationIndexer(Hs; order=sign_convention == :spin_first ? ColSnake() : RowSnake())

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

truncate(x, threshold) = ifelse(abs(x) < threshold, 0.0, x)

function project_hermitian(H, v::AbstractVector, target_eig_idx::Int, all_eigs::Vector{<:Real}; safety_factor=50.0)
    if norm(v) < 1e-9
        return v
    end
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
    dt = 2 * π / (spectral_width + 1.0) # slightly smaller step

    # Total time T:
    # To resolve gap ΔE, we need T ~ 1/ΔE. 
    # With a window, we need more cycles.
    T_required = (2 * π / min_gap) * safety_factor

    # Number of steps K
    K = ceil(Int, T_required / dt)

    # 3. Perform the Projection Sum
    # P = (1/K) * sum_{k=0}^{K-1} W(t_k) * exp(-i * t_k * target) * exp(i * t_k * H)

    v_proj = zeros(ComplexF64, size(v))

    # Current evolved vector v(t) initialized at t=0 -> v
    v_evolved = copy(v)

    for k in 0:K
        # t = k * dt
        # Normalized time tau in [0, 1]
        tau = k / K

        # Hanning window (0 at endpoints)
        window = 0.5 * (1 - cos(2 * π * tau))

        if abs(window) > 1e-10
            # Coefficient: cancel the phase of the target eigenvalue
            # c_k = exp(-i * \lambda_{target} * t)
            # t = k * dt
            phase_factor = exp(-im * target_eig * k * dt)
            coeff = window * phase_factor

            v_proj .+= coeff .* v_evolved
        end

        # Evolve for next step: v(t+dt) = exp(i*H*dt) * v(t)
        # We do this for all k < K. For k=K we don't need next step.
        if k < K
            try
                v_evolved = expv(im * dt, H, v_evolved)
            catch
                # println("Hey")
                # save("debug.jld2", H=H, v_evolved=v_evolved)
                # error("hey")
                v_evolved = exponentiate(H, im * dt, v_evolved)
            end
        end
    end

    if norm(v_proj) < 1e-9
        return v_proj
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


"""
    _map_symmetry_groups(t_vals_small, sym_small, sym_large, t_keys_small, t_keys_large)

Constructs a set of symmetrically constrained parameters (`t_vals`) for a larger system size using 
the converged solution from a smaller system size, padded with zeros for operators exclusive to the larger system.

# Arguments
- `t_vals_small`: A vector of parameter amplitudes mapped to the smaller system's symmetry groups.
- `sym_small`: The symmetry group indices structure for the smaller system. Expected to be the output of `find_symmetry_groups()`.
- `sym_large`: The symmetry group indices structure for the larger system. Expected to be the output of `find_symmetry_groups()`.
- `t_keys_small`: The ordered list of string representation keys (operators) defining the parameters in the smaller system.
- `t_keys_large`: The ordered list of string representation keys (operators) defining the parameters in the larger system.

# Returns
- `t_vals_large`: The mapped subset of parameters embedded inside a vector sized for `sym_large` representation.
- `index_mapping`: An integer vector equal in length to `t_keys_large` denoting which index in `t_keys_small` corresponds to `t_keys_large[i]`. `0` denotes an operator exclusive to the larger system.
"""
function _map_symmetry_groups(t_vals_small, sym_small, sym_large, t_keys_small, t_keys_large)
    # Create a lookup for the smaller system's t_keys
    small_keys_dict = Dict(key => idx for (idx, key) in enumerate(t_keys_small))

    # Pre-allocate the index mapping (size of large system's t_keys)
    index_mapping = zeros(Int, length(t_keys_large))

    # Populate the index mapping
    for (i, key_large) in enumerate(t_keys_large)
        if haskey(small_keys_dict, key_large)
            index_mapping[i] = small_keys_dict[key_large]
        end
    end

    # Pre-allocate the resulting t_vals for the larger system
    t_vals_large = zeros(Float64, length(sym_large[1]))

    # sym_large[1] contains vectors of indices in t_keys_large that belong to the same symmetry group
    # sym_small[2] is the inverse map for the small system: index in t_keys_small -> group index
    for (group_idx_large, group_large) in enumerate(sym_large[1])
        # Check if any index in this large group maps to the smaller system
        mapped_idx_small = 0
        for idx_large in group_large
            idx_small = index_mapping[idx_large]
            if idx_small > 0
                mapped_idx_small = idx_small
                break
            end
        end

        if mapped_idx_small > 0
            # Find the group index in the smaller system
            group_idx_small = sym_small[2][mapped_idx_small]

            # Since t_vals is indexed by group index, we can just assign the corresponding value
            t_vals_large[group_idx_large] = t_vals_small[group_idx_small]
        end
    end

    return t_vals_large, index_mapping
end

"""
    map_symmetry_groups(t_vals_small, subspace_small, subspace_large; 
                                order=1, antihermitian=false, spin_symmetry=true, trans_x=true, trans_y=true, conserve_spin=true, omit_H_conj=true, cache=Dict{Symbol, Any}())

Automates the parameter symmetry mapping routine between two system sizes by dynamically evaluating 
their Hubbard subspace configurations and invoking `_map_symmetry_groups()`.

This function abstracts away generating operator representations and finding symmetry groups by interpreting them 
directly from the supplied subspace geometry and symmetric boundary considerations. 

# Arguments
- `t_vals_small`: The solved initial parameters corresponding to `subspace_small` (typically length equals number of symmetry groups in the small system).
- `subspace_small`: `HubbardSubspace` instance representing the smaller lattice size.
- `subspace_large`: `HubbardSubspace` instance representing the targeted larger lattice size.
- `order`: The order limit of `create_randomized_nth_order_operator`. Defaults to 1 (hopping operators).
- Keyword symmetries (`antihermitian`, `spin_symmetry`, `trans_x`, `trans_y`, `conserve_spin`, `omit_H_conj`): Dictate what physics to compute symmetry groups by.
- `cache`: A dictionary to cache time-consuming setups like symmetry group determination and operator dictionary generation, enhancing optimization speeds.

# Returns
- `t_vals_large`: Solved and padded values generalized onto `subspace_large` groups.
- `index_mapping`: Diagnostic list containing exact mappings for each target index operator into the smaller space.
- `t_keys_large`: Generator structure corresponding to the output parameter configuration.
- `sym_large`: Returns the raw calculated symmetry representation from `find_symmetry_groups` for `subspace_large`.
"""
function map_symmetry_groups(t_vals_small, subspace_small, subspace_large;
    order=1, antihermitian=false, spin_symmetry=true, trans_x=true, trans_y=true, conserve_spin=true, omit_H_conj=true,
    cache=Dict{Symbol,Any}())

    if !haskey(cache, :setup_done)
        lattice_small = subspace_small.lattice
        lattice_large = subspace_large.lattice
        Lx_small, Ly_small = size(lattice_small)
        Lx_large, Ly_large = size(lattice_large)

        # Create combinations indexers
        indexer_small = CombinationIndexer(subspace_small)
        indexer_large = CombinationIndexer(subspace_large)

        # Generate t_keys (operators)
        t_dict_small,t_keys_small = create_randomized_nth_order_operator(order, indexer_small, true; omit_H_conj=omit_H_conj, conserve_spin=conserve_spin)

        t_dict_large, t_keys_large = create_randomized_nth_order_operator(order, indexer_large, true; omit_H_conj=omit_H_conj, conserve_spin=conserve_spin)

        # Find symmetry groups
        sym_small = find_symmetry_groups(t_keys_small, Lx_small, Ly_small;
            trans_x=trans_x, trans_y=trans_y, spin_symmetry=spin_symmetry,
            hermitian=!antihermitian, antihermitian=antihermitian)

        sym_large = find_symmetry_groups(t_keys_large, Lx_large, Ly_large;
            trans_x=trans_x, trans_y=trans_y, spin_symmetry=spin_symmetry,
            hermitian=!antihermitian, antihermitian=antihermitian)

        cache[:indexer_small] = indexer_small
        cache[:indexer_large] = indexer_large
        cache[:t_dict_small] = t_dict_small
        cache[:t_dict_large] = t_dict_large
        cache[:t_keys_small] = t_keys_small
        cache[:t_keys_large] = t_keys_large
        cache[:sym_small] = sym_small
        cache[:sym_large] = sym_large
        cache[:setup_done] = true
    end

    t_keys_small = cache[:t_keys_small]
    t_keys_large = cache[:t_keys_large]
    sym_small = cache[:sym_small]
    sym_large = cache[:sym_large]

    # Perform mapping
    t_vals_large, index_mapping = _map_symmetry_groups(t_vals_small, sym_small, sym_large, t_keys_small, t_keys_large)

    return t_vals_large, index_mapping, t_keys_large, sym_large
end
