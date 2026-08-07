import sys

def rewrite_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # We need to replace load_h5_ED_data
    # Let's find the start and end of it.
    import re
    
    # Match function load_h5_ED_data(...) ... end\nend
    pattern_h5 = re.compile(r'function load_h5_ED_data\(folder; verbose=false, kwargs\.\.\.\).*?return U_values.*?end\nend\n', re.DOTALL)
    
    new_h5 = """function load_h5_ED_data(folder; verbose=false, kwargs...)
    omit_indexer = get(kwargs, :omit_indexer, false)
    use_slater_reference = get(kwargs, :use_slater_reference, true)
    su2_symmetry = get(kwargs, :su2_symmetry, false)

    valid_files = [f for f in readdir(folder) if occursin("HubbardED", f)]
    if isempty(valid_files)
        error("No meta_data_and_E.jld2 file, and no HubbardED HDF5 files found in folder: $folder")
    end

    file_path = joinpath(folder, valid_files[1])
    if verbose
        println("Loading hdf5 data: $file_path")
    end

    return h5open(file_path, "r") do data
        N = (read(data, "metadata/nup"), read(data, "metadata/ndown"))
        spin_conserved = true
        use_symmetry = false

        Lvec = read(data, "metadata/Lvec")
        U_values = read(data, "data/uvec")
        kvecs = read(data, "metadata/kvecs")

        key_labels = [parse(Int, k) for k in keys(data["data/energies"])]
        all_E = [real.(read(data, "data/energies/$(k)"))[1, :] for k in key_labels] # Needed for energy selection
        k_min = find_best_energy_sector(all_E, U_values; labels=key_labels, data=data, su2_symmetry=su2_symmetry)
        if verbose
            println(all_E)
        end
        separate_spins = (read(data, "metadata/slater_labels/$k_min") isa Dict)
        if separate_spins
            sl_up = read(data, "metadata/slater_labels/$k_min/up")
            sl_dn = read(data, "metadata/slater_labels/$k_min/dn")
            H_dim = size(sl_up, 2)
        else
            sl_all = read(data, "metadata/slater_labels/$k_min")
            H_dim = size(sl_all, 2)
        end

        evecs_dataset = read(data, "data/evecs/$(k_min)")
        gs_slice = 1
        raw_evecs = evecs_dataset[:, gs_slice, :] # shape (H_dim, n_U)
        target_vecs = transpose(raw_evecs) # shape (n_U, H_dim)

        use_slater_ref = (use_slater_reference !== false && use_slater_reference !== nothing)
        if use_slater_ref
            slater_index = get_slater_ground_state(data, k_min; custom_ref=use_slater_reference)
            if slater_index == -1
                error("No Slater ground state could be found in sector $k_min.")
            end
            reference_state = zeros(ComplexF64, H_dim)
            reference_state[slater_index] = 1.0
            target_vecs = vcat(transpose(reference_state), target_vecs) # shape (n_U + 1, H_dim)
        end

        # Native HDF5 uses ColSnake, L[end:-1:1], and :spin_first
        if omit_indexer
            indexer = nothing
            return U_values, target_vecs, indexer, Dict(), N, spin_conserved, use_symmetry, :spin_first, reverse(Lvec), ColSnake()
        end

        # Build native indexer
        if separate_spins
            up_sample = sl_up[:, 1]
            dn_sample = sl_dn[:, 1]
        else
            up_sample = sl_all[:, 1, 1]
            dn_sample = sl_all[:, 1, 2]
        end
        tot_k_sec = zeros(Int, length(Lvec))
        for o in up_sample
            tot_k_sec .+= kvecs[:, o+1]
        end
        for o in dn_sample
            tot_k_sec .+= kvecs[:, o+1]
        end
        
        # Native lattice dimension is reversed for HDF5 (e.g., 3x2 is constructed as 2x3 for ColSnake alignment)
        Lvec_native = reverse(Lvec)
        k_sector = tuple([(reverse(tot_k_sec)[d] % Lvec_native[d]) + 1 for d in 1:length(Lvec_native)]...)
        lattice = Square(tuple(Lvec_native...), Periodic())
        subspace = HubbardSubspace(N..., lattice; k=k_sector)
        
        if verbose
            println("Computing native indexer for sector k = $k_sector")
        end
        order_native = ColSnake()
        indexer = CombinationIndexer(subspace; order=order_native)

        return U_values, target_vecs, indexer, Dict(), N, spin_conserved, use_symmetry, :spin_first, Lvec_native, order_native
    end
end
"""

    content = pattern_h5.sub(new_h5, content)
    
    with open(filepath, 'w') as f:
        f.write(content)

if __name__ == "__main__":
    rewrite_file(sys.argv[1])
