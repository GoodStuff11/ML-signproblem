using Lattices
using HDF5
using SparseArrays
using LinearAlgebra
using Combinatorics
using JLD2

# Include the source files
include("../ed_objects.jl")
include("../utility_functions.jl")
include("../ed_functions.jl")

function run_eigenvalue_test()
    folder = "/home/jek354/research/ML-signproblem/experimenting/ed/data_new_sign/N=(3, 3)_3x3"
    println("Loading ED data from folder: ", folder)
    U_values, target_vecs, indexer, precomputed_structures, N, spin_conserved, use_symmetry, sign_convention = load_ED_data(folder; verbose=true)
    
    file_path = joinpath(folder, readdir(folder)[findfirst(f -> occursin("HubbardED", f), readdir(folder))])
    h5open(file_path, "r") do data
        Lvec = read(data, "metadata/Lvec")
        U_values = read(data, "data/uvec")
        kvecs = read(data, "metadata/kvecs")
        
        # Best energy sector (usually sector 0 or k_min)
        key_labels = [parse(Int, k) for k in keys(data["data/energies"])]
        all_E = [real.(read(data, "data/energies/$(k)"))[:, 1] for k in key_labels]
        k_min = find_best_energy_sector(all_E, U_values; labels=key_labels)
        println("Best energy sector (k_min): ", k_min)
        
        # 1. Obtain SU(2) ground state indices & coefficients
        target_S = 0.0 # N=(3,3) singlet state
        println("\n--- Step 1: Obtaining SU(2) ground state ---")
        indices, coeffs = get_su2_ground_state(data, k_min, target_S; tol=1e-8, sign_convention=sign_convention)
        println("SU(2) ground state has ", length(indices), " active configurations.")
        
        # Construct the dense state vector in the full subspace Hilbert space
        dim = length(indexer.inv_comb_dict)
        state_vec = zeros(ComplexF64, dim)
        state_vec[indices] = coeffs
        state_vec ./= norm(state_vec) # Ensure it is perfectly normalized
        
        # 2. Construct the S² operator using the indexer
        println("\n--- Step 2: Constructing S² operator ---")
        rows_S2 = Int[]; cols_S2 = Int[]; vals_S2 = Float64[]
        create_S2!(rows_S2, cols_S2, vals_S2, 1.0, indexer; momentum_basis=true, sign_convention=sign_convention)
        S2_op = sparse(rows_S2, cols_S2, vals_S2, dim, dim)
        
        # Compute <state| S² |state>
        s2_eigenval = real(dot(state_vec, S2_op * state_vec))
        println("Expected S² eigenvalue: ", target_S * (target_S + 1.0))
        println("Computed S² eigenvalue: ", s2_eigenval)
        
        # 3. Construct the tight-binding Hamiltonian and compute its energy
        println("\n--- Step 3: Constructing Hamiltonian ---")
        lattice = Square(tuple(Lvec...), Periodic())
        subspace = HubbardSubspace(N..., lattice; k=tuple((kvecs[:, k_min+1] .+ 1)...))
        
        # HubbardModel with U=0, t=1 to represent H_tb
        Hm = HubbardModel(1.0, 0.0, 0.0, false)
        H_hopping = create_Hubbard(Hm, subspace; indexer=indexer, momentum_basis=true, sign_convention=sign_convention)
        
        # Compute expectation value <state| H |state>
        energy = real(dot(state_vec, H_hopping * state_vec))
        println("Computed Tight-Binding Ground State Energy: ", energy)
    end
end

run_eigenvalue_test()
