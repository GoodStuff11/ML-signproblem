using Lattices
using SparseArrays
using HDF5
using LinearAlgebra

include("ed_objects.jl")
include("utility_functions.jl")
include("trotter.jl")
include("ed_functions.jl")
using .Trotter

folder_new_sign = "/home/jek354/research/data/new_data/data_new_sign/N=(3, 2)_3x2"
valid_files = [f for f in readdir(folder_new_sign) if occursin("HubbardED", f)]
file_path = joinpath(folder_new_sign, valid_files[1])

local N
local U_values
local k_min
local evecs_dataset
local kvecs
h5open(file_path, "r") do data
    global N = (read(data, "metadata/nup"), read(data, "metadata/ndown"))
    Lvec = read(data, "metadata/Lvec")
    global U_values = read(data, "data/uvec")
    global kvecs = read(data, "metadata/kvecs")

    key_labels = [parse(Int, k) for k in keys(data["data/energies"])]
    all_E = [real.(read(data, "data/energies/$(k)"))[1, :] for k in key_labels] 
    global k_min = find_best_energy_sector(all_E, U_values; labels=key_labels, data=data)
    
    global evecs_dataset = read(data, "data/evecs/$(k_min)")
end

# In Julia, reading 3D hdf5 array (60, 2, 50) gives shape (50, 2, 60).
# The state at U_idx=1 for ground state (slice 1) is [:, 1, 1]
state = evecs_dataset[:, 1, 1] 
U1 = U_values[1] # 0.25
println("State length: ", length(state))
println("U1: ", U1)

# Now we need to use `create_Hubbard` and `indexer` to find the proper basis.
# The user says we can use a "specific sign convention" with create_Hubbard and indexer.
# The h5 file uses ColSnake (which matches :spin_first). But the h5 states are actually represented in F-order bits?
# Wait! In my documentation I established that the h5 states are represented with ColSnake bits, which maps exactly to ColSnake lattice.
# Wait, if we use `sign_convention=:spin_first` and `ColSnake()`, the basis states are ordered properly, but what about the Hamiltonian?
# The `target_vecs` from load_ED_data was mapped using `h5_ordering_and_signs` because `HubbardMomentumBasis` creates states in a different order.
# But `create_Hubbard` might create it in the CORRECT order directly!
Lvec = [3,2]
Hs = HubbardSubspace(N..., Square(tuple(Lvec...), Periodic()); k=nothing)
# wait, k=nothing gives the full Hilbert space!
# We need the momentum sector! The `.h5` file gives `evecs_dataset` which is already projected to `k_min`.
# Actually, the user says:
# H_hop, _, _ = Trotter.TamFermion.HubbardMomentumBasis(1.0, 0.0, [3,2], N_elec2; q_target=2)
# H_int, _, _ = Trotter.TamFermion.HubbardMomentumBasis(0.0, 1.0, [3,2], N_elec2; q_target=2)
# So the user wants me to use HubbardMomentumBasis! But HubbardMomentumBasis is incapable of producing a Hamiltonian with the same ordering convention as is present in the .h5 files.
# "However, using a specific sign convention with create_Hubbard and indexer, you can find the proper basis in which the .h5 data corresponds to the ground state of the Hamiltonian."
# Wait. `create_Hubbard` takes `momentum_basis=true`. 
# If `momentum_basis=true`, it will create a Hamiltonian IN THE MOMENTUM BASIS!
# Let's try `create_Hubbard` with `momentum_basis=true`!

println("Building Full Hubbard Model with create_Hubbard")
Hm_hop = HubbardModel(Lvec, 1.0, 0.0, 0.0, false)
Hm_int = HubbardModel(Lvec, 0.0, 1.0, 0.0, false)
# We can pass `sign_convention=:spin_first`
# But wait, `create_Hubbard` needs the specific subspace!
# For `k_min`, what is the tuple?
# Let's just use `get_indexer` or something to get the sector.
