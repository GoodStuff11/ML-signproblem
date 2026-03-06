# Overview
The overall goal of this calculation is to minimize the loss function by tuning the real-valued parameters $A_j$ (expressed in terms of physics language and pure linear algebra language)
```math
\mathcal{L}(A) = 1-|\langle \psi_f | \mathcal{U} | \psi_i \rangle|^2=1-|\mathbf{v}_f\cdot \mathcal{U}\mathbf{v}_i|^2\in [0, 1]
```
where
```math
\mathcal{U} = \exp(i[\sum_{k_1, k_2} A^{(1)}_{k_1k_2}c^\dagger_{k_1} c_{k_2}+\sum_{k_1, k_2,k_3,k_4} A^{(2)}_{k_1k_2k_3k_4}c^\dagger_{k_1} c^\dagger_{k_2} c_{k_3} c_{k_4}])=\exp(i\sum_{j=1}^N A_j H_j)
```
Here $H_j$ are sparse Hermitian matrices and $A_j$ are real-valued parameters. $\mathbf{v}_f$ ($|\psi_f\rangle$) and $\mathbf{v}_i$ ($|\psi_i\rangle$) are the normalized final and initial states, respectively, represented as real vectors. The unitary is represented in the momentum basis, in a subspace corresponding to a momentum eigenstate with the lowest Hubbard model energy. The $A$ values are non-zero only when the fermionic operators which it is a coefficient are lattice momentum conserving, and spin conserving.

This optimization is performed for a selected initial state (the ground state of the non-interacting Hubbard model) and a set of final states which are each perturbations of previous final states (the ground state of the interacting Hubbard model with the same symmetries as the initial state). These states are stored in a file. To make the optimization perform better, when we optimize over a single final state, we can re-use those tuned parameters for the next optimization, which will generally give a good initial guess. 

For the optimization, the gradient is computed with adjoint gradients to make it as efficient as possible. It is also implemented by exactly computing the exponential as supposed to trotterizing it in any way. 


# Setup
Run setup with (note the threads can be changed to suit the system)

```bash
git clone -b simplified https://github.com/GoodStuff11/ML-signproblem.git
cd ML-signproblem/experimenting/ed
julia -e 'import Pkg; Pkg.activate("../"); Pkg.instantiate(); Pkg.update()'
julia --project=../ --threads=10 run_lanczos_scan_optimization.jl true
```

Or alternatively in julia, once the github repository is cloned, you can run the following commands in julia on two .ipynb cells:

```julia
import Pkg
Pkg.activate("../")
Pkg.instantiate()
Pkg.update()
```

```julia
using Lattices
using ChainRulesCore
using ExponentialUtilities
using LinearAlgebra
using Combinatorics
using SparseArrays
using Plots
import Graphs
using LaTeXStrings
using Statistics
using Random
using Zygote
using Optimization, OptimizationOptimisers
using JSON
using OptimizationOptimJL
using JLD2


include("ed_objects.jl")
include("ed_functions.jl")
include("ed_optimization.jl")
include("utility_functions.jl")

# folder = "/home/jek354/research/ML-signproblem/experimenting/ed/data/N=(3, 3)_3x2"
folder = "data/N=(3, 3)_3x2"
file_path = joinpath(folder, "meta_data_and_E.jld2")

dic = load_saved_dict(file_path)

meta_data = dic["meta_data"]
U_values = meta_data["U_values"]
all_full_eig_vecs = dic["all_full_eig_vecs"] 
all_E = dic["E"] # Needed for energy selection
indexer = dic["indexer"]

println("Meta data:")
display(meta_data)

# Extract N for saving
N = meta_data["electron count"]
spin_conserved = !isa(meta_data["electron count"], Number) # True if tuple (N_up, N_down)
use_symmetry = true

min_E = Inf
k_min = 1
for (k, E_vec) in enumerate(all_E)
    if !isempty(E_vec)
        E_ground = E_vec[1]
        if E_ground < min_E
            min_E = E_ground
            k_min = k
        end
    end
end
println("Selected lowest energy symmetry sector: $k_min with Energy $(min_E)")

# Select the eigenvectors for this sector
# all_full_eig_vecs is a list of sectors. each sector is a list of vectors (per U).
target_vecs = all_full_eig_vecs[k_min]

scan_instructions = Dict(
    "starting level" => 1,
    "ending level" => 1, # level index for targets
    "u_range" => 50:length(U_values),
    "optimization_scheme" => [2],
    "use symmetry" => use_symmetry
)

interaction_scan_map_to_state(target_vecs, scan_instructions, indexer,
    spin_conserved;
    maxiters=50, gradient=:adjoint_gradient,
    optimizer=[:GradientDescent, :LBFGS, :GradientDescent, :LBFGS],
    save_folder=nothing, save_name="unitary_map_energy_symmetry=$(use_symmetry)_N=$N")

````


