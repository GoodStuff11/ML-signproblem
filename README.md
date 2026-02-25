# ML-signproblem

Run setup with

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
