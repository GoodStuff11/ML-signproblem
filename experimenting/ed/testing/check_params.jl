using Lattices
using LinearAlgebra
using SparseArrays
using JLD2
include("../utility_functions.jl")
include("../ed_objects.jl")
include("../ed_functions.jl")
include("../data_path.jl")

const FILE_LABEL_PAIRS = [
    ("3x2 (2,2)",   "N=(2, 2)_3x2", (2,2)),
    ("3x2 (3,2)",  "N=(3, 2)_3x2", (3,2)),
    ("3x2 (3,3)",  "N=(3, 3)_3x2", (3,3)),
    ("3x3 (3,2)",   "N=(3, 2)_3x3", (3,2)),
    ("4x2 (3,3)",   "N=(3, 3)_4x2", (3,3)),
    ("3x3 (3,3)",   "N=(3, 3)_3x3", (3,3)),
    ("3x3 (4,3)",   "N=(4, 3)_3x3", (4,3)),
    ("3x3 (4,4)",   "N=(4, 4)_3x3", (4,4)),
    ("3x3 (5,4)",   "N=(5, 4)_3x3", (5,4)),
    ("4x3 (5,4)",   "N=(5, 4)_4x3", (5,4)),
]

folder = get_data_root()
for (label, file_label, nelectrons) in FILE_LABEL_PAIRS
    dirpath = joinpath(folder, file_label)
    files = readdir(dirpath)
    nsites = prod(parse_lattice_dimension(file_label))
    
    # Check pruning analysis file
    filename = build_save_name_prefix("pruning_analysis_trotter"; sites=nsites, antihermitian=true, custom_ref_state_arg="slater", num_exponentials=2)
    path = joinpath(dirpath, "$(filename).jld2")
    num_exp = 2
    if !isfile(path)
        filename = build_save_name_prefix("pruning_analysis_trotter"; sites=nsites, antihermitian=true, custom_ref_state_arg="slater", num_exponentials=1)
        path = joinpath(dirpath, "$(filename).jld2")
        num_exp = 1
    end
    d = load(path)
    rem = d["removed_terms"]
    
    # Check matching trotter files
    trotter_files = if num_exp == 2
        filter(f -> contains(f, "num_exponentials=2") && contains(f, "_u_") && contains(f, "slater"), files)
    else
        filter(f -> !contains(f, "num_exponentials") && contains(f, "_u_") && contains(f, "slater"), files)
    end
    coeff_len = if !isempty(trotter_files)
        f_sample = joinpath(dirpath, trotter_files[1])
        d_sample = load(f_sample)
        haskey(d_sample, "coefficients") ? length(d_sample["coefficients"]) : "no coeff"
    else
        "no trotter files"
    end
    println(rpad(label, 12), " | num_exp: $num_exp | coeff_len: $coeff_len | max(rem): ", maximum(rem))
end
