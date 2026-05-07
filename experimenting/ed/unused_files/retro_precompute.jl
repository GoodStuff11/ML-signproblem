using Lattices
using LinearAlgebra
using Combinatorics
using SparseArrays
import Graphs
using Statistics
using Random
using JSON
using JLD2
using KrylovKit
using Zygote
using ExponentialUtilities

include("ed_objects.jl")
include("ed_functions.jl")
include("utility_functions.jl")

function update_file(file_path::String)
    println("Checking file: ", file_path)
    
    # 1. Fast check: is it already at the root level?
    # This completely avoids parsing the massive "dict" object into memory if the script already ran on it.
    already_at_root = false
    jldopen(file_path, "r") do file
        if haskey(file, "precomputed_structures")
            already_at_root = true
        end
    end
    
    if already_at_root
        println("File already contains precomputed_structures at root level. Skipping.")
        return
    end
    # error("DID NOT FIND PRECOMPUTED STRUCTURES")

    # 2. Check if it's natively inside the serialized dictionary
    println("Reading full dict into memory for: ", file_path)
    dic = load_saved_dict(file_path)
    
    if haskey(dic, "precomputed_structures")
        println("File already contains precomputed_structures natively inside dict. Skipping.")
        return
    end

    # Extract info
    indexer = dic["indexer"][1] # Because we do dic["indexer"] => [all_indexers[selected_index]]
    meta_data = dic["meta_data"]
    N = meta_data["electron count"]
    
    println("Precomputing...")
    spin_conserved = !isa(N, Number)
    momentum_basis = true
    precomputed_structures = precompute_n_body_structures(indexer, 2; spin_conserved=spin_conserved, momentum_basis=momentum_basis)
    
    # We add this directly to the file at the root level to avoid rewriting the massive "dict"
    # This takes advantage of the modified load_saved_dict in utility_functions.jl
    println("Saving precomputed_structures to the JLD2 root to avoid memory rewrite overhead...")
    jldopen(file_path, "a+") do file
        file["precomputed_structures"] = precomputed_structures
    end
    println("Finished updating file in place at root level!")
end

if length(ARGS) == 1
    update_file(ARGS[1])
else
    println("Usage: julia retro_precompute.jl <path/to/meta_data_and_E.jld2>")
end
