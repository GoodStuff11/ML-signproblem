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

function benchmark_file(file_path::String)
    println("\n==================================================")
    println("Benchmarking H5 file: ", basename(file_path))
    
    if !isfile(file_path)
        println("File not found! Skipping...")
        return
    end

    h5open(file_path, "r") do data
        Lvec = read(data, "metadata/Lvec")
        Ne_up = read(data, "metadata/nup")
        Ne_dn = read(data, "metadata/ndown")
        kvecs = read(data, "metadata/kvecs")
        num_sectors = size(kvecs, 2)
        
        println("Lattice size: ", Lvec[1], "x", Lvec[2])
        println("Electrons: Up=", Ne_up, ", Down=", Ne_dn)
        println("Number of momentum sectors in file: ", num_sectors)
        
        target_S = (Ne_up + Ne_dn) % 2 == 0 ? 0.0 : 0.5
        println("Target spin S: ", target_S)
        
        # We will benchmark the first sector available (sector 0)
        sector = 0
        println("Running on Sector ", sector, "...")
        
        # JIT compile run (warm up)
        try
            get_su2_ground_state(data, sector, target_S; tol=1e-8)
        catch e
            println("Warmup failed: ", e)
        end
        
        # Timed run
        t_run = @elapsed indices, coeffs = get_su2_ground_state(data, sector, target_S; tol=1e-8)
        
        println("Result:")
        println("  Time taken: ", round(t_run, digits=4), " seconds")
        println("  Ground state has ", length(indices), " configurations with non-zero coefficients.")
        if length(indices) > 0
            println("  Configuration indices: ", indices[1:min(5, end)], length(indices) > 5 ? "..." : "")
        end
    end
end

function run_benchmarks()
    base_dir = "/home/jek354/research/ML-signproblem/experimenting/ed/data_new_sign"
    
    files = [
        joinpath(base_dir, "N=(3, 3)_3x3", "HubbardED_Slater_3x3_(3,3)_t_1_m_2.h5"),
        joinpath(base_dir, "N=(5, 4)_4x3", "HubbardED_Slater_4x3_(5,4)_t_1_m_2.h5"),
        joinpath(base_dir, "N=(6, 6)_4x4", "HubbardED_Slater_4x4_(6,6)_t_1_m_2_sectors_(0)-002.h5")
    ]
    
    for f in files
        benchmark_file(f)
    end
end

run_benchmarks()
