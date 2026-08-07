using HDF5
using LinearAlgebra

function main()
    folder_path = "/home/jek354/research/data/new_data/data_new_sign/N=(3, 2)_3x2"
    h5_files = filter(f -> endswith(f, ".h5"), readdir(folder_path))
    h5_path = joinpath(folder_path, h5_files[1])
    
    h5open(h5_path, "r") do data
        U_vals = read(data, "data/uvec")
        energies = read(data, "data/energies/2")
        println("U values: ", U_vals[1:5])
        println("Energies in sector 2 for U = 0.25 (first U index):")
        for idx in 1:min(10, size(energies, 2))
            println("  State $idx: ", energies[:, idx])
        end
    end
end

main()
