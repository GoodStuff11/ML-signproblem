using HDF5
folder_new_sign = "/home/jek354/research/data/new_data/data_new_sign/N=(2, 2)_3x2"
valid_files = [f for f in readdir(folder_new_sign) if occursin("HubbardED", f)]
file_path = joinpath(folder_new_sign, valid_files[1])
h5open(file_path, "r") do data
    key_labels = [parse(Int, k) for k in keys(data["data/energies"])]
    all_E = [real.(read(data, "data/energies/$(k)"))[1, :] for k in key_labels] 
    U_values = read(data, "data/uvec")
    println("U values: ", U_values)
    println("Energies: ", all_E)
end
