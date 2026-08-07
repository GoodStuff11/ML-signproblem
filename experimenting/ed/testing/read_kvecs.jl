using HDF5
folder_path = "/home/jek354/research/data/new_data/data_new_sign/N=(3, 2)_3x2"
h5_files = filter(f -> endswith(f, ".h5"), readdir(folder_path))
h5open(joinpath(folder_path, h5_files[1]), "r") do data
    println("kvecs: ", read(data, "metadata/kvecs"))
end
