using HDF5
using LinearAlgebra

folder_new_sign = "/home/jek354/research/data/new_data/data_new_sign/N=(3, 2)_3x2"
file_path = joinpath(folder_new_sign, "HubbardED_Slater_3x2_(3,2)_t_1_m_2.h5")
data = h5open(file_path, "r")

evecs_dataset = read(data, "data/evecs/2")
target_vecs = evecs_dataset[:, 1, 1] 

println("Norm: ", norm(target_vecs))
println("First 5 amplitudes: ", target_vecs[1:5])
println("Max amplitude index: ", argmax(abs.(target_vecs)), " value: ", target_vecs[argmax(abs.(target_vecs))])
close(data)
