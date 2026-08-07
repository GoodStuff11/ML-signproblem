using HDF5
using LinearAlgebra
using Lattices
using SparseArrays
include("../ed_objects.jl")
include("../utility_functions.jl")
include("../ed_functions.jl")

folder_new_sign = "/home/jek354/research/data/new_data/data_new_sign/N=(3, 2)_3x2"
file_path = joinpath(folder_new_sign, "HubbardED_Slater_3x2_(3,2)_t_1_m_2.h5")
data = h5open(file_path, "r")

Lvec = read(data, "metadata/Lvec")
kvecs = read(data, "metadata/kvecs")
k_min = 2
uvec = read(data, "data/uvec")
evecs_dataset = read(data, "data/evecs/$(k_min)")

sl_up = read(data, "metadata/slater_labels/2/up")
sl_dn = read(data, "metadata/slater_labels/2/dn")

println("uvec[1] = ", uvec[1], ", uvec[end] = ", uvec[end])

v_u1 = evecs_dataset[:, 1, 1]
v_u60 = evecs_dataset[:, 1, 60]

println("v_u1 top 3 amplitudes:")
p1 = sortperm(abs.(v_u1), rev=true)
for i in p1[1:5]
    println("  h5_idx=", i, " abs=", abs(v_u1[i]), " val=", v_u1[i], " up=", sl_up[:, i], " dn=", sl_dn[:, i])
end

println("v_u60 top 3 amplitudes:")
p60 = sortperm(abs.(v_u60), rev=true)
for i in p60[1:5]
    println("  h5_idx=", i, " abs=", abs(v_u60[i]), " val=", v_u60[i], " up=", sl_up[:, i], " dn=", sl_dn[:, i])
end

close(data)
