using HDF5

file_path = "/home/jek354/research/data/new_data/data_new_sign/N=(3, 2)_3x2/HubbardED_Slater_3x2_(3,2)_t_1_m_2.h5"
h5open(file_path, "r") do data
    sl_up = read(data, "metadata/slater_labels/2/up")
    sl_dn = read(data, "metadata/slater_labels/2/dn")
    kvecs = read(data, "metadata/kvecs")
    evecs = read(data, "data/evecs/2")

    raw_gs = evecs[:, 1, 1]
    H_dim = size(sl_up, 2)
    Lx, Ly = 3, 2

    println("Top 10 components of raw_gs in sector 2:")
    perm = sortperm(abs.(raw_gs), rev=true)
    for k in 1:10
        idx = perm[k]
        up = sl_up[:, idx]
        dn = sl_dn[:, idx]
        
        e_sum = 0.0
        for o in up
            kx, ky = kvecs[1, o+1], kvecs[2, o+1]
            e_sum += -2.0 * (cos(2 * pi * kx / Lx) + cos(2 * pi * ky / Ly))
        end
        for o in dn
            kx, ky = kvecs[1, o+1], kvecs[2, o+1]
            e_sum += -2.0 * (cos(2 * pi * kx / Lx) + cos(2 * pi * ky / Ly))
        end
        
        println("  rank $k: idx=$idx, |val|^2 = $(abs2(raw_gs[idx])), val = $(raw_gs[idx]), up=$up, dn=$dn, diag_E = $e_sum")
    end
end
