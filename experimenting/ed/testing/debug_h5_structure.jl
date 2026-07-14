# Debug script to inspect H5 slater_labels and kvecs
using HDF5

h5_file = "data_new_sign/N=(2, 2)_3x2/HubbardED_Slater_3x2_(2,2)_t_1_m_2.h5"

h5open(h5_file, "r") do data
    kvecs = read(data, "metadata/kvecs")
    println("kvecs shape: $(size(kvecs))")
    println("kvecs:")
    for col in 1:size(kvecs, 2)
        println("  orbital $(col-1): k = $(kvecs[:, col])")
    end

    Lvec = read(data, "metadata/Lvec")
    println("\nLvec = $Lvec")

    # Find the best sector
    key_labels = [parse(Int, k) for k in keys(data["data/energies"])]
    sort!(key_labels)
    println("\nSectors: $key_labels")
    for k in key_labels
        E = real.(read(data, "data/energies/$k"))[:, 1]
        println("  sector $k: E[1] = $(E[1]), E[end] = $(E[end])")
    end

    # Look at slater_labels for sector 0
    sector = 0
    println("\n--- Slater labels for sector $sector ---")
    sl = read(data, "metadata/slater_labels/$sector")
    println("Type: $(typeof(sl))")
    if sl isa Dict
        sl_up = read(data, "metadata/slater_labels/$sector/up")
        sl_dn = read(data, "metadata/slater_labels/$sector/dn")
        println("sl_up shape: $(size(sl_up))")
        println("sl_dn shape: $(size(sl_dn))")
        println("First 5 states:")
        for i in 1:min(5, size(sl_up, 2))
            println("  state $i: up=$(sl_up[:, i]), dn=$(sl_dn[:, i])")
        end
        println("Last 5 states:")
        for i in max(1, size(sl_up, 2)-4):size(sl_up, 2)
            println("  state $i: up=$(sl_up[:, i]), dn=$(sl_dn[:, i])")
        end
        
        # Check: are these 0-based orbital indices?
        println("\nAll unique orbital indices: $(sort(unique(vcat(vec(sl_up), vec(sl_dn)))))")
    else
        println("Non-dict slater labels, shape: $(size(sl))")
        println("Element type: $(typeof(sl[1]))")
        if sl[1] isa UInt
            println("First 5 states (UInt format):")
            for i in 1:min(5, size(sl, 2))
                println("  state $i: up=$(sl[1, i]) = $(digits(sl[1, i], base=2, pad=6)), dn=$(sl[2, i]) = $(digits(sl[2, i], base=2, pad=6))")
            end
        else
            println("First 5 states:")
            for i in 1:min(5, size(sl, 2))
                println("  state $i: up=$(sl[:, i, 1]), dn=$(sl[:, i, 2])")
            end
        end
    end
end
