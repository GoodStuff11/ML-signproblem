using JLD2
data = load("/home/jek354/research/data/new_data/data_new_sign/N=(3, 2)_3x2/unitary_map_energy_symmetry=false_N=(3, 2)_ref_slater_antihermitian_shared.jld2")
println("Keys in JLD2 dictionary:")
for (k, v) in data
    println("- ", k, ": ", typeof(v))
    if v isa Dict
        for (k2, v2) in v
            println("  - ", k2, ": ", typeof(v2))
        end
    end
end
