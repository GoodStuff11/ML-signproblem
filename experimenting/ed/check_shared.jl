using JLD2
data = load("/home/jek354/research/data/new_data/data/N=(4, 4)_3x3_2/trotter_N=9_loss_energy_shared.jld2")
println(keys(data))
println("antihermitian: ", get(data, "antihermitian", "not found"))
println("use_slater_ref: ", get(data, "use_slater_reference", "not found"))
