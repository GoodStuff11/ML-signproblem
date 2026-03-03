using JLD2

function verify()
    folder = "data/tmp"
    u_28_file = joinpath(folder, "unitary_map_energy_symmetry=false_N=(3, 3)_u_28.jld2")
    u_28_dict = load(u_28_file)
    if haskey(u_28_dict, "dict")
        u_28_dict = u_28_dict["dict"]
    end

    println("u_28 final loss: ", u_28_dict["metrics"]["loss"][end])
    println("u_28 L1 norm of matrices: ", u_28_dict["norm1"])
end

verify()
