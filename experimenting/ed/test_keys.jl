using JLD2

function test()
    folder = "data/tmp"
    load_file = joinpath(folder, "unitary_map_energy_symmetry=false_N=(3, 3)_u_28.jld2")

    dict = load(load_file)
    if haskey(dict, "dict")
        dict = dict["dict"]
    end
    current_coeffs = dict["coefficients"]

    shared_file = replace(load_file, r"_u_\d+\.jld2$" => "_shared.jld2")
    println("Shared file is: ", shared_file, " exists: ", isfile(shared_file))

    if isfile(shared_file)
        shared_dict = load(shared_file)
        if haskey(shared_dict, "dict")
            shared_dict = shared_dict["dict"]
        end
        initial_coefficient_labels = shared_dict["coefficient_labels"]
        println("Is current_coeffs length equal to initial_coefficient_labels length: ", length(current_coeffs) == length(initial_coefficient_labels))
        for i in 1:length(current_coeffs)
            println("Order $i:")
            if isnothing(current_coeffs[i])
                println("  current_coeffs: nothing")
            else
                println("  current_coeffs length: ", length(current_coeffs[i]))
            end
            if isnothing(initial_coefficient_labels[i])
                println("  labels: nothing")
            else
                println("  labels length: ", length(initial_coefficient_labels[i]))
            end
        end
    end
end

test()
