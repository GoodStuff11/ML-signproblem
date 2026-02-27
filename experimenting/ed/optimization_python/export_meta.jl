using JLD2
using JSON

function load_saved_dict(load_name::String)
    file = jldopen(load_name, "r")
    dict = file["dict"]
    close(file)
    return dict
end

folder = "../data/N=(3, 3)_3x2"
file_path = joinpath(folder, "meta_data_and_E.jld2")
dic = load_saved_dict(file_path)

# Meta data includes U_values, indexer, E, all_full_eig_vecs
meta_data = dic["meta_data"]
U_values = meta_data["U_values"]
N = meta_data["electron count"]
spin_conserved = !isa(meta_data["electron count"], Number)

jldsave(joinpath(folder, "meta_data_python_export.jld2");
    meta_data=meta_data,
    U_values=U_values,
    N=N,
    spin_conserved=spin_conserved,
    E=dic["E"],
    all_full_eig_vecs=dic["all_full_eig_vecs"]
)

println("Successfully extracted meta_data to meta_data_python_export.jld2")
