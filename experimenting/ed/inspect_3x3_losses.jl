using JLD2
include("data_path.jl")

const FOLDER = get_data_root()
const sys_dir = joinpath(FOLDER, "N=(5, 4)_3x3")

function inspect_runs(prefix::String, name::String)
    println("--- ", name, " ---")
    for u_idx in 2:60
        f = joinpath(sys_dir, "$(prefix)_u_$(u_idx).jld2")
        if isfile(f)
            d = load(f)["dict"]
            m = d["metrics"]
            loss_history = m["loss"]
            println("u_idx: ", u_idx, " U: ", (u_idx - 1) * 0.25, " Loss history: ", loss_history)
        end
    end
end

inspect_runs("trotter_N=9_ref_slater_antihermitian", "Slater Reference")
inspect_runs("trotter_N=9_antihermitian", "No Slater Reference")
