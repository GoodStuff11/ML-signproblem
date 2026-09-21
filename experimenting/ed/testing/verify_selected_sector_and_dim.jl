"""
verify_selected_sector_and_dim.jl

One-off verification for the system-summary table: confirm which momentum sector
`load_h5_ED_data` actually selects for a given dataset folder, and the Hilbert
space dimension of that sector.

Prints, per folder: the HDF5 file the loader chose, the Hilbert space dimension
(number of columns of `target_vecs`, which is `(n_U, H_dim)`), and `N=(nu, nd)`.
This cross-checks the Python-side replication of `find_best_energy_sector` used
to build the table.

Arguments (positional, optional):
  ARGS...  One or more dataset folder paths. Defaults to four small systems in
           /home/jek354/research/data/new_data/data_h5_fixed that are cheap to load.

Run from `experimenting/ed/` as:  julia --project=.. testing/verify_selected_sector_and_dim.jl
"""

using HDF5, LinearAlgebra, SparseArrays, Lattices

include(joinpath(@__DIR__, "..", "ed_objects.jl"))
include(joinpath(@__DIR__, "..", "utility_functions.jl"))
include(joinpath(@__DIR__, "..", "trotter.jl"))
include(joinpath(@__DIR__, "..", "ed_functions.jl"))
include(joinpath(@__DIR__, "..", "logging.jl"))
using .Trotter

function parse_arguments(args)
    root = "/home/jek354/research/data/new_data/data_h5_fixed"
    default_folders = [
        joinpath(root, "N=(2, 2)_3x2"),
        joinpath(root, "N=(3, 2)_3x2"),
        joinpath(root, "N=(3, 3)_3x2"),
        joinpath(root, "N=(4, 3)_3x3"),
    ]
    return isempty(args) ? default_folders : collect(args)
end

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "verify_selected_sector_and_dim")
    with_logging(log_path) do
        folders = parse_arguments(ARGS)
        for folder in folders
            println("\n=== ", basename(folder), " ===")
            U_values, target_vecs, _, _, N, _, _, _, Lvec, _ =
                load_h5_ED_data(folder; verbose=true, omit_indexer=true,
                                use_slater_reference=false)
            println("RESULT folder=", basename(folder),
                    " N=", N, " Lvec=", Lvec,
                    " n_U=", length(U_values),
                    " H_dim=", size(target_vecs, 2))
        end
    end
end
