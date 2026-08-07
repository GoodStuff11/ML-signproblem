using Lattices
using LinearAlgebra
using SparseArrays
using JLD2
using HDF5

include(joinpath(@__DIR__, "..", "logging.jl"))
include(joinpath(@__DIR__, "..", "utility_functions.jl"))
using .UtilityFunctions
include(joinpath(@__DIR__, "..", "trotter.jl"))
using .Trotter
include(joinpath(@__DIR__, "..", "ed_objects.jl"))
include(joinpath(@__DIR__, "..", "ed_functions.jl"))

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "test_real_vs_complex_eigvecs")
    with_logging(log_path) do
        println("=== Testing LAPACK Real vs Complex Eigenvector Solver ===")

        folder_new_sign = "/home/jek354/research/data/new_data/data_h5_fixed/N=(3, 2)_3x3"
        sign_conv = :coordinate_first

        U_vals, vecs, indexer, _, N_elec, _, _, actual_sign = load_ED_data(
            folder_new_sign; sign_convention=sign_conv, verbose=false, use_slater_reference=false
        )

        Lvec = parse_lattice_dimension(folder_new_sign)
        order = RowSnake()
        u_val = U_vals[1]
        gs = vecs[1, :]

        H_trotter, _, _ = Trotter.HubbardMomentumBasis(
            1.0, u_val, Tuple(Lvec), N_elec;
            indexer=indexer, sign_convention=actual_sign, lattice_ordering=order
        )

        M_sparse = H_trotter
        M_dense_complex = Matrix(H_trotter) # Matrix{ComplexF64}
        M_dense_real = real.(M_dense_complex) # Matrix{Float64}

        println("Is M_dense_complex purely real? $(maximum(abs.(imag.(M_dense_complex))) == 0.0)")

        V_complex = eigvecs(M_dense_complex)
        V_real = eigvecs(M_dense_real)
        V_hermitian = eigvecs(Hermitian(M_dense_real))

        println("Overlap using complex solver eigvecs(Matrix{ComplexF64}): $(abs(V_complex[:, 1]' * gs))")
        println("Overlap using real solver eigvecs(Matrix{Float64}):       $(abs(V_real[:, 1]' * gs))")
        println("Overlap using real Hermitian eigvecs(Hermitian{Float64}): $(abs(V_hermitian[:, 1]' * gs))")
    end
end
