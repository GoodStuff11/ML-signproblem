using Lattices
using LinearAlgebra
using SparseArrays

include(joinpath(@__DIR__, "..", "logging.jl"))
include(joinpath(@__DIR__, "..", "utility_functions.jl"))
using .UtilityFunctions
include(joinpath(@__DIR__, "..", "trotter.jl"))
using .Trotter
include(joinpath(@__DIR__, "..", "ed_objects.jl"))
include(joinpath(@__DIR__, "..", "ed_functions.jl"))

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "test_sign_debug")
    with_logging(log_path) do
        Lvec = (2, 2)
        N_sites = prod(Lvec)
        N_elec = (2, 1)
        u_val = 1.0
        lattice = Square(Lvec, Periodic())

        for sign_conv in [:spin_first, :coordinate_first]
            order = sign_conv == :spin_first ? ColSnake() : RowSnake()
            subspace = HubbardSubspace(N_elec..., lattice; k=(1, 1))
            indexer = CombinationIndexer(subspace; order=order)

            H_trotter, basis_trotter, _ = Trotter.TamFermion.HubbardMomentumBasis(
                1.0, u_val, Lvec, N_elec;
                indexer=indexer, sign_convention=sign_conv, lattice_ordering=order
            )
            M_trotter = real.(Matrix(H_trotter))
            V_trotter = eigvecs(M_trotter)

            new_hopping = create_Hubbard(HubbardModel(1.0, 0.0, 0.0, false), subspace; 
                                indexer=indexer, momentum_basis=true, sign_convention=sign_conv, lattice_ordering=order)
            new_interaction = create_Hubbard(HubbardModel(0.0, 1.0, 0.0, false), subspace; 
                                indexer=indexer, momentum_basis=true, sign_convention=sign_conv, lattice_ordering=order)
            H_create = Matrix(new_hopping .+ u_val .* new_interaction)
            V_create = eigvecs(H_create)

            diff = M_trotter .- H_create
            max_diff = maximum(abs.(diff))
            ov = abs(V_trotter[:, 1]' * V_create[:, 1])
            println("sign_conv = $sign_conv, order = $order")
            println("  Max diff |M_trotter - H_create|: $max_diff")
            println("  Eigenvector overlap between trotter and create: $ov")

            if sign_conv == :coordinate_first
                sorted_sites = sort(indexer.a, order=order)
                println("  Basis state sign comparison:")
                for (i, conf) in enumerate(indexer.inv_comb_dict)
                    ops = [(s, 1, :create) for s in conf[1]]
                    append!(ops, [(s, 2, :create) for s in conf[2]])
                    s_spin = compute_jw_sign(conf, sorted_sites, ops; sign_convention=:spin_first)
                    s_coord = compute_jw_sign(conf, sorted_sites, ops; sign_convention=:coordinate_first)
                    ratio_ed = s_coord * s_spin

                    # Compute TamFermion state sign for state i
                    s_int = basis_trotter["ints"][i]
                    N_s = prod(Lvec)
                    s_up = UInt(s_int & ((one(s_int) << N_s) - one(s_int)))
                    s_dn = UInt(s_int >> N_s)
                    col_sites_by_row_order = [Trotter.TamFermion.ravel_c(Trotter.TamFermion.unravel_f(r, Lvec), Lvec) for r in 0:N_s-1]
                    swaps = 0
                    for r_dn_idx in 0:N_s-1
                        c_dn_idx = col_sites_by_row_order[r_dn_idx + 1]
                        if ((s_dn >> c_dn_idx) & 1) == 1
                            for r_up_idx in r_dn_idx+1:N_s-1
                                c_up_idx = col_sites_by_row_order[r_up_idx + 1]
                                if ((s_up >> c_up_idx) & 1) == 1
                                    swaps += 1
                                end
                            end
                        end
                    end
                    tam_sgn = iseven(swaps) ? 1.0 : -1.0
                    println("    state $i: ed_ratio = $ratio_ed, tam_sgn = $tam_sgn")
                end
            end
        end
    end
end
