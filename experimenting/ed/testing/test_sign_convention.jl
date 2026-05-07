using Lattices, HDF5, SparseArrays, LinearAlgebra, Combinatorics, Random
using Zygote, KrylovKit, ExponentialUtilities, Statistics
import Graphs
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using JSON, JLD2

include("../ed_objects.jl"); include("../ed_functions.jl")
include("../ed_optimization.jl"); include("../utility_functions.jl")

lattice = Square((3,2), Periodic())
subspace = HubbardSubspace(2, 2, lattice; k=(1,1))

# H_hopping is diagonal in k-basis — JW-convention-independent
H_hopping, indexer = create_Hubbard(HubbardModel(1.0,0.0,0.0,false), subspace;
                                    get_indexer=true, momentum_basis=true)
sorted_sites = sort(indexer.a)
our_dim = length(indexer.inv_comb_dict)

# ── Function to rebuild H_interaction with arbitrary JW ordering ──────────────
function build_interaction_with_jw(indexer, sorted_sites, order_fn)
    N = prod(indexer.lattice_dims)
    V = 1.0 / N   # U=1, factor out U for normalisation
    rows_t = Int[]; cols_t = Int[]; vals_t = Float64[]

    for (i1, conf) in enumerate(indexer.inv_comb_dict)
        for k_ann_up in conf[1], k_ann_dn in conf[2]
            for k_cre_up in indexer.a
                k_cre_dn_coords = mod.(k_ann_up.coordinates .+ k_ann_dn.coordinates
                                       .- k_cre_up.coordinates .- 1, indexer.lattice_dims) .+ 1
                k_cre_dn = Coordinate(k_cre_dn_coords...)

                if k_cre_up == k_ann_up && k_cre_dn == k_ann_dn
                    push!(rows_t, i1); push!(cols_t, i1); push!(vals_t, V)
                elseif k_cre_up ∉ conf[1] && k_cre_dn ∉ conf[2]
                    new_up   = union(setdiff(conf[1], [k_ann_up]),  [k_cre_up])
                    new_down = union(setdiff(conf[2], [k_ann_dn]),  [k_cre_dn])
                    i2 = index(indexer, new_up, new_down)

                    # JW sign with chosen ordering
                    ops = [(k_cre_up, 1, :create),   (k_ann_up, 1, :annihilate),
                           (k_cre_dn, 2, :create),   (k_ann_dn, 2, :annihilate)]
                    jw_order  = order_fn(sorted_sites)
                    jw_index  = Dict((sσ, i) for (i,sσ) in enumerate(jw_order))
                    occupied  = Set((s,sp) for sp in 1:2 for s in sorted_sites if s ∈ conf[sp])
                    s_val = 1
                    for (site, spin, op) in reverse(ops)
                        n_b = count(m -> jw_index[m] < jw_index[(site,spin)], occupied)
                        s_val *= (-1)^n_b
                        op == :annihilate ? delete!(occupied,(site,spin)) : push!(occupied,(site,spin))
                    end

                    push!(rows_t, i1); push!(cols_t, i2); push!(vals_t, V * s_val)
                end
            end
        end
    end
    return sparse(rows_t, cols_t, vals_t, our_dim, our_dim)
end

# ── Load external eigenvector (confirmed ±1 direct comparison in sign_convention.jl)
h5open("/home/jek354/research/ML-signproblem/experimenting/ed/data/N=(2, 2)_3x2/HubbardED_Slater_3x2_(2,2)_t_1.h5","r") do data
    u_idx = 1; mom_sec = 0; state_i = 1
    uval = read(data["data/uvec"])[u_idx]   # 0.25
    v1 = read(data["data/evecs/$mom_sec"])[:,state_i,u_idx]
    mask = abs.(v1) .> 1e-10
    println("U=$uval,  nonzero ext entries=$(count(mask))")

    jw_variants = [
        ("interleaved_up_first (current)",  ss -> [(s,σ) for s  in ss        for σ in (1,2)]),
        ("spin_major_up_first",             ss -> [(s,σ) for σ  in (1,2)     for s  in ss]),
        ("spin_major_dn_first",             ss -> [(s,σ) for σ  in (2,1)     for s  in ss]),
        ("interleaved_dn_first",            ss -> [(s,σ) for s  in ss        for σ in (2,1)]),
        ("rev_site_interleaved_up",         ss -> [(s,σ) for s  in reverse(ss) for σ in (1,2)]),
        ("rev_site_spin_major_up",          ss -> [(s,σ) for σ  in (1,2)     for s  in reverse(ss)]),
    ]

    println("\n=== Testing JW orderings for the interaction term only ===")
    println("(H_hopping is diagonal & JW-independent in k-basis)")
    for (name, order_fn) in jw_variants
        H_int = build_interaction_with_jw(indexer, sorted_sites, order_fn)
        H_full = uval * H_int + H_hopping

        # Check eigenvalues are real & sensible
        E_check = sort(real.(eigvals(Matrix(H_full))))
        Random.seed!(42)
        _, Vt = eigsolve(H_full, rand(ComplexF64, our_dim), 1, :SR)
        vt = Vt[1]

        # Phase-align to v1
        pd = (v1 ./ vt)[findfirst(abs.(v1) .> 1e-10)]
        st = round.(Int, real.(v1 ./ vt ./ pd))
        np = count(==(1), st[mask]); nm = count(==(-1), st[mask]); no = count(s->s!=1&&s!=-1, st[mask])
        println("  [$name]  GS_E=$(round(E_check[1],digits=4))  +1=$np  -1=$nm  other=$no")
    end

    # Also show the reference: current convention
    H_int_cur = create_Hubbard(HubbardModel(0.0,1.0,0.0,false), subspace; indexer=indexer, momentum_basis=true)
    H_full_cur = uval * H_int_cur + H_hopping
    Random.seed!(42)
    _, Vcur = eigsolve(H_full_cur, rand(ComplexF64, our_dim), 1, :SR)
    vcur = Vcur[1]
    pd = (v1 ./ vcur)[findfirst(abs.(v1) .> 1e-10)]
    scur = round.(Int, real.(v1 ./ vcur ./ pd))
    println("\n  [current create_Hubbard (reference)]  +1=$(count(==(1),scur[mask]))  -1=$(count(==(-1),scur[mask]))")
end
