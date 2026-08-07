using Test

function fast_coord_first_state_sign(s_up::UInt, s_dn::UInt, N_sites::Int)
    swaps = 0
    for d in 0:N_sites-1
        if ((s_dn >> d) & 1) == 1
            swaps += count_ones(s_up >> (d + 1))
        end
    end
    return iseven(swaps) ? 1.0 : -1.0
end

function reference_coord_first_state_sign(c_up_mask::UInt, c_dn_mask::UInt, N_sites::Int)
    c_pairs_coord = Tuple{Int, Int}[]
    for s in 0:N_sites-1
        if ((c_up_mask >> s) & 1) == 1
            push!(c_pairs_coord, (s, 1))
        end
        if ((c_dn_mask >> s) & 1) == 1
            push!(c_pairs_coord, (s, 2))
        end
    end

    c_pairs_spin = Tuple{Int, Int}[]
    for s in 0:N_sites-1
        if ((c_up_mask >> s) & 1) == 1
            push!(c_pairs_spin, (s, 1))
        end
    end
    for s in 0:N_sites-1
        if ((c_dn_mask >> s) & 1) == 1
            push!(c_pairs_spin, (s, 2))
        end
    end

    perm = [findfirst(==(p), c_pairs_coord) for p in c_pairs_spin]
    parity = 0
    for i in 1:length(perm), j in i+1:length(perm)
        if perm[i] > perm[j]
            parity += 1
        end
    end
    return iseven(parity) ? 1.0 : -1.0
end

function (@main)(ARGS)
    println("=== Testing Fast JW State Sign Formula ===")
    N_sites = 9
    matching = true
    for s_up in UInt(0):UInt(2^N_sites - 1)
        for s_dn in UInt(0):UInt(2^N_sites - 1)
            f_sign = fast_coord_first_state_sign(s_up, s_dn, N_sites)
            r_sign = reference_coord_first_state_sign(s_up, s_dn, N_sites)
            if f_sign != r_sign
                println("Mismatch for up=$s_up, dn=$s_dn: fast=$f_sign, ref=$r_sign")
                matching = false
                break
            end
        end
        if !matching; break; end
    end
    if matching
        println("SUCCESS! Fast formula matches reference 100% across all 262,144 configurations!")
    end
end
