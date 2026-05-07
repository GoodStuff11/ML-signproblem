# Test the new ordering algorithm
@inline function _each_comb(f::F, a::Vector{T}, b::Int) where {F,T}
    n = length(a)
    b == 0 && (f(T[]); return)
    b > n  && return
    indices = collect(1:b)
    buf = Vector{T}(undef, b)
    while true
        @inbounds for j in 1:b
            buf[j] = a[indices[j]]
        end
        f(buf)
        # Advance: find the LEFTMOST index that can be incremented
        j = 1
        while j <= b
            max_val = j < b ? indices[j+1] - 1 : n
            if indices[j] < max_val
                indices[j] += 1
                # Reset everything to the left back to 1, 2, ..., j-1
                @inbounds for k in 1:j-1
                    indices[k] = k
                end
                break
            end
            j += 1
        end
        j > b && break  # no index could advance → done
    end
end

# ── Verify the exact ordering the user wants ─────────────────────────────────
println("=== b=3, n=5: checking exact ordering ===")
expected = [
    (1,2,3),(1,2,4),(1,3,4),(2,3,4),
    (1,2,5),(1,3,5),(2,3,5),(1,4,5),(2,4,5),(3,4,5)
]
got = NTuple{3,Int}[]
_each_comb(collect(1:5), 3) do comb
    push!(got, (comb[1], comb[2], comb[3]))
end
for (i, (e, g)) in enumerate(zip(expected, got))
    mark = e == g ? "✓" : "✗ expected $e"
    println("  $i: $g  $mark")
end
println(got == expected ? "  All match ✓" : "  MISMATCH ✗")

# ── b=2, n=4: show full sequence ─────────────────────────────────────────────
println("\n=== b=2, n=4: full sequence ===")
_each_comb(collect(1:4), 2) do comb
    print("  $(tuple(comb...))  ")
end
println()

# ── count check: must equal binomial(n,b) ────────────────────────────────────
println("\n=== count == binomial(n,b) for all (n,b) in 0..6 ===")
count_ok = true
for n in 0:6, b in 0:n
    cnt = Ref(0)
    _each_comb(collect(1:n), b) do _; cnt[] += 1; end
    if cnt[] != binomial(n, b)
        println("  FAIL n=$n b=$b: got $(cnt[]) expected $(binomial(n,b))")
        count_ok = false
    end
end
println(count_ok ? "  All counts correct ✓" : "  Some counts wrong ✗")

# ── strictly ascending within each combo ────────────────────────────────────
println("\n=== strictly ascending within each combo ===")
asc_ok = Ref(true)
_each_comb(collect(1:6), 3) do comb
    for j in 1:length(comb)-1
        comb[j] >= comb[j+1] && (asc_ok[] = false)
    end
end
println("  All strictly ascending? $(asc_ok[] ? "yes ✓" : "NO ✗")")
