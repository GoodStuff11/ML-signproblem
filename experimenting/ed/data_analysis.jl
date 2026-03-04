# using Pkg
# Pkg.activate("/home/jek354/research/ML-signproblem")
# Pkg.update()

using Lattices
using LinearAlgebra
using Combinatorics
using SparseArrays
using Plots
import Graphs
using LaTeXStrings
using Statistics
using Random
using Zygote
using Optimization, OptimizationOptimisers
using JSON
using OptimizationOptimJL
using JLD2
using ExponentialUtilities


include("ed_objects.jl")
include("ed_functions.jl")
include("ed_optimization.jl")
include("utility_functions.jl")


"""
Computes the 4-point tensor transformation A_out = P * A_in * P_conj.
This handles both the forward Fourier transform (momentum to position) and the inverse Fourier transform (position to momentum).

Arguments:
- `A_tensor`: Input tensor of size (N1, N1, N1, N1, num_u_vals)
- `P_mat`: The single-particle transformation matrix. For momentum->position, P[k, r] = exp(i * k ⋅ r)/sqrt(V).
    For position->momentum (inverse), P[r, k] = exp(-i * k ⋅ r)/sqrt(V).

Mathematically, we want to compute the following sum efficiently:
    A_out[r1, r2, r3, r4] = ∑_{k1, k2, k3, k4} A_in[k1, k2, k3, k4] * P[k1, r1] * P[k2, r2] * P*[k3, r3] * P*[k4, r4]

To avoid an O(V^8) loop, we factor out the problem by grouping the pairs of indices:
1. Group the first two indices (e.g. creations):
   Let index K12 = (k1, k2) and R12 = (r1, r2).
   Define the batched projection matrix P12[K12, R12] = P[k1, r1] * P[k2, r2].
2. Group the last two indices (e.g. annihilations):
   Let index K34 = (k3, k4) and R34 = (r3, r4).
   Define the conjugated batched projection matrix P34_conj[K34, R34] = P*[k3, r3] * P*[k4, r4].

By flattening A_in into a matrix over (K12, K34), the summation simplifies into a chain of two matrix multiplications:
    A_out[R12, R34] = ∑_{K12, K34} P12[K12, R12] * A_in[K12, K34] * P34_conj[K34, R34]
This can be written precisely as the matrix product:
    A_out_mat = (P12^T) * A_in_mat * (P34_conj)

Yielding the result in O(V^6) time by passing the heavy lifting to BLAS algorithm arrays.

Returns:
- `A_out`: Output tensor of size (N2, N2, N2, N2, num_u_vals), where N2 = size(P_mat, 2)
"""
function transform_4point_tensor(A_tensor::AbstractArray{ComplexF64,5}, P_mat::AbstractMatrix{ComplexF64})
    N1 = size(P_mat, 1) # Size of input basis (e.g., number of k-points)
    N2 = size(P_mat, 2) # Size of output basis (e.g., number of real-space sites, V)
    num_u_vals = size(A_tensor, 5)

    A_out = zeros(ComplexF64, N2, N2, N2, N2, num_u_vals)

    # 1. First, we compute the grouped projection matrices for the pairs of indices.
    P_conj = conj.(P_mat)
    P12 = zeros(ComplexF64, N1^2, N2^2)
    P34_conj = zeros(ComplexF64, N1^2, N2^2)
    for i2 in 1:N1, i1 in 1:N1
        idx_in = (i2 - 1) * N1 + i1 # flattened inner index (e.g. K12 or K34)
        for o2 in 1:N2, o1 in 1:N2
            idx_out = (o2 - 1) * N2 + o1 # flattened outer index (e.g. R12 or R34)

            # P12[K12, R12] = P[k1, r1] * P[k2, r2]
            P12[idx_in, idx_out] = P_mat[i1, o1] * P_mat[i2, o2]

            # P34_conj[K34, R34] = P*[k3, r3] * P*[k4, r4]
            P34_conj[idx_in, idx_out] = P_conj[i1, o1] * P_conj[i2, o2]
        end
    end

    # 2. We use P12^T so that when multiplied, the inner dimension matches the rows of A_mat
    P12_T = transpose(P12)

    # 3. Iterate through each U value (hyperparameter index), reshape to matrices, and multiply
    for u in 1:num_u_vals
        # Flatten the input tensor slice A[k1,k2,k3,k4] into A_mat[K12, K34]
        A_mat = reshape(@view(A_tensor[:, :, :, :, u]), N1^2, N1^2)

        # Dense Complex Matrix Multiply: (N2^2 x N1^2) * (N1^2 x N1^2) * (N1^2 x N2^2) -> (N2^2 x N2^2)
        A_out_mat = P12_T * A_mat * P34_conj

        # Reshape the output matrix back into a 4-dimensional tensor slice (r1,r2,r3,r4)
        A_out[:, :, :, :, u] = reshape(A_out_mat, N2, N2, N2, N2)
    end
    return A_out
end



folder = "data/N=(3, 3)_3x2"
# folder="data/tmp"

e_metadata = load_saved_dict(joinpath(folder, "meta_data_and_E.jld2"))
U_values = e_metadata["meta_data"]["U_values"]
dim = [parse(Int, x) for x in split(e_metadata["meta_data"]["sites"], "x")]
shared_data = load_saved_dict(joinpath(folder, "unitary_map_energy_symmetry=false_N=(3, 3)_shared.jld2"))
coefficient_data = []
interaction_data = []
loss = []
initial_loss = []
for i = 6:61#2:61
    dic = load_saved_dict(joinpath(folder, "unitary_map_energy_symmetry=false_N=(3, 3)_u_$i.jld2"))
    push!(coefficient_data, dic["coefficients"][2])
    push!(interaction_data, U_values[i])
    push!(loss, dic["metrics"]["loss"][2])
    push!(initial_loss, dic["metrics"]["loss"][1])
end
coefficient_data = reduce(hcat, coefficient_data);


order = 2
x = []
y = []
z = []
z1 = []
z2 = []
target_spin = 3 .- [1, 2, 1, 2]
selected_coefficient_index = falses(length(shared_data["coefficient_labels"][2])) # to filter spins
for i in eachindex(if shared_data["coefficient_labels"][order] !== nothing
    shared_data["coefficient_labels"][order]
else
    []
end)
    # k1 + k2 -> k1' + k2'
    if !all(shared_data["coefficient_labels"][2][i][term][2] == target_spin[term] for term in 1:4)
        continue
    end
    # display(shared_data["coefficient_labels"][2][i])
    s = sum(abs2, (collect((shared_data["coefficient_labels"][2][i][1][1] + shared_data["coefficient_labels"][2][i][2][1]).coordinates) .- 2)) # (k1 + k2)^2
    t = sum(abs2, (shared_data["coefficient_labels"][2][i][1][1] - shared_data["coefficient_labels"][2][i][1+order][1]).coordinates) # (k1 - k1')^2
    u = sum(abs2, (shared_data["coefficient_labels"][2][i][1][1] - shared_data["coefficient_labels"][2][i][2+order][1]).coordinates) # (k1 - k2')^2
    # println("($s, $t, $u): $(u_data["coefficient_values"][2][i])")

    # push!(x, (u -t)+(rand()*2-1)*0.2)
    push!(x, s)
    push!(y, t)
    push!(z, u)
    # push!(y, s+(rand()*2-1)*0.2)
    selected_coefficient_index[i] = true

end


# scatter(x,y,z, marker_z=coefficient_data[selected_coefficient_index, 1], color=:balance,xlabel=L"s", ylabel=L"t", zlabel="u")

high_range_mask = abs.(coefficient_data[:, 50]) .> 0.2
mid_range_mask = 0.05 .< abs.(coefficient_data[:, 50]) .< 0.2
low_range_mask = abs.(coefficient_data[:, 50]) .< 0.05
# histogram(abs.(coefficient_data[selected_coefficient_index, 50]), bins=30)

for i in eachindex(selected_coefficient_index)
    if !selected_coefficient_index[i] || !high_range_mask[i]
        continue
    end
    display(shared_data["coefficient_labels"][2][i])
end

# --- Compute Fourier transform to real space A_{r1, r2, r3, r4} ---
println("Computing Fourier transform of coefficients to real space...")

# Positions are linearly indexed 1:V
V = prod(dim)
lattice_sites = CartesianIndices(tuple(dim...))

# We'll map all unique k vectors to integers to form dense arrays
all_k = unique([label[term][1] for i in eachindex(shared_data["coefficient_labels"][2])
                for label in (shared_data["coefficient_labels"][2][i],)
                for term in 1:4])
num_k = length(all_k)
k_to_idx = Dict(k => idx for (idx, k) in enumerate(all_k))

# Precompute the single-particle spatial phase transformation matrix
# P_mat[i, r] = e^{i * k_i ⋅ r} / sqrt(V)
P_mat = zeros(ComplexF64, num_k, V)
for (i, k) in enumerate(all_k)
    for (r_idx, r) in enumerate(lattice_sites)
        phase_exponent = 2 * π * sum((k.coordinates[d] - 1) * (r[d] - 1) / dim[d] for d in 1:length(dim))
        P_mat[i, r_idx] = exp(im * phase_exponent) / sqrt(V)
    end
end
println("P_mat spatial FT projectors initialized.")

# Allocate A_k tensor over the k-space index grid: (k1_D, k2_U, k3_D, k4_U, U_val)
num_u_vals = size(coefficient_data, 2)
A_k = zeros(ComplexF64, num_k, num_k, num_k, num_k, num_u_vals)

# Use ALL terms in shared_data that have exactly one up/down spin in creation & annihilation
for i in eachindex(shared_data["coefficient_labels"][2])
    label = shared_data["coefficient_labels"][2][i]

    c_spins = [label[1][2], label[2][2]]
    a_spins = [label[3][2], label[4][2]]

    # We want to reconstruct the tensor A_{r1_down, r2_up, r3_down, r4_up}
    # We only include terms mapping to down-up-down-up 
    if sort(c_spins) == [1, 2] && sort(a_spins) == [1, 2]

        # Determine the k indices matching down (spin=2) and up (spin=1)
        idx_c_down = findfirst(==(2), c_spins)
        idx_c_up = findfirst(==(1), c_spins)
        k1_D = label[idx_c_down][1]
        k2_U = label[idx_c_up][1]

        idx_a_down = findfirst(==(2), a_spins)
        idx_a_up = findfirst(==(1), a_spins)
        k3_D = label[2+idx_a_down][1]
        k4_U = label[2+idx_a_up][1]

        # Anti-commutation sign to rearrange to order (down, up, down, up)
        c_swaps = idx_c_down == 2 ? 1 : 0
        a_swaps = idx_a_down == 2 ? 1 : 0
        canonical_sign = (-1)^(c_swaps + a_swaps)

        # Map k vectors to dense array indices
        ik1 = k_to_idx[k1_D]
        ik2 = k_to_idx[k2_U]
        ik3 = k_to_idx[k3_D]
        ik4 = k_to_idx[k4_U]

        # Populate the A_k grid with base and Hermitian conjugate components
        @inbounds for u in 1:num_u_vals
            term = canonical_sign * coefficient_data[i, u]
            A_k[ik1, ik2, ik3, ik4, u] += term
            A_k[ik3, ik4, ik1, ik2, u] -= term # H.C maps creation elements to annihilation and applies anti-hermitian negation
        end
    end
end
println("A_k tensor constructed in momentum space.")

# We compute the position-space tensor via batched tensor contraction/matrix multiplication
A_real_all = transform_4point_tensor(A_k, P_mat)

println("Fourier transform complete. The tensor `A_real_all` has size ", size(A_real_all),
    " where the first 4 indices are spatial locations (1 to $V) and the last index is the U value index.")

# -------------------------------------------------------------
# Verify by applying the Inverse Fourier Transform back to momentum space
# P_inv[r, k] = exp(-i * k * r) / sqrt(V)
# Because P_mat has elements P_mat[k, r], the inverse transformation matrix is just P_mat^dagger.
# So P_inv = conj.(transpose(P_mat))
println("Running Inverse Fourier Transform to verify accuracy...")
P_inv = conj.(transpose(P_mat))

# Transform back from size (V, V, V, V, num_u_vals) to (num_k, num_k, num_k, num_k, num_u_vals)
A_k_recovered = transform_4point_tensor(A_real_all, P_inv)

max_error = maximum(abs.(A_k .- A_k_recovered))
println("Maximum deviation between original A_k and recovered A_k: ", max_error)
if max_error < 1e-10
    println("Verification Successful! The Fourier transforms are numerically exact.")
else
    println("WARNING: Substantial error detected in transformation recovery.")
end

# --- Extract and Label Real-Space Coefficients ---
println("\nExtracting independent real-space coefficients and labels...")

# Function to shift a Coordinate on a periodic lattice
function shift_coord(coord::Coordinate, shift_vec, dim)
    shifted = mod.(coord.coordinates .+ shift_vec .- 1, dim) .+ 1
    return Coordinate(Tuple(shifted))
end

function get_coord(idx, dim)
    c = lattice_sites[idx]
    return Coordinate(Tuple(c))
end

function extract_spatial_coefficients(A_real_all, dim, V, lattice_sites)
    pos_coefficient_labels = []
    num_u_vals = size(A_real_all, 5)
    pos_coefficient_data = zeros(ComplexF64, 0, num_u_vals)
    visited_canonical_translations = Set()

    for r1 in 1:V, r2 in 1:V, r3 in 1:V, r4 in 1:V
        mag = maximum(abs.(A_real_all[r1, r2, r3, r4, :]))
        if mag > 1e-10
            c1 = get_coord(r1, dim)
            c2 = get_coord(r2, dim)
            c3 = get_coord(r3, dim)
            c4 = get_coord(r4, dim)

            shift = (1, 1) .- c4.coordinates

            c1_rel = shift_coord(c1, shift, dim)
            c2_rel = shift_coord(c2, shift, dim)
            c3_rel = shift_coord(c3, shift, dim)
            c4_rel = Coordinate((1, 1))

            canonical_key = (c1_rel, c2_rel, c3_rel, c4_rel)

            if !(canonical_key in visited_canonical_translations)
                push!(visited_canonical_translations, canonical_key)

                label = (
                    (c1_rel, 2, :create),
                    (c2_rel, 1, :create),
                    (c3_rel, 2, :annihilate),
                    (c4_rel, 1, :annihilate)
                )
                push!(pos_coefficient_labels, label)

                val_vector = A_real_all[r1, r2, r3, r4, :]
                pos_coefficient_data = vcat(pos_coefficient_data, transpose(val_vector))
            end
        end
    end
    return pos_coefficient_labels, pos_coefficient_data
end

pos_coefficient_labels, pos_coefficient_data = extract_spatial_coefficients(A_real_all, dim, V, lattice_sites)

println("Found $(length(pos_coefficient_labels)) independent spatial coefficients (translation invariant groups).")

# Sanity Check: if we re-expand these independent spatial coefficients over all translations,
# do we exactly recover A_real_all?
println("\nSanity checking real-space coefficients reconstruction...")
A_real_reconstructed = zeros(ComplexF64, V, V, V, V, num_u_vals)

for (idx, label) in enumerate(pos_coefficient_labels)
    c1_rel, c2_rel, c3_rel, c4_rel = label[1][1], label[2][1], label[3][1], label[4][1]
    val_vector = pos_coefficient_data[idx, :]

    # Apply all possible translations
    for r_shift in lattice_sites
        shift_vec = Tuple(r_shift) .- (1, 1)

        c1_t = shift_coord(c1_rel, shift_vec, dim)
        c2_t = shift_coord(c2_rel, shift_vec, dim)
        c3_t = shift_coord(c3_rel, shift_vec, dim)
        c4_t = shift_coord(c4_rel, shift_vec, dim)

        # Map Coordinates back to linear indices
        lin_inds = LinearIndices(Tuple(dim))
        r1_t = lin_inds[c1_t.coordinates...]
        r2_t = lin_inds[c2_t.coordinates...]
        r3_t = lin_inds[c3_t.coordinates...]
        r4_t = lin_inds[c4_t.coordinates...]

        A_real_reconstructed[r1_t, r2_t, r3_t, r4_t, :] .= val_vector
    end
end

max_recon_error = maximum(abs.(A_real_all .- A_real_reconstructed))
println("Maximum deviation in position-space reconstruction: ", max_recon_error)
if max_recon_error < 1e-10
    println("Sanity Check Passed! The extracted labels and coefficients perfectly represent the tensor.")
else
    println("WARNING: Sanity Check Failed! Translation invariance assumption is flawed.")
end
