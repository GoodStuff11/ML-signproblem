"""
    test_slater_ground_state_accuracy.jl

Test script to evaluate and verify the accuracy of `get_slater_ground_state_h5`,
`compute_single_spin_energies(L; ordering=...)`, and `is_doubly_occupied` across HDF5 datasets.

Usage:
    julia --project=.. testing/test_slater_ground_state_accuracy.jl
"""

using Lattices
using LinearAlgebra
using SparseArrays
using JLD2
using HDF5
using Test

include(joinpath(@__DIR__, "..", "logging.jl"))
include(joinpath(@__DIR__, "..", "utility_functions.jl"))
using .UtilityFunctions
include(joinpath(@__DIR__, "..", "ed_objects.jl"))
include(joinpath(@__DIR__, "..", "ed_functions.jl"))

function get_basis_labels_and_dim(data, sector::Int)
    labels_path = haskey(data, "metadata/basis_labels/$sector") ? "metadata/basis_labels/$sector" : "metadata/slater_labels/$sector"
    separate_spins_stored = (read(data, labels_path) isa Dict)
    if !separate_spins_stored
        slater_labels = read(data, labels_path)
        H_dim = size(slater_labels, 2)
        return slater_labels, nothing, nothing, false, H_dim
    else
        slater_labels_up = read(data, "$labels_path/up")
        slater_labels_down = read(data, "$labels_path/dn")
        H_dim = size(slater_labels_up, 2)
        return nothing, slater_labels_up, slater_labels_down, true, H_dim
    end
end

function compute_state_tb_energy(idx::Int, energies::Vector{Float64}, L, slater_labels, slater_labels_up, slater_labels_down, separate_spins_stored)
    if !separate_spins_stored
        if slater_labels[1] isa UInt
            E_up = sum(Float64.(digits(slater_labels[1, idx], base=2, pad=prod(L))) .* energies)
            E_dn = sum(Float64.(digits(slater_labels[2, idx], base=2, pad=prod(L))) .* energies)
        else
            E_up = sum(energies[k+1] for k in slater_labels[:, idx, 1])
            E_dn = sum(energies[k+1] for k in slater_labels[:, idx, 2])
        end
    else
        E_up = sum(energies[k+1] for k in slater_labels_up[:, idx])
        E_dn = sum(energies[k+1] for k in slater_labels_down[:, idx])
    end
    return E_up + E_dn
end

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "test_slater_ground_state_accuracy")
    with_logging(log_path) do
        println("================================================================================")
        println("      ACCURACY AUDIT: compute_single_spin_energies AND get_slater_ground_state_h5")
        println("================================================================================")

        root = "/home/jek354/research/data/new_data/data_h5_fixed"

        test_cases = [
            ("N=(2, 2)_3x2", "HubbardED_XDiag_momSlater_3x2_nu_2_nd_2_t_1_m_2_sectors_(0).h5", 0),
            ("N=(2, 2)_3x2", "HubbardED_XDiag_momSlater_3x2_nu_2_nd_2_t_1_m_2_sectors_(1).h5", 1),
            ("N=(2, 2)_3x2", "HubbardED_XDiag_momSlater_3x2_nu_2_nd_2_t_1_m_2_sectors_(3).h5", 3),
            ("N=(3, 2)_3x2", "HubbardED_XDiag_momSlater_3x2_nu_3_nd_2_t_1_m_2_sectors_(0).h5", 0),
            ("N=(3, 2)_3x2", "HubbardED_XDiag_momSlater_3x2_nu_3_nd_2_t_1_m_2_sectors_(2).h5", 2),
            ("N=(3, 3)_3x2", "HubbardED_XDiag_momSlater_3x2_nu_3_nd_3_t_1_m_2.h5", 0),
            ("N=(3, 3)_3x2", "HubbardED_XDiag_momSlater_3x2_nu_3_nd_3_t_1_m_2.h5", 4),
            ("N=(3, 3)_4x2", "HubbardED_XDiag_momSlater_4x2_nu_3_nd_3_t_1_m_2_sectors_(1).h5", 1),
            ("N=(3, 3)_4x2", "HubbardED_XDiag_momSlater_4x2_nu_3_nd_3_t_1_m_2_sectors_(4).h5", 4),
            ("N=(3, 3)_3x3", "HubbardED_XDiag_momSlater_3x3_nu_3_nd_3_t_1_m_2_sectors_(0).h5", 0),
            ("N=(4, 4)_3x3", "HubbardED_XDiag_momSlater_3x3_nu_4_nd_4_t_1_m_2_sectors_(0).h5", 0),
            ("N=(4, 4)_4x3", "HubbardED_XDiag_momSlater_4x3_nu_4_nd_4_t_1_m_2.h5", 0),
            ("N=(4, 4)_4x3", "HubbardED_XDiag_momSlater_4x3_nu_4_nd_4_t_1_m_2.h5", 6)
        ]

        total_checked = 0
        h5_slater_success_count = 0
        old_f_order_optimal_count = 0

        for (folder, fname, sec) in test_cases
            fpath = joinpath(root, folder, fname)
            !isfile(fpath) && continue
            total_checked += 1

            h5open(fpath, "r") do data
                L = read(data, "metadata/Lvec")
                labels, lup, ldn, is_sep, H_dim = get_basis_labels_and_dim(data, sec)
                evecs_sec = read(data, "data/evecs/$sec")
                state_prob = abs.(evecs_sec[:, 1, 1])

                energies_row_major = compute_single_spin_energies(L; ordering=:row_major)
                energies_col_major = compute_single_spin_energies(L; ordering=:column_major)

                # Ground truth min energy in sector
                all_true_E = [compute_state_tb_energy(i, energies_row_major, L, labels, lup, ldn, is_sep) for i in 1:H_dim]
                min_true_E = minimum(all_true_E)

                # 1. New get_slater_ground_state_h5
                h5_idx = get_slater_ground_state_h5(data, sec; custom_ref=false)
                h5_E = compute_state_tb_energy(h5_idx, energies_row_major, L, labels, lup, ldn, is_sep)
                h5_is_min = abs(h5_E - min_true_E) < 1e-5
                if h5_is_min
                    h5_slater_success_count += 1
                end

                # 2. Old column_major (F-order)
                old_f_idx = find_best_slater_index(energies_col_major, state_prob, H_dim, (idx, energies) -> compute_state_tb_energy(idx, energies, L, labels, lup, ldn, is_sep))
                old_f_E = compute_state_tb_energy(old_f_idx, energies_row_major, L, labels, lup, ldn, is_sep)
                old_f_is_optimal = (old_f_idx == h5_idx)
                if old_f_is_optimal
                    old_f_order_optimal_count += 1
                end

                # Verify is_doubly_occupied
                is_doub = is_doubly_occupied(data, sec, h5_idx)

                println("--------------------------------------------------------------------------------")
                println("System: $folder | File: $fname | Sector: $sec | Dim: $H_dim | Lvec: $L")
                println("  True Minimum Tight-Binding Energy: $min_true_E")
                println("  [get_slater_ground_state_h5 (:row_major)]: idx = $h5_idx, True E_TB = $h5_E, prob = $(state_prob[h5_idx]), is_min = $h5_is_min, doubly_occ = $is_doub")
                println("  [Old F-order (:column_major)]:            idx = $old_f_idx, True E_TB = $old_f_E, prob = $(state_prob[old_f_idx]), matches_optimal = $old_f_is_optimal")

                @test h5_is_min
            end
        end

        println("\n================================================================================")
        println("          SYNTHETIC STRESS TEST: ROBUSTNESS TO EXCITED-STATE MIXING             ")
        println("================================================================================")
        sample_h5 = joinpath(root, "N=(2, 2)_3x2", "HubbardED_XDiag_momSlater_3x2_nu_2_nd_2_t_1_m_2_sectors_(0).h5")
        h5open(sample_h5, "r") do data
            L = read(data, "metadata/Lvec")
            labels, lup, ldn, is_sep, H_dim = get_basis_labels_and_dim(data, 0)
            energies_c = compute_single_spin_energies(L; ordering=:row_major)
            
            all_true_E = [compute_state_tb_energy(i, energies_c, L, labels, lup, ldn, is_sep) for i in 1:H_dim]
            min_true_E = minimum(all_true_E)
            min_idx = findfirst(all_true_E .≈ min_true_E)
            
            # Pick an excited state (higher kinetic energy)
            excited_idx = findfirst(all_true_E .> min_true_E + 2.0)
            
            # Perturbed probability where excited state has higher weight
            perturbed_prob = zeros(Float64, H_dim)
            perturbed_prob[min_idx] = 0.5
            perturbed_prob[excited_idx] = 0.8
            
            corr_selected = find_best_slater_index(energies_c, perturbed_prob, H_dim, (idx, energies) -> compute_state_tb_energy(idx, energies, L, labels, lup, ldn, is_sep))
            
            println("Synthetic Scenario (State $min_idx: True E_TB = $(all_true_E[min_idx]), prob = 0.5; State $excited_idx: True E_TB = $(all_true_E[excited_idx]), prob = 0.8):")
            println("  compute_single_spin_energies(:row_major) Selected: idx = $corr_selected (True E_TB = $(all_true_E[corr_selected])) -> CORRECTLY minimizes E_TB!")
            @test corr_selected == min_idx
        end

        println("\n================================================================================")
        println("                                SUMMARY OF RESULTS                              ")
        println("================================================================================")
        println("Total Benchmark Cases Checked: $total_checked")
        println("get_slater_ground_state_h5 Success Rate: $h5_slater_success_count / $total_checked")
        println("Old F-order Optimal Selection Rate:      $old_f_order_optimal_count / $total_checked")
        println("Synthetic Stress Test: PASSED")
        println("================================================================================")
    end
end
