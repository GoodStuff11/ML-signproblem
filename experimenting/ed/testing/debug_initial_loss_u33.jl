#=
debug_initial_loss_u33.jl

Diagnostic test to investigate why the initial loss is high when loading
trotter_N=16_ref_slater_antihermitian_u_33.jld2 and growing from 1 to 2 exponentials.
=#

ENV["JULIA_CUDA_USE_COMPAT"] = "true"

using Lattices
using LinearAlgebra
using SparseArrays
using JLD2
using HDF5
using CUDA

include(joinpath(@__DIR__, "..", "data_path.jl"))
include(joinpath(@__DIR__, "..", "logging.jl"))
include(joinpath(@__DIR__, "..", "utility_functions.jl"))
using .UtilityFunctions
include(joinpath(@__DIR__, "..", "trotter.jl"))
using .Trotter
include(joinpath(@__DIR__, "..", "ed_objects.jl"))
include(joinpath(@__DIR__, "..", "ed_functions.jl"))

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "debug_initial_loss_u33")
    with_logging(log_path) do
        folder = data_folder("N=(6, 6)_4x4")
        old_file = joinpath(folder, "trotter_N=16_ref_slater_antihermitian_u_33.jld2")
        
        println("=== Debug Initial Loss for U_idx=33 ===")
        println("Old file: $old_file")
        old_dict = JLD2.load(old_file)["dict"]
        old_coeffs = old_dict["coefficients"]
        println("Old coeffs length: ", length(old_coeffs))
        println("Old coeffs norm: ", norm(old_coeffs))
        if haskey(old_dict, "loss_metrics")
            println("Old final loss in dict: ", old_dict["loss_metrics"])
        end
        if haskey(old_dict, "metrics") && haskey(old_dict["metrics"], "optimization_losses")
            opt_losses = old_dict["metrics"]["optimization_losses"]
            if !isempty(opt_losses)
                println("Old recorded last iter loss: ", opt_losses[1][end])
            end
        end

        # Load ED data
        U_values, state_vecs, indexer, _, N_elec, _, _, sign_convention =
            load_ED_data(folder; verbose=true, sign_convention=:spin_first, use_slater_reference=true)
        
        Lvec = parse_lattice_dimension(folder)
        N_sites = prod(Lvec)
        basis_sector = Trotter.get_basis_sector(indexer, Lvec, N_sites)
        gates = Trotter.enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true, include_diagonal=false)
        num_gates = length(gates)
        println("Number of gates: $num_gates")
        println("Basis sector size: $(length(basis_sector))")

        # Grow coefficients
        grown_coeffs = grow_coefficients(old_coeffs, 1, 2, num_gates)
        println("Grown coeffs length: ", length(grown_coeffs))

        u_idx = 33
        state1 = state_vecs[1, :] # Slater reference
        state2 = state_vecs[u_idx, :] # Target state at U=8.0
        
        println("\n--- ComplexF64 Analysis ---")
        state1_c64 = ComplexF64.(state1)
        state2_c64 = ComplexF64.(state2)
        println("Raw <state2, state1> = ", dot(state2_c64, state1_c64))
        println("Raw overlap loss (un-evolved): ", 1.0 - abs2(dot(state2_c64, state1_c64)))

        # Phase stripping check
        ref_stripped, phase_ref = strip_global_phase(state1)
        target_stripped, phase_tgt = strip_global_phase(state2)
        println("Ref phase: $phase_ref")
        println("Target phase: $phase_tgt")
        println("Max imag in stripped ref: ", maximum(abs, imag.(state1 .* conj(phase_ref))))
        println("Max imag in stripped target: ", maximum(abs, imag.(state2 .* conj(phase_tgt))))
        println("Stripped overlap loss (un-evolved): ", 1.0 - abs2(dot(target_stripped, ref_stripped)))

        # GPU Evaluations
        println("\n--- GPU Loss Evaluations ---")
        tau_terms = Trotter.fgateToTauSector(gates, N_sites, basis_sector; antihermitian=true)
        
        # 1. GPU ComplexF64, num_exponentials=1 (old_coeffs)
        loss_gpu_c64_1 = adjoint_loss(old_coeffs, gates, tau_terms, state1_c64, state2_c64, basis_sector, N_sites;
            num_exponentials=1, antihermitian=true, use_gpu=true, datatype=ComplexF64)
        println("GPU (ComplexF64, num_exp=1, old_coeffs) loss: $loss_gpu_c64_1")

        # 2. GPU ComplexF64, num_exponentials=2 (grown_coeffs)
        loss_gpu_c64_2 = adjoint_loss(grown_coeffs, gates, tau_terms, state1_c64, state2_c64, basis_sector, N_sites;
            num_exponentials=2, antihermitian=true, use_gpu=true, datatype=ComplexF64)
        println("GPU (ComplexF64, num_exp=2, grown_coeffs) loss: $loss_gpu_c64_2")

        # 3. GPU Float32, num_exponentials=1 (old_coeffs)
        loss_gpu_f32_1 = adjoint_loss(old_coeffs, gates, tau_terms, ref_stripped, target_stripped, basis_sector, N_sites;
            num_exponentials=1, antihermitian=true, use_gpu=true, datatype=Float32)
        println("GPU (Float32, num_exp=1, old_coeffs) loss: $loss_gpu_f32_1")

        # 4. GPU Float32, num_exponentials=2 (grown_coeffs)
        loss_gpu_f32_2 = adjoint_loss(grown_coeffs, gates, tau_terms, ref_stripped, target_stripped, basis_sector, N_sites;
            num_exponentials=2, antihermitian=true, use_gpu=true, datatype=Float32)
        println("GPU (Float32, num_exp=2, grown_coeffs) loss: $loss_gpu_f32_2")

        # 5. Check evolved state norm and dot product with Float32 vs ComplexF64
        evolved_f32 = apply_unitary(old_coeffs, gates, ref_stripped, basis_sector, N_sites, 1;
            antihermitian=true, use_gpu=true, datatype=Float32)
        evolved_f32_cpu = Array(evolved_f32)
        println("Float32 evolved state norm: ", norm(evolved_f32_cpu))
        target_f32 = Float32.(target_stripped)
        println("Float32 dot(target, evolved): ", dot(target_f32, evolved_f32_cpu))
        println("Float32 fidelity (overlap^2): ", abs2(dot(target_f32, evolved_f32_cpu)))
        println("Float32 infidelity (1 - overlap^2): ", 1.0f0 - abs2(dot(target_f32, evolved_f32_cpu)))
    end
end
