#=
barren_plateau.jl

Run a gradient variance barren plateaus scaling study on a list of system sizes.
# For each system, multiple configurable tests are performed (all with num_samples=80 by default):
#   1. NN-predicted initial coefficients + ED ground-state ref (ϵ = epsilon)
#   2. Zero initial coefficients + ED ground-state ref (ϵ = epsilon)
#   3. Zero initial coefficients + ED ground-state ref (ϵ = epsilon_large = 10.0)
#   4. Zero initial coefficients + uniformly random normalized ref state (ϵ = epsilon)
#   5. Zero initial coefficients + first canonical basis vector as ref state (ϵ = epsilon)

Usage:
  julia --project=.. barren_plateau.jl [options]

Options:
  --U=<val> (optional): Interaction strength U. Default: 4.0.
  --epsilon=<val> (optional): Standard deviation for perturbing coefficients. Default: 0.005.
    Used for tests 1, 2, 4, and 5. Test 3 always uses epsilon_large = 100.0.
  --nn=<val> (optional): Comma-separated list of names or paths of neural network strategy files to load.
    Default: "pure_low_u_mild".
    Each name may be a bare name resolved under trained_neural_networks/ as "trained_neural_network_<name>.jld2" or a direct file path.
  --use_gpu=<bool> (optional): Whether to use GPU for calculations. Default: false.
  --num_exponentials=<int> (optional): Number of exponentials in the unitary ansatz. Default: 1.
  --spin_conserved=<bool> (optional): Whether to conserve spin (fixed N_up, N_down). Default: true.
  --momentum_conserved=<bool> (optional): Whether to conserve total momentum k. Default: true.
  --system_set=<val> (optional): Preset list of system sizes to study. Default: "all".
    Valid options:
    - "all": Runs all target system sizes.
    - "test1": Runs the target systems list corresponding to the original Test 1.
    - "test2": Runs the target systems list corresponding to the original Test 2.
  --random_std=<val> (optional): Comma-separated list of standard deviations for random initialization (init == :random). Default: "10.0".

Examples:
  julia --project=.. barren_plateau.jl --U=4.0 --epsilon=0.005 --num_exponentials=3 --spin_conserved=true
  julia --project=.. barren_plateau.jl --nn=pure_low_u_mild,pure_low_u_strong --num_exponentials=3 --system_set=single_per
=#

using Lattices
using LinearAlgebra
using Combinatorics
using SparseArrays
using Statistics
using Random
using Zygote
using Optimization, OptimizationOptimisers
using OptimizationOptimJL
using JLD2
using HDF5
using KrylovKit
using ExponentialUtilities
using Printf
using Dates

# Pre-scan ARGS for --use_gpu before loading any CUDA package.
# If --use_gpu=false is explicitly passed, skip loading CUDA entirely so that
# CPU-only code paths (including Threads.@threads in adjoint pullbacks) are safe.
# Any other value (true or absent) triggers the normal CUDA load-and-probe sequence.
_pre_scan_use_gpu = let val = false
    for arg in ARGS
        if startswith(arg, "--use_gpu=")
            val = parse(Bool, split(arg, "=", limit=2)[2])
        end
    end
    val  # nothing = auto-detect
end

if _pre_scan_use_gpu == true
    using CUDA
    try
        if !CUDA.functional()
            @info "CUDA not functional yet — trying local_toolkit mode"
            CUDA.set_runtime_version!(local_toolkit=true)
        end
        if CUDA.functional()
            @info "GPU available: $(CUDA.name(CUDA.CuDevice(0)))"
        end
    catch e
        @warn "CUDA setup warning: $e"
    end
end

# Include source files
include("utility_functions.jl")
include("ed_objects.jl")
include("ed_functions.jl")
include("ed_optimization.jl")
include("nn_strategy.jl")
include("logging.jl")

"""
    parse_arguments(args::Vector{String})

Parse command line arguments for the barren plateau study.
"""
function parse_arguments(args::Vector{String})
    U = 4.0
    epsilon = 0.005
    nn_files = String["trained_neural_networks/trained_neural_network_pure_low_u_mild.jld2"]
    use_gpu = @isdefined(CUDA) && CUDA.functional()
    num_exponentials_list = Int[1]
    spin_conserved = true
    momentum_conserved = true
    system_set = "all"
    random_stds = Float64[10.0]

    for arg in args
        if startswith(arg, "--U=")
            U = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--epsilon=")
            epsilon = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--nn=")
            val = String(split(arg, "=", limit=2)[2])
            nn_files = String[]
            for item in split(val, "|")
                item_str = strip(String(item), ['"', '\''])
                if isfile(item_str)
                    push!(nn_files, item_str)
                else
                    resolved = joinpath("trained_neural_networks", "trained_neural_network_$(item_str).jld2")
                    if isfile(resolved)
                        push!(nn_files, resolved)
                    else
                        error("Neural network strategy file not found: '$(item_str)' (also tried '$(resolved)')")
                    end
                end
            end
        elseif startswith(arg, "--use_gpu=")
            use_gpu = parse(Bool, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--num_exponentials=")
            val = String(split(arg, "=", limit=2)[2])
            num_exponentials_list = [parse(Int, strip(x, ['"', '\''])) for x in split(val, "|")]
        elseif startswith(arg, "--spin_conserved=")
            spin_conserved = parse(Bool, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--momentum_conserved=")
            momentum_conserved = parse(Bool, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--system_set=")
            system_set = String(split(arg, "=", limit=2)[2])
            if system_set != "all" && system_set != "test1" && system_set != "test2" && system_set != "smaller" && system_set != "single_per"
                error("Invalid system_set option: $(system_set). Must be 'all', 'test1', 'test2', 'smaller', or 'single_per'.")
            end
        elseif startswith(arg, "--random_std=")
            val = String(split(arg, "=", limit=2)[2])
            random_stds = [parse(Float64, strip(x, ['"', '\''])) for x in split(val, "|")]
        end
    end
    return U, epsilon, nn_files, use_gpu, num_exponentials_list, spin_conserved, momentum_conserved, system_set, random_stds
end


"""
Example command:
```bash
julia --project=.. barren_plateau.jl --U=4.0 --epsilon=0.05
```
"""
function (@main)(ARGS)
    # Prevent redundant execution on multi-task Slurm allocations
    if haskey(ENV, "SLURM_PROCID") && ENV["SLURM_PROCID"] != "0"
        return 0
    end

    log_path = make_log_path(@__DIR__, "barren_plateau")
    with_logging(log_path) do
        U, epsilon, nn_files, use_gpu, num_exponentials_list, spin_conserved, momentum_conserved, system_set, random_stds = parse_arguments(ARGS)
        large_epsilon = 10.0
        nparams_listed = 20

        # Load all specified neural network strategies
        strategies = Dict{String,NeuralNetStrategy}()
        for nn_file in nn_files
            base = basename(nn_file)
            if endswith(base, ".jld2")
                base = base[1:end-5]
            end
            if startswith(base, "trained_neural_network_")
                base = base[24:end]
            end
            strategies[base] = load_neural_network(nn_file)
        end

        # Define the experiments to perform. 
        # Each test must specify:
        #   - label: Used for printing table headers
        #   - init: :nn (interpolate NN coefficients), :random, or :zeros (zero coefficients)
        #   - eps: The perturbation standard deviation (typically epsilon or epsilon_large)
        #   - ref: :ground_state (loaded ED state), :random (normalized random state), or :basis (first basis vector)
        #   - nn_name: The name of the neural network strategy to use (empty for non-NN tests)
        #   - num_exps: The number of exponentials for this specific test
        #   - random_std: The standard deviation for random initialization (only used for init == :random)
        tests = NamedTuple{(:label, :init, :eps, :ref, :nn_name, :num_exps, :random_std),Tuple{String,Symbol,Float64,Symbol,String,Int,Float64}}[]
        for num_exps in num_exponentials_list
            exp_suffix = " ($(num_exps) exp)"
            push!(tests, (label="0s basis, ϵ=$large_epsilon" * exp_suffix, init=:zeros, eps=large_epsilon, ref=:basis, nn_name="", num_exps=num_exps, random_std=0.0))
            for r_std in random_stds
                push!(tests, (label="random basis (std=$r_std), ϵ=0.0" * exp_suffix, init=:random, eps=0.0, ref=:basis, nn_name="", num_exps=num_exps, random_std=r_std))
            end
        end
        sorted_nn_names = sort(collect(keys(strategies)))
        for nn_name in sorted_nn_names
            push!(tests, (label="nn ($nn_name) gs, ϵ=0.0", init=:nn, eps=0.0, ref=:ground_state, nn_name=nn_name, num_exps=1, random_std=0.0))
        end

        println("=========================================================================")
        println("              BARREN PLATEAUS GRADIENT VARIANCE SCALING STUDY            ")
        println("=========================================================================")
        println("Parameters selected:")
        println("  - U (interaction strength): $U")
        println("  - Epsilon (perturbation SD): $epsilon")
        println("  - Using GPU: $use_gpu")
        println("  - Num Exponentials: $(join(num_exponentials_list, ", "))")
        println("  - Random Std Devs: $(join(random_stds, ", "))")
        println("  - Spin Conserved: $spin_conserved")
        println("  - Momentum Conserved: $momentum_conserved")
        println("  - System Set: $system_set")
        println("  - Neural Networks: $(join(nn_files, ", "))")
        println("=========================================================================\n")

        # Defined list of target systems to study scaling behavior
        # ((N_up, N_down), (Lx, Ly), folder_name)
        systems = if system_set == "smaller"
            [
                ((2, 2), (2, 2), "N=(2, 2)_2x2"),
                ((2, 2), (3, 2), "N=(2, 2)_3x2"),
                ((3, 3), (3, 2), "N=(3, 3)_3x2"),
                ((3, 3), (4, 2), "N=(3, 3)_4x2"),
                ((3, 3), (3, 3), "N=(3, 3)_3x3"),
                ((4, 4), (4, 2), "N=(4, 4)_4x2"),
                ((4, 4), (3, 3), "N=(4, 4)_3x3"),
            ]
        elseif system_set == "single_per"
            [
                ((2, 2), (2, 2), "N=(2, 2)_2x2"),
                ((3, 3), (3, 2), "N=(3, 3)_3x2"),
                ((4, 4), (4, 2), "N=(4, 4)_4x2"),
                ((4, 4), (3, 3), "N=(4, 4)_3x3"),
                ((4, 4), (4, 3), "N=(4, 4)_4x3"),
            ]
        else # "all"
            [
                ((2, 2), (2, 2), "N=(2, 2)_2x2"),
                ((2, 2), (3, 2), "N=(2, 2)_3x2"),
                ((3, 3), (3, 2), "N=(3, 3)_3x2"),
                ((3, 3), (4, 2), "N=(3, 3)_4x2"),
                ((3, 3), (3, 3), "N=(3, 3)_3x3"),
                ((4, 4), (4, 2), "N=(4, 4)_4x2"),
                ((4, 4), (3, 3), "N=(4, 4)_3x3"),
                ((4, 4), (4, 3), "N=(4, 4)_4x3"),
            ]
        end

        results = []

        for (electrons, dims, folder_name) in systems
            system_size = collect(dims)
            folder = joinpath("data", folder_name)
            if !isdir(folder)
                @warn "Directory $folder does not exist. Skipping."
                continue
            end

            println("Evaluating system from folder: $folder_name...")

            # 1. Load ED data to obtain the correct sign convention, etc.
            local_data = try
                @time load_ED_data(folder)
            catch e
                @warn "Failed to load ED data from $folder: $e. Skipping this system size."
                continue
            end
            U_values, target_vecs, indexer, precomputed_structures, N, spin_conserved, use_symmetry, sign_convention = local_data

            # 2. Initialize square periodic lattice and Hubbard subspace
            lattice = Square(dims, Periodic())

            if !spin_conserved
                # Subspace with fixed total number of electrons, no spin/momentum conservation
                N_tot = electrons[1] + electrons[2]
                subspace = HubbardSubspace(N_tot, lattice; k=nothing)
                my_indexer = CombinationIndexer(subspace; order=sign_convention == :spin_first ? ColSnake() : RowSnake())
                dim = get_subspace_dimension(subspace)
            else
                # Standard subspace
                k_sec = momentum_conserved ? indexer.k : nothing
                subspace = HubbardSubspace(electrons[1], electrons[2], lattice; k=k_sec)
                my_indexer = momentum_conserved ? indexer : CombinationIndexer(subspace; order=sign_convention == :spin_first ? ColSnake() : RowSnake())
                dim = get_subspace_dimension(subspace)
            end

            # 3. Construct H_hopping and H_interaction using standard conventions
            @time H_hopping, H_interaction = create_hubbard_matrices(subspace; indexer=my_indexer, get_indexer=false, sign_convention=:coordinate_first)

            # 4. Generate operator structures (order 2)
            precomputed_structures = precompute_n_body_structures(my_indexer, [2]; use_symmetry=[false], spin_conserved=spin_conserved, sign_convention=:coordinate_first)
            struct_cache = precomputed_structures[(2, false)]
            t_keys = struct_cache[:t_keys]

            num_samples = 80
            test_results = []
            for test in tests
                # Resolve initial coefficients
                c_init = if test.init == :zeros
                    zeros(length(t_keys) * test.num_exps)
                elseif test.init == :random
                    randn(length(t_keys) * test.num_exps) * test.random_std
                elseif test.init == :nn
                    strategy = strategies[test.nn_name]
                    # Check if the model expects Trotter features
                    n_ctx_expected = 1 + (strategy.include_dim ? 2 : 0) + (strategy.include_electrons ? 2 : 0)
                    n_ctx_actual = strategy.use_scale_head ? size(strategy.model.scale[1].weight, 2) : (size(strategy.model.context[1].weight, 2) - size(strategy.model.base[end].weight, 1))
                    include_trotter = (n_ctx_actual == n_ctx_expected + 2)
                    
                    if include_trotter
                        c_init = Float64[]
                        for l in 1:test.num_exps
                            ctx = NeuralNetContext(U, electrons, strategy.U_max, test.num_exps, l, 10)
                            c_l = interpolate_coefficients(strategy, ctx, t_keys, system_size)
                            append!(c_init, c_l)
                        end
                        c_init
                    else
                        ctx = NeuralNetContext(U, electrons, strategy.U_max)
                        c_one = interpolate_coefficients(strategy, ctx, t_keys, system_size)
                        repeat(c_one, test.num_exps)
                    end
                else
                    error("Unknown init mode: $(test.init)")
                end

                # Resolve reference state
                ref_state = if test.ref == :basis
                    r = zeros(ComplexF64, dim)
                    r[1] = 1.0
                    r
                elseif test.ref == :ground_state
                    local_data = try
                        load_ED_data(folder)
                    catch e
                        @warn "Failed to load ED data from $folder: $e. Skipping this system size."
                        continue
                    end
                    U_values, target_vecs, indexer, precomputed_structures, N, spin_conserved_gs, use_symmetry, sign_convention = local_data

                    if target_vecs isa AbstractVector && eltype(target_vecs) <: AbstractVector
                        ref = target_vecs[1]
                    elseif size(target_vecs, 2) == dim
                        ref = target_vecs[1, :]
                    elseif size(target_vecs, 1) == dim
                        ref = target_vecs[:, 1]
                    else
                        if size(target_vecs, 1) == length(U_values)
                            ref = target_vecs[1, :]
                        else
                            ref = target_vecs[:, 1]
                        end
                    end
                else
                    error("Unknown ref mode: $(test.ref)")
                end

                println("Running test: $(test.label) (Hilbert Dim: $dim, Params: $(length(c_init)))...")
                @time variances_t, max_var_t, loss_var_t, precomputed_structures, all_grads, all_losses = test_barren_plateaus(
                    (H_hopping, H_interaction),
                    subspace,
                    my_indexer,
                    U,
                    c_init,
                    test.eps,
                    ref_state;
                    num_samples=num_samples,
                    use_gpu=use_gpu,
                    sign_convention=:coordinate_first,
                    precomputed_structures=precomputed_structures,
                    verbose=true,
                    num_exponentials=test.num_exps,
                    spin_conserved=spin_conserved,
                )

                push!(test_results, (
                    eps=test.eps,
                    label=test.label,
                    variances=variances_t,
                    max_var=max_var_t,
                    loss_var=loss_var_t,
                    grads=all_grads,
                    losses=all_losses
                ))

                # Save all_grads and all_losses to a JLD2 file in the system's folder
                # Append a suffix based on the loaded neural networks to prevent race conditions when running in parallel
                nn_suffix = ""
                if !isempty(nn_files)
                    names = String[]
                    for f in nn_files
                        b = basename(f)
                        if endswith(b, ".jld2")
                            b = b[1:end-5]
                        end
                        if startswith(b, "trained_neural_network_")
                            b = b[24:end]
                        end
                        push!(names, b)
                    end
                    nn_suffix = "_" * join(names, "_")
                end
                clean_label = replace(test.label, r"[^a-zA-Z0-9_.-]" => "_")
                save_filename = "barren_plateau_grads_losses_$(clean_label)$(nn_suffix).jld2"
                save_filepath = joinpath(folder, save_filename)
                println("Saving barren plateau data to $save_filepath...")
                JLD2.jldsave(
                    save_filepath;
                    all_grads=all_grads,
                    all_losses=all_losses,
                    variances=variances_t,
                    max_var=max_var_t,
                    loss_var=loss_var_t,
                    dim=dim,
                    num_params=length(c_init),
                    U=U,
                    epsilon=epsilon,
                    test_label=test.label,
                    electrons=electrons,
                    dims=dims,
                    num_exponentials=test.num_exps,
                    spin_conserved=spin_conserved,
                    momentum_conserved=momentum_conserved,
                    system_set=system_set
                )
            end

            push!(results, (
                system_size="$(dims[1])x$(dims[2])",
                dim=dim,
                num_params=length(t_keys) * maximum(t -> t.num_exps, tests),
                test_results=test_results,
            ))
        end

        # Print summary scaling table comparing NN prediction, random sampling, and zero-init with alternate ref states
        headers = ["System Size", "Hilbert Dim", "Param Count", "Param Index"]
        for test in tests
            if test.eps == 0.0
                push!(headers, "Grad ($(test.label))")
                push!(headers, "Loss ($(test.label))")
                push!(headers, "norm(g)^2 ($(test.label))")
            else
                push!(headers, "Var (ϵ=$(test.label))")
                push!(headers, "E[g^2] (ϵ=$(test.label))")
                push!(headers, "Mean(L) (ϵ=$(test.label))")
                push!(headers, "Var(L) (ϵ=$(test.label))")
                push!(headers, "E[norm(g)^2] (ϵ=$(test.label))")
                push!(headers, "Std[norm(g)^2] (ϵ=$(test.label))")
            end
        end

        col_widths = [15, 15, 15, 12]
        for test in tests
            if test.eps == 0.0
                push!(col_widths, 18)
                push!(col_widths, 18)
                push!(col_widths, 18)
            else
                push!(col_widths, 18)
                push!(col_widths, 18)
                push!(col_widths, 18)
                push!(col_widths, 18)
                push!(col_widths, 18)
                push!(col_widths, 18)
            end
        end

        total_width = sum(col_widths) + 3 * (length(col_widths) - 1)
        half_title_padding = max(0, div(total_width - length("SCALING STUDY SUMMARY RESULTS"), 2))

        println("\n" * "="^total_width)
        println(" "^half_title_padding * "SCALING STUDY SUMMARY RESULTS")
        println("="^total_width)

        # Print header
        for (i, h) in enumerate(headers)
            print(rpad(h, col_widths[i]))
            if i < length(headers)
                print(" | ")
            end
        end
        println()

        println("-"^total_width)

        # Print rows
        for res in results
            # Find the ranking of parameter indices
            ranking_test_idx = findfirst(tr -> tr.eps > 0.0, res.test_results)
            sort_indices = if ranking_test_idx !== nothing
                vars = res.test_results[ranking_test_idx].variances
                sortperm(vars, rev=true)
            else
                # Fallback: sort by first test's absolute gradient
                first_tr = res.test_results[1]
                grads_avg = mean(first_tr.grads, dims=1)[1, :]
                sortperm(abs.(grads_avg), rev=true)
            end

            # Display the top N parameters. Let's show top 5 parameters.
            top_n = min(nparams_listed, length(sort_indices))
            for rank_idx in 1:top_n
                p_idx = sort_indices[rank_idx]

                # Print basic columns
                print(rpad(res.system_size, 15), " | ", rpad(string(res.dim), 15), " | ", rpad(string(res.num_params), 15), " | ", rpad(string(p_idx), 12))

                # Print test columns
                for (t_idx, tr) in enumerate(res.test_results)
                    print(" | ")
                    if tr.eps == 0.0
                        if p_idx <= size(tr.grads, 2)
                            grad_val = tr.grads[1, p_idx]
                            loss_val = tr.losses[1]
                            norm2_val = sum(tr.grads[1, :] .^ 2)
                            @printf("%-18.6e", grad_val)
                            print(" | ")
                            @printf("%-18.6e", loss_val)
                            print(" | ")
                            @printf("%-18.6e", norm2_val)
                        else
                            @printf("%-18s", "N/A")
                            print(" | ")
                            @printf("%-18s", "N/A")
                            print(" | ")
                            @printf("%-18s", "N/A")
                        end
                    else
                        if p_idx <= length(tr.variances)
                            var_val = tr.variances[p_idx]
                            eg2_val = mean(tr.grads[:, p_idx] .^ 2)
                            mean_loss = mean(tr.losses)
                            var_loss = tr.loss_var
                            eg2_total = sum(mean(tr.grads .^ 2, dims=1))
                            sample_norms = sum(tr.grads .^ 2, dims=2)[:, 1]
                            std_eg2_total = std(sample_norms)
                            @printf("%-18.6e", var_val)
                            print(" | ")
                            @printf("%-18.6e", eg2_val)
                            print(" | ")
                            @printf("%-18.6e", mean_loss)
                            print(" | ")
                            @printf("%-18.6e", var_loss)
                            print(" | ")
                            @printf("%-18.6e", eg2_total)
                            print(" | ")
                            @printf("%-18.6e", std_eg2_total)
                        else
                            @printf("%-18s", "N/A")
                            print(" | ")
                            @printf("%-18s", "N/A")
                            print(" | ")
                            @printf("%-18s", "N/A")
                            print(" | ")
                            @printf("%-18s", "N/A")
                            print(" | ")
                            @printf("%-18s", "N/A")
                            print(" | ")
                            @printf("%-18s", "N/A")
                        end
                    end
                end
                println()
            end
            println("-"^total_width)
        end
        println("="^total_width * "\n")

        return 0
    end # with_logging
end
