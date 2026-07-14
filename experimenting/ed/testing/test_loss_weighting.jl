# testing/test_loss_weighting.jl
using Test
using Lattices
using LinearAlgebra
using Combinatorics
using SparseArrays
using Statistics
using Random
using Zygote
using JLD2
using Flux
using CUDA

include("../logging.jl")
include("../ed_objects.jl")
include("../ed_functions.jl")
include("../ed_optimization.jl")
include("../utility_functions.jl")
include("../nn_strategy.jl")

@testset "Loss Weighting Tests" begin
    # 1. Load a small mock folder specs
    folder_specs = [
        ((2, 2), [2, 2], 2:4, "") # N=(2,2)_2x2, U-indices 2 to 4
    ]
    
    println("--- Testing prepare_dataset_nn ---")
    @time X1, X2, X3, X4, Ctx, Y, Y_log_scale, Y_opt_loss = prepare_dataset_nn(folder_specs;
        U_max=20.0,
        include_dim=true,
        include_electrons=true,
        dim_max=4,
        use_scale_head=true
    )
    
    @test size(X1, 2) == length(Y)
    @test length(Y_log_scale) == length(Y)
    @test length(Y_opt_loss) == length(Y)
    
    println("Dataset loaded successfully!")
    println("Y_opt_loss sample values: ", Y_opt_loss[1:min(10, end)])
    @test any(y -> y > 0.0, Y_opt_loss) # Ensure we loaded non-zero optimized losses
    
    # 2. Run a dry-run train_mlp! with 2 epochs for each of the new weighting schemes
    n_feat = size(X1, 1)
    n_ctx = size(Ctx, 1)
    
    for scheme in ["loss_mild", "loss_std", "loss_power_mild", "loss_power_std"]
        println("\n--- Testing weighting_scheme = $scheme ---")
        
        # Build model
        model = build_two_stage_mlp(;
            feat_dim=n_feat, base_hidden=[16, 16], embed_dim=16,
            context_hidden=[16, 8], scale_hidden=[16, 8], n_context=n_ctx,
            use_scale_head=true
        )
        
        # Train for 2 epochs on CPU
        try
            loss_history = train_mlp!(model, X1, X2, X3, X4, Ctx, Y, Y_log_scale, Y_opt_loss;
                n_epochs=2, batch_size=64, lr=1e-3,
                use_scale_head=true, verbose=true,
                weighting_scheme=scheme
            )
            @test length(loss_history) == 2
            println("✓ Weighting scheme $scheme trained successfully without errors!")
        catch e
            println("✗ Weighting scheme $scheme failed with error: ", e)
            rethrow(e)
        end
    end
end
