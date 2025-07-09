function optimize_unitary(state1::Vector, state2::Vector, indexer::CombinationIndexer; maxiters=10, ϵ=1e-5, max_order=2)
    computed_matrices = []
    
    loss = 1-abs2(state1'*state2)
    losses = [loss]
    if loss < 1e-8
        println("States are already equal")
        return [], losses
    end


    for order = 1:max_order
        magnitude_esimate = loss/2
        learning_rate = loss/10
        println("magnitude: $magnitude_esimate")
        println("learning rate: $learning_rate")
        @time t_dict = create_randomized_nth_order_operator(order,indexer;magnitude=magnitude_esimate)
        t_keys = collect(keys(t_dict))
        t_vals = collect(values(t_dict))
        @time rows, cols, signs, ops_list = build_n_body_structure(t_dict, indexer)



        function f(t_vals, p=nothing)
            vals = update_values(signs, ops_list, Dict(zip(t_keys,t_vals)))
            mat = Matrix(Hermitian(sparse(rows, cols, vals, dim, dim)))
            if p isa AbstractMatrix
                mat += p
            end
            loss = 1-abs2(state2'*exp(1im*mat)*state1)
            println(loss)
            return loss
        end

        optf = OptimizationFunction(f, Optimization.AutoZygote())

        if length(computed_matrices) > 0
            prob = OptimizationProblem(optf, t_vals,sum(computed_matrices))
        else
            prob = OptimizationProblem(optf, t_vals)
        end
        opt = OptimizationOptimisers.Adam(learning_rate)
        @time sol = solve(prob, opt, maxiters = maxiters)
        
        vals = update_values(signs, ops_list, Dict(zip(t_keys,sol.u)))
        loss = f(sol.u, if length(computed_matrices) > 0 sum(computed_matrices) else nothing end)
        push!(computed_matrices,Hermitian(sparse(rows, cols, vals, dim, dim)))
        println("Finished order $order")
        push!(losses, loss) 
        if loss < ϵ
            break
        end
    end
    return computed_matrices, losses
end

function test_map_to_gs(degen_rm_U::Vector, instructions::Dict{String, Any}, indexer::CombinationIndexer; maxiters=100)
    # meta_data = Dict("starting state"=>Dict("U index"=>1, "levels"=>1:5),
    #             "ending state"=>Dict("U index"=>5, "levels"=>1),
    #             "electron count"=>3, "sites"=>"2x3", "bc"=>"periodic", "basis"=>"adiabatic", 
    #             "U_values"=>U_values)
    data_dict = Dict{String, Any}("norm1_metrics"=>[],"norm2_metrics"=>[],
                    "loss_metrics"=>[], "labels"=>[])
    for i in instructions["starting state"]["levels"]
        for j in instructions["ending state"]["levels"]
            state1 = degen_rm_U[instructions["starting state"]["U index"]][:,i] # ground state
            state2 = degen_rm_U[instructions["ending state"]["U index"]][:,j] # these states are non-degenerate and slater determinants
            computed_matrices, losses = optimize_unitary(state1, state2, indexer; 
                    maxiters=maxiters, max_order=get!(instructions, "max_order", 2))
            push!(data_dict["norm1_metrics"],[norm(cm, 1) for cm in computed_matrices])
            push!(data_dict["norm2_metrics"],[norm(cm, 2) for cm in computed_matrices])
            push!(data_dict["loss_metrics"], losses)
            push!(data_dict["labels"], Dict(
                "starting state"=>Dict("level"=>i, "U index"=>instructions["starting state"]["U index"]), 
                "ending state"=> Dict("level"=>j, "U index"=>instructions["ending state"]["U index"]))
            )
        end
    end
    return data_dict
end