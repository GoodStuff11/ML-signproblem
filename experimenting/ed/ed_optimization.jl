function make_hermitian(A::SparseMatrixCSC)
    # acts similar to Hermitian(A) but is when only one of A[i,j] and A[j,i] are non-zero
    # This function doesn't override non-zero values with zero values like Hermitian(A) can
    I, J, V = findnz(A)
    return sparse(
        vcat(I, J),
        vcat(J, I),
        vcat(V, conj.(V)),
        size(A,1), size(A,2)
    )
end

function optimize_sd_sum(goal_state::Vector, indexer::CombinationIndexer; maxiters=100)
    # this approach optimizes U|psi> to be close, subtracts it from the state and then
    # finds a new slater determinant state that's the close. This optimization doesn't work
    # and the optimization must be simultaneously as supposed to sequentially.
    computed_matrices = []
    dim = length(indexer.inv_comb_dict)
    # accumulated_matrix = zeros(ComplexF64, length(goal_state), length(goal_state))
    coefficients = []

    losses = []

    loss = 1
    starting_state = zeros(length(goal_state))
    starting_state[1] = 1.0
    while loss > 1e-6
        magnitude_esimate = loss/2
        learning_rate = loss/10
        println("magnitude: $magnitude_esimate")
        println("learning rate: $learning_rate")
        t_dict = create_randomized_nth_order_operator(1,indexer;magnitude=magnitude_esimate)
        t_keys = collect(keys(t_dict))
        t_vals = collect(values(t_dict))
        rows, cols, signs, ops_list = build_n_body_structure(t_dict, indexer)


        function f(t_vals, p=nothing)
            vals = update_values(signs, ops_list, Dict(zip(t_keys,t_vals)))
            mat = exp(1im*Matrix(make_hermitian(sparse(rows, cols, vals, dim, dim))))
            loss = abs2(1-abs2(goal_state'*mat*starting_state))
            println(sqrt(loss))
            return loss
        end

        optf = OptimizationFunction(f, Optimization.AutoZygote())
        prob = OptimizationProblem(optf, t_vals)

        opt = OptimizationOptimisers.Adam(learning_rate)
        @time sol = solve(prob, opt, maxiters = maxiters)
        
        vals = update_values(signs, ops_list, Dict(zip(t_keys,sol.u)))

        loss = sqrt(f(sol.u))
        push!(computed_matrices, make_hermitian(sparse(rows, cols, vals, dim, dim)))
        starting_state -= exp(1im*Matrix(computed_matrices[end]))*starting_state
        push!(coefficients, 1/sqrt(sum(abs2.(starting_state))))
        starting_state *= coefficients[end]
        println("Finished iteration $(length(computed_matrices))")
        println("coefficient $(coefficients[end])")
        push!(losses, loss) 
        # if loss < ϵ
        #     break
        # end
        if length(computed_matrices) > 5
            break
        end
    end
    println(cumprod(coefficients))
    return computed_matrices, coefficients, losses
end

function optimize_sd_sum_2(goal_state::Vector, indexer::CombinationIndexer, sd_count::Int=2; maxiters=100, optimization=:zygote)
    computed_matrices = []
    dim = length(goal_state)

    starting_state = zeros(dim)
    starting_state[1] = 1.0
    
    t_dict = create_randomized_nth_order_operator(1,indexer;magnitude=0.5)
    t_keys = collect(keys(t_dict))
    t_vals = collect(values(t_dict))
    rows, cols, signs, ops_list = build_n_body_structure(t_dict, indexer)
    N = length(t_vals)
    params = rand(sd_count*N+(sd_count-1))


    function f(params, p=nothing)
        mat = zeros(dim,dim)
        for i in 1:sd_count
            if i == 1
                coeff = 1
            else
                coeff = params[sd_count*N+i-1]
            end
            vals = update_values(signs, ops_list, Dict(zip(t_keys,params[1+N*(i-1):N*i])))
            mat += coeff*exp(1im*Matrix(make_hermitian(sparse(rows, cols, vals, dim, dim))))
        end
        state = mat*starting_state
        state /= sqrt(sum(abs2.(state)))
        loss = abs2(1-abs2(goal_state'*state))
        println(sqrt(loss))
        return loss
    end

    # optimization
    if optimization == :zygote
        optf = OptimizationFunction(f, Optimization.AutoZygote())
    else
        optf = OptimizationFunction(f, Optim.NelderMead())
    end
    prob = OptimizationProblem(optf, params)

    opt = OptimizationOptimisers.Adam(0.1)
    @time sol = solve(prob, opt, maxiters = maxiters)
    

    # evaluating what the coefficients are
    loss = sqrt(f(sol.u))
    total_matrix = zeros(ComplexF64, length(goal_state), length(goal_state))
    coefficients = [1;sol.u[sd_count*N+1:end]]
    for (k,coeff) in zip(0:(sd_count-1),coefficients)
        vals = update_values(signs, ops_list, Dict(zip(t_keys,sol.u[1+k*N:N*(k+1)])))
        push!(computed_matrices, make_hermitian(sparse(rows, cols, vals, dim, dim)))
        total_matrix += coeff*exp(1im*computed_matrices[end])
    end
    coeff = 1/sqrt(sum(abs2.(total_matrix*starting_state)))
    coefficients .*= coeff
    println("Finished iteration $(length(computed_matrices))")
    
    println(coefficients)
    return computed_matrices, coefficients, loss
end

function optimize_unitary(state1::Vector, state2::Vector, indexer::CombinationIndexer; 
    maxiters=10, ϵ=1e-5, max_order=2, optimization=:gradient, metric_functions::Dict{String, Function}=Dict{String, Function}())
    computed_matrices = []
    computed_coefficients = []
    coefficient_labels = []
    dim = length(indexer.inv_comb_dict)
    
    metrics = Dict{String,Vector{Any}}()
    loss = 1-abs2(state1'*state2)
    metrics["loss"] = Float64[loss]
    metrics["loss_std"] = Float64[0.0]
    for k in keys(metric_functions)
        metrics[k] = Any[]
    end
    if loss < 1e-8
        println("States are already equal")
        return [], metrics
    end

    for order = 1:max_order
        magnitude_esimate = loss/2
        learning_rate = loss/10
        println("magnitude: $magnitude_esimate")
        println("learning rate: $learning_rate")
        @time t_dict = create_randomized_nth_order_operator(order,indexer;magnitude=magnitude_esimate, hermitian=true)
        t_keys = collect(keys(t_dict))
        t_vals = collect(values(t_dict))
        push!(coefficient_labels, t_keys)
        @time rows, cols, signs, ops_list = build_n_body_structure(t_dict, indexer)

        
        tmp_losses = []
        function callback(args...)
            N = 20
            push!(tmp_losses, args[end]) 
            if length(tmp_losses) > N && std(tmp_losses[end-N:end]) < 1e-8
                println("std: $(std(tmp_losses[end-N:end]))")
                return true
            end
            return false
        end

        function f_nongradient(t_vals, p=nothing)
            vals = update_values(signs, ops_list, Dict(zip(t_keys,t_vals)))
            mat = make_hermitian(sparse(rows, cols, vals, dim, dim))
            if p isa AbstractMatrix
                mat += p
            end
            loss = 1-abs2(state2'*expv(1im,mat,state1))
            println(loss)
            return loss
        end
        function f(t_vals, p=nothing)
            vals = update_values(signs, ops_list, Dict(zip(t_keys,t_vals)))
            mat = Matrix(make_hermitian(sparse(rows, cols, vals, dim, dim)))
            if p isa AbstractMatrix
                mat += p
            end
            loss = 1-abs2(state2'*exp(1im*mat)*state1)
            println(loss)
            return loss
        end

        if optimization == :gradient
            optf = Optimization.OptimizationFunction(f, Optimization.AutoZygote())
        else
            optf = Optimization.OptimizationFunction(f_nongradient)
        end

        if length(computed_matrices) > 0
            prob = Optimization.OptimizationProblem(optf, t_vals,sum(computed_matrices))
        else
            prob = Optimization.OptimizationProblem(optf, t_vals)
        end

        if optimization == :gradient
            # opt = OptimizationOptimisers.Adam(learning_rate)
            @time sol = Optimization.solve(prob, Optim.LBFGS(), maxiters=maxiters, callback=callback)
            s = sol
        else
            function prob_func(prob, i, repeat)
                remake(prob, u0 = t_vals)
            end

            ensembleproblem = Optimization.EnsembleProblem(prob; prob_func)
            @time sol = Optimization.solve(ensembleproblem, OptimizationOptimJL.ParticleSwarm(), EnsembleThreads(), trajectories=Threads.nthreads(), maxiters=maxiters, callback=callback)
            s = sol[argmin([s.objectives for s in sol])]
        end
        
        vals = update_values(signs, ops_list, Dict(zip(t_keys,sol.u)))

        # loss = f(new_tvals, if length(computed_matrices) > 0 sum(computed_matrices) else nothing end)
        loss = sol.objective
        push!(computed_matrices, make_hermitian(sparse(rows, cols, vals, dim, dim)))
        println("Finished order $order")
        push!(metrics["loss"], loss) 
        push!(metrics["loss_std"], std(last(tmp_losses,20)))
        push!(computed_coefficients, sol.u)
        for (k,func) in metric_functions
            push!(metrics[k], func(state1, state2, computed_matrices, tmp_losses))
        end
        println("loss std: $(metrics["loss_std"][end])")
        # if loss < ϵ
        #     break
        # end
    end
    return computed_matrices, coefficient_labels, computed_coefficients, metrics
end


function test_map_to_state(degen_rm_U::Vector, instructions::Dict{String, Any}, indexer::CombinationIndexer; maxiters=100, 
        optimization=:gradient,metric_functions::Dict{String, Function}=Dict{String, Function}()
        )
    # meta_data = Dict("starting state"=>Dict("U index"=>1, "levels"=>1:5),
    #             "ending state"=>Dict("U index"=>5, "levels"=>1),
    #             "electron count"=>3, "sites"=>"2x3", "bc"=>"periodic", "basis"=>"adiabatic", 
    #             "U_values"=>U_values)
    data_dict = Dict{String, Any}("norm1_metrics"=>[],"norm2_metrics"=>[],
                    "loss_metrics"=>[], "labels"=>[], "loss_std_metrics"=>[], "all_matrices"=>[],
                    "coefficients"=>[], "coefficient_labels"=>nothing)


    for i in instructions["starting state"]["levels"]
        for j in instructions["ending state"]["levels"]
            state1 = degen_rm_U[instructions["starting state"]["U index"]][:,i]
            state2 = degen_rm_U[instructions["ending state"]["U index"]][:,j]
            computed_matrices, coefficient_labels, coefficient_values, metrics = optimize_unitary(state1, state2, indexer; 
                    maxiters=maxiters, max_order=get!(instructions, "max_order", 2), optimization=optimization,
                    metric_functions=metric_functions)
            push!(data_dict["norm1_metrics"],[norm(cm, 1) for cm in computed_matrices])
            push!(data_dict["norm2_metrics"],[norm(cm, 2) for cm in computed_matrices])
            push!(data_dict["all_matrices"], computed_matrices)
            push!(data_dict["coefficients"], coefficient_values)
            if isnothing(data_dict["coefficient_labels"])
                data_dict["coefficient_labels"] = coefficient_labels
            end

            for (k, val) in metrics
                if k*"_metrics" ∉ keys(data_dict)
                    data_dict[k*"_metrics"] = [val]
                else
                    push!(data_dict[k*"_metrics"], val)
                end
            end
            push!(data_dict["labels"], Dict(
                "starting state"=>Dict("level"=>i, "U index"=>instructions["starting state"]["U index"]), 
                "ending state"=> Dict("level"=>j, "U index"=>instructions["ending state"]["U index"]))
            )
        end
    end
    return data_dict
end

function test_map_sd_sum(degen_rm_U::Vector, instructions::Dict{String, Any}, indexer::CombinationIndexer;maxiters=maxiters)
    data_dict = Dict{String, Any}("norm1_metrics"=>[],"norm2_metrics"=>[],
                    "loss_metrics"=>[], "labels"=>[], "coefficients"=>[])
    for j in instructions["goal state"]["levels"]
        goal_state = degen_rm_U[instructions["goal state"]["U index"]][:,j]
        computed_matrices, coefficients, losses = optimize_sd_sum_2(goal_state, indexer; maxiters=maxiters)
        push!(data_dict["norm1_metrics"],[norm(cm, 1) for cm in computed_matrices])
        push!(data_dict["norm2_metrics"],[norm(cm, 2) for cm in computed_matrices])
        push!(data_dict["coefficients"], coefficients)
        push!(data_dict["loss_metrics"], losses)
        push!(data_dict["labels"], Dict(
            "goal state"=> Dict("level"=>j, "U index"=>instructions["goal state"]["U index"]))
        )
    end
    return data_dict
end