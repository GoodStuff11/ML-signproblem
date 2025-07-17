using Optimization, OptimizationOptimisers, OptimizationOptimJL
using Base.Threads, ThreadsX, NLSolvers


function f(x,p=nothing)
    # Simulate an expensive function
    # sleep(0.1)
    return sum(x .^ 2)
end


function (@main)(ARGS)
    x0  =rand(2)
    optf = OptimizationFunction(f)
    prob = Optimization.OptimizationProblem(optf, x0)
    function prob_func(prob, i, repeat)
        remake(prob, u0 = rand(2))
    end

    ensembleproblem = Optimization.EnsembleProblem(prob; prob_func)
    @time sol = Optimization.solve(ensembleproblem, OptimizationOptimJL.ParticleSwarm(), EnsembleThreads(), trajectories=Threads.nthreads(), maxiters=parse(Int,ARGS[1]))
    # @show findmin(i -> sol[i].objective, 1:4)[1]
    i = argmin([s.objective for s in sol])
    println(sol[i].u)
end