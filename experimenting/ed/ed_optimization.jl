function optimize_unitary(output_state::Vector, input_state::Vector, H::Function, p0::Vector)
    f(x, p=nothing) = 1-abs2(output_state'*exp(1im*H(x))*input_state)
    optf = OptimizationFunction(f, Optimization.AutoZygote())
    prob = OptimizationProblem(optf, p0)
    opt = OptimizationOptimisers.Adam(0.1)
    sol = solve(prob, opt, maxiters = 1000)

    phase = output_state'*exp(1im*H(x))*input_state
    return sol.u, phase
end