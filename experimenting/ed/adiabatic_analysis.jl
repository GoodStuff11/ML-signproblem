
"""
time_ordered_unitary(T, V, t0, tf; tol=1e-9, max_substeps=10^6, verbose=false)

Compute the full unitary U = T exp(-i ∫_{t0}^{tf} H(t) dt)
for a Hamiltonian of the form H(t) = T + t V, where T and V are dense matrices.

Inputs:
  T, V  :: Hermitian matrices (Matrix{ComplexF64})
  t0    :: initial time
  tf    :: final time

Keyword args:
  tol         :: error tolerance for adaptive step (default = 1e-9)
  max_substeps :: safety cap
  verbose     :: print accepted/rejected steps

Returns:
  U :: Unitary matrix (ComplexF64) approximating the time-ordered exponential
"""
function time_ordered_unitary(T::Matrix{Float64},
                              V::Matrix{Float64},
                              t0::Float64, tf::Float64;
                              tol::Float64 = 1e-9,
                              max_substeps::Int = 1_000_000,
                              verbose::Bool = false)

    n = size(T,1)
    U = Matrix{ComplexF64}(I, n, n)

    t = t0
    dt = (tf - t0) / 100   # initial guess

    substeps = 0

    while t < tf - 1e-15
        substeps += 1
        if substeps > max_substeps
            error("Exceeded max_substeps = $max_substeps")
        end

        # Prevent stepping past final time
        if t + dt > tf
            dt = tf - t
        end

        # --------- midpoint method (1 step) ----------
        tmid = t + dt/2
        Hmid = T .+ tmid .* V
        U1 = exp(-1im * Hmid * dt)

        # --------- midpoint method (2 half steps) ----------
        # First half
        tmid1 = t + dt/4
        Hmid1 = T .+ tmid1 .* V
        U_half1 = exp(-1im * Hmid1 * (dt/2))

        # Second half
        tmid2 = t + 3dt/4
        Hmid2 = T .+ tmid2 .* V
        U_half2 = exp(-1im * Hmid2 * (dt/2))

        U2 = U_half2 * U_half1   # order matters

        # --------- error estimate ----------
        err = opnorm(U2 - U1, Inf) / max(opnorm(U2, Inf), eps(Float64))

        if err <= tol
            # accept two-half-steps version
            U = U2 * U
            t += dt

            # update dt for next step
            s = err == 0 ? 2.0 : min(2.0, 0.9 * (tol/err)^(1/3))
            dt = min(dt * s, tf - t)

            if verbose
                @info "ACCEPT t=$(t - dt) -> $t, dt_new=$dt, err=$err"
            end
        else
            # reject and shrink dt
            s = max(0.1, 0.9 * (tol/err)^(1/3))
            dt *= s

            if dt < 1e-16
                error("dt underflow. Consider loosening tol.")
            end

            if verbose
                @info "REJECT at t=$t, new dt=$dt, err=$err"
            end
        end
    end

    return U
end