# run_scaling.jl — runs system_scaling.jl outside a Jupyter notebook.
# Intercepts `using IJulia` by providing a stub module before the include.

# 1. Stub IJulia so the `using IJulia` at the top of system_scaling.jl is a no-op.
@eval Main module IJulia end

# 2. Patch the load path so Julia finds our stub first.
pushfirst!(LOAD_PATH, "@stdlib")   # keep stdlib; IJulia stub lives in Main already

# 3. Fix CUDA runtime on HPC clusters.
#    When CUDA.jl is precompiled on a login node (no GPU present), it doesn't know
#    which CUDA runtime to use on compute nodes.  `local_toolkit=true` tells CUDA.jl
#    to use the system CUDA toolkit loaded via `module load cuda` rather than its
#    bundled JLLs.  This is a no-op if CUDA.jl already has a valid runtime.
try
    using CUDA
    if !CUDA.functional()
        @info "CUDA not functional yet — trying local_toolkit mode"
        CUDA.set_runtime_version!(local_toolkit=true)
        # Force re-evaluation so the preference takes effect in this session.
        # (set_runtime_version! writes a LocalPreferences.toml; a restart is the
        #  reliable way to apply it, but the call below sometimes works in-process.)
    end
    if CUDA.functional()
        @info "GPU available: $(CUDA.name(CUDA.CuDevice(0)))"
    else
        @warn "GPU still not functional — falling back to CPU training"
    end
catch e
    @warn "CUDA.jl not loaded or errored: $e — will train on CPU"
end

# 4. Include the real script. Julia's `using IJulia` will resolve to Main.IJulia.
src = read(joinpath(@__DIR__, "system_scaling.jl"), String)
# Remove just the `using IJulia` line.
src = replace(src, r"^using IJulia\r?\n"m => "")
include_string(Main, src, "system_scaling.jl")

