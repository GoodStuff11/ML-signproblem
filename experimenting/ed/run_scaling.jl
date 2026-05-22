# run_scaling.jl — runs system_scaling.jl outside a Jupyter notebook.
# Intercepts `using IJulia` by providing a stub module before the include.

# 1. Stub IJulia so the `using IJulia` at the top of system_scaling.jl is a no-op.
@eval Main module IJulia end

# 2. Patch the load path so Julia finds our stub first.
pushfirst!(LOAD_PATH, "@stdlib")   # keep stdlib; IJulia stub lives in Main already

# 3. Include the real script. Julia's `using IJulia` will resolve to Main.IJulia.
#    We skip line 1 by reading the file ourselves.
src = read(joinpath(@__DIR__, "system_scaling.jl"), String)
# Remove just the `using IJulia` line.
src = replace(src, r"^using IJulia\r?\n"m => "")
include_string(Main, src, "system_scaling.jl")
