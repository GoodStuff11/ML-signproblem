module Trotter

include("utility_functions.jl")
include("TamLib.jl")
include("TamFermion.jl")
include("trotter_optimization.jl")

using .UtilityFunctions
using .TamLib
using .TamFermion
using .TrotterOptimization

export UtilityFunctions, TamLib, TamFermion, TrotterOptimization

end # module Trotter
