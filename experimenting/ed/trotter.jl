module Trotter

using ChainRulesCore
using LinearAlgebra
using SparseArrays
using Zygote
using Optimization
using OptimizationOptimJL
using Optim
using OptimizationOptimisers
using JLD2
using Statistics

include("utility_functions.jl")
include("TamLib.jl")
include("TamFermion.jl")

using .UtilityFunctions
using .TamLib
using .TamFermion

include("trotter_optimization.jl")
using .TrotterOptimization

export UtilityFunctions, TamLib, TamFermion, TrotterOptimization

# Re-export key Trotter optimization APIs directly at Trotter level
export adjoint_loss, energy_loss, optimize_unitary, interaction_scan_map_to_state, extract_convergence_info, grow_coefficients,
       StridedCheckpoints, determine_checkpoint_stride, apply_unitary_checkpoints, backward_adjoint_propagation, apply_unitary,
       strip_global_phase, get_gpu_gate_ops, gpu_apply_gate_exp!, to_device_vector

# Re-export TamFermion basis and gate utilities at Trotter level
export get_basis_sector, enumerate_ferm_excitations, fgateToTauSector

end # module Trotter
