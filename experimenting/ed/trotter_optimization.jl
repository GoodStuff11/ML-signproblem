#=
trotter_optimization.jl

Trotter optimization module aggregating GPU kernels, state evolution, loss functions,
core parameter optimization routines, and interaction parameter scans.
=#

module TrotterOptimization

using ChainRulesCore
using LinearAlgebra
using SparseArrays
using Zygote
using Optimization
using OptimizationOptimJL
using Optim
using OptimizationOptimisers
import ..TamFermion
using ..Trotter: @safe_threads
using JLD2
using Statistics

# Sub-components
include("trotter_gpu_kernels.jl")
include("trotter_evolution.jl")
include("trotter_loss.jl")
include("trotter_core_optimization.jl")
include("trotter_scan.jl")

export adjoint_loss, energy_loss, optimize_unitary, interaction_scan_map_to_state, extract_convergence_info, grow_coefficients,
       StridedCheckpoints, determine_checkpoint_stride, apply_unitary_checkpoints, backward_adjoint_propagation, apply_unitary,
       strip_global_phase, get_gpu_gate_ops, gpu_apply_gate_exp!, to_device_vector

end # module TrotterOptimization