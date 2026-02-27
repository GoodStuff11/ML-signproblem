using JLD2
using Optimization, OptimizationOptimJL, Optim
using SparseArrays

function fix_jld2_keys(directory_path)
    if !isdir(directory_path)
        println("Directory not found: $directory_path")
        return
    end

    files = filter(f -> endswith(f, ".jld2"), readdir(directory_path, join=true))

    for file in files[1:3]
        if contains(file, "meta_data_and_E.jld2") || contains(file, "shared")
            continue
        end

        println("Processing $file...")
        try
            target_dict = nothing
            jldopen(file, "r") do f
                if haskey(f, "dict")
                    target_dict = f["dict"]
                    # Unwrap nested dicts if they were created by previous failed runs
                    while target_dict isa Dict && haskey(target_dict, "dict") && length(target_dict) == 1
                        println("  Unwrapping nested 'dict' in $file")
                        target_dict = target_dict["dict"]
                    end
                else
                    # Try to recover directly if 'dict' is missing but data is there
                    target_dict = Dict{String,Any}()
                    for k in keys(f)
                        target_dict[k] = f[k]
                    end
                end
            end

            if isnothing(target_dict) || !(target_dict isa Dict)
                println("  SKIP: Could not find valid data dict in $file.")
                continue
            end

            # Ensure coefficients is present
            if haskey(target_dict, "coefficient_values")
                target_dict["coefficients"] = target_dict["coefficient_values"]
                println("  Added 'coefficients' from 'coefficient_values'.")
            end

            # Save correctly with ONE top-level "dict" key
            # We overwrite to be sure
            jldopen(file, "w") do f
                f["dict"] = target_dict
            end

            # Verification load
            test_ok = false
            jldopen(file, "r") do f
                if haskey(f, "dict") && f["dict"] isa Dict && haskey(f["dict"], "coefficients")
                    test_ok = true
                end
            end

            if test_ok
                println("  SUCCESS: Verified load(file)[\"dict\"][\"coefficients\"]")
            else
                println("  FAILURE: Verification failed for $file")
            end

        catch e
            println("  ERROR with $file: $e")
        end
    end
end

dir = "/Users/jonathonkambulow/Library/CloudStorage/Dropbox/programming/cornell courses/research/experimenting/ed/data/N=(3, 3)_3x2"
fix_jld2_keys(dir)
