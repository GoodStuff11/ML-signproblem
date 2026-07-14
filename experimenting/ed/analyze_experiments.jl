# analyze_experiments.jl
using Printf
using Dates
using Statistics

function parse_log_file(filepath)
    if !isfile(filepath)
        return nothing
    end
    
    content = read(filepath, String)
    
    # Extract weighting scheme and name
    weighting_m = match(r"Training NN with weighting scheme:\s+(\w+)", content)
    name_m = match(r"No existing neural network found at trained_neural_networks/trained_neural_network_([\w\_]+)\.jld2", content)
    
    # If the network already existed, it will load it instead:
    if isnothing(name_m)
        name_m = match(r"Loaded neural network strategy from:\s+trained_neural_networks/trained_neural_network_([\w\_]+)\.jld2", content)
    end
    
    if isnothing(weighting_m) || isnothing(name_m)
        return nothing
    end
    
    weighting = weighting_m[1]
    name = name_m[1]
    
    # Parse evaluation sections
    # Look for "Testing on folder: data/<folder>"
    sections = split(content, "Testing on folder: data/")
    
    results = Dict{String, Dict{String, Any}}()
    
    for i in 2:length(sections)
        lines = split(sections[i], "\n")
        folder_name = strip(lines[1])
        
        # Parse table rows under "Evaluating adjoint_loss per U-index" or "Evaluating accuracy (pred/baseline) per U-index"
        table_started = false
        u_values = Float64[]
        pred_rand_ratios = Float64[]
        rand_baseline_ratios = Float64[]
        
        for line in lines
            if occursin("Evaluating adjoint_loss per U-index", line) || occursin("Evaluating accuracy", line)
                table_started = true
                continue
            end
            if table_started && startswith(line, "---")
                continue
            end
            
            if table_started
                tokens = split(strip(line))
                if length(tokens) >= 13
                    try
                        u_val = parse(Float64, tokens[1])
                        pred_loss = parse(Float64, tokens[3])
                        ratio = parse(Float64, tokens[4]) # pred/baseline
                        rand_loss = parse(Float64, tokens[9])
                        rand_ratio = parse(Float64, tokens[12]) # pred/rand
                        rand_baseline = parse(Float64, tokens[13]) # rand/baseline
                        
                        # Only analyze around U ≈ 8 (7.5 <= U <= 9.5)
                        if 7.5 <= u_val <= 9.5
                            push!(u_values, u_val)
                            push!(pred_rand_ratios, rand_ratio)
                            push!(rand_baseline_ratios, rand_baseline)
                        end
                    catch e
                        # Skip header or malformed rows
                    end
                elseif length(tokens) > 0 && occursin("Done.", line)
                    break
                end
            end
        end
        
        if !isempty(pred_rand_ratios)
            results[folder_name] = Dict(
                "mean_pred_rand" => mean(pred_rand_ratios),
                "max_pred_rand" => maximum(pred_rand_ratios),
                "mean_rand_baseline" => mean(rand_baseline_ratios),
                "max_rand_baseline" => maximum(rand_baseline_ratios)
            )
        end
    end
    
    return Dict(
        "weighting" => weighting,
        "name" => name,
        "results" => results,
        "filepath" => filepath
    )
end

# Find all log files
log_dir = "/home/jek354/research/ML-signproblem/experimenting/ed/logs"
log_files = readdir(log_dir; join=true)

parsed_runs = []
for f in log_files
    if endswith(f, ".log") && occursin("system_scaling_", f)
        parsed = parse_log_file(f)
        if !isnothing(parsed)
            push!(parsed_runs, parsed)
        end
    end
end

println("Found $(length(parsed_runs)) relevant experiment logs.")

if isempty(parsed_runs)
    println("No logs found. Exiting.")
    exit(0)
end

# Keep only the best run (with the most evaluation results) for each weighting scheme
best_runs = Dict{String, Any}()
for run in parsed_runs
    w = run["weighting"]
    if !haskey(best_runs, w) || length(run["results"]) > length(best_runs[w]["results"])
        best_runs[w] = run
    end
end
parsed_runs = collect(values(best_runs))

# Generate Markdown Comparison
md_content = """# Weighting Scheme Comparison (Target U ≈ 8)

This table compares the performance of the alternative optimized loss-based weighting schemes against the baseline.
A lower **Mean Pred/Rand** ratio indicates better generalization to that target dataset. Ratios below 1.0 (marked in bold) mean the network outperformed the random baseline.

## Summary Table

"""

# Get all unique datasets
datasets = Set{String}()
for run in parsed_runs
    for ds in keys(run["results"])
        push!(datasets, ds)
    end
end
sorted_ds = sort(collect(datasets))

# Build headers
header1 = "| Weighting Scheme | Model Name |" * join([" $(ds) Mean (Max) |" for ds in sorted_ds], "")
header2 = "| :--- | :--- |" * join([" :---: |" for ds in sorted_ds], "")
md_content *= header1 * "\n" * header2 * "\n"

for run in parsed_runs
    row = "| $(run["weighting"]) | `$(run["name"])` |"
    for ds in sorted_ds
        if haskey(run["results"], ds)
            r = run["results"][ds]
            mean_str = @sprintf("%.4f", r["mean_pred_rand"])
            max_str = @sprintf("%.4f", r["max_pred_rand"])
            
            # Highlight if mean < 1.0
            if r["mean_pred_rand"] < 1.0
                row *= " **$(mean_str)** ($(max_str)) |"
            else
                row *= " $(mean_str) ($(max_str)) |"
            end
        else
            row *= " N/A |"
        end
    end
    global md_content *= row * "\n"
end

# Write comparison to file
out_path = "/home/jek354/.gemini/antigravity-ide/brain/e25eecbc-5752-410b-8558-d8280ef464cc/walkthrough_experiments.md"
write(out_path, md_content)
println("Wrote comparison to: $out_path")
println("\n" * md_content)
