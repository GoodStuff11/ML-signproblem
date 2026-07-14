# testing/compile_detailed_table.jl
using Statistics
using Printf

# Maps weighting scheme flags to simplified math formulas
const WEIGHTING_FORMULAS = Dict(
    "loss_power_mild" => "\$L^{-0.07}\$",
    "loss_power_std"  => "\$L^{-0.12}\$",
    "loss_mild"        => "\$(-\\log_{10} L)^{1.15}\$",
    "loss_std"         => "\$(-\\log_{10} L)^2\$",
    "low_u_mild"       => "\$U^{-0.4}\$",
    "low_u"            => "\$U^{-1.0}\$",
    "low_u_strong"     => "\$U^{-2.0}\$",
    "low_u_very_strong"=> "\$U^{-3.0}\$",
    "uniform"          => "\$1.0\$",
    "focus_u8"         => "\$\\exp(-5(U-8)^2)\$",
    "high_u"           => "\$U^{1.0}\$",
    "high_u_strong"    => "\$U^{2.0}\$"
)

# Maps folder set flags to compact training system descriptions
const SYSTEM_SIZES = Dict(
    "small_only"       => "(2,2) 3x2, 2x2",
    "exclude_2x2"      => "(3,3) 3x2, 3x3; (4,4) 4x2",
    "large_only"       => "(3,3) 3x3; (4,4) 4x2",
    "small_nonsplit"   => "(3,3) 3x2 dupl",
    "medium_nonsplit"  => "(3,3) 3x2 dupl; (4,4) 4x2 dupl",
    "mixed_nonsplit"   => "(3,3) 3x2, 4x2; (4,4) 4x2 dupl",
    "square_nonsplit"  => "(3,3), (4,4), (4,5) 3x3",
    "square_pure"      => "(3,3), (4,5) 3x3",
    "square_extended"  => "(3,3), (4,5) 3x3 ext",
    "square_with_2x2"  => "(3,3), (4,5) 3x3; (2,2) 2x2",
    "square_and_rect"  => "(3,3), (4,4), (4,5) 3x3; (3,3) 3x2 dupl",
    "multiscale_3x3"   => "(3,3), (4,4), (4,5) 3x3; (4,4), (3,3) rect",
    "all"              => "(3,3) 3x2, 3x3; (2,2) 3x2, 2x2; (4,4) 4x2"
)

function parse_detailed_log(filepath)
    if !isfile(filepath)
        return nothing
    end
    content = read(filepath, String)
    
    # 1. Weighting Scheme
    weighting_m = match(r"Training NN with weighting scheme:\s+(\w+)", content)
    if isnothing(weighting_m)
        return nothing
    end
    w_flag = weighting_m[1]
    formula = get(WEIGHTING_FORMULAS, w_flag, w_flag)
    
    # 2. Network name
    name_m = match(r"No existing neural network found at trained_neural_networks/trained_neural_network_([\w\_]+)\.jld2", content)
    if isnothing(name_m)
        name_m = match(r"Loaded neural network strategy from:\s+trained_neural_networks/trained_neural_network_([\w\_]+)\.jld2", content)
    end
    net_name = isnothing(name_m) ? "unknown" : name_m[1]
    
    # 3. Architecture
    arch_m = match(r"Training NN with architecture parameters: base_hidden=\[([\d\s,]+)\], embed_dim=(\d+), context_hidden=\[([\d\s,]+)\], scale_hidden=\[([\d\s,]+)\]", content)
    arch_str = if !isnothing(arch_m)
        base = replace(arch_m[1], " " => "")
        embed = arch_m[2]
        ctx = replace(arch_m[3], " " => "")
        scale = replace(arch_m[4], " " => "")
        "[$base]-$embed-[$ctx]-[$scale]"
    else
        # Default architecture from system_scaling.jl
        "[128,128]-64-[64,32]-[32,16]"
    end
    
    # 4. Training config: U-range
    urange_m = match(r"Training NN with U index range:\s+([\d:]+)", content)
    u_range = isnothing(urange_m) ? "2:52" : urange_m[1]
    
    # 5. Training config: folder set
    folderset_m = match(r"Training NN with folder set:\s+(\w+)", content)
    folder_flag = isnothing(folderset_m) ? "all" : folderset_m[1]
    systems = get(SYSTEM_SIZES, folder_flag, folder_flag)
    
    config_str = "U: $u_range; sizes: $systems"
    
    # Parse folder evaluations
    sections = split(content, "Testing on folder: data/")
    if length(sections) <= 1
        return nothing # not evaluated at all
    end
    
    folder_results = Dict{String, Any}()
    
    # For overall fraction: count all data points across all folders and all U values
    total_points = 0
    better_points = 0
    
    for i in 2:length(sections)
        lines = split(sections[i], "\n")
        folder_name = strip(lines[1])
        
        table_started = false
        u_values = Float64[]
        pred_baseline_ratios = Float64[]
        
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
                        pred_baseline = parse(Float64, tokens[4]) # pred/baseline
                        
                        # Add to overall statistics
                        total_points += 1
                        if pred_baseline < 1.0
                            better_points += 1
                        end
                        
                        # Filter for U around 8 (7.5 <= U <= 9.5)
                        if 7.5 <= u_val <= 9.5
                            push!(u_values, u_val)
                            push!(pred_baseline_ratios, pred_baseline)
                        end
                    catch e
                    end
                elseif length(tokens) > 0 && occursin("Done.", line)
                    break
                end
            end
        end
        
        if !isempty(pred_baseline_ratios)
            folder_results[folder_name] = Dict(
                "mean" => mean(pred_baseline_ratios),
                "max" => maximum(pred_baseline_ratios)
            )
        end
    end
    
    if isempty(folder_results)
        return nothing
    end
    
    # Calculate U≈8 metrics
    folder_means = Float64[]
    pass_folders_count = 0
    for (f, res) in folder_results
        push!(folder_means, res["mean"])
        if res["mean"] < 1.0
            pass_folders_count += 1
        end
    end
    
    pass_rate_str = "$(pass_folders_count)/$(length(folder_results))"
    best_mean = minimum(folder_means)
    worst_mean = maximum(folder_means)
    best_worst_str = @sprintf("%.3f / %.3f", best_mean, worst_mean)
    
    overall_fraction = total_points == 0 ? 0.0 : (better_points / total_points)
    
    return Dict(
        "weighting" => formula,
        "name" => net_name,
        "arch" => arch_str,
        "config" => config_str,
        "folder_results" => folder_results,
        "pass_rate" => pass_rate_str,
        "best_worst" => best_worst_str,
        "overall_fraction" => overall_fraction,
        "filepath" => basename(filepath)
    )
end

log_dir = "/home/jek354/research/ML-signproblem/experimenting/ed/logs"
log_files = readdir(log_dir; join=true)

runs = []
for f in log_files
    if endswith(f, ".log") && occursin("system_scaling_", f)
        parsed = parse_detailed_log(f)
        if !isnothing(parsed)
            push!(runs, parsed)
        end
    end
end

# Filter out runs that don't have results for N=(4, 4)_3x3_2
valid_runs = []
for r in runs
    if haskey(r["folder_results"], "N=(4, 4)_3x3_2")
        push!(valid_runs, r)
    end
end

# Sort by Mean Pred/Baseline ratio on N=(4, 4)_3x3_2 (ascending)
sort!(valid_runs, by=x->x["folder_results"]["N=(4, 4)_3x3_2"]["mean"])

# Generate markdown table
md = """| Weighting Formula | Network Architecture (Hyperparams) | Training Config (U-range; Train sizes) | 4x3 (3,3) Mean (Max) | 3x3 (4,4) Mean (Max) | U≈8 Pass Rate | U≈8 Best / Worst Mean | Overall Pass Rate |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
"""

for r in valid_runs
    ds1 = "N=(3, 3)_4x3"
    ds2 = "N=(4, 4)_3x3_2"
    
    val_str1 = "N/A"
    if haskey(r["folder_results"], ds1)
        val_str1 = @sprintf("**%.4f** (%.4f)", r["folder_results"][ds1]["mean"], r["folder_results"][ds1]["max"])
    end
    
    val_str2 = @sprintf("**%.4f** (%.4f)", r["folder_results"][ds2]["mean"], r["folder_results"][ds2]["max"])
    
    overall_pct = @sprintf("%.1f%%", r["overall_fraction"] * 100)
    
    global md *= "| $(r["weighting"]) | `$(r["arch"])` | $(r["config"]) | $val_str1 | $val_str2 | $(r["pass_rate"]) | $(r["best_worst"]) | $overall_pct |\n"
end

println("COMPILE COMPLETE!")
println(md)
write("testing/compiled_markdown_table.md", md)
