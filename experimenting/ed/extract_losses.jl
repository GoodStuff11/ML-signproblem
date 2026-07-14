using JLD2
using DataFrames
using Markdown
using Statistics

function parse_loss_log(path)
    rows = []
    for line in eachline(path)
        isempty(strip(line)) && continue
        pairs = split(strip(line), ' ')
        row = Dict{String,Float64}()
        for p in pairs
            k, v = split(p, '=')
            row[k] = parse(Float64, v)
        end
        push!(rows, row)
    end
    return DataFrame(rows)
end


"""
    extract_losses_from_file(filepath::String; order_idx::Int=1)

Extracts and returns the optimization loss history from the JLD2 results file, along with the opposite metric (energy or overlap).
If multi-start was used, it retrieves the trajectory of the best-performing start
and prepends/merges it into the subsequent final optimization losses.

Returns:
- `all_losses::Vector{Float64}`: The concatenated sequence of losses.
- `was_multistart::Bool`: Whether multi-start was run and merged.
- `opposite_val::Float64`: The opposite loss metric value after optimization (energy or overlap).
"""
function extract_losses_from_file(filepath::String; order_idx::Int=1)
    if !isfile(filepath)
        error("File not found: $filepath")
    end

    # Load the JLD2 data
    data = load(filepath)
    if !haskey(data, "dict")
        error("JLD2 file does not contain 'dict' key.")
    end
    dic = data["dict"]

    if !haskey(dic, "metrics")
        error("Dictionary does not contain 'metrics' key.")
    end
    metrics = dic["metrics"]

    # Extract final optimization losses for the specified order index
    if !haskey(metrics, "optimization_losses") || isempty(metrics["optimization_losses"])
        error("No optimization losses found in metrics.")
    end

    final_losses = metrics["optimization_losses"][order_idx]

    # Check if multistart was run
    was_multistart = false
    best_start_losses = Float64[]

    if haskey(metrics, "multistart_losses") && length(metrics["multistart_losses"]) >= order_idx
        ms_list = metrics["multistart_losses"][order_idx]
        if !isempty(ms_list)
            best_idx = metrics["best_start_idx"][order_idx]
            if best_idx > 0 && best_idx <= length(ms_list)
                best_start_losses = ms_list[best_idx]
                was_multistart = true
            end
        end
    end

    # Concatenate the best start losses and the final optimization losses
    all_losses = vcat(best_start_losses, final_losses)

    # Extract opposite loss metric if available
    opposite_val = NaN
    if haskey(metrics, "energy") && !isempty(metrics["energy"])
        opposite_val = metrics["energy"][min(end, order_idx + 1)]
    elseif haskey(metrics, "overlap") && !isempty(metrics["overlap"])
        opposite_val = metrics["overlap"][min(end, order_idx + 1)]
    end

    return all_losses, was_multistart, opposite_val
end

"""
    collect_scaling_results(logs_dir::String; subfolder=nothing, u_values=nothing, nn_labels=nothing)

Searches recursively through the `logs_dir` for `system_scaling_*.log` files, parses their
adjoint loss evaluation tables, and returns a consolidated `DataFrame` of results.

# Arguments
- `logs_dir::String`: Path to the logs directory.
- `subfolder::Union{String, Nothing}`: Optional filter for the system folder (e.g. `"N=(4, 4)_4x2"`).
- `u_values::Union{AbstractVector{<:Real}, Nothing}`: Optional filter for interaction strengths (U values).
- `nn_labels::Union{AbstractVector{String}, Nothing}`: Optional filter for the neural network name.

# Returns
- `df::DataFrame`: Consolidate table containing columns:
    - `"log_file"`: The source log filename.
    - `"nn_name"`: The identified neural network strategy name.
    - `"folder"`: The tested system folder name.
    - And all parsed table columns, including `"U-value"`, `"True E"`, `"Base loss"`, `"Base E"`, `"Pred loss"`, `"Pred E"`, `"pr/ba loss"`, `"pr-ba E"`, `"rand min loss"`, `"rand min E"`, etc.
"""
function collect_scaling_results(
    logs_dir::String;
    subfolder::Union{String,Nothing}=nothing,
    u_values::Union{AbstractVector{<:Real},Nothing}=nothing,
    nn_labels::Union{AbstractVector{String},Nothing}=nothing
)
    # Define the 22 columns present in the log file tables (including energy difference metrics)
    column_names = [
        "U-value",
        "True E",
        "Base loss",
        "Base E",
        "Pred loss",
        "Pred E",
        "pr/ba loss",
        "pr-ba E",
        "MeanAbs Stor",
        "MeanAbs Pred",
        "RMS Stored",
        "RMS Pred",
        "rand min loss",
        "rand min E",
        "mean loss",
        "mean E",
        "max loss",
        "max E",
        "pr/rd loss",
        "pr-rd E",
        "rd/ba loss",
        "rd-ba E"
    ]

    # Initialize the DataFrame with source metadata columns and the parsed columns
    df = DataFrame(
        log_file=String[],
        nn_name=String[],
        folder=String[]
    )
    for col in column_names
        df[!, col] = Float64[]
    end

    if !isdir(logs_dir)
        @warn "Logs directory not found: $logs_dir"
        return df
    end

    # Find all system_scaling_*.log files recursively
    log_files = String[]
    for (root, dirs, files) in walkdir(logs_dir)
        for file in files
            if startswith(file, "system_scaling_") && endswith(file, ".log")
                push!(log_files, joinpath(root, file))
            end
        end
    end

    for filepath in log_files
        lines = try
            readlines(filepath)
        catch e
            @warn "Failed to read $filepath: $e"
            continue
        end

        # 1. Parse the neural network strategy/name
        nn_name = "unknown"
        for line in lines
            m = match(r"trained_neural_network_([a-zA-Z0-9_\- \(\),]+)\.jld2", line)
            if m !== nothing
                nn_name = m.captures[1]
                nn_name = replace(nn_name, ".jld2" => "")
            end
        end

        # Filter by NN labels if specified (exact matching)
        if nn_labels !== nothing
            matched_nn = false
            for target in nn_labels
                if nn_name == target
                    matched_nn = true
                    break
                end
            end
            if !matched_nn
                continue
            end
        end

        # 2. Parse table entries
        current_folder = ""
        in_table = false

        for line in lines
            trimmed = strip(line)

            # Check for folder demarcation (e.g. "Testing on folder: data/N=(4, 4)_4x2")
            m_folder = match(r"Testing on folder:\s*(?:data/)?(.*)", trimmed)
            if m_folder !== nothing
                current_folder = m_folder.captures[1]
                in_table = false
                continue
            end

            # Skip if we are filtering by subfolder and this is not the target subfolder
            if subfolder !== nothing && !isempty(current_folder)
                if current_folder != subfolder && current_folder != "data/$subfolder"
                    continue
                end
            end

            # Check for table boundaries (robust matching for and/or energy logs)
            if occursin("Evaluating adjoint_loss", trimmed) && occursin("per U-index", trimmed)
                in_table = false
                continue
            end

            if startswith(trimmed, "----")
                if !isempty(current_folder) && (subfolder === nothing || current_folder == subfolder || current_folder == "data/$subfolder")
                    in_table = true
                end
                continue
            end

            if in_table
                if isempty(trimmed) || startswith(trimmed, "Done") || startswith(trimmed, "===")
                    in_table = false
                    continue
                end

                parts = split(trimmed)
                if length(parts) == 22
                    vals = map(x -> tryparse(Float64, x), parts)
                    if any(x -> x === nothing, vals)
                        continue
                    end

                    u_val = vals[1]

                    # Filter by U value if specified (tolerance check)
                    if u_values !== nothing
                        matched_u = any(isapprox(u_val, target_u, atol=1e-4, rtol=1e-4) for target_u in u_values)
                        if !matched_u
                            continue
                        end
                    end

                    row_data = (
                        basename(filepath),
                        nn_name,
                        current_folder,
                        vals...
                    )
                    push!(df, row_data)
                elseif length(parts) == 13
                    vals = map(x -> tryparse(Float64, x), parts)
                    if any(x -> x === nothing, vals)
                        continue
                    end

                    u_val = vals[1]

                    # Filter by U value if specified (tolerance check)
                    if u_values !== nothing
                        matched_u = any(isapprox(u_val, target_u, atol=1e-4, rtol=1e-4) for target_u in u_values)
                        if !matched_u
                            continue
                        end
                    end

                    row_vals = [
                        vals[1],  # U-value
                        NaN,      # True E
                        vals[2],  # Base loss
                        NaN,      # Base E
                        vals[3],  # Pred loss
                        NaN,      # Pred E
                        vals[4],  # pr/ba loss
                        NaN,      # pr-ba E
                        vals[5],  # MeanAbs Stor
                        vals[6],  # MeanAbs Pred
                        vals[7],  # RMS Stored
                        vals[8],  # RMS Pred
                        vals[9],  # rand min loss
                        NaN,      # rand min E
                        vals[10], # mean loss
                        NaN,      # mean E
                        vals[11], # max loss
                        NaN,      # max E
                        vals[12], # pr/rd loss
                        NaN,      # pr-rd E
                        vals[13], # rd/ba loss
                        NaN       # rd-ba E
                    ]
                    row_data = (
                        basename(filepath),
                        nn_name,
                        current_folder,
                        row_vals...
                    )
                    push!(df, row_data)
                end
            end
        end
    end

    return df
end

"""
    dataframe_to_markdown(df::DataFrame) -> String

Formats a `DataFrame` into a Markdown table string using Julia's built-in `Markdown` standard library.
Aligns numeric columns to the right and other columns to the left.

In notebook, use `println(dataframe_to_markdown(df))` to display properly
"""
function dataframe_to_markdown(df::DataFrame)
    cols = names(df)
    nr = nrow(df)
    nc = ncol(df)

    if nr == 0
        return "No data in DataFrame."
    end

    # Content holds the header row followed by all data rows
    content = Vector{Vector{Any}}()

    # 1. Header Row
    push!(content, Any[string(c) for c in cols])

    # 2. Data Rows
    for i in 1:nr
        push!(content, Any[df[i, j] for j in 1:nc])
    end

    # 3. Alignments (Right-align numeric columns, left-align others)
    alignments = Symbol[]
    for j in 1:nc
        col_type = eltype(df[!, j])
        if col_type <: Number
            push!(alignments, :r)
        else
            push!(alignments, :l)
        end
    end

    # 4. Construct Table and render to Markdown String
    tbl = Markdown.Table(content, alignments)
    md = Markdown.MD(tbl)
    return sprint(Markdown.print, md)
end

"""
    merge_losses_and_scaling(df_scaling::DataFrame, df_losses::DataFrame; 
                             nn_rep::Union{String, Nothing}="sq_uniform",
                             cols::AbstractVector=["rand min loss", "mean loss", "max loss"]) -> DataFrame

Merges the scaling results table and the optimization losses table. 
Creates the specified new columns (`cols`) matched on `folder` 
and `U_val` (rounded to 2 decimal places). The values are only populated when the row's 
`initial/optimized` column is `"initial"`.

If `nn_rep` is a `String` (e.g., `"sq_uniform"`), it uses the random baseline losses from that 
representative model run. If `nn_rep` is `nothing`, it averages the baseline losses across all models.
"""
function merge_losses_and_scaling(
    df_scaling::DataFrame,
    df_losses::DataFrame;
    nn_rep::Union{String,Nothing}="sq_uniform",
    cols::AbstractVector=["rand min loss", "mean loss", "max loss"]
)
    # 1. Prepare copy of inputs to avoid mutating the original dataframes
    df_sc = copy(df_scaling)
    df_ls = copy(df_losses)

    # Ensure all columns to be added are treated as strings
    cols_str = string.(cols)

    # 2. Add U-matching helper column (rounding to 2 decimal places to match 3.888 with 3.89)
    df_sc[!, "U_match"] = [round(u; digits=2) for u in df_sc[!, "U-va"]]
    df_ls[!, "U_match"] = [round(u; digits=2) for u in df_ls[!, "U_val"]]

    # 3. Handle model representation / aggregation for baseline losses
    if nn_rep !== nothing
        df_sc_filtered = filter(row -> row.nn_name == nn_rep, df_sc)
    else
        # Aggregate across different NNs by taking the mean of the specified columns
        df_sc_filtered = combine(
            groupby(df_sc, ["folder", "U_match"]),
            [col => mean => col for col in cols_str]...
        )
    end

    # 4. Join the dataframes on folder and the U_match column
    keep_cols = vcat(["folder", "U_match"], cols_str)
    merged = leftjoin(df_ls, df_sc_filtered[:, keep_cols], on=["folder", "U_match"])

    # Allow missing values for the custom columns so we can clear optimized rows
    allowmissing!(merged, cols_str)

    # 5. Clear values for "optimized" rows (only keep when initial/optimized is "initial")
    for i in 1:nrow(merged)
        if merged[i, "initial/optimized"] != "initial"
            for col in cols_str
                merged[i, col] = missing
            end
        end
    end

    # Clean up the matching helper column
    select!(merged, Not("U_match"))

    return merged
end


