function serialize_data(obj)
    if obj isa AbstractRange
        return collect(obj)
    elseif obj isa Dict
        return Dict(k => serialize_data(v) for (k, v) in obj)
    elseif obj isa Vector
        return [serialize_data(x) for x in obj]
    else
        return obj
    end
end

function save_json(data::Any, folder::String, filename::String)
    #filename should end with .json
    open(joinpath(folder, filename), "w") do io
        JSON.print(io, serialize_data(data), 4)
    end
end


function load_json_folder(path::String)
    data_dict = Dict{String, Any}()

    for entry in readdir(path)
        if endswith(entry, ".json")
            full_path = joinpath(path, entry)
            name = splitext(entry)[1]  # Remove .json extension
            open(full_path, "r") do io
                data_dict[name] = JSON.parse(IOBuffer(read(io, String)))
            end
        end
    end

    return data_dict
end

function append_to_json_files(new_data::Dict{String, Any}, folder::String)
    new_data = serialize_data(new_data)
    @assert haskey(new_data, "meta_data") "Missing 'meta_data' key in new_data"

    new_meta = new_data["meta_data"]
    final_folder = folder

    function meta_matches(path::String)
        meta_file = joinpath(path, "meta_data.json")
        if !isfile(meta_file)
            return false
        end
        existing_meta = open(meta_file, "r") do io
            JSON.parse(IOBuffer(read(io, String)))
        end
        
        return existing_meta == new_meta
    end

    # Check if we need to switch to a different folder due to meta mismatch
    if isfile(joinpath(folder, "meta_data.json"))
        existing_meta = open(joinpath(folder, "meta_data.json"), "r") do io
            JSON.parse(IOBuffer(read(io, String)))
        end

        if existing_meta != new_meta
            # Search for next available matching or new folder
            i = 1
            while true
                candidate = folder * "_$i"
                if !isdir(candidate)
                    mkpath(candidate)
                    final_folder = candidate
                    break
                elseif meta_matches(candidate)
                    final_folder = candidate
                    break
                end
                i += 1
            end
        end
    else
        # If no meta_data.json yet, create the folder if needed
        mkpath(folder)
    end

    # Proceed to write/append all keys to final_folder
    for (key, value) in new_data
        # if key == "" || key == ""
        #     file_path = joinpath(final_folder, key * ".jld2")
        #     existing = nothing

        # else
            file_path = joinpath(final_folder, key * ".json")
            existing = nothing

            if isfile(file_path)
                open(file_path, "r") do io
                    existing = JSON.parse(IOBuffer(read(io, String)))
                end
            end

            # Merge/append logic
            updated = if existing === nothing
                value
            elseif isa(existing, Vector) && isa(value, Vector)
                vcat(existing, value)
            elseif isa(existing, Dict) && isa(value, Dict)
                merge(existing, value)
            else
                error("Incompatible types for key '$key': can't append $(typeof(value)) to $(typeof(existing))")
            end

            open(file_path, "w") do io
                JSON.print(io, updated, 4)
            end
        # end
    end

    return final_folder
end

"""
    save_dict_with_metadata(dict, folder::String, filename::String)

Saves `dict` as a JLD2 file inside `folder/filename.jld2`, following the rules:

1. If `folder` does not exist: create it and write the file.
2. If `folder` exists:
   - If `filename.jld2` does not exist: write the file.
   - If it exists:
       * Compare metadata:
           identical → create a new file with incremented name.
           different → create a new folder with incremented name.

"""
function save_with_metadata(dict, folder::String, filename::String)

    # --- Helper functions ---
    # Increment names: base, base_1, base_2, ...
    function increment_name(path::String)
        base = path
        i = 1
        while ispath(path)
            path = base * "_" * string(i)
            i += 1
        end
        return path
    end

    function increment_file(folder::String, filename::String)
        base = filename
        i = 1
        newfile = joinpath(folder, filename * ".jld2")
        while isfile(newfile)
            newfile = joinpath(folder, base * "_" * string(i) * ".jld2")
            i += 1
        end
        return newfile
    end

    # --- Ensure folder exists ---
    if !isdir(folder)
        mkpath(folder)
        @info "Folder created: $folder"
        filepath = joinpath(folder, filename * ".jld2")
        @save filepath dict
        return filepath
    end

    # Folder exists
    filepath = joinpath(folder, filename * ".jld2")

    # If file does not exist, write it
    if !isfile(filepath)
        @save filepath dict
        return filepath
    end

    # File exists → load and compare metadata
    existing = load(filepath)["dict"]
    if !haskey(existing, "meta_data")
        error("Existing file does not contain meta_data key.")
    end

    same_meta = existing["meta_data"] == dict["meta_data"]

    if same_meta
        # Metadata matches → increment filename
        newfile = increment_file(folder, filename)
        @info "Metadata matches. Saving to new file: $newfile"
        @save newfile dict
        return newfile
    else
        # Metadata differs → increment folder
        newfolder = increment_name(folder)
        mkpath(newfolder)
        newfile = joinpath(newfolder, filename * ".jld2")
        @info "Metadata differs. Creating new folder: $newfolder"
        @save newfile dict
        return newfile
    end
end
"""
    merge_jld2_folder(folder; omit_keys = String[])

Merge all JLD2 files inside `folder` into a single dictionary.

Rules:
- All files must contain `meta_data`, and all meta_data must match.
- All non-omitted keys (except meta_data) must be vectors and will be concatenated.
- The merged dictionary has:
      merged["meta_data"] = shared_meta
      merged[key] = vcat(all vectors across files…)

Arguments:
- `folder`      : Folder containing JLD2 files.
- `omit_keys`   : Vector of keys to exclude from merging. (Default: none)
"""
function merge_jld2_folder(folder::String; omit_keys=String[])
    # Locate all JLD2 files
    files = filter(f -> endswith(f, ".jld2"), readdir(folder, join=true))
    isempty(files) && error("No JLD2 files found in folder: $folder")

    merged = Dict{String,Any}()
    shared_meta = nothing
    first_file = true

    for file in files
        dict = load(file)["dict"]

        # Ensure meta_data exists
        @assert haskey(dict, "meta_data") "File $file has no meta_data key."

        # Establish or validate shared metadata
        if first_file
            shared_meta = dict["meta_data"]
            merged["meta_data"] = shared_meta
            first_file = false
        else
            @assert dict["meta_data"] == shared_meta "meta_data mismatch in file: $file"
        end

        # Merge all non-omitted keys except meta_data
        for (k, v) in dict
            if k == "meta_data" || k in omit_keys
                continue
            end

            if haskey(merged, k)
                # append to existing vector
                try
                    merged[k] = vcat(merged[k], v)
                catch
                    error("Key '$k' cannot be appended. Check that all values are vectors. Error in file: $file")
                end
            else
                # initialize
                merged[k] = v
            end
        end
    end

    return merged
end