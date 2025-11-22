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
    save_energy_with_metadata(folder::String, dict::Dict)

Save a dictionary containing meta_data and other values.

Behavior:
1. If folder does NOT exist:
       - create it
       - save dict as `meta_data_and_E.jld2` inside it
2. If folder exists:
       - if meta_data_and_E.jld2 exists:
             * if meta_data matches → do nothing
             * if meta_data differs → create new incremented folder and save there
       - if meta_data_and_E.jld2 does NOT exist:
             * save dict normally

Returns the final save path.
"""
function save_energy_with_metadata(folder::String, dict::Dict)

    file_name = "meta_data_and_E.jld2"
    file_path = joinpath(folder, file_name)

    # Helper: increment folder name (folder, folder_1, folder_2, ...)
    function increment_folder_name(base::String)
        new = base
        i = 1
        while isdir(new)
            new = base * "_" * string(i)
            i += 1
        end
        return new
    end

    # --- Case 1: folder does not exist ---
    if !isdir(folder)
        mkpath(folder)
        @save file_path dict
        return file_path
    end

    # --- Case 2: folder exists ---

    # If file does NOT exist → save normally
    if !isfile(file_path)
        @save file_path dict
        return file_path
    end

    # File exists → load and compare metadata
    existing = load(file_path)

    if !haskey(existing, "meta_data")
        error("Existing file in folder $folder does not contain meta_data.")
    end

    # Compare metadata
    if existing["meta_data"] == dict["meta_data"]
        # meta_data match → do nothing and finish
        return file_path
    else
        # meta_data mismatch → create incremented folder
        new_folder = increment_folder_name(folder)
        mkpath(new_folder)
        new_path = joinpath(new_folder, file_name)
        @save new_path dict
        return new_path
    end
end

"""
    save_incrementing(folder::String, filename::String, dict::Dict)

Save `dict` into an existing folder.

Rules:
1. If the folder does NOT exist → error.
2. If `filename.jld2` does NOT exist → save directly.
3. If it DOES exist → increment the name (filename_1, filename_2, …) until free, then save.

This version uses ONLY JLD2, no FileIO.
"""
function save_dictionary(folder::String, filename::String, dict::Dict)

    # Ensure folder exists
    if !isdir(folder)
        error("Folder does not exist: $folder")
    end

    # Base path
    file_path = joinpath(folder, filename * ".jld2")

    # If file doesn't exist, save immediately
    if !isfile(file_path)
        @save file_path dict
        return file_path
    end

    # Otherwise increment filename until one is free
    base = filename
    i = 1
    new_path = joinpath(folder, base * "_" * string(i) * ".jld2")

    while isfile(new_path)
        i += 1
        new_path = joinpath(folder, base * "_" * string(i) * ".jld2")
    end

    # Save to the incremented filename
    @save new_path dict

    return new_path
end


"""
    load_saved_dict(filename::AbstractString) -> Dict

Load the dictionary saved using `save_with_metadata`.

Returns the dictionary stored under the key `"dict"`.

Throws an error if the file does not exist or if it does not contain `"dict"`.
"""
function load_saved_dict(filename::AbstractString)
    @assert isfile(filename) "File does not exist: $filename"

    return JLD2.jldopen(filename, "r") do file
        if !haskey(file, "dict")
            error("File '$filename' does not contain a saved dictionary under key \"dict\".")
        end
        file["dict"]
    end
end