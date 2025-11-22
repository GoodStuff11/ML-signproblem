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
    save_with_metadata(dict::Dict, filename::AbstractString)

Save a dictionary to a JLD2 file with metadata checking and append behavior.

Dictionary format:
- Must contain the key `"meta_data"`
- All other keys contain *vectors* of data that should be appended on match.
- If `filename` does not exist → save entire dictionary.
- If `filename` exists:
    * metadata matches → append to the vector keys
    * metadata does not match → generate new filename filename_1, filename_2, ...
"""
function save_with_metadata(dict::Dict, filename::AbstractString)
    @assert haskey(dict, "meta_data") "Dictionary must contain `\"meta_data\"`."

    # ------------------------------------------------------------
    # Case 1: The file does not exist — save new dictionary
    # ------------------------------------------------------------
    if !isfile(filename)
        println("Saving new file: $filename")
        JLD2.@save filename dict
        return filename
    end

    # ------------------------------------------------------------
    # Case 2: File exists — load and compare metadata
    # ------------------------------------------------------------
    existing = JLD2.jldopen(filename, "r") do file
        file["dict"]
    end

    if existing["meta_data"] == dict["meta_data"]
        println("Metadata matches — appending data to $filename")

        # Append vector data
        JLD2.jldopen(filename, "a") do file
            d = file["dict"]              # load stored dictionary into memory

            # mutate the in-memory dict
            for (k, v_new) in dict
                k == "meta_data" && continue
                d[k] = vcat(d[k], v_new)
            end
            delete!(file, "dict")
            file["dict"] = d              # WRITE BACK to the file  <-- crucial
        end

        return filename
    end

    # ------------------------------------------------------------
    # Case 3: Metadata mismatch — create new incrementing filename
    # ------------------------------------------------------------
    println("Metadata mismatch — creating new file")

    # Extract directory, stem, and extension manually
    dir = dirname(filename)
    base = basename(filename)
    stem, ext = splitext(base)

    # Construct incrementing new filename
    i = 1
    new_filename = joinpath(dir, stem * "_" * string(i) * ext)

    while isfile(new_filename)
        i += 1
        new_filename = joinpath(dir, stem * "_" * string(i) * ext)
    end

    println("Saving to new file: $new_filename")
    JLD2.@save new_filename dict

    return new_filename
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

function safe_load_dict(filename::String; retries=20, delay=0.2)
    for i in 1:retries
        if !isfile(filename)
            error("File $filename does not exist")
        end
        dict = jldopen(filename, "r") do f
            haskey(f, "dict") ? f["dict"] : nothing
        end
        if dict !== nothing
            return dict
        end
        sleep(delay)
    end
    error("Could not read dict from $filename after $retries retries — file may be in use")
end