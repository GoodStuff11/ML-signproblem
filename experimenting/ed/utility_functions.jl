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
    end

    return final_folder
end