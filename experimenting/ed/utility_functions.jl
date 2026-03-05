# don't modify this file
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
