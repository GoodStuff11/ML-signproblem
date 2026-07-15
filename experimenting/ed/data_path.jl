"""
data_path.jl

Resolves the root data directory for all ED scripts.

Priority order:
  1. Environment variable ED_DATA_ROOT (highest priority)
  2. File `data_config.txt` in the same directory as this file (one path per file)
  3. Fallback: "data" subdirectory relative to this file (legacy behaviour)

To configure, either:
  - Set the environment variable:   export ED_DATA_ROOT=/path/to/data
  - Or edit data_config.txt:        echo /path/to/data > experimenting/ed/data_config.txt
"""

const _CONFIG_FILE = joinpath(@__DIR__, "data_config.txt")

"""
    get_data_root() -> String

Return the root directory that contains all ED data sub-folders (e.g. `N=(2,2)_2x2/`).
Reads from (in priority order):
  1. Environment variable `ED_DATA_ROOT`
  2. `data_config.txt` next to this file
  3. `"data"` subdirectory relative to this script (legacy fallback)
"""
function get_data_root()::String
    # 1. Environment variable
    env_val = get(ENV, "ED_DATA_ROOT", "")
    if !isempty(env_val)
        return env_val
    end

    # 2. Config file
    if isfile(_CONFIG_FILE)
        line = strip(readline(_CONFIG_FILE))
        if !isempty(line)
            return line
        end
    end

    # 3. Legacy fallback: "data" relative to this script
    return joinpath(@__DIR__, "data")
end

"""
    data_folder(subfolder::String) -> String

Join the data root with `subfolder`, e.g. `data_folder("N=(2, 2)_2x2")`.
If `subfolder` is already an absolute path it is returned unchanged.
"""
function data_folder(subfolder::String)::String
    isabspath(subfolder) && return subfolder
    return joinpath(get_data_root(), subfolder)
end
