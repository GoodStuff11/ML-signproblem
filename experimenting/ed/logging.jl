"""
    TeeStream

An IO stream that forwards all writes to two underlying streams simultaneously.
Used to mirror stdout/stderr to a log file while still printing to the terminal.
"""
struct TeeStream <: IO
    primary::IO
    secondary::IO
end

Base.write(t::TeeStream, x::UInt8) = (write(t.primary, x); write(t.secondary, x); 1)
function Base.write(t::TeeStream, x::AbstractVector{UInt8})
    write(t.primary, x)
    n = write(t.secondary, x)
    flush(t.secondary)  # flush after each write so log file is updated in real-time
    return n
end
Base.flush(t::TeeStream) = (flush(t.primary); flush(t.secondary))
Base.isopen(t::TeeStream) = isopen(t.primary)

"""
    with_logging(f, log_path)

Execute `f()` while teeing all stdout and stderr output to `log_path`.
The log file is created (or appended to) for the duration of `f`.
Covers println, print, @printf, @time, and any macro that writes to stdout/stderr.
"""
function with_logging(f, log_path::String)
    mkpath(dirname(log_path))
    open(log_path, "a") do log_file
        # redirect_stdout/stderr only accept OS-level stream types and reject
        # custom IO subtypes. Instead, we directly rebind the Julia-level globals
        # in Base, which is what redirect_stdout does internally after the dup2.
        old_stdout = Base.stdout
        old_stderr = Base.stderr
        @eval Base stdout = $(TeeStream(old_stdout, log_file))
        @eval Base stderr = $(TeeStream(old_stderr, log_file))
        try
            f()
        finally
            flush(Base.stdout)
            flush(Base.stderr)
            @eval Base stdout = $old_stdout
            @eval Base stderr = $old_stderr
        end
    end
end

"""
    make_log_path(script_dir, script_name) -> String

Construct a timestamped log file path of the form:
    <script_dir>/logs/<script_name>_YYYY-MM-DD_HH-MM-SS.log
"""
function make_log_path(script_dir::String, script_name::String)
    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
    return joinpath(script_dir, "logs", "$(script_name)_$(timestamp).log")
end
