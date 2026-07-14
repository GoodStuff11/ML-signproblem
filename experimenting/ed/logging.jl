using Dates

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
Covers println, print, @printf, @time, and any macro that writes to stdout/stderr,
including C-level output, because we redirect at the OS file-descriptor level.
Output is written to the log in real-time (flushed after every chunk).
"""
function with_logging(f, log_path::String)
    mkpath(dirname(log_path))
    open(log_path, "a") do log_file
        prog_file = isempty(Base.PROGRAM_FILE) ? "interactive" : Base.PROGRAM_FILE
        command_str = "Command run: julia " * prog_file * " " * join(ARGS, " ")
        println(log_file, "================================================================================")
        println(log_file, "Timestamp: ", Dates.now())
        println(log_file, command_str)
        println(log_file, "================================================================================")
        flush(log_file)

        orig_stdout = Base.stdout
        orig_stderr = Base.stderr

        # redirect_stdout() (no-arg form) creates an OS pipe, redirects fd 1 to
        # its write end, and returns the read end. Base.stdout becomes the write end.
        out_rd = redirect_stdout()
        out_wr = Base.stdout
        err_rd = redirect_stderr()
        err_wr = Base.stderr

        # Helper: drain a pipe read-end to two sinks in real-time.
        function make_tee_task(rd, primary, secondary)
            @async try
                while !eof(rd)
                    data = readavailable(rd)
                    isempty(data) && continue
                    write(primary, data)
                    flush(primary)
                    write(secondary, data)
                    flush(secondary)
                end
            catch e
                (e isa EOFError || e isa Base.IOError) || rethrow()
            end
        end

        out_task = make_tee_task(out_rd, orig_stdout, log_file)
        err_task = make_tee_task(err_rd, orig_stderr, log_file)

        try
            f()
        catch e
            print(stderr, "ERROR: ")
            showerror(stderr, e, catch_backtrace())
            println(stderr)
            flush(stderr)
            rethrow()
        finally
            flush(out_wr)
            flush(err_wr)
            redirect_stdout(orig_stdout)
            redirect_stderr(orig_stderr)
            close(out_wr)   # signal EOF to reader tasks
            close(err_wr)
            wait(out_task)
            wait(err_task)
        end
    end
end

"""
    make_log_path(script_dir, script_name) -> String

Construct a timestamped log file path of the form:
    <script_dir>/logs/<yyyy-mm-dd>/<script_name>_YYYY-MM-DD_HH-MM-SS.log
"""
function make_log_path(script_dir::String, script_name::String)
    day_folder = Dates.format(Dates.now(), "yyyy-mm-dd")
    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
    return joinpath(script_dir, "logs", day_folder, "$(script_name)_$(timestamp).log")
end
