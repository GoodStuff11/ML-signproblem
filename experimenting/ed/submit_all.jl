# submit_all.jl
# Programmatically find all folders in data/ (excluding tmp, N=(3, 3)_3x3_newsign, and any copies/backups)
# and submit sbatch jobs for both overlap and energy loss.

using Dates

function submit_sbatch_jobs()
    data_dir = joinpath(@__DIR__, "data")
    folders = readdir(data_dir)
    
    # Exclude files, tmp, N=(3, 3)_3x3_newsign, and anything containing "copy"
    target_folders = String[]
    for f in folders
        path = joinpath(data_dir, f)
        if !isdir(path)
            continue
        end
        if f == "tmp" || f == "N=(3, 3)_3x3_newsign" || occursin("copy", f)
            continue
        end
        push!(target_folders, f)
    end
    
    println("Found $(length(target_folders)) target folders to submit:")
    for f in target_folders
        println("  - $f")
    end
    
    # Ensure the jobs directory exists for error and output logs
    jobs_dir = "/home/jek354/research/ML-signproblem/jobs"
    mkpath(jobs_dir)
    println("Job log files will be written to: $jobs_dir")

    for folder in target_folders
        for loss in ["overlap", "energy"]
            job_name = "trotter_$(folder)_$(loss)"
            # Sanitize folder name for filenames
            safe_name = replace(folder, " " => "_", "=" => "_", "(" => "", ")" => "", "," => "_")
            out_log = joinpath(jobs_dir, "trotter_$(safe_name)_$(loss).out")
            err_log = joinpath(jobs_dir, "trotter_$(safe_name)_$(loss).err")
            
            cmd_str = "julia --project=.. run_trotter_scan_optimization.jl \"data/$(folder)\" 60 2 --maxiters=300 --loss=$(loss)"
            
            sbatch_cmd = `sbatch --mem=20G --cpus-per-task=60 --time=7-00:00:00 --partition=kim --job-name=$(job_name) --output=$(out_log) --error=$(err_log) --wrap=$(cmd_str)`
            
            println("Submitting job for folder='$folder', loss='$loss'...")
            run(sbatch_cmd)
        end
    end
    println("All jobs submitted successfully!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    submit_sbatch_jobs()
end
