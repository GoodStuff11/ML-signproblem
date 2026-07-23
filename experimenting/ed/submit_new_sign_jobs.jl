using Dates

function main()
    data_dir = "/home/jek354/research/data/new_data/data_new_sign"
    jobs_dir = "/home/jek354/research/ML-signproblem/jobs"
    mkpath(jobs_dir)
    
    folders = readdir(data_dir)
    
    # Filter out 4x4 and invalid folders
    valid_folders = filter(f -> isdir(joinpath(data_dir, f)) && !occursin("4x4", f), folders)
    
    # Sort folders so that 4x3 is at the end
    regular_folders = filter(f -> !occursin("4x3", f), valid_folders)
    four_by_three = filter(f -> occursin("4x3", f), valid_folders)
    
    ordered_folders = vcat(regular_folders, four_by_three)
    
    sbatch_commands = String[]
    
    for folder in ordered_folders
        full_path = joinpath(data_dir, folder)
        safe_folder_name = replace(folder, " " => "_", "=" => "_", "(" => "", ")" => "", "," => "_")
        
        # 1. Trotter Job
        trotter_prefix = "trotter_antihermitian_slater_$(safe_folder_name)"
        out_log_t = joinpath(jobs_dir, "$(trotter_prefix).out")
        err_log_t = joinpath(jobs_dir, "$(trotter_prefix).err")
        
        # We will use u_start=60 and u_end=2 as the standard scan range
        cmd_args_t = ["\"$full_path\"", "60", "2", "--antihermitian", "--custom_ref_state=slater", "--loss=overlap", "--use_gpu=false"]
        cmd_str_t = "cd /home/jek354/research/ML-signproblem/experimenting/ed && julia --project=.. run_trotter_scan_optimization.jl " * join(cmd_args_t, " ")
        sbatch_cmd_t = "sbatch --mem=20G --cpus-per-task=20 --time=7-00:00:00 --partition=kim --job-name=$(trotter_prefix) --output=$(out_log_t) --error=$(err_log_t) --wrap='$(cmd_str_t)'"
        push!(sbatch_commands, sbatch_cmd_t)
        
        # 2. Exact Exponential Job
        exact_prefix = "exact_antihermitian_slater_$(safe_folder_name)"
        out_log_e = joinpath(jobs_dir, "$(exact_prefix).out")
        err_log_e = joinpath(jobs_dir, "$(exact_prefix).err")
        
        cmd_args_e = ["\"$full_path\"", "60", "2", "--antihermitian", "--custom_ref_state=slater", "--loss=overlap", "--use_gpu=false"]
        cmd_str_e = "cd /home/jek354/research/ML-signproblem/experimenting/ed && julia --project=.. run_lanczos_scan_optimization.jl " * join(cmd_args_e, " ")
        sbatch_cmd_e = "sbatch --mem=20G --cpus-per-task=20 --time=7-00:00:00 --partition=kim --job-name=$(exact_prefix) --output=$(out_log_e) --error=$(err_log_e) --wrap='$(cmd_str_e)'"
        push!(sbatch_commands, sbatch_cmd_e)
    end
    
    # Write submission script
    submit_script_path = joinpath(@__DIR__, "submit_new_sign.sh")
    open(submit_script_path, "w") do io
        println(io, "#!/bin/bash")
        println(io, "echo \"Submitting $(length(sbatch_commands)) jobs to Slurm...\"")
        for cmd in sbatch_commands
            println(io, cmd)
            println(io, "sleep 0.1")
        end
        println(io, "echo \"All jobs submitted successfully!\"")
    end
    chmod(submit_script_path, 0o755)
    
    println("Generated $(length(sbatch_commands)) jobs.")
    println("Submission script saved to: $submit_script_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
