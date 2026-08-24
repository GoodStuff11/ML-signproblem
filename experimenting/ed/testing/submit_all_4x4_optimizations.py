#!/usr/bin/env python3
"""
submit_all_4x4_optimizations.py
===============================
Submits Slurm optimization jobs for 4 systems across 4 option configurations:
Systems:
  - N=(5, 4)_3x3
  - N=(4, 4)_3x3
  - N=(3, 3)_3x2
  - N=(3, 2)_3x2

Options:
  1. trotter_slater: Trotter with Slater reference state
  2. trotter_default: Trotter with default reference state (eigenstate)
  3. exact_slater: Exact exponential with Slater reference state
  4. exact_default: Exact exponential with default reference state (eigenstate)
"""

import os
import re
import subprocess
import sys

DATA_ROOT = "/home/jek354/research/data/new_data/data_h5_fixed"
EXP_DIR = "/home/jek354/research/ML-signproblem/experimenting/ed"
JOBS_DIR = "/home/jek354/research/ML-signproblem/jobs"

SYSTEMS = [
    "N=(5, 4)_3x3",
    "N=(4, 4)_3x3",
    "N=(3, 3)_3x2",
    "N=(3, 2)_3x2",
]

OPTIONS = [
    {
        "label": "trotter_slater",
        "script": "run_trotter_scan_optimization.jl",
        "cli_extra": ["--antihermitian", "--custom_ref_state=slater", "--loss=overlap", "--use_gpu=false"]
    },
    {
        "label": "trotter_default",
        "script": "run_trotter_scan_optimization.jl",
        "cli_extra": ["--antihermitian", "--loss=overlap", "--use_gpu=false"]
    },
    {
        "label": "exact_slater",
        "script": "run_lanczos_scan_optimization.jl",
        "cli_extra": ["--antihermitian", "--custom_ref_state=slater", "--loss=overlap", "--use-gpu=false"]
    },
    {
        "label": "exact_default",
        "script": "run_lanczos_scan_optimization.jl",
        "cli_extra": ["--antihermitian", "--loss=overlap", "--use-gpu=false"]
    },
]


def sanitize(name):
    return name.replace(" ", "_").replace("=", "_").replace("(", "").replace(")", "").replace(",", "_")


def main():
    os.makedirs(JOBS_DIR, exist_ok=True)
    results = []

    for sys_name in SYSTEMS:
        full_sys_path = os.path.join(DATA_ROOT, sys_name)
        safe_sys = sanitize(sys_name)

        for opt in OPTIONS:
            job_name = f"{opt['label']}_{safe_sys}"
            out_log = os.path.join(JOBS_DIR, f"{job_name}.out")
            err_log = os.path.join(JOBS_DIR, f"{job_name}.err")

            cmd_args = [f'"{full_sys_path}"', "60", "2"] + opt["cli_extra"]
            cmd_str = f"cd {EXP_DIR} && julia --project=.. {opt['script']} " + " ".join(cmd_args)

            sbatch_cmd = [
                "sbatch",
                "--mem=20G",
                "--cpus-per-task=20",
                "--time=7-00:00:00",
                "--partition=kim",
                f"--job-name={job_name}",
                f"--output={out_log}",
                f"--error={err_log}",
                f"--wrap={cmd_str}"
            ]

            print(f"Submitting job: {job_name} ...")
            try:
                proc = subprocess.run(sbatch_cmd, capture_output=True, text=True, check=True)
                stdout = proc.stdout.strip()
                print(f"  {stdout}")
                m = re.search(r"Submitted batch job (\d+)", stdout)
                job_id = m.group(1) if m else "UNKNOWN"
                results.append((sys_name, opt["label"], job_id, "SUBMITTED"))
            except subprocess.CalledProcessError as e:
                print(f"  Error submitting {job_name}: {e.stderr}", file=sys.stderr)
                results.append((sys_name, opt["label"], "ERROR", "FAILED"))

    print("\n" + "=" * 60)
    print("SLURM SUBMISSION SUMMARY")
    print("=" * 60)
    for sys_name, opt_label, job_id, status in results:
        print(f"[{status:9s}] {sys_name:14s} | {opt_label:16s} -> Job ID: {job_id}")
    print("=" * 60)


if __name__ == "__main__":
    main()
