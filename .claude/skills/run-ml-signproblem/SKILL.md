---
name: run-ml-signproblem
description: Build, run, and drive the ML-signproblem Julia ED (exact diagonalization) pipeline. Use when asked to run ML-signproblem, compute or load Hubbard-model ED ground states, submit or debug a SLURM/GPU job for this repo, verify ed_functions.jl / ed_optimization.jl changes, or smoke-test the experimenting/ed codebase.
---

ML-signproblem is a Julia research codebase (no server, no GUI) that computes exact-diagonalization (ED) ground states of the Hubbard model and uses them as training targets for ML models of the fermion sign problem. There's no single "app" to launch — the two things you drive are (1) `run_*.jl` scripts under `experimenting/ed/` that compute ED data from scratch via CLI args, and (2) the `ed_functions.jl` library (`load_h5_ED_data` / `load_jld2_ED_data`) that other code imports to read that data back. Drive both via `.claude/skills/run-ml-signproblem/smoke.sh`, which does one real end-to-end run: compute → save → reload → verify.

All paths below are relative to `ML-signproblem/` (this unit's root). This repo also carries agent-wide operating rules in `/home/jek354/research/.agents/rules/*.md` (outside this unit, at the research/ root) — they are binding on any agent working here and are folded into this skill below; treat them as the source of truth if this skill and a rule file ever disagree.

## Before you touch anything: check `.agent_docs/` and existing output

- **Read `.agent_docs/_INDEX.md` first**, before parsing any `.h5`/`.jld2` file or writing loader code — it routes to `.agent_docs/core_conventions.md` (vector ordering, sign conventions, `HubbardMomentumBasis` gotchas) and `.agent_docs/data_pipeline.md` (full `.h5`/`.jld2` schema, `target_vecs` row-indexing rule). This documentation exists specifically so you never have to guess a file format or re-derive a convention by reading parsing code from scratch (`.agents/rules/memory.md`). If you extend or correct these docs, keep the exhaustive, no-ambiguity, fully-mapped-schema style already there, and never merge unrelated domains into one file. If you're unsure of a format/schema/physical meaning while writing this documentation, stop and ask the user rather than inventing details.
- **Markdown formatting**: this user's markdown (including `.agent_docs/*.md`) gets read in Obsidian — avoid excess blank lines. No blank line between a display equation and the paragraph above/below it, none between two consecutive equations, and none right after a heading; only put a blank line between two paragraphs or above a `---`/table so it renders correctly (`.agents/rules/markdown-documentation.md`).
- **Never blindly recompute.** Before running any `run_*.jl` script for real (not the disposable `smoke.sh` scratch runs), check whether the target data folder already has the output the script would produce (same `N=(nu, nd)_LxxLy` folder / `meta_data_and_E.jld2` / `HubbardED_*.h5`). If it's already there, don't overwrite it — and if you're unsure whether overwriting is intended, ask the user rather than guessing (`.agents/rules/rerunning-jobs.md`). Likewise, never cancel a long-running job whose output file hasn't been written yet — that's the one signal that it hasn't finished.

## Prerequisites

Julia 1.12 via juliaup (already installed on this host: `julia --version` → `1.12.5`). No `apt-get` packages were needed — `HDF5.jl` and friends use bundled JLL artifacts.

## Setup

The project environment at `experimenting/Project.toml` is already resolved and precompiled on this host (`julia --project=experimenting -e 'using Pkg; Pkg.status()'` lists everything with no missing/red entries). On a fresh checkout, instantiate it first:

```bash
julia --project=experimenting -e 'using Pkg; Pkg.instantiate()'
```

This pulls ~25 packages including two git-sourced ones (`Lattices.jl`, `QuantumLattices.jl`), plus CUDA/Flux/Makie/Zygote — expect this to take several minutes and a few GB of disk on a cold cache.

**Julia invocation convention**: always run Julia commands inside `experimenting/ed/` (or its subfolders) as `julia --project=.. <command>`, where `..` resolves to `experimenting/` (where `Project.toml`/`Manifest.toml` live). Never build a roundabout `LOAD_PATH`/`include` workaround to dodge this — it's the one supported invocation form (`.agents/rules/julia-command-format.md`). `smoke.sh` follows this convention.

## Build

No separate build step. First invocation of any script pays Julia's JIT/precompile cost (dominates the ~60-90s wall time of the smoke test below).

## Run (agent path)

```bash
.claude/skills/run-ml-signproblem/smoke.sh          # writes to a fresh mktemp -d
.claude/skills/run-ml-signproblem/smoke.sh /tmp/my_out   # or a dir you choose
```

What it does (verified working end-to-end in this container):
1. Prints a memory-headroom preflight (see Gotchas — this host has a hard per-user memory cap that this pipeline can hit). If headroom is low or the run gets killed, prefer testing via `srun` instead of retrying locally — see Gotchas for the exact command.
2. Runs `julia --project=.. run_ed_lanczos_momentum.jl 3 2 2 2 <out_dir>` from `experimenting/ed/` — a **real, from-scratch ED computation**: Lanczos diagonalization with momentum-sector projection for a 3×2-site Hubbard model at half filling (N_up=N_down=2), scanning 61 U values across 6 momentum sectors. Writes `<out_dir>/N=(2, 2)_3x2/meta_data_and_E.jld2`.
3. Loads that file back with `load_jld2_ED_data` (the same loader real experiments use) and asserts: 61 U values, `target_vecs` shape `(61, 39)`, every returned eigenvector normalized to 1.

`run_ed_lanczos_momentum.jl` only `using CUDA` for the `@safe_threads` GPU-loaded check (see GPU section below) — it never calls `CUDA.functional()`/`CuArray` and does all its math on CPU, so running it locally (as `smoke.sh` does) is correct and doesn't fall under the "GPU work must go through SLURM" rule below.

A clean run looks like:
```
== Step 1/2: computing ED ground states for a 3x2 lattice, N=(2,2) Hubbard model ==
[... "overlap: ..." lines from the script's own adiabatic-tracking sanity checks, "k: N, indexer: ..." per momentum sector ...]
Precomputing n_body_structure for optimization...
== Step 2/2: loading the computed data back through load_jld2_ED_data and verifying it ==
OK: loaded 61 ground states, Hilbert space dim=39, N=(2, 2)
SMOKE TEST PASSED (output left in: /tmp/tmp.XXXXXXXXXX)
```
Took ~76s wall time in a clean successful run in this container.

## Direct invocation (exploring the library without running a full computation)

Most PRs touch `ed_functions.jl`/`ed_optimization.jl` loading or physics logic, not the run scripts. Load an existing cached dataset directly from a Julia REPL or `-e` snippet:

```julia
using HDF5, LinearAlgebra, SparseArrays, Lattices
include("experimenting/ed/ed_objects.jl")
include("experimenting/ed/utility_functions.jl")
include("experimenting/ed/trotter.jl")
include("experimenting/ed/ed_functions.jl")
using .Trotter

# HDF5-backed dataset (folder of HubbardED_*.h5 files):
U_values, target_vecs, indexer, precomp, N, spin_conserved, use_symmetry,
    sign_conv, Lvec, order_native = load_h5_ED_data("/path/to/N=(nu, nd)_LxxLy"; omit_indexer=true)

# JLD2-backed dataset (a single meta_data_and_E.jld2, e.g. one smoke.sh produced):
U_values, target_vecs, indexer, precomp, N, spin_conserved, use_symmetry,
    sign_conv = load_jld2_ED_data("/path/to/meta_data_and_E.jld2")
```
`target_vecs` is `(n_U [+1 if use_slater_reference], H_dim)` — index rows, never columns (see `.agent_docs/data_pipeline.md`).

## Run (human path) — SLURM / GPU jobs

The `run_*.jl` scripts under `experimenting/ed/` are normally submitted as SLURM batch jobs (see `jobs/*.sh`), e.g.:
```bash
julia --project=.. run_ed_lanczos_momentum.jl 4 3 4 4 my_data_dir   # bigger lattice, real experiment
```
For CPU-only scripts like this one, running locally the way `smoke.sh` does is fine at small sizes.

**GPU rule**: for any script that actually exercises the GPU (calls `CUDA.functional()`, `CUDA.has_cuda_gpu()`, or allocates `CuArray`s — currently `ed_optimization.jl` and everything that uses it: `run_optimization_experiments.jl`, `run_custom_ref_experiments.jl`, `run_lanczos_scan_optimization.jl`, `system_scaling.jl`, `trotter_optimization.jl`, `barren_plateau.jl`, `nn_strategy.jl`), do **not** run it locally. Use:
```bash
srun --mem=20g --gres=gpu:1 --cpus-per-task=1 --time=0:30:00 --partition=kim <cmd>
```
adjusting `--mem`/`--time`/`--cpus-per-task` as needed, or `sbatch` with a throwaway submission script for longer runs (`.agents/rules/gpu.md`). Either way, know what output/log you'll inspect if the job crashes.

**SLURM submission conventions** (`.agents/rules/srun-arguments.md`): give `srun`/`sbatch` a lenient time window (e.g. a week, not the job's expected runtime), an informative `--job-name`, and redirect stderr to a file under `ML-signproblem/jobs/`.

**Fire-and-forget once submitted** (`.agents/rules/checking-runs.md`): after `sbatch`/`srun` returns a job ID, report the ID to the user and stop — do not poll `squeue`, tail logs in a loop, sleep-and-recheck, or otherwise babysit the job. The user has their own external notification for job completion. Only check `squeue --user=$(whoami) --format="%i|%j|%T|%P|%D|%M|%l|%R" --noheader` if the user explicitly asks for a fresh status check right then.

## GPU & threading hazards

- **Never mix `Threads.@threads` with a loaded CUDA context** — GC/pinned-memory interaction between Julia's threaded GC and CUDA's memory can segfault the process. Use `@safe_threads` from `utility_functions.jl` instead anywhere you'd otherwise reach for `Threads.@threads`; it detects whether CUDA is loaded and falls back to serial execution automatically (`.agents/rules/GPU-multithreading.md`) — this is exactly what produces the benign `Warning: CUDA is loaded. Running loop serially...` message you'll see in `smoke.sh`'s output.
- **Known opaque CUDA errors and their fixes are logged in `research/.agents/rules/gpu-debugging-process.md`** (one level above this unit, at the `research/` root, alongside all the other `.agents/rules/*.md` files referenced throughout this skill) — covers `CUDA error: device kernel image is invalid (code 200)`, `CUDA error: unknown error (code 999)` on specific SLURM nodes, and a SIGSEGV root-caused to `Threads.@threads` + CUDA coexisting in `ed_optimization.jl`'s adjoint pullbacks. Check there before re-debugging a GPU error from scratch. If you resolve a new opaque GPU error, append the procedure to that file (or a numbered `gpu-debugging-process2.md` if it gets too long) rather than losing the knowledge (`.agents/rules/gpu-debugging.md`).

## Writing or modifying scripts in this repo

- **Standalone scripts** (ones that aren't just `include`d by another script) must wrap their body in:
  ```julia
  function (@main)(ARGS)
      log_path = make_log_path(@__DIR__, "<script_name_without_.jl>")
      with_logging(log_path) do
          ...
      end
  end
  ```
  so stdout/stderr get tee'd to a log file in real time (`.agents/rules/logging.md`; see `logging.jl`). `run_ed_lanczos_momentum.jl` already follows this.
- **CLI argument parsing** goes in its own `parse_arguments(args)` function called from inside `(@main)(ARGS)`, never inlined. Give every script a docstring at the top documenting every positional/keyword argument, its default, and — for any argument with a fixed set of valid values — what each value does, in enough detail that you never need to read the code to know what the argument does or what you can safely change. Follow the format in `experimenting/ed/system_scaling.jl` (`.agents/rules/command-line.md`).

## Test

There is no formal `runtests.jl` / `Pkg.test()` suite for this codebase. `experimenting/ed/testing/` holds ad hoc one-off investigation scripts (not a maintained regression suite). `smoke.sh` is the closest thing to a test — run it after touching `ed_functions.jl`, `ed_objects.jl`, or `utility_functions.jl`.

**Backing claims** (`.agents/rules/backing-claims.md`): when you claim a fix or a piece of behavior works, back it with a test script that is actually run, whose real (unfaked) output you show — don't assert correctness from reading code alone, and don't work around a question the user actually asked. Put such one-off verification scripts in `experimenting/ed/testing/`, not scattered among the library files — that's exactly the pattern `smoke.sh` follows (it's the skill's own persistent version of this).

## Gotchas

- **This host enforces a hard 4GiB memory cap per user account**, via a `memory.max` on the `user-<uid>.slice` cgroup — completely separate from (and far smaller than) the 125GiB `free -h` reports for the machine as a whole. Check it directly:
  ```bash
  cat /sys/fs/cgroup/user.slice/user-$(id -u).slice/memory.max      # cap, bytes ("max" = uncapped)
  cat /sys/fs/cgroup/user.slice/user-$(id -u).slice/memory.current  # current usage, bytes
  cat /sys/fs/cgroup/user.slice/user-$(id -u).slice/memory.events   # oom_kill count — nonzero here is normal on this host
  ```
  VSCode/editor extension-host processes under the same account routinely consume most of that 4GiB on their own, leaving very little headroom. The Julia run in step 1 only peaks around ~900MB RSS (measured with `/usr/bin/time -v`), but across repeated attempts in this session it was SIGKILL'd by the cgroup OOM killer 3 times out of 4 purely because other processes under the same account had already eaten the rest of the 4GiB — `free -h` looked fine (tens of GiB "available") the whole time. `smoke.sh` prints a headroom preflight for exactly this reason. **If it warns of low headroom or the run gets killed, just retry later** (headroom fluctuates as the user's other VSCode windows/extensions come and go) — it is not a bug in the script or the library.
  **The reliable way to sidestep this entirely: test via `srun` instead of running directly on this shared login node.** A compute-node allocation has its own memory, not shared with everyone else's VSCode/editor processes the way the login node's per-user cgroup is, so `smoke.sh` (or any verification script) run under `srun` won't compete with those other processes at all:
  ```bash
  srun --mem=4g --cpus-per-task=2 --time=1-0 --job-name=ml-signproblem-smoketest --partition=kim \
      .claude/skills/run-ml-signproblem/smoke.sh
  ```
  This doesn't need `--gres=gpu:1` — `smoke.sh` is CPU-only (see the "Run (agent path)" note on why). Prefer this over retrying locally when headroom is low or a run keeps getting killed.
- **`using CUDA` prints a warning even with no GPU present** (`nvidia-smi` is not installed on this host): `Warning: CUDA is loaded. Running loop serially to prevent segmentation faults...`. This is expected/benign — the scripts detect the *package* being loaded, not an actual device, and fall back to serial execution (see GPU & threading hazards above).
- **Don't pick a lattice/filling too small.** A 2×2 lattice with N_up=N_down=1 fails partway through with `ERROR: error is bad: <overlap>` — the invariant-subspace is small enough that the script's own overlap-based adiabatic state-tracking sanity check spuriously trips. `smoke.sh` uses the known-good 3×2, N=(2,2) combination.
- **`smoke.sh` always writes to a fresh scratch directory**, so the "check before recomputing" rule above doesn't apply to it directly — but it does apply the moment you invoke `run_*.jl` against a real, persistent data directory (`experimenting/ed/data`, or wherever `ED_DATA_ROOT`/`data_config.txt` points).

## Troubleshooting

- **`Killed` printed mid-run, or the process just vanishes with no `SMOKE TEST PASSED`**: SIGKILL from the per-user cgroup OOM killer (see Gotchas), not a script bug. Retry.
- **`ERROR: InvalidDataException: Did not find a Superblock`** when loading a `.jld2` file: the file is truncated because the process that wrote it was killed mid-write (usually the same OOM cause above). Delete the partial output dir and rerun `smoke.sh`.
- **`ERROR: error is bad: 0.4790...`** (or similar) from `run_ed_lanczos_momentum.jl`: system size too small for the script's degeneracy-tracking assumptions (see Gotchas). Use a larger `(Lx, Ly, N_up, N_down)` combination, e.g. the smoke test's `3 2 2 2`.
- **Opaque CUDA errors** (`code 200`, `code 999`, GPU-context SIGSEGVs): don't debug from scratch — check `.agents/rules/gpu-debugging-process.md` first (see GPU & threading hazards above).
