# Cluster Remote Workflow Robustness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the VSCode Remote-SSH workflow into the Unicorn cluster resilient to disconnects — no more agent tasks silently dying and no more multi-minute-plus stuck reconnects — without changing the toolchain (VSCode Desktop, Remote-SSH, Antigravity, Claude Code's VSCode extension).

**Architecture:** Three independent layers get hardened: (1) the local SSH connection config, (2) VSCode's remote server process on the login node plus a recovery runbook for when it hangs, and (3) tmux-based persistent sessions on the login node so long-running agent/job/Jupyter processes survive a dropped GUI connection. Each layer is a self-contained set of file edits with no shared interfaces between tasks.

**Tech Stack:** OpenSSH (client config, ControlMaster multiplexing), VSCode Remote-SSH extension settings, tmux, zsh (local shell).

**Spec:** [docs/superpowers/specs/2026-08-26-cluster-remote-workflow-design.md](../specs/2026-08-26-cluster-remote-workflow-design.md)

## Global Constraints

- Cluster host alias: `unicorn` (also matches `unicorn-login-01.coecis.cornell.edu`), user `jek354`.
- Do not touch Antigravity's configuration or process model — it has no CLI/headless mode and is intentionally left GUI-only (see spec, "Out of scope").
- Do not remove or reorder existing lines in `~/.ssh/config`, `~/.zshrc`, or VSCode `settings.json` — only add/modify the specific keys named in each task.
- Local machine is macOS, local shell is zsh (`~/.zshrc` confirmed present).
- `~/.ssh/sockets/` already exists locally — no need to create it, only verify it's usable.

---

### Task 1: SSH connection hardening (local)

**Files:**
- Modify: `~/.ssh/config` (the `Host unicorn-login-01.coecis.cornell.edu unicorn` block)

**Interfaces:** None — self-contained, no dependency on other tasks.

- [ ] **Step 1: Read the current SSH config block**

Run: `grep -n -A 10 "Host unicorn-login-01" ~/.ssh/config`

Expected output (current state):
```
Host unicorn-login-01.coecis.cornell.edu unicorn
  HostName unicorn-login-01.coecis.cornell.edu
  User jek354
  ServerAliveInterval 30
  ServerAliveCountMax 3
  TCPKeepAlive yes
  ControlMaster auto
  ControlPath ~/.ssh/sockets/%r@%h-%p
  ControlPersist 600
```

- [ ] **Step 2: Change `ControlPersist` from `600` to `4h`**

Edit `~/.ssh/config`, changing only this line inside the block found in Step 1:
```
  ControlPersist 4h
```

- [ ] **Step 3: Verify the config parses and the value took effect**

Run: `ssh -G unicorn | grep -i controlpersist`
Expected: `controlpersist 14400` (`ssh -G` normalizes to seconds; 4h = 14400s)

- [ ] **Step 4: Verify a real connection multiplexes correctly**

Run:
```bash
ssh unicorn true
ssh -O check unicorn
```
Expected: the second command prints something like `Master running (pid=NNNNN)` — confirming the ControlMaster socket is live and reusable. (No git commit — `~/.ssh/config` is outside the repo.)

---

### Task 2: VSCode preventive settings (local)

**Files:**
- Modify: `~/Library/Application Support/Code/User/settings.json`

**Interfaces:** None — self-contained, no dependency on other tasks.

- [ ] **Step 1: Add the two Remote-SSH settings**

Add these two keys to the existing JSON object in `~/Library/Application Support/Code/User/settings.json` (alongside the existing keys — do not remove or reorder any current entries):
```json
    "remote.SSH.connectTimeout": 60,
    "remote.SSH.showLoginTerminal": true,
```

- [ ] **Step 2: Verify the file is still valid JSON**

Run: `python3 -m json.tool ~/"Library/Application Support/Code/User/settings.json" > /dev/null && echo VALID`
Expected: `VALID`

- [ ] **Step 3: Verify both keys are present with correct values**

Run: `python3 -c "import json; d=json.load(open('/Users/jonathonkambulow/Library/Application Support/Code/User/settings.json')); print(d['remote.SSH.connectTimeout'], d['remote.SSH.showLoginTerminal'])"`
Expected: `60 True`

(No git commit — this file is outside the repo. Restart VSCode, or reload window via `Cmd+Shift+P` → "Developer: Reload Window", for the settings to take effect.)

---

### Task 3: Recovery runbook + shell alias

**Files:**
- Modify: `~/.zshrc`
- Create: `docs/cluster-workflow-runbook.md`

**Interfaces:** None — self-contained, no dependency on other tasks. References the alias defined in Step 1 by name (`fix-vscode-remote`) so the runbook doc must use that exact name.

- [ ] **Step 1: Add the recovery alias to `~/.zshrc`**

Append to `~/.zshrc`:
```bash
alias fix-vscode-remote='ssh unicorn pkill -f vscode-server'
```

- [ ] **Step 2: Verify the alias is defined without executing it**

Run: `zsh -c 'source ~/.zshrc && type fix-vscode-remote'`
Expected: `fix-vscode-remote is an alias for ssh unicorn pkill -f vscode-server`

(Do not actually run `fix-vscode-remote` during setup — it would kill any currently-active, working VSCode remote server process on the login node.)

- [ ] **Step 3: Write the runbook document**

Create `docs/cluster-workflow-runbook.md`:
```markdown
# Cluster Remote Workflow Runbook

Quick-reference for the day-to-day habits and recovery steps from
[docs/superpowers/specs/2026-08-26-cluster-remote-workflow-design.md](superpowers/specs/2026-08-26-cluster-remote-workflow-design.md).

## When a VSCode Remote-SSH reconnect looks stuck

Do **not** just close and reopen the VSCode window — that kills the
local client but leaves the hung remote `vscode-server` process
untouched, so the next attempt reconnects to the same broken state.

Instead:

1. Run `fix-vscode-remote` in a local terminal (defined in `~/.zshrc`).
   This runs `ssh unicorn pkill -f vscode-server` to kill the stale
   remote process directly.
2. Retry the VSCode Remote-SSH connection.
3. If VSCode's command palette is reachable despite the stuck state,
   `Remote-SSH: Kill VS Code Server on Host...` does the same thing
   without needing a separate terminal.

If the connection itself dropped (e.g. a VPN transition), this won't
prevent the drop — it makes recovery fast instead of an open-ended hang.

## Persistent tmux sessions on the login node

Long-running or agent-driven work should live in a named tmux session
on the login node, not directly in a VSCode terminal, so it survives a
dropped GUI connection:

- `tm-agent` — attach/create the `agent` session, for Claude Code CLI
  runs on anything long or unattended.
- `tm-jobs` — attach/create the `jobs` session, for `sbatch`/`srun`
  submission and monitoring.
- `tm-jupyter` — attach/create the `jupyter` session, for the
  notebook/kernel server process.

These aliases are defined in the login node's shell rc file. Detach
with `Ctrl-b d`; the session and everything running in it keeps going
after you disconnect.

Antigravity has no CLI mode and is not part of this — it stays a
GUI-only tool in VSCode.
```

- [ ] **Step 4: Commit the runbook**

```bash
git add docs/cluster-workflow-runbook.md
git commit -m "Add cluster remote workflow runbook"
```

---

### Task 4: Remote tmux session setup

**Files:**
- Create/Modify (remote, on `unicorn`): `~/.tmux.conf`
- Modify (remote, on `unicorn`): the remote shell's rc file (path determined in Step 1)

**Interfaces:** Produces three named tmux sessions (`agent`, `jobs`, `jupyter`) and matching remote aliases (`tm-agent`, `tm-jobs`, `tm-jupyter`) — these exact names are referenced in Task 3's runbook doc (already written; if these names change, update the runbook too).

- [ ] **Step 1: Determine the remote login shell**

Run: `ssh unicorn 'echo $SHELL'`
Expected: either `/bin/bash` or `/bin/zsh`. Use the matching rc file for Step 4 below (`~/.bashrc` for bash, `~/.zshrc` for zsh).

- [ ] **Step 2: Check for an existing remote `~/.tmux.conf`**

Run: `ssh unicorn 'test -f ~/.tmux.conf && cat ~/.tmux.conf || echo "(none)"'`
Note the output — Step 3 appends rather than overwrites if the file already has content.

- [ ] **Step 3: Append tmux config for scrollback and status visibility**

Run:
```bash
ssh unicorn 'cat >> ~/.tmux.conf << "EOF"

# Cluster remote workflow: longer scrollback, visible session name
set -g history-limit 10000
set -g status on
set -g status-left-length 20
set -g status-left "#[fg=green]#S #[default]"
EOF'
```

- [ ] **Step 4: Add the named-session aliases to the remote rc file**

Using the rc file identified in Step 1 (example below uses `~/.bashrc`; substitute `~/.zshrc` if Step 1 found zsh):
```bash
ssh unicorn 'cat >> ~/.bashrc << "EOF"

# Cluster remote workflow: persistent named tmux sessions
alias tm-agent="tmux new-session -A -s agent"
alias tm-jobs="tmux new-session -A -s jobs"
alias tm-jupyter="tmux new-session -A -s jupyter"
EOF'
```

- [ ] **Step 5: Verify the tmux config loads without error**

Run: `ssh unicorn 'tmux new-session -d -s _verify && tmux kill-session -t _verify && echo TMUX_CONFIG_OK'`
Expected: `TMUX_CONFIG_OK` (a `tmux.conf` syntax error would print a parse error instead of reaching this line).

- [ ] **Step 6: Verify session creation, status bar, and reattachment**

Run:
```bash
ssh unicorn 'tmux new-session -d -s agent && tmux list-sessions'
```
Expected: output includes a line starting `agent:` — confirming the named session was created and persists independent of this SSH call returning.

Run: `ssh unicorn 'tmux has-session -t agent && echo SESSION_PERSISTS'`
Expected: `SESSION_PERSISTS` — confirming the session is still alive on a fresh connection, demonstrating the core property this task exists for (a process in this session survives any single SSH/VSCode connection dropping).

(No git commit — these are remote dotfiles outside the repo.)

---

## Manual validation (not automatable now)

The spec notes the underlying failure is intermittent and couldn't be
reproduced during design. After all four tasks are applied, confirm
during normal use (tracked outside this plan, in the runbook):

- Next VPN switch or slow reconnect: use `fix-vscode-remote` instead of
  close/reopen, and note whether recovery is fast.
- Next longer Claude Code task: run it via `tm-agent` and deliberately
  disconnect (e.g. close the laptop lid) to confirm the task keeps
  running and is reattachable afterward.
