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
