# Cluster Remote Workflow Robustness — Design

## Problem

The research workflow relies on VSCode's Remote-SSH extension into the
Unicorn login node (`unicorn-login-01.coecis.cornell.edu`), used for:

- GUI editing with AI agent extensions (Antigravity, Claude Code's
  VSCode extension)
- Running `sbatch`/`srun` and monitoring jobs
- Connecting to a Jupyter kernel/notebook running on a compute node

Two recurring failures make this workflow unreliable:

1. **AI agents randomly stop mid-task** when the Remote-SSH connection
   drops (e.g. on a network/VPN transition), losing in-progress agent
   work.
2. **Reconnecting can hang for extended periods** (observed: 30+
   minutes), with repeated close/reopen of the VSCode window not
   helping.

Plain terminal SSH to the same login node does not exhibit either
problem — the failures are specific to VSCode's Remote-SSH connection
and process model, not the network path itself.

## Root cause analysis

- Terminal SSH being reliable while VSCode Remote-SSH is not points to
  the extension's own connection/reconnection handling and remote
  server process, not the underlying SSH transport.
- The "stuck connecting" pattern (as opposed to stuck installing)
  combined with "closing and reopening doesn't help" indicates the
  hang is server-side: closing the VSCode client does nothing to the
  remote `vscode-server` process it's trying to reconnect to. If that
  process is left in a stale/half-dead state (common after a VPN
  transition drops the connection mid-session), every reconnect
  attempt just reconnects to the same broken state.
- Agents dying with the connection is expected for anything whose
  process lifetime is tied to the VSCode remote extension host — a
  dropped connection can take the extension host, and anything running
  under it, down with it.

The fix targets three independent layers: the SSH connection itself,
the remote VS Code Server process, and decoupling long-running/agent
processes from the GUI connection's lifetime.

## Design

### 1. Connection layer (SSH config)

Existing `~/.ssh/config` entry already has `ControlMaster`,
`ControlPersist`, `ServerAliveInterval`/`CountMax`, and `TCPKeepAlive`
configured — this is largely already correct. Two adjustments:

- Increase `ControlPersist` from `600` (10 min) to `4h`, so the shared
  multiplexed connection survives idle periods between sessions
  (closing all terminals/VSCode windows for >10 min currently forces a
  fresh connection negotiation on the next session).
- Ensure `~/.ssh/sockets/` exists locally (`mkdir -p ~/.ssh/sockets`) —
  `ControlPath` silently fails to multiplex without it, with no
  obvious error.

VSCode's Remote-SSH extension reads the same `~/.ssh/config` by
default, so it automatically inherits the shared, hardened connection
once these are in place — no separate VSCode-side connection config is
needed for this layer.

A VPN-vs-direct network transition is a genuine connection loss no SSH
config can prevent; the goal here is a fast, uneventful reconnect, not
eliminating the disconnect event itself.

### 2. Remote server layer (VS Code Server on the login node)

**Recovery runbook** (replaces the close/reopen loop, which doesn't
touch the hung remote process):

- From a plain terminal SSH session, run:
  ```
  ssh unicorn pkill -f vscode-server
  ```
  then retry the VSCode Remote-SSH connection.
- Add a local shell alias/function for this so it's zero-friction:
  ```
  alias fix-vscode-remote='ssh unicorn pkill -f vscode-server'
  ```
- If VSCode's command palette is reachable despite the stuck state,
  `Remote-SSH: Kill VS Code Server on Host...` does the same thing
  without needing a separate terminal.

**Preventive VSCode settings:**

- `remote.SSH.connectTimeout`: increase from the default (15s) to `60`
  so a slow-but-working connection isn't abandoned and retried
  prematurely in a way that compounds into a long hang.
- `remote.SSH.showLoginTerminal: true`: surfaces the SSH handshake
  output during connect, so a hang shows *why* (e.g. waiting on a
  stale process) instead of an opaque spinner.

This runbook is the primary response the moment a reconnect looks
slow, rather than waiting and blindly retrying.

### 3. Process layer (tmux-based decoupling)

This targets "agent randomly stops" directly: anything that shouldn't
die with the GUI connection runs inside a `tmux` session on the login
node, so VSCode becomes a window into persistent state rather than the
thing keeping that state alive.

Named sessions (`tmux new-session -A -s <name>`, which attaches if the
session exists or creates it if not — strictly more convenient than
plain `attach`), not ad-hoc ones:

- `agent` — Claude Code CLI runs for anything expected to run long or
  unattended (job submission loops, long analysis tasks). The VSCode
  Claude Code extension remains available for quick interactive edits
  where a drop just means re-asking; it is not moved into tmux.
- `jobs` — `sbatch`/`srun` submission and monitoring. `sbatch` jobs
  already survive a disconnect on their own; running submission and
  monitoring from `jobs` keeps a persistent, reattachable scrollback of
  what's been launched and its output.
- `jupyter` — the notebook/kernel server process on the compute node
  allocation, launched from inside this session (directly or via
  `srun`) so the kernel survives independent of the SSH/VSCode
  connection. Only the port-forward becomes fragile, and re-forwarding
  is cheap since the underlying process never died.

Antigravity has no CLI/headless mode and is not folded into tmux — it
stays a GUI-only tool, relying on layers 1 and 2 for reliability. A
mid-task drop there means restarting that task, not losing cluster-side
state.

One-time `~/.tmux.conf` tweak: longer scrollback history and a status
bar showing the session name, so it's easy to orient after
reattaching cold.

## Validation

The failure is intermittent, so validation is a mix of one-time checks
and behavior to confirm during normal use:

- One-time: confirm `~/.ssh/sockets/` exists and the `ControlPersist`
  bump takes effect (`ssh -O check unicorn` shows an active master
  after connecting).
- One-time: set up the `agent`/`jobs`/`jupyter` tmux session habit
  before it's needed under pressure.
- During use: next VPN switch or slow reconnect, use the runbook
  (`fix-vscode-remote` alias) instead of close/reopen, and note
  whether recovery is fast.
- During use: run a longer Claude Code task inside the `agent` tmux
  session and deliberately disconnect (e.g. close the laptop lid) to
  confirm the task keeps running and is reattachable afterward.

## Out of scope

- Replacing VSCode Remote-SSH with Remote Tunnels or code-server
  (considered as alternative approaches; not pursued now since this
  hardening approach preserves the existing toolchain without
  extension-compatibility risk).
- Any change to Antigravity's process model — it has no CLI/headless
  mode to decouple into tmux.
