# Claude Code — Day 1 Learning Session Reference

**Date:** 2026-05-09
**Topic:** CLI Fluency, Session Management, VS Code Extension, Flags, Worktrees, Slash Commands, /branch

---

## 1. Can You Run `claude` From the VS Code Extension?

**Answer:** No — and you don't need to. The VS Code extension **is** Claude Code.

The `claude` CLI and the VS Code extension are two surfaces of the same tool:

| Surface | How to use |
|---------|-----------|
| VS Code extension panel | Graphical UI — type directly in the chat panel |
| Terminal (`claude`) | Text-mode — launch from any terminal |

- Both share the same engine, settings, CLAUDE.md, and memory
- If you open the integrated terminal (`Ctrl+\`) and type `claude`, you get a **second separate session** running in text mode alongside the graphical one
- For Day-to-day use, the extension panel is sufficient — no need to launch `claude` separately

---

## 2. Slash Commands & `Shift+Tab` in the VS Code Extension

**Slash commands** work directly in the extension chat input:
- Type `/` in the prompt box → autocomplete menu appears with all available commands
- All slash commands work the same as in the CLI

**`Shift+Tab`** works in the extension:
- Press it while your cursor is in the chat input box (not while focused on the editor)
- Cycles through permission modes: `default` → `plan` → `acceptEdits` → `auto`
- Mode indicator updates in the prompt area

**CLI-only exception:** `claude -r` (session picker) is CLI-only. In the extension, use the **session history sidebar** (history/clock icon in the extension panel) — same idea, different UI.

---

## 3. Stop, Rewind & Keyboard Shortcuts

### Keyboard Shortcuts — Complete Reference

| Shortcut | Where | What it does |
|----------|-------|-------------|
| `Shift+Tab` | CLI + Extension | Cycle permission modes: default → plan → acceptEdits → auto |
| `Esc` (once) | CLI | Stop Claude mid-response |
| `Esc` `Esc` (twice) | CLI | Open the rewind menu |
| `Ctrl+G` | CLI | Open external editor to write a long multi-line prompt |
| `Ctrl+T` | CLI | Open the task list (TodoWrite tasks) |
| `Ctrl+O` | CLI | Open the session transcript viewer |
| `Cmd/Ctrl+Esc` | Extension | Focus the Claude Code extension panel from the editor |
| `Alt+K` / `Option+K` | Extension | Insert an `@file` reference at cursor |
| Hover message → rewind icon | Extension | Rewind to that message (see below) |

---

### Rewind — Time Machine for Session + Files

Rewind takes you back to any previous point in the conversation **and restores all file changes made after that point**. It undoes both the conversation history and filesystem edits together.

**In the CLI — `Esc` `Esc`:**
```
1. Press Esc once  → stops Claude if it's running
2. Press Esc again → opens rewind menu listing previous turns
3. Select a turn   → conversation truncates to that point, files restored
```

**In the VS Code extension:**
- Hover over any previous message in the chat panel
- A rewind icon appears on that message
- Click it → conversation truncates to that point, files restored

---

### What Gets Rewound

| What | Rewound? |
|------|----------|
| Conversation history after the chosen point | ✅ truncated |
| File edits made after the chosen point | ✅ restored to prior state |
| Shell side-effects (git commits, npm installs, API calls) | ❌ not reversed |

Rewind is **filesystem rewind, not process rewind** — only changes to files tracked by Claude Code are restored. External effects stay.

---

### Rewind vs `/branch`

| | Rewind | `/branch` |
|--|--------|-----------|
| Goes back in time | Yes — discards what came after | No — forks forward |
| Keeps the "future" | No | Yes (original session intact) |
| Use when | You want to undo and retry | You want to try two approaches in parallel |

**Combined pattern:** use `/branch` to fork first, then if one branch goes wrong use rewind inside that branch — the other branch is untouched.

---

### Checkpoints (VS Code Extension)

The extension lets you **explicitly set a checkpoint** before a risky operation so you have a clean rewind target rather than hunting through the message list.

- Hover any message → set checkpoint
- Before any non-trivial refactor, set a checkpoint first
- Rewind to checkpoint = one click restore

---

## 4. Key CLI Launch Flags

These flags are passed when launching `claude` from a terminal:

### `--model`
Override which Claude model the session uses.
```bash
claude --model claude-opus-4-7
claude --model claude-haiku-4-5-20251001
```
- Use a faster/cheaper model for simple tasks
- Use the most capable model for complex reasoning
- In the extension: use `/model` command or the model picker instead

---

### `--add-dir`
Grant Claude access to a directory outside your current working directory.
```bash
claude --add-dir ../shared-utils
claude --add-dir /path/to/other-repo
```
- By default Claude can only read/write within your project folder
- This extends the boundary without making everything visible
- Keeps context focused while enabling cross-repo work

---

### `--worktree`
Creates an isolated git worktree for the session — a separate folder and branch.
```bash
claude --worktree
```
- Detailed explanation in Section 4 below

---

### `--bare`
Starts faster with minimal context load — skips CLAUDE.md and memory initialization.
```bash
claude --bare
```
- Good for quick one-off tasks
- Useful in scripts where you don't need project context
- Avoids loading CLAUDE.md, auto-memory, hooks

---

### `--permission-mode`
Set the starting permission mode at launch instead of cycling with `Shift+Tab`.
```bash
claude --permission-mode plan             # read-only, cannot touch files
claude --permission-mode acceptEdits      # auto-approve all file edits
claude --permission-mode bypassPermissions  # no prompts (sandboxes only — dangerous)
```
- Saves having to `Shift+Tab` to your preferred mode every session
- In the extension: set `initialPermissionMode` in extension settings

---

## 5. `--worktree` Deep Dive

### The Problem It Solves
Without worktrees, if you have two Claude sessions on the same repo:
- Both sessions work on the same files and same branch
- One session's edits overwrite the other's

### What a Git Worktree Is
A worktree is git's way of checking out the same repo into a **second folder on a different branch**, at the same time. Both folders share the same `.git` history — they are **not clones**.

```
my-repo\              ← normal checkout  (branch: main)
my-repo-worktree\     ← isolated checkout (branch: fix-login-bug)
```

### What `claude --worktree` Does
Claude Code automatically:
1. Creates a new git branch
2. Checks out the repo into a new temp folder (sibling of current project)
3. Runs the session inside that isolated copy

**Result:** Two Claude sessions can work on the same repo in parallel with zero conflicts.

### Branch Naming
- Derived from your task description: `"fix login timeout"` → `fix-login-timeout`
- If no description given: uses a timestamp or random slug
- Override with: `claude --worktree --branch my-branch-name`

### Folder Location
- Always created as a **sibling** of your current project folder:
  ```
  C:\ma\ccode\worktree\         ← your project
  C:\ma\ccode\worktree-abc123\  ← the new worktree (suffix = branch name or hash)
  ```
- Folder location is not directly configurable — always goes next to the current project

### Cleanup Behaviour
- **No changes made** → Claude auto-cleans up the worktree folder after the session
- **Changes made** → folder and branch persist so you can review, merge, or raise a PR

### Typical Use Pattern
```
Extension panel     → main task (your current branch)
Integrated terminal → claude --worktree  (parallel isolated task on new branch)
```

---

## 6. `--worktree` in the VS Code Extension

**Short answer:** Not supported directly in the extension UI — it's a CLI-only launch flag.

**How to use it from VS Code:**
1. Open the integrated terminal: `Ctrl+\``
2. Run: `claude --worktree`
3. A text-mode Claude session starts in an isolated worktree

**Parallel workflow:**
| Where | What |
|-------|------|
| Extension panel | Main task in your current working directory |
| Integrated terminal (`claude --worktree`) | Parallel task in isolated branch/folder |

Both share the same CLAUDE.md and settings — just different branches and folders.

> The graphical extension panel always runs in your current working directory. Worktree support in the UI is not available yet.

---

---

## 7. Slash Commands — Complete Reference

### Group 1: Context Management (Most Important)

These control your context window — the #1 thing that affects Claude's performance.

**`/clear`**
Wipes the entire conversation. Claude forgets everything discussed.
- Use when: switching to a completely different task
- Don't use when: you want to keep going on the same task but lighten the load

**`/compact`**
Summarizes the conversation history to free up context, but keeps working.
```
/compact                            # summarize everything
/compact focus on the auth changes  # summarize but preserve this thread
```
- Use when: same task, context getting heavy
- Key fact: project-root CLAUDE.md and auto-memory survive compaction — conversation history gets summarized

**`/context`**
Shows a breakdown of what's consuming your context window right now.
- Use before you hit the limit, not after
- Helps spot: are files being re-read? Is memory bloated?

**`/btw`**
Ask a side question without it becoming part of the main conversation history.
```
/btw what does the --bare flag do?
```
- The answer appears but doesn't bloat your working context
- Perfect for quick lookups mid-task

---

### Group 2: Session Control

**`/plan`**
Enters plan mode — Claude becomes read-only. It can explore and think but cannot edit files or run write commands.
- Use before any non-trivial implementation
- Switch back to default mode with `Shift+Tab`

**`/branch`**
Forks the current session — creates a new session starting from this exact point.
- Use when: you want to try a different approach without losing the current one
- Like git branches, but for conversations
- Detailed explanation in Section 7 below

**`/resume`**
Opens the session picker to jump back into a past conversation.

**`/diff`**
Shows all file changes made in the current session — a running summary of what Claude has touched.

---

### Group 3: Code Quality

**`/review`**
Fast in-session code review of your current changes (git diff).
```
/review
/review 123    # review PR #123
```
- Runs in your current context — quick feedback
- Use during iteration

**`/ultrareview`**
Launches a fleet of agents in a cloud sandbox to deeply review your branch.
- Independent agents verify every finding
- Takes 5–10 minutes, runs in background
- Use before merging, not during iteration

**`/simplify`**
Spawns 3 parallel agents to review recently changed files for code reuse, quality, and efficiency. Applies fixes automatically.
```
/simplify
/simplify focus on memory efficiency
```

**`/security-review`**
Security-focused review of pending changes — looks for OWASP top 10, injection risks, credential exposure, etc.

---

### Group 4: Setup & Configuration

**`/init`**
Analyzes your codebase and generates a starter CLAUDE.md.
- Run this in every new project
- Then refine what it generates

**`/memory`**
Opens an editor to view/edit all CLAUDE.md files and auto-memory entries.
- Use to audit what Claude has learned about your project
- Toggle auto-memory on/off from here

**`/mcp`**
Add, enable, disable, and manage MCP servers (external integrations like GitHub, databases, Slack).

**`/agents`**
Manage custom subagent configurations.

**`/skills`**
List and manage available skills (reusable procedures invoked with `/skill-name`).

**`/permissions`**
View and edit current permission allow/deny rules.

**`/config`**
Configure UI settings: theme, editor mode (vim/normal), diff tool, etc.

---

### Group 5: Automation

**`/loop`**
Run a prompt on a recurring interval — Claude keeps repeating the task autonomously.
```
/loop 30m check if CI is passing
/loop        # Claude self-paces based on what it's doing
```

**`/schedule`**
Create a cloud Routine — runs on Anthropic's infrastructure (no local machine needed). Supports cron, API triggers, and GitHub event triggers.

---

### Group 6: Utilities

| Command | Purpose |
|---------|---------|
| `/model` | Switch the Claude model mid-session |
| `/effort` | Set reasoning effort: `low`, `medium`, `high`, `xhigh` |
| `/add-dir` | Grant access to an additional directory mid-session |
| `/editor` | Open external editor for writing a long multi-line prompt |
| `/hooks` | View currently configured hooks |
| `/doctor` | Run diagnostics on your Claude Code installation |
| `/help` | Full command reference |

---

### The 6 You'll Use Every Day

| Command | When |
|---------|------|
| `/clear` | Switching to a new unrelated task |
| `/compact` | Same task, context getting heavy |
| `/context` | Check what's consuming context window |
| `/btw` | Quick side question without bloating history |
| `/plan` | Before any non-trivial implementation |
| `/review` | Quick check after making changes |

---

## 8. `/branch` — Forking a Conversation

Think of it like `git branch` but for your Claude conversation.

When you run `/branch`, Claude creates a **new session that starts from exactly this point** in the conversation — with full history up to here, but from now on the two sessions are independent.

### Why It Exists

Imagine you're mid-task and want to try two different approaches:
- Approach A: Refactor using a class-based design
- Approach B: Refactor using pure functions

Without `/branch`: pick one, see if it works, undo everything if it doesn't.

With `/branch`:
1. Reach the decision point
2. Run `/branch` — creates a fork
3. Try Approach A in the original session
4. Switch to the forked session, try Approach B
5. Compare results, keep the better one

### Practical Example

```
You: "Refactor the auth module. Here are the requirements..."
Claude: [explores, reads files, understands the code]

You: /branch    ← fork here, before any changes are made

Original session → "Use a class-based AuthService"
Forked session   → "Use standalone pure functions instead"

Both sessions start with the same understanding of the codebase.
Each goes its own direction from here.
```

### How It Differs From `/resume`

| | `/branch` | `/resume` |
|--|-----------|-----------|
| Creates new session | Yes — from current point | No — opens existing past session |
| Shares history up to now | Yes | Yes (that session's history) |
| Independent going forward | Yes | Yes |
| Use case | Trying alternatives | Continuing interrupted work |

### How It Differs From `--worktree`

| | `/branch` | `--worktree` |
|--|-----------|--------------|
| What it isolates | The **conversation** | The **git files** |
| Prevents | Conversation divergence | File edit conflicts |
| Use together? | Yes — they complement each other |

For full parallel isolation: use `/branch` for the conversation fork + `--worktree` so file changes don't collide.

### Finding Your Branches

Branched sessions appear in the session history picker (`/resume` or the history sidebar in the extension) — named by timestamp or the first message after the fork.

---

## 9. Slash Commands — Advanced Group (Module 1 continuation)

### `/compact` — Smart Summarization
```
/compact
/compact focus on the auth changes
```
- Summarizes conversation history to free context; optional focus string tells Claude what to preserve in the summary
- **What survives:** project-root `CLAUDE.md` and auto-memory — NOT conversation history
- Use when: same task, context getting heavy (not when switching tasks — use `/clear` instead)

---

### `/btw` — Side Question Without Bloating Context
```
/btw what does the --bare flag do?
```
- The answer appears but is NOT added to the working conversation history
- Perfect for quick lookups mid-task

---

### `/loop` — Recurring Autonomous Task
```
/loop 30m check if CI is passing
/loop          ← Claude self-paces
```
- With interval: runs the prompt every N minutes
- Without interval: Claude uses `ScheduleWakeup` internally to self-pace based on what it's doing
- Cache-aware pacing: sleeping under 5 min (≤270s) keeps prompt cache warm; over 5 min pays a cache miss
- Good for: polling CI, monitoring a process, repeated checks while you do other work

---

### `/schedule` — Cloud Routine (Runs Without Your Machine)
- Creates a Routine that runs on Anthropic's infrastructure
- Trigger types: **cron** (time-based), **API** (HTTP webhook), **GitHub event** (push, PR, issue)
- No local machine needed — survives computer shutdowns
- Use for: nightly reports, auto-responding to GitHub events, scheduled checks

---

### `/ultrareview` — Deep Multi-Agent Code Review
```
/ultrareview           ← reviews current branch
/ultrareview 123       ← reviews PR #123
```
- Launches a fleet of independent agents in a cloud sandbox
- Each agent reviews independently — cross-verifies findings
- Takes 5–10 minutes, runs in background; use **before merging**, not during iteration
- Contrast with `/review`: fast, in-session, single pass — use during iteration

---

### `/ultraplan` — Multi-Agent Planning
- Like `/plan` but dispatches multiple planning agents in parallel
- Each agent approaches the problem from a different angle
- Synthesizes into a single recommended approach
- Use for complex architecture decisions before touching code

---

### `/effort` — Reasoning Depth Control
```
/effort low      ← fast, less reasoning
/effort medium
/effort high
/effort xhigh    ← extended thinking, slowest
```
- Controls how much reasoning Claude applies to responses
- `xhigh` enables extended thinking — useful for hard algorithmic or architectural problems
- Balance: `low`/`medium` for routine tasks, `high`/`xhigh` for complex reasoning

---

## 10. Skills vs Commands

Both are invoked with `/name` in the chat, but they are very different under the hood.

### Commands — Simple Prompt Templates
- Location: `.claude/commands/name.md`
- Just markdown — the file content IS the prompt sent to Claude
- No frontmatter power, no tool restrictions, no forking
- Good for: quick reusable prompt snippets

### Skills — Full Workflow Definitions
- Location: `.claude/skills/<name>/SKILL.md`
- Rich frontmatter controls behavior:

```yaml
---
name: my-skill
description: What this skill does — Claude reads this to decide when to auto-invoke
model: claude-opus-4-7        # override model for this skill
effort: high                  # reasoning level
allowed-tools: [Bash, Read]   # restrict what tools can be used
context: fork                 # run in isolated context, doesn't pollute main conversation
arguments:
  - name: target
    description: The file to process
---
```

- **Dynamic context injection:** `` !`git log --oneline -5` `` in the skill body runs a shell command and injects its output before Claude reads the skill — live data at invocation time
- **Auto-invocation:** Claude reads the `description` and can invoke a skill automatically when it matches the task
- **`context: fork`:** skill runs in its own context window — high-context skills won't bloat your main conversation

### When to Use Each

| | Commands | Skills |
|--|----------|--------|
| Complexity | Simple prompt | Multi-step workflow |
| Frontmatter | None | Full (model, effort, tools, context) |
| Dynamic content | No | Yes (`` !`cmd` `` injection) |
| Auto-invocation | No | Yes (via `description`) |
| Context isolation | No | Yes (`context: fork`) |

---

## 11. Plugins — Packaged Distribution Units

A **plugin** is an installable package that bundles multiple Claude Code capabilities together: skills, agents, hooks, MCP servers.

### Enabling Plugins
In `settings.json`:
```json
{
  "enabledPlugins": {
    "plugin-name@marketplace-id": true
  }
}
```

Marketplace IDs:
- `claude-code-marketplace` — community marketplace
- `claude-plugins-official` — Anthropic-maintained
- `anthropic-tools` — Anthropic's own tooling plugins
- `builtin` — shipped with Claude Code itself

### Plugin Directory Structure

A plugin is a folder. The `.claude-plugin/` subfolder is the required marker that identifies it as a plugin and holds its metadata:

```
.claude/plugins/<plugin-name>/
├── .claude-plugin/
│   └── plugin.json          ← required manifest (name, description, version, author)
├── commands/                ← slash command definitions (.md files, optional)
├── agents/                  ← subagent definitions (optional)
├── skills/                  ← skill definitions (optional)
├── hooks/                   ← hook configurations (optional)
├── .mcp.json                ← MCP server config for this plugin (optional)
└── README.md
```

**`plugin.json` example** (from the official `code-review` plugin):
```json
{
  "name": "code-review",
  "description": "Automated code review for pull requests using multiple specialized agents",
  "version": "1.0.0",
  "author": {
    "name": "Your Name",
    "email": "you@example.com"
  }
}
```

### What a Plugin Can Include

| Folder / File | What it provides |
|---------------|-----------------|
| `commands/` | Slash command `.md` files (e.g. `code-review.md` → `/code-review`) |
| `agents/` | Subagent definitions with restricted tools/model |
| `skills/` | Reusable skill definitions |
| `hooks/` | Lifecycle automation (PreToolUse, PostToolUse, SessionStart, etc.) |
| `mcp-server/` | MCP server implementation code (optional — see below) |
| `.mcp.json` | MCP server connection config |

### Self-Contained Plugins with `stdio` MCP Servers

For `http`/`sse` MCP servers the server process runs externally — the plugin's `.mcp.json` just points at it.

For `stdio` MCP servers Claude Code **spawns the process itself**, so the server script can live inside the plugin folder. This makes the plugin fully self-contained — no separate install step:

```
.claude/plugins/my-plugin/
├── .claude-plugin/
│   └── plugin.json
├── commands/
├── mcp-server/
│   └── server.py        ← MCP server implementation, shipped with the plugin
├── .mcp.json            ← references the bundled server
└── README.md
```

**.mcp.json referencing the bundled server:**
```json
{
  "mcpServers": {
    "my-server": {
      "type": "stdio",
      "command": "python",
      "args": ["mcp-server/server.py"]
    }
  }
}
```

> **Note:** The exact path resolution mechanism (whether args are relative to the plugin folder, the project root, or require an absolute path) is not confirmed from the official repo — no official plugin currently bundles a stdio server this way. The pattern is architecturally valid but verify path handling before relying on it.

A skill is a single workflow; a plugin is a distributable, versioned collection of related workflows + tooling — shareable across teams or published to a marketplace.

### `.mcp.json` — Dedicated MCP Server Configuration

MCP server definitions live in a **separate file** from `settings.json`, at the same scopes:

| Scope | File |
|-------|------|
| User | `~/.claude/.mcp.json` |
| Project | `.claude/.mcp.json` |
| Managed | Set by org policy |

**Why separate from `settings.json`?** MCP configs are often team-shared or auto-generated; keeping them in their own file makes them easier to manage, gitignore selectively, or rotate.

**`.mcp.json` format:**
```json
{
  "mcpServers": {
    "github": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
      }
    },
    "postgres": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres", "${DATABASE_URL}"]
    },
    "my-api": {
      "type": "http",
      "url": "https://my-internal-api/mcp",
      "headers": {
        "Authorization": "Bearer ${MY_API_TOKEN}"
      }
    }
  }
}
```

Transport types:
- `stdio` — local process, Claude spawns it
- `http` — remote HTTP endpoint
- `sse` — Server-Sent Events (streaming remote)

Manage servers with `/mcp` or via `claude mcp add <name>`.

### Enabling/Disabling Specific MCP Servers
```json
{
  "enableAllProjectMcpServers": true,
  "enabledMcpjsonServers": ["github", "postgres"],
  "disabledMcpjsonServers": ["blocked-server"]
}
```

---

## 12. Configuration & Permissions

### Settings.json — 4 Scopes (Highest to Lowest Precedence)

| Scope | File | Committed? | Purpose |
|-------|------|-----------|---------|
| Managed | Set by org policy (registry / managed file) | N/A | Org-wide enforced rules — read-only overrides |
| User | `~/.claude/settings.json` | No | Personal preferences across all projects |
| Project | `.claude/settings.json` | Yes | Team-wide settings, checked into git |
| Local | `.claude/settings.local.json` | No (gitignore) | Personal overrides for this project only |

Later scopes override earlier ones. Managed always wins — you cannot override org policy.

**Add JSON schema for autocomplete:**
```json
{
  "$schema": "https://json.schemastore.org/claude-code-settings.json"
}
```

---

### CLAUDE.md — Instruction Hierarchy

All levels are loaded simultaneously — they stack, not override:

```
~/.claude/CLAUDE.md          ← personal, all projects
./CLAUDE.md                  ← project root (survives /compact)
./CLAUDE.local.md            ← personal project overrides (gitignored)
.claude/rules/<name>.md      ← path-scoped rules (frontmatter: paths: ["src/**/*.ts"])
```

**What survives `/compact`:** only project-root `CLAUDE.md` and auto-memory. Subdirectory CLAUDE.md files and conversation history are summarized away.

**Import syntax inside CLAUDE.md:**
```
@path/to/file   ← inlines another file's content
```

**Keep under 200 lines** — Claude ignores rules buried deep. Add a `## Compact Instructions` section for rules that must persist after compaction.

---

### 6 Permission Modes

| Mode | What Claude Can Do |
|------|--------------------|
| `default` | Prompts for anything potentially destructive |
| `plan` | Read-only — can explore files but cannot edit or run write commands |
| `acceptEdits` | Auto-approves all file edits; still prompts for shell commands |
| `auto` | Auto-approves everything except explicitly denied rules |
| `dontAsk` | Never prompts — approves all tool use silently |
| `bypassPermissions` | Skips all permission checks entirely — **sandboxes only, dangerous** |

**Cycle modes with `Shift+Tab`** (cursor in chat input): `default` → `plan` → `acceptEdits` → `auto`

**Set at launch:**
```bash
claude --permission-mode plan
claude --permission-mode acceptEdits
```

**Set as default in settings.json:**
```json
{ "permissions": { "defaultMode": "acceptEdits" } }
```

---

### Permission Rule Syntax

Rules go in `permissions.allow` or `permissions.deny` arrays in settings.json:

```json
{
  "permissions": {
    "allow": [
      "Bash(npm run *)",           // prefix wildcard — all npm run commands
      "Bash(git *)",               // all git commands
      "Bash(powershell *)",        // all PowerShell commands (including piped/compound)
      "Edit(/src/**)",             // edit anything under /src/ (project-root anchored)
      "Read"                       // all Read operations — no argument = any
    ],
    "deny": [
      "Bash(git push --force *)",  // block force push
      "Edit(/.env)"                // block editing .env
    ]
  }
}
```

**Path anchoring:**
- `/src/**` — anchored to project root
- `~/` — anchored to home directory
- `//` — anchored to filesystem root (absolute path)

**Precedence:** `deny` > `ask` > `allow` — deny always wins.

**Compound/piped commands:** Claude evaluates each subcommand independently. `Bash(powershell *)` covers `powershell Get-Date`, `powershell -Command "..."`, and any compound PS expression — because the initial `powershell` invocation matches the prefix wildcard.

**MCP tool rules:**
```json
"mcp__github"                  // all GitHub MCP tools
"mcp__github__read_file"       // specific MCP tool
```

---

### `/fewer-permission-prompts` — Auto-Build Your Allowlist

Scans your session transcript history and generates `allow` rules for every command you've approved before. Run it once per project after a few sessions to reduce future prompts.

---

### `/update-config` Skill — Safe Settings Editing

A built-in skill that handles settings.json edits safely:

1. **Read before write** — always reads the existing file first
2. **Merge, never replace** — adds to existing arrays rather than overwriting them
3. **Validates JSON** — checks syntax after editing
4. **Hook verification** — pipe-tests new hooks before writing

Invoke it by describing what you want:
```
/update-config allow all PowerShell commands in local settings
/update-config add a PostToolUse hook to run prettier after file edits
```

---

### Practical Example — What Was Set Up

Created `.claude/settings.local.json` (local scope, not committed):
```json
{
  "permissions": {
    "allow": [
      "Bash(powershell *)"
    ]
  }
}
```
Result: all PowerShell commands — including piped and compound expressions — run without a permission prompt in this project.

---

## Quick Recap — Day 1 Key Takeaways

| Concept | Key Point |
|---------|-----------|
| VS Code extension vs CLI | Same engine — extension is the graphical surface |
| Slash commands | Type `/` in prompt box — works in extension and CLI |
| `Shift+Tab` | Cycles permission modes — works in extension (cursor must be in input) |
| `Esc` once | Stop Claude mid-response |
| `Esc` `Esc` | Open rewind menu (CLI) |
| Rewind | Restores conversation + files to a prior point; does NOT undo shell side-effects |
| `Ctrl+G` | Open external editor for long prompts (CLI) |
| `Ctrl+T` | Open task list (CLI) |
| `Ctrl+O` | Open transcript viewer (CLI) |
| Checkpoints (extension) | Hover message → set checkpoint before risky ops; one-click rewind target |
| `--model` | Override model at launch; use `/model` in extension |
| `--add-dir` | Extend file access to another directory |
| `--worktree` | Isolated branch + folder for parallel sessions; CLI-only |
| `--bare` | Fast startup, skips CLAUDE.md/memory |
| `--permission-mode` | Set starting mode at launch |
| Worktree branch name | Auto-generated from task description; override with `--branch` |
| Worktree folder | Always sibling of current project; not configurable |
| Worktree in extension | Use integrated terminal + `claude --worktree` |
| `/compact` | Summarize context; optionally focus on a thread; CLAUDE.md survives |
| `/btw` | Side question — answer doesn't enter conversation history |
| `/loop` | Recurring autonomous task; self-paced or fixed interval |
| `/schedule` | Cloud Routine — runs without your machine; cron/API/GitHub triggers |
| `/ultrareview` | Deep fleet review before merging; 5–10 min background run |
| `/ultraplan` | Multi-agent planning for complex decisions |
| `/effort` | Reasoning depth: low / medium / high / xhigh (extended thinking) |
| Skills vs Commands | Skills have frontmatter, tool control, context isolation, auto-invocation |
| Plugins | Installable bundles of skills + agents + hooks + MCP servers |
| Settings scopes | managed > user > project > local (local = gitignored personal override) |
| CLAUDE.md | All levels load simultaneously; only project-root survives `/compact` |
| Permission modes | 6 modes: default / plan / acceptEdits / auto / dontAsk / bypassPermissions |
| Rule syntax | `Bash(npm *)` prefix wildcard; `Edit(/src/**)` path wildcard; `Read` tool-only |
| Rule precedence | deny > ask > allow — deny always wins |
| Compound commands | Each subcommand evaluated independently; `Bash(powershell *)` covers all PS |
| `/fewer-permission-prompts` | Scans transcript history and auto-builds your allowlist |
| `/update-config` | Safe settings editing: read-before-write, merge arrays, validates JSON |
