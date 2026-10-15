# Learn Claude Code — 5-Day Plan

---

## Day 1 — CLI Fluency & Session Management

**Goal:** Navigate Claude Code confidently from the terminal and IDE.

| What | How |
|------|-----|
| Start / resume sessions | `claude`, `claude -c`, `claude -r` |
| One-shot queries | `claude -p "query"` |
| Parallel work | `claude --worktree` (separate git checkout per session) |
| Key slash commands | `/clear`, `/compact`, `/btw`, `/context`, `/plan`, `/review`, `/init` |
| Mode cycling | `Shift+Tab` → plan → acceptEdits → auto → default |
| Stop / rewind | `Esc` once to stop, twice to open rewind menu |
| VS Code extension | `Cmd/Ctrl+Esc` to focus, `Alt+K` for `@file` references, inline diffs, checkpoints |

**Practice:** Complete one real coding task using the full cycle — plan mode to understand scope, switch to default mode to implement, `/review` to check the result.

---

## Day 2 — Configuration, Memory & Permissions

**Goal:** Make Claude remember your preferences and stop asking for permission on safe commands.

**Memory hierarchy** (highest to lowest precedence):
- `~/.claude/settings.json` — personal, all projects
- `.claude/settings.json` — team, checked into git
- `CLAUDE.md` at project root — instructions that survive `/compact`
- Auto-memory — Claude writes learnings to `~/.claude/projects/.../memory/`

**Key CLAUDE.md rules:**
- Keep under 200 lines — Claude ignores buried rules
- Add a `## Compact Instructions` section for rules that must persist after compaction
- Use `.claude/rules/` with `paths:` frontmatter for file-type-specific rules

**Permissions to configure in `.claude/settings.json`:**
```json
{
  "permissions": {
    "allow": ["Bash(npm run *)", "Bash(git commit *)", "Bash(git status)", "Edit(/src/**)"],
    "deny": ["Bash(git push --force *)", "Edit(/.env)"]
  }
}
```

**Shortcut:** Run `/fewer-permission-prompts` — Claude scans your session history and builds the allowlist for you.

**Practice:** Run `/init` on a project, refine the generated CLAUDE.md, and run `/fewer-permission-prompts` to reduce future prompts.

---

## Day 3 — Hooks & MCP Servers

**Goal:** Automate enforcement of project rules and connect Claude to external systems.

**Hooks — deterministic shell commands at lifecycle events:**

| Event | Common use |
|-------|-----------|
| `PostToolUse` (Write/Edit) | Auto-run linter/formatter after file edits |
| `PreToolUse` (Bash) | Block dangerous commands |
| `PreCompact` | Re-inject critical context before compaction |
| `SessionStart` | Load env vars, credentials |

Hook exit codes: `0` = allow, `1` = show error, `2` = block.

```json
{
  "hooks": [
    {
      "event": "PostToolUse",
      "if": { "tool": "Write" },
      "command": "prettier --write {{path}}"
    }
  ]
}
```

**MCP Servers — connect Claude to external tools:**
- Transport types: `stdio` (local process), `http`/`sse` (remote)
- Use `/mcp` to add and manage servers
- Popular: GitHub (PRs/issues), PostgreSQL (queries), Slack, Sentry, Linear

**Practice:** Write one `PostToolUse` hook (e.g., auto-lint on save) and configure the GitHub MCP server to comment on a PR from Claude.

---

## Day 4 — Subagents, Skills & IDE Integration

**Goal:** Delegate work to specialized agents and build reusable workflows.

**Subagents — isolated context windows for focused tasks:**
- `Explore` subagent: read-only research, keeps main context clean
- `Plan` subagent: architecture and design thinking
- Custom agents: define in `.claude/agents/<name>/AGENT.md` with restricted tools
- Key insight: subagents summarize back — they don't bloat your main context

**Skills — reusable procedures invoked with `/skill-name`:**
- **Reference skills** (`~/.claude/skills/`): conventions, patterns Claude applies automatically
- **Task skills** (`.claude/skills/`): step-by-step procedures (deploy, create-pr, code-review)
- `SKILL.md` frontmatter: `description` (triggers auto-invocation), `allowed-tools`, `context: fork`
- Dynamic injection: `` !`git log --oneline -5` `` runs a shell command before Claude sees the skill

**IDE (VS Code) essentials:**
- Inline diffs before accepting changes
- `@browser` for live UI testing against your dev server
- Checkpoints: hover any message → rewind to restore code state
- Multiple tabs for parallel tasks

**Practice:** Create an `Explore` subagent for your codebase and write a task skill for your PR workflow (push + create PR + fill description template).

---

## Day 5 — Multi-Agent Patterns, Automation & Expert Habits

**Goal:** Orchestrate parallel agents and build autonomous workflows.

**Multi-agent patterns:**
- **Writer + Reviewer**: two sessions, one implements, one does a fresh-context review
- **Agent teams**: 3 parallel reviewers (security, performance, tests) on the same PR
- **Worktrees**: `--worktree` gives each agent its own git checkout, no conflicts

**Automation:**
- `/loop 30m check if CI passed` — polls autonomously while you work
- `/schedule` — create a cloud Routine (cron schedule, API trigger, or GitHub event trigger)
- Cloud Routines run on Anthropic infrastructure — no local machine needed

**Expert habits that separate novice from advanced users:**

| Novice | Expert |
|--------|--------|
| One long kitchen-sink session | `/clear` between unrelated tasks |
| Correct Claude 3+ times | `/clear` + rewrite the prompt |
| Instructions in conversation | Instructions in CLAUDE.md |
| Read files ad-hoc | `@file` references, or delegate to Explore subagent |
| No success criteria | Give Claude a test or expected output to verify against |
| CLAUDE.md keeps growing | Prune ruthlessly — if removing it won't cause mistakes, delete it |

**The expert cycle for any non-trivial task:**
1. Explore subagent → understand scope
2. `/plan` mode → design before touching code
3. Implementation → with success criteria
4. `/review` → fast in-session check
5. `/ultrareview` → deep multi-agent check before merging

**Practice:** Pick a real feature or bug. Run the full expert cycle end-to-end. Set up one `/loop` or scheduled Routine for something you check repeatedly.

---

## Quick Reference

| Need | Command |
|------|---------|
| Start fresh session | `claude` |
| Resume last session | `claude -c` |
| One-off query | `claude -p "..."` |
| Enter plan mode | `Shift+Tab Shift+Tab` |
| Compact context | `/compact "focus on X"` |
| Side question (no history bloat) | `/btw your question` |
| Build permission allowlist | `/fewer-permission-prompts` |
| Generate CLAUDE.md | `/init` |
| Audit auto-memory | `/memory` |
| Check context usage | `/context` |
| Deep code review | `/ultrareview` |
