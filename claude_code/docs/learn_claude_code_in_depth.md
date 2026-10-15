# Learn Claude Code In Depth — Learning Curriculum

## Context
Claude Code is Anthropic's CLI/IDE tool for agentic software development. Learning it in depth means mastering: the CLI and its interaction model, configuration and memory, the permissions and hooks systems, MCP integrations, the subagents/skills/agent-teams features, automation (scheduling, loops), and the underlying Claude API and Agent SDK for building custom agents. This plan structures that learning progressively — foundations first, then customization, then advanced agent orchestration, finishing with production API usage.

---

## Module 1 — Core CLI & Session Management (Days 1–3)

**Goal:** Be fluent in starting, continuing, and managing Claude Code sessions.

### Topics
- Starting Claude: `claude`, `claude -p "query"`, `claude -c`, `claude -r`
- Key flags: `--model`, `--add-dir`, `--worktree`, `--bare`, `--permission-mode`
- All slash commands grouped by category:
  - Session: `/clear`, `/compact`, `/btw`, `/branch`, `/diff`, `/context`, `/resume`
  - Mode & model: `/plan`, `/model`, `/effort`, `/config`, `/permissions`
  - Code quality: `/review`, `/simplify`, `/security-review`
  - Setup: `/init`, `/memory`, `/agents`, `/mcp`, `/skills`
  - Advanced: `/loop`, `/schedule`, `/ultrareview`, `/ultraplan`
    - `/loop` — repeat a task autonomously on an interval or let Claude self-pace:
      ```
      /loop 30m check if CI is passing and summarize any failures
      /loop      # Claude decides its own wake-up cadence based on what it's doing
      ```
    - `/schedule` — create a cloud Routine that runs on Anthropic infra (no local machine needed):
      ```
      /schedule every day at 9am summarize open PRs in this repo
      /schedule on push to main run the security review skill
      /schedule when issue labeled "needs-triage" draft a response with context
      ```
      Trigger types: cron (time-based), API webhook, GitHub event (push / PR / issue)
    - `/ultrareview` — dispatch a fleet of independent agents for deep review; use before merging, not during iteration:
      ```
      /ultrareview           # reviews current branch vs main
      /ultrareview 123       # reviews GitHub PR #123
      ```
      Takes 5–10 min, runs in background. Each agent reviews independently and findings are cross-verified.
    - `/ultraplan` — multi-agent planning; each agent attacks the problem from a different angle, then synthesizes:
      ```
      /ultraplan refactor the auth module to support OAuth and API key auth side by side
      ```
      Use for complex architecture decisions before touching any code.
- Essential keyboard shortcuts: `Shift+Tab` (cycle modes), `Esc` (stop/rewind), `Ctrl+G` (external editor), `Ctrl+T` (task list), `Ctrl+O` (transcript)
- Session workflow: plan mode → implementation mode → review cycle
- Parallel sessions with `--worktree` and git worktrees

### Milestones
- [ ] Start a session and complete a real coding task end-to-end
- [ ] Use plan mode (`Shift+Tab Shift+Tab`) before implementing a change
- [ ] Compact a long session with `/compact`, then verify CLAUDE.md survived
- [ ] Run two parallel sessions with `--worktree` on separate branches

---

## Module 2 — Configuration & Memory (Days 3–5)

**Goal:** Understand all persistence mechanisms so Claude retains context across sessions.

### Topics
- **CLAUDE.md hierarchy**: managed policy → user (`~/.claude/CLAUDE.md`) → project (`./CLAUDE.md`) → local (`./CLAUDE.local.md`)
- Path-scoped rules in `.claude/rules/` with frontmatter `paths:` globs
- `@path/to/file` imports inside CLAUDE.md
- What survives compaction: only project-root CLAUDE.md + auto-memory
- **Auto-memory**: what it stores, `MEMORY.md` index (200 lines cap), topic files, `/memory` command to audit
- **settings.json** configuration scopes (user → project → local → managed)
- Key settings categories: permissions, model/AI behavior, UI/display, hooks, env vars, MCP, git attribution
- Using JSON schema: `"$schema": "https://json.schemastore.org/claude-code-settings.json"`
- Running `/init` to generate a starter CLAUDE.md from codebase analysis
- `/fewer-permission-prompts` to auto-build your allowlist from transcripts

### Milestones
- [ ] Write a CLAUDE.md with project conventions, build commands, and a "Compact Instructions" section
- [ ] Create a path-scoped rule file in `.claude/rules/tests.md` for test-writing conventions
- [ ] Review auto-memory with `/memory` and verify it captured useful insights
- [ ] Configure settings.json with 5+ permission allow rules and 2+ deny rules

---

## Module 3 — Permissions & Security (Days 5–7)

**Goal:** Configure Claude to work autonomously without constant permission prompts, safely.

### Topics
- **Permission modes**: `default`, `plan`, `acceptEdits`, `auto`, `dontAsk`, `bypassPermissions`
- Permission rule syntax: exact match, wildcards (`npm run *`, `git * main`), path anchoring (`/src/**` = project root, `~/` = home, `//` = filesystem root)
- Read-only commands that never prompt: `ls`, `cat`, `grep`, `find`, `git status`, etc.
- Compound commands: rules per subcommand; process wrappers auto-stripped
- MCP tool rules: `mcp__github`, `mcp__github__read_file`
- Rule precedence: deny > ask > allow
- Sandbox mode: OS-level filesystem + network isolation
- Symlink handling: allow requires both sides, deny triggers on either
- Managed settings for org-wide policy (read-only overrides)
- Security: prompt injection safeguards, credential handling, `bypassPermissions` only in sandboxes

### Milestones
- [ ] Set up a project settings.json with granular allow/deny rules covering your common workflow
- [ ] Run `/fewer-permission-prompts` and review its suggestions
- [ ] Test each permission mode and understand when to use each
- [ ] Enable sandbox mode for an untrusted script experiment

---

## Module 4 — Hooks System (Days 7–10)

**Goal:** Use hooks for deterministic, automated enforcement of project rules.

### Topics
- **Hook types**: command (shell), prompt (LLM-evaluated), agent (subagent), HTTP (webhook), async (background)
- Hook exit codes: 0 = allow, 1 = error/fall-through, 2 = block
- **Key hook events**:
  - `SessionStart` — initialize env, load credentials
  - `UserPromptSubmit` — intercept/modify user input
  - `PreToolUse` — validate/modify/block tool calls
  - `PostToolUse` — react to tool results (run linter, notify)
  - `PermissionRequest` — auto-approve programmatically
  - `SubagentStart/Stop`, `TaskCreated/Completed` — agent lifecycle
  - `PreCompact/PostCompact` — inject context around compaction
  - `FileChanged`, `CwdChanged`, `ConfigChange` — react to events
- Hook `if` matchers: `tool`, `args` (glob), `regex`
- Hook configuration in settings.json at project/user/managed scope

### Milestones
- [ ] Write a `PostToolUse` hook that auto-runs `prettier` after any `.ts` file edit
- [ ] Write a `PreToolUse` hook that blocks edits to a protected directory
- [ ] Write a `PreCompact` prompt hook that re-injects critical context
- [ ] Write an async `PostToolUse` hook that runs tests after file changes

---

## Module 5 — MCP Servers & External Integrations (Days 10–13)

**Goal:** Connect Claude to external systems via Model Context Protocol.

### Topics
- What MCP is: standard protocol to expose tools, resources, and prompts to Claude
- Transport types: `stdio` (local process), `http` (remote HTTP), `sse` (Server-Sent Events)
- Configuration scopes: user (`~/.claude/.mcp.json`), project (`.claude/.mcp.json`), managed
- Authentication: env vars, OAuth, custom headers, `apiKeyHelper` script
- Tool discovery: deferred schema loading via MCP Tool Search (saves context)
- Popular server categories: GitHub, databases (PostgreSQL, MongoDB), Slack, Sentry, Figma, Linear
- `allowedMcpServers` / `deniedMcpServers` for org policy
- `/mcp` command: add, enable/disable, manage servers

### Milestones
- [ ] Configure the GitHub MCP server and use it to comment on a PR from Claude
- [ ] Configure a database MCP server and run a query through Claude
- [ ] Add a custom HTTP MCP server (or Slack) and use it in a workflow
- [ ] Verify MCP tool search deferral is working with `/context`

---

## Module 6 — IDE Integration (Days 13–15)

**Goal:** Use Claude Code seamlessly inside VS Code or JetBrains.

### Topics
- **VS Code**: graphical panel, inline diffs, `@`-mentions with fuzzy matching, session history, `Option/Alt+K` for file references, `Cmd/Ctrl+Esc` to focus, checkpoints (rewind), browser automation via `@browser`
- **JetBrains**: quick launch (`Cmd/Ctrl+Esc`), IDE diff viewer, diagnostics auto-shared, file references (`Cmd+Option+K`), `/ide` command to connect from external terminal
- Extension settings vs `~/.claude/settings.json` — what lives where
- Remote development: plugin on remote host, WSL2 networking fix
- Session management from IDE: multiple tabs, parallel tasks
- VS Code-specific: `autosave`, `initialPermissionMode`, `preferredLocation`

### Milestones
- [ ] Complete a full feature implementation using only the VS Code extension (no terminal)
- [ ] Use `@browser` for UI testing against a local dev server
- [ ] Set up JetBrains integration with IDE diff viewer and diagnostics sharing
- [ ] Create a rewind checkpoint before a risky refactor and test restoring it

---

## Module 7 — Subagents & Skills (Days 15–20)

**Goal:** Delegate work to specialized agents and create reusable workflows.

### Topics
- **Subagent types**: `Explore` (read-only research), `Plan` (architecture design), `general-purpose` (default), custom
- Subagent configuration: `.claude/agents/<name>/AGENT.md` with frontmatter (name, model, tools, skills, permissions)
- Foreground vs background agents; when to spawn vs handle directly
- Context isolation: subagents summarize back, don't fill main context
- **Skills**: reference (knowledge Claude applies) vs task (step-by-step procedures)
- Skill locations: `~/.claude/skills/` (user) and `.claude/skills/` (project)
- `SKILL.md` frontmatter: `description`, `disable-model-invocation`, `allowed-tools`, `context: fork`, `model`, `effort`, `arguments`
- Dynamic context injection: `` !`shell command` `` runs before Claude sees the skill
- Skills invoked with `/skill-name` or auto-invoked by Claude based on `description`
- Built-in skills: `/review`, `/init`, `/simplify`, `/security-review`, `/ultrareview`

### Milestones
- [ ] Create an `Explore` subagent scoped to your codebase to answer architecture questions
- [ ] Create a `security-reviewer` custom agent with restricted tools
- [ ] Write a task skill for your team's PR workflow (push + create PR + fill template)
- [ ] Write a reference skill for project API conventions with dynamic context injection
- [ ] Use `context: fork` to run a high-context skill without polluting main conversation

---

## Module 8 — Multi-Agent Orchestration & Agent Teams (Days 20–24)

**Goal:** Coordinate multiple Claude agents for complex parallel work.

### Topics
- **Subagents vs Agent Teams**: subagents report to main only; teams communicate peer-to-peer
- Agent team coordination: shared task list, claim/release work, dependency tracking
- Lead agent pattern: one lead synthesizes; teammates own domains
- Display modes: in-process (terminal cycling) or split panes (tmux/iTerm2)
- Plan approval gates: require teammates to plan before implementing
- Parallel workers with worktrees: one git worktree per agent, no conflicts
- When teams win: code review (security + perf + tests in parallel), bug investigation (competing hypotheses), large refactors (frontend + backend + tests teams)
- Cost/token considerations: each agent has its own context window

### Milestones
- [ ] Run a 3-agent code review team (security, performance, tests) on a real PR
- [ ] Use a 2-agent pattern (writer + reviewer) for a feature implementation
- [ ] Investigate a bug with 3 agents exploring competing hypotheses
- [ ] Set up a tmux split-pane display for an agent team session

---

## Module 9 — Scheduling & Automation (Days 24–27)

**Goal:** Run Claude autonomously on schedules and in CI/CD pipelines.

### Topics
- **`/loop`**: session-scoped recurring tasks, self-paced vs interval, `ScheduleWakeup` for cache-aware timing (< 270s = warm cache, ≥ 300s = cache miss)
- **Desktop scheduled tasks**: machine-local, persistent, 1-minute minimum, access to local MCP servers
- **Cloud Routines** (`/schedule`): Anthropic-hosted, persistent, 1-hour minimum, no local machine needed
  - Trigger types: schedule (cron), API endpoint (POST with bearer token), GitHub events (PR opened, push, release)
- Channels (webhooks): push external messages into a running session
- Remote triggers from CI/CD: curl a Routine API endpoint from deploy pipelines
- Auto-memory and hooks for state tracking across scheduled runs

### Milestones
- [ ] Use `/loop 30m` to poll CI status while working on something else
- [ ] Create a Desktop scheduled task that runs a daily codebase health check
- [ ] Create a cloud Routine with both a schedule trigger and an API trigger
- [ ] Call a Routine from a GitHub Actions workflow after a deploy

---

## Module 10 — Claude API & Anthropic SDK (Days 27–33)

**Goal:** Build custom Claude-powered applications using the full API feature set.

### Topics
- **Tool use / function calling**: define tools (JSON schema), tool calling loop, parallel tool use, strict tool use, built-in tools (code execution, web search, bash)
- **Streaming**: stream text tokens, fine-grained tool streaming, real-time UI patterns
- **Prompt caching**: `cache_control` tokens, cache hit rates, cost optimization (10% cost on cache reads), when to cache (long docs, system prompts, few-shot examples)
- **Extended thinking**: thinking tokens, budget control, adaptive thinking, use cases (complex reasoning, math, debugging)
- **Batch API**: JSONL format, async submit/poll/retrieve, 50% cost reduction, per-request error handling
- **Files API**: upload documents/images, reference by file ID, 24-hour expiration
- **Structured outputs**: Pydantic (Python) / Zod (TypeScript) schema validation
- **Vision & PDF support**: image analysis, multi-page PDF processing
- **Context editing**: modify past messages without resubmitting
- **Citations**: source attribution in responses
- Model selection: `claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5-20251001`

### Milestones
- [ ] Build an agentic loop: 3 custom tools, parallel tool calling, error handling
- [ ] Implement streaming in a web UI showing tool execution progress
- [ ] Add prompt caching to a multi-turn agent; measure cache hit rates
- [ ] Use extended thinking on a complex reasoning problem; compare results
- [ ] Submit a 50-item batch job; retrieve and handle results
- [ ] Build a structured output pipeline with Pydantic/Zod validation

---

## Module 11 — Claude Agent SDK (Days 33–40)

**Goal:** Build, deploy, and operate production-grade custom agents.

### Topics
- **Core SDK concepts** (Node.js/TypeScript + Python): the agentic loop, turns, messages (user/assistant/system/result), tool execution pipeline
- Sessions: continue, resume by ID, fork (explore alternatives without losing original)
- Custom tools: tool annotations, error handling, returning images/resources
- MCP integration in SDK: stdio/HTTP/SSE, OAuth, tool discovery
- Streaming vs single-mode I/O; user input handling (approval requests, clarifying questions)
- Hooks in SDK: `PreToolUse`, `PostToolUse`, `PermissionRequest` for deterministic control
- Permission modes in SDK: `acceptEdits`, `dontAsk`, `bypassPermissions`, `plan`
- Cost tracking: per-step, per-model usage, cache tokens
- Observability: telemetry export, tracing, user attribution
- File checkpointing: restore to pre-operation state on failure
- **Deployment patterns**: ephemeral (new container per request), long-running (persistent), hybrid
- **Security**: gVisor/container/VM isolation, read-only code mounts, credential proxy pattern, MCP tool access restrictions

### Milestones
- [ ] Build an SDK agent with 3 custom tools + 1 MCP server; test parallel tool use
- [ ] Implement streaming I/O, user clarification questions, and hook-based auto-approval
- [ ] Add structured JSON output validation with Pydantic/Zod
- [ ] Track cost per request and export telemetry
- [ ] Deploy the agent to a container with proper isolation and credential proxy
- [ ] Implement file checkpointing before a risky multi-file operation

---

## Module 12 — Expert Patterns & Anti-Patterns (Days 40–45)

**Goal:** Internalize what separates expert Claude Code users from novices.

### Expert Workflows
- **Explore → Plan → Code → Review** as the default cycle
- Parallel sessions (Writer + Reviewer) for higher-quality output
- Research via subagents to keep main context clean
- Batch autonomous work with `/loop` or Routines
- Giving Claude verification criteria upfront (tests, expected output, screenshots)

### Critical Context Habits
- Monitor `/context` every 30–60 min in long sessions
- `/clear` between unrelated tasks (not `/compact`)
- `/compact "focus on X"` when heavy but continuing
- `/btw` for side questions that shouldn't bloat history
- Put persistent rules in CLAUDE.md, not conversation turns

### CLAUDE.md Discipline
- Keep under 200 lines; prune like code
- Ask: "Would removing this line cause a mistake?" — if no, delete it
- Move deterministic rules to hooks; CLAUDE.md is for judgment and context
- Add "Compact Instructions" section for rules that must survive compaction

### Common Pitfalls
- **Kitchen sink sessions**: unrelated tasks in one session → context degrades. Fix: `/clear`.
- **Correction loops**: correct twice with no progress → `/clear` + better prompt.
- **Oversized CLAUDE.md**: Claude ignores buried rules. Fix: prune ruthlessly.
- **Infinite exploration**: no scope on "investigate" → context explodes. Fix: use subagents or `@file` references.
- **Trust-then-ship gap**: no verification criteria → edge cases slip. Fix: always give Claude a test or expected output.
- **Wrong problem solved**: jumping to code without understanding scope. Fix: plan mode first.

### Milestones
- [ ] Build and internalize a personal daily workflow: session naming, mode cycling, review cadence
- [ ] Audit your CLAUDE.md and prune it to under 100 lines
- [ ] Convert 3 CLAUDE.md advisory rules to enforcing hooks
- [ ] Complete a complex feature using the full Expert cycle: Explore subagent → Plan mode → Implementation → `/review` → `/ultrareview`

---

## Verification: How to Know You've Mastered Each Module

| Module | Verification Signal |
|--------|---------------------|
| 1. Core CLI | Can start/resume/fork sessions, switch modes fluently |
| 2. Config & Memory | CLAUDE.md survives compaction; auto-memory captures useful insights |
| 3. Permissions | Zero unexpected permission prompts in daily workflow |
| 4. Hooks | Auto-formatting, test running, context injection all work without manual triggers |
| 5. MCP | GitHub and 1 database server integrated; queries and PR actions work |
| 6. IDE | Full feature built inside VS Code extension; checkpoints used for risky changes |
| 7. Subagents & Skills | Custom agents and skills reduce repetitive prompting by >50% |
| 8. Agent Teams | 3-agent parallel review produces richer feedback than single-pass |
| 9. Scheduling | Daily automated check runs without opening Claude; CI calls a Routine |
| 10. Claude API | Streaming agent with prompt caching, structured output, and tool use working |
| 11. Agent SDK | Deployed agent handles multi-turn sessions with cost tracking and observability |
| 12. Expert Patterns | Context stays clean; CLAUDE.md is lean; full cycle used for complex tasks |

---

## Time Estimate
- **Focused daily learning (2–3 hrs/day)**: ~45 days
- **Alongside regular work**: ~3 months
- Modules 1–6 are usable skills; Modules 7–12 are mastery-level

## Key Resources
- Claude Code docs: `claude.ai/code` / in-product `/help`
- Use `/init` to generate a CLAUDE.md for any new codebase
- Use `/fewer-permission-prompts` to reduce friction immediately
- Use `/memory` to audit what Claude has learned about your project
