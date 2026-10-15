# Claude Code — Agents Deep Dive

**Date:** 2026-05-11
**Topic:** Subagents & Agent Teams — full reference
**Curriculum:** Module 7 (Days 15–20) + Module 8 (Days 20–24)

---

## 1. Recap — What Day 5 Covered

See `day5_skills_deep_dive.md` sections 14–17 for the foundation. Quick summary:

| Topic | Coverage in Day 5 |
|-------|-------------------|
| Agent types | Explore (read-only), Plan (design), general-purpose, custom |
| AGENT.md fields | name, description, model, tools, permissions.deny, skills |
| Foreground vs background | Foreground = wait for result; background = continue working |
| Skill vs agent | Skill = same Claude instance; agent = separate Claude instance |

**What Day 5 did not cover:** agent lifecycle, context isolation mechanics, auto-spawning, parallel agents, worktree-per-agent pattern, agent teams (peer-to-peer), lead agent pattern, plan approval gates, display modes, cost/token model, debugging, passing context effectively.

---

## 2. Agent Lifecycle

### What happens when an agent is spawned

```
Main Claude decides to spawn an agent
    │
    ▼
Claude Code creates a new session:
    - Fresh context window (0 tokens)
    - AGENT.md body loaded as system prompt
    - Frontmatter constraints applied (tools, permissions, model)
    │
    ▼
Agent runs autonomously:
    - Makes tool calls in its own context
    - Can spawn its own sub-agents (nested)
    - Internal chain of thought never visible to main
    │
    ▼
Agent completes:
    - Writes a summary of findings/results
    - Summary injected into main conversation as a tool result
    - Agent session is discarded (not resumable by default)
```

### Session IDs

Every agent session has an ID. If you need to resume an agent (rare — usually you just re-spawn):

```
claude -r <session-id>    # resume a specific agent session
```

Claude Code logs agent session IDs if you need to recover a long-running agent's work after a crash.

---

## 3. Context Isolation — How Summaries Work

The main conversation never sees the agent's internal work. What it gets is a **summary** — but the quality of that summary depends on how you instruct the agent.

### What transfers back

```
Agent's internal context (invisible):
    - All tool calls and their results
    - Agent's reasoning chain
    - Files read, searches run, errors encountered

What main conversation receives:
    - A single text block (the agent's final response)
    - Nothing else
```

### Controlling summary quality

By default the agent summarizes however it sees fit. To get structured, useful output, instruct the agent in AGENT.md or in the spawn prompt:

**In AGENT.md body:**
```markdown
When you finish, report in this format:
## Summary
One paragraph of what you found.

## Findings
Bullet list: file path, issue, severity.

## Recommended Next Steps
Ordered list of actions for the main session to take.
```

**In the spawn prompt (ad-hoc):**
```
"Explore the authentication module. Report back: (1) which files handle auth,
(2) the JWT validation flow step by step, (3) any security gaps you notice.
Be specific — include file paths and line numbers."
```

Vague prompts → vague summaries. The main session is blind to the agent's internal work, so the summary is your only output — design it explicitly.

---

## 4. AGENT.md — Complete Frontmatter Reference

```yaml
---
name: agent-name                        # required — identifier for spawning

description: >                          # required — Claude reads to decide auto-spawning
  What this agent does and when to use it.
  Be specific for reliable auto-spawning.

model: claude-opus-4-7                  # optional — model for this agent session
effort: high                            # optional — low / medium / high / xhigh

tools:                                  # optional — whitelist (all others blocked)
  - Read
  - Glob
  - Grep
  - Bash(git log *)
  - Bash(git diff *)
  - mcp__github__list_issues
  - mcp__github__*                      # wildcard

permissions:                            # optional — explicit denies
  deny:
    - Write
    - Edit
    - Bash(git push *)
    - Bash(rm *)

skills:                                 # optional — skills this agent can invoke
  - security-review
  - code-review

context: isolated                       # optional — always isolated (default for agents)
---
```

### `tools` vs `permissions.deny` — precedence

Both restrict what the agent can do. They interact as follows:
- `tools` is a **whitelist** — only listed tools are available
- `permissions.deny` is a **blacklist** — explicitly blocks regardless of whitelist
- Deny always wins

```yaml
tools:
  - Bash              # all Bash allowed by whitelist...
permissions:
  deny:
    - Bash(git push *) # ...except git push, blocked by deny
```

### `skills` field

An agent can invoke skills during its work. List which skills it has access to:

```yaml
skills:
  - security-review   # agent can run /security-review as part of its workflow
  - format-imports
```

Without this field, agents inherit no skills by default.

---

## 5. Auto-Spawning — How Claude Decides

Like skills, Claude reads agent `description` fields and can auto-spawn an agent when the task matches. The same rules apply:

```yaml
# Too vague — never auto-spawns reliably
description: Helps with code stuff

# Good — specific trigger condition
description: >
  Research agent for understanding unfamiliar code. Spawned when the user asks
  to explore, understand, or map out how a module or system works.
  Read-only — never modifies files.
```

**Auto-spawn vs explicit spawn:**

```
# Explicit
"Spawn the security-reviewer agent to audit src/auth/"

# Auto-spawn trigger (Claude decides)
"Can you check the auth module for security issues?"
→ Claude reads agent descriptions, matches security-reviewer, spawns it
```

For critical workflows, explicit spawning is more reliable than trusting auto-spawn.

### Can you invoke an agent with a slash command?

No — agents have no direct `/agent-name` slash command equivalent. Only skills support that syntax.

| | Skills | Agents |
|--|--------|--------|
| Direct slash invocation | ✅ `/skill-name` | ❌ None |
| Auto-invocation | ✅ via `description` | ✅ via `description` |
| Explicit invocation | `/skill-name` | Conversational: "spawn the X agent" |

**Workaround — skill wrapper:** Create a thin skill whose body tells Claude to spawn the agent. This gives you slash-command convenience:

```markdown
---
name: security-review
description: Run the security-reviewer agent on the current codebase
---

Spawn the security-reviewer agent. Scope: the entire src/ directory.
Report findings grouped by severity: CRITICAL, HIGH, MEDIUM, LOW.
```

Now `/security-review` → skill executes → Claude spawns the agent. The skill also lets you pre-set scope, output format, and context so every invocation is consistent.

---

## 6. Passing Context to Agents

The agent starts with a blank context — it only knows what's in its AGENT.md and what you tell it in the spawn prompt. If the main session has discovered important context, you must explicitly pass it.

### What to pass

```
# ❌ Bad — agent has no context
"Spawn the db-analyst agent to investigate the slow query"

# ✅ Good — agent has what it needs
"Spawn the db-analyst agent to investigate this slow query:
- Query: SELECT * FROM orders WHERE customer_id = ? AND status = 'pending'
- Table: orders (2.3M rows, no index on status)
- Observed latency: 4.2s average
- Recent change: added ORDER BY created_at DESC last Tuesday
Identify the root cause and propose a fix."
```

### Dynamic context via AGENT.md injection

Use shell injection in AGENT.md to give the agent live project context at spawn time:

```markdown
---
name: pr-reviewer
description: Reviews a pull request for correctness and style
tools: [Read, Glob, Grep, Bash]
---

## Current PR context

Branch: !`git branch --show-current`
Changed files: !`git diff --name-only main..HEAD`
Recent commits: !`git log main..HEAD --oneline`

## Review Instructions

Review the above changes for...
```

The shell commands run when the agent is spawned — agent gets current, live data without you having to copy-paste it.

---

## 7. Nested Agents

An agent can itself spawn sub-agents, creating a hierarchy. Context flows upward via summaries only at each level — the parent never sees the child's raw tool calls or internal reasoning.

```
Main session
    └── Agent A (Lead)
            ├── Agent B (Specialist 1)   ← B's tool calls invisible to A
            │       └── Agent C (Sub-specialist)  ← C's tool calls invisible to B
            └── Agent D (Specialist 2)

Context flow (summaries only):
    C summarizes → B receives C's summary → B summarizes
    B and D summarize → A receives both summaries → A summarizes
    A summarizes → Main receives A's summary only
```

### Controlling whether an agent can spawn sub-agents

The `Agent` tool is what enables spawning. Include or exclude it in `tools` to control nesting:

```yaml
# Agent CAN spawn sub-agents
tools:
  - Read
  - Glob
  - Agent     # explicitly allowed to spawn sub-agents

# Agent CANNOT spawn sub-agents
tools:
  - Read
  - Glob
  - Grep
  # Agent tool not listed → spawning blocked
```

Excluding the `Agent` tool is how you prevent a specialist agent from unilaterally spawning its own sub-agents — keeping the hierarchy under the lead agent's control.

### Cost at each nesting level

Each nested level is a separate billing context:

```
Main → Agent A → Agent B → Agent C
         ↓           ↓          ↓
    own billing  own billing  own billing
```

Three levels of nesting = four concurrent billing sessions. Depth should be deliberate.

### Depth limit

There is a nesting depth limit to prevent runaway spawning (exact number not publicly documented). In practice, rarely go beyond two levels — lead agent + specialists. Three or more levels usually signals the task should be restructured.

### When nesting makes sense vs doesn't

| Scenario | Use |
|----------|-----|
| Lead agent orchestrating domain specialists | ✅ one level of nesting |
| Specialist that needs a focused sub-researcher | ✅ two levels — deliberate |
| Agent spawning agents spawning agents | ❌ restructure — too complex, high cost |
| Simple parallel work | ❌ flat parallel agents from main is cleaner |

---

## 8. Parallel Agents — Spawning Multiple Simultaneously

You can spawn multiple agents simultaneously. Claude Code runs them concurrently — each in its own context window, each making tool calls independently.

### Spawning in parallel

```
"Spawn three agents in parallel:
1. security-reviewer — audit src/auth/ for vulnerabilities
2. performance-analyst — profile the database query layer
3. test-coverage — find untested code paths in src/

Report back when all three are done."
```

Claude spawns all three, waits for all summaries, then synthesizes the combined findings.

### Results come back independently

Each agent completes when it's done — they don't wait for each other. The main Claude collects results as they arrive and synthesizes after all complete (or proceeds with partial results if you ask).

---

## 9. Parallel Agents with Worktrees

When multiple agents need to **edit files**, you must give each its own git worktree to avoid conflicts. Two agents editing the same file simultaneously = corruption.

### Setup

```bash
# Claude Code handles this automatically with --worktree flag
claude --worktree

# Or manually:
git worktree add ../feature-agent-1 feature/auth-refactor
git worktree add ../feature-agent-2 feature/auth-refactor
```

### Pattern

```
Main session (main worktree: current branch)
    │
    ├── Agent 1 (worktree: ../agent-frontend — edits frontend files)
    ├── Agent 2 (worktree: ../agent-backend  — edits backend files)
    └── Agent 3 (worktree: ../agent-tests    — writes tests)
```

Each agent works in its own directory with its own git state. When all three finish:
1. Review each agent's changes with `git diff`
2. Merge or cherry-pick into the main branch

### When you need worktrees vs when you don't

| Scenario | Need worktrees? |
|----------|----------------|
| Multiple read-only agents (Explore, Plan) | No — reads don't conflict |
| One writing agent + reading agents | No — only writers conflict |
| Multiple agents editing different files | No — different files don't conflict |
| Multiple agents editing overlapping files | Yes — required |
| Large refactor split across agents | Yes — each owns a domain |

---

## 10. Agent Teams — Coordination Pattern, Not a Separate Feature

**Important clarification:** "Agent teams" is a **usage pattern** built on top of regular Claude sessions — not a distinct feature type with special infrastructure. There is no direct peer-to-peer messaging API between agents.

```
What it sounds like (does NOT exist):
    Agent A ←──── direct message ────→ Agent B

What actually happens:
    Agent A reads/writes TASKS.md
    Agent B reads/writes TASKS.md
    ↑ shared file in a shared worktree = the only coordination channel
```

### Subagents vs Agent Teams — the real difference is topology

| | Subagents | Agent Teams |
|--|-----------|-------------|
| **What it is** | Claude Code feature — Agent tool, AGENT.md | Usage pattern — multiple independent `claude` sessions |
| **Communication** | Hub-and-spoke: each reports to main only | Peer-via-files: agents read/write a shared task list |
| **Setup** | AGENT.md + Agent tool call | Multiple terminal tabs, each running `claude`, shared worktree |
| **Coordination** | Main session orchestrates | Each session claims tasks from shared file independently |

### What "peer-to-peer" actually means in practice

Open multiple terminal tabs, each running its own Claude session in the same worktree:

```
Tab 1: claude --worktree    ← "Agent A" — given a role, reads TASKS.md
Tab 2: claude --worktree    ← "Agent B" — given a role, reads TASKS.md
Tab 3: claude --worktree    ← "Agent C" — given a role, reads TASKS.md
```

All three share the same filesystem. The shared task file is the entire peer-to-peer mechanism.

### How teams coordinate

Team members share a **task list** — a structured file (e.g., `TASKS.md` or a JSON file) in the shared worktree that all sessions read and write:

```markdown
# TASKS.md

## Pending
- [ ] Audit auth module for SQL injection (unassigned)
- [ ] Check rate limiting implementation (unassigned)
- [ ] Review session token handling (unassigned)

## In Progress
- [~] Check for hardcoded secrets (Agent B, started 14:02)

## Completed
- [x] Scan dependencies for CVEs (Agent A — found 2 HIGH issues, see findings.md)
```

Each agent:
1. Reads TASKS.md to find unclaimed work
2. Claims a task (writes its name + timestamp to mark in-progress)
3. Does the work
4. Writes findings to a shared findings file
5. Marks task complete in TASKS.md
6. Claims the next task

This prevents two agents from doing the same work simultaneously.

---

## 11. Lead Agent Pattern

For complex work, one agent acts as the **lead** — it spawns and orchestrates the teammates, synthesizes their findings, and reports a single unified result to the main session.

```
Main session
    │
    └── Lead Agent (general-purpose, reads from all teammates)
            │
            ├── Teammate A: Security Reviewer
            ├── Teammate B: Performance Analyst
            └── Teammate C: Test Coverage Checker

Lead synthesizes all three reports → single coherent result to main
```

### Lead agent AGENT.md

```markdown
---
name: review-lead
description: >
  Lead agent for comprehensive code review. Spawns specialist reviewers
  in parallel, synthesizes their findings, and reports a unified review.
model: claude-opus-4-7
tools: [Read, Glob, Grep]
---

You are the lead reviewer. Your job:

1. Spawn three specialist agents in parallel:
   - security-reviewer: audit for vulnerabilities
   - performance-analyst: identify bottlenecks
   - test-reviewer: check test coverage and quality

2. Wait for all three to complete.

3. Synthesize their findings into a single review:
   - CRITICAL issues (any severity from any reviewer)
   - Conflicts between reviewers (e.g., perf fix that creates security risk)
   - Priority order for the main session to act on

Do not duplicate findings — deduplicate and rank by impact.
```

---

## 12. Plan Approval Gates

For high-stakes work, require agents to **plan before implementing**. The agent proposes a plan, the main session (or you) reviews it, then approves before the agent proceeds.

### Pattern

```markdown
---
name: refactor-agent
description: Refactors a module according to the approved plan
tools: [Read, Edit, Write, Bash]
---

## Phase 1 — Planning (ALWAYS DO THIS FIRST)

Before making any changes:
1. Read all files in the target module
2. Produce a detailed refactor plan:
   - List every file you will modify
   - Describe each change and why
   - Identify risks and dependencies
3. Output the plan and STOP. Write "AWAITING APPROVAL" at the end.
4. Do not make any edits until the user confirms the plan.

## Phase 2 — Implementation (only after approval)

After the user says "approved" or "proceed":
1. Implement each change from the plan in order
2. Run tests after each file change
3. Report any deviations from the plan immediately
```

The agent writes its plan and stops. You review it in the main session, then say "proceed" to continue. This is the equivalent of `/plan` mode but for a delegated agent.

---

## 13. Display Modes for Agent Teams

When running multiple agents simultaneously, you need to see what each is doing.

### In-process terminal cycling (default)

Claude Code cycles through agent outputs in the same terminal. Simple but you can't see all agents at once:

```
[Agent A] Reading src/auth/login.ts...
[Agent B] Running npm audit...
[Agent C] Searching for test files...
[Agent A] Found JWT validation at line 47
[Agent B] 2 HIGH vulnerabilities found
```

### tmux split panes (recommended for teams)

Run each agent in its own pane so you can watch them all simultaneously:

```bash
# Create a 3-pane layout
tmux new-session -d -s review
tmux split-window -h
tmux split-window -v
tmux select-layout even-horizontal

# Run each agent in its pane
tmux send-keys -t review:0.0 'claude --worktree "Run the security-reviewer agent"' Enter
tmux send-keys -t review:0.1 'claude --worktree "Run the performance-analyst agent"' Enter
tmux send-keys -t review:0.2 'claude --worktree "Run the test-reviewer agent"' Enter

tmux attach -t review
```

Each pane shows a live agent session. You can interact with any of them independently.

### iTerm2 (macOS)

Same concept — use iTerm2's split panes. Each pane runs `claude --worktree` with a specific agent task.

---

## 14. When Agent Teams Win

### Code review (parallel specialists)

```
Single-pass review (one agent, sequential):
    Reads code → checks security → checks perf → checks tests → reports
    Time: 5 min, depth: limited by one context window

3-agent parallel review:
    Agent A: security (full context, full depth)
    Agent B: performance (full context, full depth)
    Agent C: test coverage (full context, full depth)
    Time: 5 min (parallel), depth: 3× deeper per domain
```

### Bug investigation (competing hypotheses)

```
"This service intermittently returns 500 errors"

Agent A: hypothesis — race condition in request handler
Agent B: hypothesis — database connection pool exhaustion
Agent C: hypothesis — memory leak causing OOM restarts

Each agent investigates its hypothesis independently.
Main session gets 3 independent analyses — picks the most supported one.
```

This is better than one agent investigating sequentially because:
- No anchoring bias (each starts fresh with no prior hypothesis)
- Faster — all three run simultaneously
- Covers more ground — each goes deep on its hypothesis

### Large refactors (domain teams)

```
Refactor: add OAuth to an app that only has API key auth

Agent A (worktree: agent-frontend):
    Updates login UI, OAuth redirect flow, token storage
Agent B (worktree: agent-backend):
    Adds OAuth endpoints, middleware, token validation
Agent C (worktree: agent-tests):
    Writes integration tests for the full OAuth flow
```

All three work simultaneously. Lead agent coordinates dependencies and merges.

---

## 15. Cost and Token Considerations

Each agent has its own context window — **its tool calls and reasoning are billed separately** from the main session.

### Cost model

```
Main session:        X tokens of context → X tokens billed to main
Agent A session:     Y tokens (its own work) → Y tokens billed to Agent A
Agent B session:     Z tokens (its own work) → Z tokens billed to Agent B

Total cost = main + Agent A + Agent B
           (not main + Agent A's summary only)
```

### Practical implications

| Decision | Impact |
|----------|--------|
| Spawn Explore agent for research | Agent pays for the research context; main only pays for the summary |
| Run analysis inline (no agent) | Main pays for everything |
| Spawn 5 agents in parallel | 5× context windows running simultaneously |
| Use Haiku model for simple agents | Much cheaper for straightforward tasks |
| Use Opus for lead/synthesis agent | Worth it for the final synthesis quality |

### Cost optimization patterns

```yaml
# ✅ Use smaller models for simple agents
---
name: find-todos
description: Find all TODO comments in the codebase
model: claude-haiku-4-5-20251001   # cheap — just grep and report
tools: [Glob, Grep, Read]
---

# ✅ Use Opus for complex reasoning agents
---
name: architecture-reviewer
description: Reviews system architecture for scalability and maintainability
model: claude-opus-4-7             # worth it for deep analysis
effort: high
---
```

### When NOT to use agents

- **Simple, fast tasks** — spawning an agent has overhead; for a 2-second task, inline is faster
- **When you need full control** — agents are autonomous; for step-by-step approval, use skill with plan gate or `/plan` mode
- **Tight cost budgets** — 5 parallel agents = 5 concurrent billing sessions

---

## 16. Debugging Agents

### See agent status

```
/mcp        ← shows MCP servers; not agents directly
Ctrl+T      ← task list — shows running agents and their status
```

### Agent isn't doing what you expected

1. **Check AGENT.md body** — is the system prompt clear and unambiguous?
2. **Check tools list** — is the agent blocked from a tool it needs?
3. **Check the spawn prompt** — did you give enough context?
4. **Use foreground mode** — watch the agent's output in real time instead of background
5. **Add explicit output format** to AGENT.md — vague instructions → vague summaries

### Agent is using the wrong tools

```yaml
# Agent keeps trying to Edit files when it should only read
# Fix: explicit deny in permissions
permissions:
  deny:
    - Edit
    - Write
```

### Agent summary is too vague

Add a structured output format to AGENT.md:

```markdown
## Required output format

Your final report MUST include:
1. **Files examined**: list with line counts
2. **Issues found**: each as `FILE:LINE — SEVERITY: description`
3. **Not examined**: anything you skipped and why
4. **Confidence**: HIGH / MEDIUM / LOW with reasoning
```

### Agent ran out of context

If an agent needs to scan a very large codebase, it may hit its context limit. Fix:
- Scope the agent more narrowly (specific directories, not entire repo)
- Use multiple agents, each scoped to a subdomain
- Use `effort: low` to reduce reasoning verbosity and save context for data

---

## 17. Explore and Plan Agents — Effective Use

### Explore agent — good prompts

The Explore agent is read-only and great for mapping unfamiliar code. Prompt quality matters:

```
# ❌ Too vague
"Explore the codebase"

# ✅ Specific question
"Map the request lifecycle for a POST /api/orders call:
starting from the route definition, trace through middleware,
controller, service layer, and database call.
Include file paths and function names at each step."

# ✅ Targeted research
"Find all places where user input is passed to a database query
without parameterization. List file, line, and the specific
unsanitized variable."

# ✅ Architecture question
"How does this application handle background jobs?
Find the job queue, worker processes, and retry logic.
Is there a dead-letter queue?"
```

### Plan agent — getting useful plans

The Plan agent proposes implementation strategies. Guide it with constraints:

```
"Using the Plan agent, design the implementation for adding
rate limiting to the API. Constraints:
- Must not require Redis (we don't have it)
- Must work in a multi-process deployment
- Must be configurable per-route
- Must not break existing tests

Produce: (1) chosen approach with rationale, (2) files to create/modify,
(3) risk assessment, (4) estimated complexity."
```

Without constraints the plan agent produces generic plans. With constraints it produces actionable ones.

---

## Quick Reference

| Concept | Key Point |
|---------|-----------|
| **Agent lifecycle** | Spawns with blank context → AGENT.md as system prompt → runs autonomously → summary returned |
| **Context isolation** | Agent's tool calls/reasoning invisible to main; only final summary returned |
| **Summary quality** | Depends on output format instructions in AGENT.md or spawn prompt |
| **AGENT.md `tools`** | Whitelist — agent physically cannot call unlisted tools |
| **AGENT.md `permissions.deny`** | Explicit deny — overrides whitelist; deny wins |
| **AGENT.md `skills`** | Skills the agent can invoke during its work |
| **Auto-spawning** | Claude matches spawn description to task — be specific |
| **Passing context** | Agent starts blank — explicitly pass relevant context in spawn prompt |
| **Dynamic injection** | `` !`cmd` `` in AGENT.md runs at spawn time — gives agent live project data |
| **Parallel agents** | Spawn multiple simultaneously; each runs in own context; results collected when all done |
| **Worktree per agent** | Required when multiple agents edit files; prevents conflicts |
| **Agent teams** | Peer-to-peer communication via shared task list file |
| **Lead agent** | Spawns and orchestrates teammates; synthesizes unified result |
| **Plan approval gate** | Agent proposes plan → stops → waits for approval → then implements |
| **tmux split panes** | Run each agent in its own pane for live visibility |
| **3-agent code review** | Security + performance + tests in parallel → 3× deeper per domain |
| **Competing hypotheses** | One agent per hypothesis — no anchoring bias, parallel investigation |
| **Cost model** | Each agent has own context window → own billing; not just the summary |
| **Model per agent** | Haiku for simple/fast; Opus for deep analysis; saves cost |
| **Debugging** | `Ctrl+T` task list; explicit output format in AGENT.md; use foreground mode |
| **Explore agent** | Read-only; prompt with specific question, not vague "explore" |
| **Plan agent** | Architecture proposals; give constraints for actionable output |
| **Don't use agents for** | Simple fast tasks, tight cost budgets, tasks needing step-by-step approval |
