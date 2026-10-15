# Claude Code — Skills Deep Dive

**Date:** 2026-05-11
**Topic:** Skills, Subagents — full reference
**Curriculum:** Module 7 (Days 15–20)

---

## 1. Recap — What Day 1 Covered

See `day1_session_reference.md` Section 10 for the foundation. Quick summary:

| Topic | Coverage in Day 1 |
|-------|-------------------|
| Skills vs Commands | Skills = full workflow (frontmatter, tools, fork); Commands = prompt template only |
| Location | `.claude/skills/<name>/SKILL.md` |
| Key frontmatter | `name`, `description`, `model`, `effort`, `allowed-tools`, `context: fork`, `arguments` |
| Dynamic injection | `` !`shell command` `` runs before Claude reads the skill |
| Auto-invocation | Claude reads `description` to decide when to invoke automatically |
| `context: fork` | Skill runs in its own context window |

**What Day 1 did not cover:** reference vs task types, full frontmatter options, argument definitions, `disable-model-invocation`, user vs project scope, built-in skills internals, subagent configuration (AGENT.md), foreground vs background agents, skill vs subagent decision, practical end-to-end examples.

---

## 2. Two Types of Skills

### Reference Skills — Conventions Claude Applies Automatically

Reference skills contain **knowledge, standards, or conventions** that Claude applies passively whenever relevant. They're not procedures — they're constraints and context.

```
~/.claude/skills/           ← user scope (all projects)
.claude/skills/             ← project scope (current project only)
```

**Examples:**
- Coding standards for your team (naming conventions, error handling patterns)
- API design rules (REST conventions, response envelope format)
- Commit message format
- Security constraints (never log PII, always sanitize inputs)

```markdown
---
name: api-conventions
description: REST API design conventions for this project — applied to all API code changes
---

## API Conventions

All endpoints return:
```json
{ "data": ..., "error": null, "meta": { "request_id": "..." } }
```

- Use plural nouns for collections: `/users`, `/orders`
- Never return raw database IDs — use opaque slugs
- All timestamps in ISO 8601 UTC: `2026-05-11T14:32:00Z`
- Validation errors use HTTP 422, not 400
```

**How "reference" skills actually work — there is no `type: reference` field.** The distinction is conceptual, not a technical marker. All skills go through the same loading mechanism:

```
Session starts
    Claude Code scans skill directories
    Loads description from each SKILL.md frontmatter (names + descriptions only)

During the session:
    Claude reads the task → checks skill descriptions
    If description matches → loads full skill body and applies it
```

A "reference skill" achieves its always-on behavior purely through its `description` being broad and unconditional enough that Claude recognizes it as relevant to almost every related task:

```yaml
# Task skill — specific trigger
description: Push branch and create a PR when the user asks to submit or open a PR

# Reference-style — broad, always-relevant
description: Python coding standards applied to all Python file edits in this project
```

**For conventions that must never be skipped — use CLAUDE.md, not a skill.** CLAUDE.md is guaranteed to be loaded every session. Skills depend on Claude's judgment matching the description to the task. If a rule is critical, put it in CLAUDE.md. Use reference skills for standards that are important but where Claude's auto-matching is reliable enough.

---

### Task Skills — Step-by-Step Procedures

Task skills are **executable workflows** with defined steps. They're invoked explicitly with `/skill-name` or auto-invoked when Claude recognizes the task matches.

```
.claude/skills/create-pr/SKILL.md
.claude/skills/deploy/SKILL.md
.claude/skills/code-review/SKILL.md
```

**Examples:**
- Create a PR: push branch, open PR, fill template, add reviewers
- Deploy: run tests, build, tag, push to registry, notify Slack
- Code review: check against standards, run security scan, summarize

```markdown
---
name: create-pr
description: Push the current branch and create a pull request with a filled template
allowed-tools: [Bash, Read, mcp__github__create_pull_request]
---

## Steps

1. Push the current branch: `git push -u origin HEAD`
2. Read the PR template from `.github/pull_request_template.md` if it exists
3. Summarize the changes from `git log main..HEAD --oneline`
4. Create the PR using the GitHub MCP tool with:
   - Title: derived from the branch name and commit summary
   - Body: filled template with actual change descriptions
   - Base: `main`
   - Reviewers: check `CODEOWNERS` for the affected files
5. Output the PR URL
```

---

## 3. Skill Locations and Scope

| Location | Scope | Use for |
|----------|-------|---------|
| `~/.claude/skills/<name>/SKILL.md` | All projects on this machine | Personal workflows, team standards you always apply |
| `.claude/skills/<name>/SKILL.md` | Current project only | Project-specific procedures, team workflows in repo |

Project skills take precedence if both scopes define a skill with the same name.

Skills in the project `.claude/skills/` directory are typically **checked into git** — they become shared team workflows that everyone gets when they clone the repo.

---

## 4. SKILL.md — Complete Frontmatter Reference

```yaml
---
name: skill-name                        # required — used as /skill-name invocation
description: >                          # required — Claude reads this to decide auto-invocation
  One to three sentences explaining
  what this skill does and when to use it.
  The more specific, the better the auto-match.

model: claude-opus-4-7                  # optional — override model for this skill only
effort: high                            # optional — low / medium / high / xhigh

allowed-tools:                          # optional — whitelist of tools (blocks all others)
  - Bash
  - Read
  - Write
  - mcp__github__create_pull_request
  - mcp__github__list_issues

disable-model-invocation: true          # optional — run as pure shell script (no Claude call)

context: fork                           # optional — run in isolated context window

arguments:                              # optional — typed inputs Claude fills
  - name: target
    description: The file or directory to process
    required: true
  - name: format
    description: Output format — one of json, markdown, text
    required: false
    default: markdown
---
```

### Field Details

**`description`** — most important field. Claude reads this to decide whether to auto-invoke the skill. Be specific: "Create a PR with filled template and CODEOWNERS-based reviewers" auto-invokes better than "PR workflow".

**`model`** — override applies only for this skill's execution. Your session model is unchanged after the skill completes. Use `claude-opus-4-7` for complex analysis skills, `claude-haiku-4-5-20251001` for fast formatting tasks.

**`effort`** — maps to reasoning depth:
- `low` — fast, minimal reasoning
- `medium` — default
- `high` — deeper reasoning
- `xhigh` — extended thinking (budget_tokens enabled)

**`allowed-tools`** — if set, Claude can ONLY use the listed tools during this skill. Anything not listed is blocked. This is a safety constraint, not just a hint. Use it to prevent a code-review skill from accidentally making commits, or a read-only research skill from editing files.

**`disable-model-invocation`** — the skill runs as a pure shell script or template expansion with no Claude call. The skill body is executed directly (if it's a shell script) or expanded as a template. Used for deterministic operations that don't need AI reasoning.

**`context: fork`** — the skill gets its own context window. The main conversation's history is not included. Useful for:
- High-context skills (large codebase analysis) that would bloat the main conversation
- Skills that need a clean slate (e.g., an unbiased security review shouldn't know your implementation history)
- Skills that run long and would exceed the main context limit

**`arguments`** — typed inputs that Claude fills from the invocation context. Each argument has `name`, `description`, `required`, and optionally `default`.

---

## 5. Skill Body — What Goes Inside

The body (below the frontmatter) is the prompt/instructions Claude receives when the skill runs. It's standard markdown.

### Instruction-style body

```markdown
---
name: security-review
description: Run a security review on the staged changes before merging
allowed-tools: [Bash, Read, Glob, Grep]
context: fork
---

You are performing a focused security review. Check the following, in order:

1. **Injection risks**: SQL, shell, HTML injection in any user-supplied input paths
2. **Authentication**: Are all routes protected that should be? Are there hardcoded credentials?
3. **Secrets**: Are any secrets, tokens, or keys hardcoded or logged?
4. **Dependencies**: Run `npm audit` or `pip-audit` and report HIGH/CRITICAL issues
5. **Permissions**: Are file operations scoped correctly? No path traversal risks?

For each issue found, report: file, line, severity (HIGH/MEDIUM/LOW), and specific fix.
Summarize as a table at the end.
```

### Step-by-step procedure body

```markdown
---
name: deploy
description: Run tests, build Docker image, push to registry, and notify Slack
allowed-tools: [Bash]
arguments:
  - name: environment
    description: Target environment — staging or production
    required: true
---

Deploy to {{environment}}:

1. Run the test suite: `npm test`
   - If tests fail, stop and report failures. Do not proceed.
2. Build the Docker image: `docker build -t myapp:{{version}} .`
   where {{version}} is the current git tag from `git describe --tags`
3. Push to registry: `docker push registry.internal/myapp:{{version}}`
4. Update the Kubernetes deployment:
   `kubectl set image deployment/myapp myapp=registry.internal/myapp:{{version}} -n {{environment}}`
5. Wait for rollout: `kubectl rollout status deployment/myapp -n {{environment}}`
6. Post to Slack: `curl -X POST $SLACK_WEBHOOK -d '{"text":"Deployed {{version}} to {{environment}}"}'`
```

Note `{{argument_name}}` syntax for argument interpolation in the body.

---

## 6. Dynamic Context Injection

`` !`shell command` `` runs a shell command at invocation time and injects the output into the skill before Claude reads it. This gives the skill live, current data.

```markdown
---
name: pr-review
description: Review the current PR against team standards
context: fork
---

## Current branch status

Recent commits:
!`git log main..HEAD --oneline`

Changed files:
!`git diff --name-only main..HEAD`

Current test results:
!`npm test 2>&1 | tail -20`

## Review Instructions

Review the above changes against our standards. Check for...
```

When `/pr-review` is invoked, Claude Code runs all three shell commands first, substitutes their output into the skill body, then sends the fully-materialized prompt to Claude.

### Multiple injections

```markdown
---
name: daily-standup
description: Summarize what changed since yesterday for standup
---

## Yesterday's work
!`git log --since="yesterday" --oneline --author="$(git config user.email)"`

## Open PRs awaiting review
!`gh pr list --reviewer @me --json title,url --jq '.[] | "\(.title) \(.url)"'`

## Failing CI
!`gh run list --limit 5 --json status,name --jq '.[] | select(.status=="failure") | .name'`

Summarize the above for a brief standup update.
```

### Injection failure handling

If a shell command fails (non-zero exit), the injection outputs the error and the skill continues. Design injections to fail gracefully — use `|| echo "unavailable"` for optional data:

```markdown
!`cat .env.example 2>/dev/null || echo "No .env.example found"`
```

---

## 7. Arguments — Full Definition

Arguments let the skill accept typed inputs. Claude fills them from context or the explicit invocation.

```yaml
arguments:
  - name: file
    description: Path to the file to analyze. Example: src/auth/login.ts
    required: true

  - name: depth
    description: Analysis depth — quick (syntax only) or deep (logic + security)
    required: false
    default: quick
```

### Invocation with arguments

```
/analyze-file file=src/auth/login.ts depth=deep
```

Or Claude fills arguments from conversation context — if you just said "analyze login.ts in depth" and trigger the skill, Claude infers `file=src/auth/login.ts` and `depth=deep`.

### Argument interpolation in body

Use `{{argument_name}}` in the body:

```markdown
Analyze `{{file}}` at depth: {{depth}}.
```

---

## 8. `context: fork` — Deep Dive

Without `context: fork`, the skill runs inside your current conversation. The skill sees your full history, and its output is appended to your history.

With `context: fork`, the skill gets a **new context window**:

```
Main conversation (100k tokens)
    │
    │  /security-review invoked with context: fork
    ▼
Forked context (0 tokens — fresh start)
    reads files, runs checks, produces report
    ▼
Summary result injected back into main conversation
```

**Use `context: fork` when:**
- The skill needs a lot of tokens (full codebase scan, large document analysis)
- The skill's internal reasoning shouldn't be visible in your main conversation
- You want the skill to have no prior bias from the current session (unbiased review)
- The skill could hit the main context limit if run inline

**Don't use `context: fork` when:**
- The skill needs to reference your current conversation history
- The skill is short and conversational (the fork overhead isn't worth it)
- The skill needs to make follow-up decisions based on what you've discussed

---

## 9. Auto-Invocation — Writing Good Descriptions

Claude scans skill descriptions to decide whether to invoke a skill automatically when you give it a task. Weak descriptions miss auto-invocations; overly broad descriptions cause false positives.

### Bad descriptions

```yaml
description: Does PR stuff         # too vague — never auto-invokes
description: Helps with code       # matches everything — fires on every task
description: Run when needed       # no signal — Claude can't extract intent
```

### Good descriptions

```yaml
description: >
  Push the current branch and create a GitHub pull request with an
  auto-filled template, CODEOWNERS-based reviewers, and a commit summary.
  Use when the user asks to open, create, or submit a PR.
```

```yaml
description: >
  Full security audit of staged changes: injection risks, hardcoded secrets,
  auth gaps, and dependency vulnerabilities. Use before merging any
  security-sensitive code or when the user asks for a security check.
```

**Pattern:** State what it does + the specific condition under which it applies. Claude matches on both.

---

## 10. Invoking Skills

### Explicit invocation

```
/skill-name
/skill-name arg1=value arg2=value
```

### In conversation

Just describe the task — Claude auto-invokes if the description matches:
```
"Create a PR for this branch"         → auto-invokes /create-pr
"Do a security review before merging" → auto-invokes /security-review
```

### List available skills

```
/skills         ← lists all skills from user and project scope
```

### Check what a skill does

```
/skill-name --help      ← shows the skill's description and arguments
```

---

## 11. `allowed-tools` — Tool Restrictions

When `allowed-tools` is set, Claude is **hard-restricted** to only those tools during the skill. This is enforced by the runtime, not just a hint.

```yaml
allowed-tools:
  - Read           # all Read calls allowed
  - Glob           # all Glob calls allowed
  - Grep           # all Grep calls allowed
  - Bash           # all Bash calls allowed
  - mcp__github__list_issues          # specific MCP tool
  - mcp__github__create_pull_request  # specific MCP tool
  - mcp__github__*                    # wildcard — all github MCP tools
```

**Common patterns:**

| Skill type | Typical `allowed-tools` |
|------------|------------------------|
| Read-only analysis | `[Read, Glob, Grep, Bash]` — no Write/Edit |
| Code review | `[Read, Glob, Grep]` — no Bash (can't run code) |
| PR workflow | `[Bash, mcp__github__*]` — no file edits |
| Refactor | `[Read, Edit, Write, Bash]` — no MCP (no external calls) |
| Research only | `[Read, Glob, Grep, WebSearch, WebFetch]` |

---

## 12. Model and Effort Overrides

**`model` override:** The skill runs with a different model than your current session. After the skill completes, your session model is unchanged.

```yaml
model: claude-haiku-4-5-20251001   # fast + cheap for formatting/linting skills
model: claude-opus-4-7             # highest capability for deep analysis skills
```

**`effort` override:** Controls reasoning depth for this skill execution only.

```yaml
effort: xhigh   # extended thinking — use for complex architecture decisions
effort: low     # fast response — use for simple template fills
```

**Combining both:**

```yaml
---
name: architecture-review
description: Deep architectural review of a module — slow but thorough
model: claude-opus-4-7
effort: xhigh
context: fork
allowed-tools: [Read, Glob, Grep]
---
```

```yaml
---
name: format-imports
description: Sort and organize import statements in a file
model: claude-haiku-4-5-20251001
effort: low
allowed-tools: [Read, Edit]
---
```

---

## 13. Built-In Skills

These ship with Claude Code and are available in every project:

| Skill | Invocation | What it does |
|-------|-----------|-------------|
| `/review` | `/review` | In-session code review — checks changed files against conventions |
| `/simplify` | `/simplify` | Reviews changed code for reuse, quality, efficiency — fixes issues found |
| `/security-review` | `/security-review` | Security audit of pending changes on current branch |
| `/init` | `/init` | Generates a CLAUDE.md for the current codebase via analysis |
| `/ultrareview` | `/ultrareview` or `/ultrareview <PR#>` | Multi-agent deep review — fleet of independent agents, background, 5–10 min |

**`/review` vs `/ultrareview`:**
- `/review` — fast, single-agent, inline. Use during iteration.
- `/ultrareview` — slow, multi-agent, background. Use before merging.

**`/simplify`:** Runs after you've made changes. Looks for over-engineered code, missed reuse opportunities, or quality issues and fixes them. Good to run after a `/review` cycle.

---

## 14. Subagents — Overview

Subagents are **isolated Claude instances** that work on focused tasks and report back. They protect your main context window from being flooded by large research or analysis tasks.

```
Main conversation
    │
    │  spawns subagent
    ▼
Subagent (own context window, own tool calls)
    does the work — reads files, searches, analyzes
    summarizes result
    ▼
Summary injected back into main conversation
    (not the full transcript — just the result)
```

The main conversation never sees the subagent's internal tool calls or chain of thought. Only the final summary comes back.

---

### Built-in subagent types

| Type | Purpose | Tools available |
|------|---------|----------------|
| `Explore` | Read-only research — find files, understand code, answer "where is X" | Read, Glob, Grep, WebSearch, WebFetch |
| `Plan` | Architecture design — proposes implementation plans, considers trade-offs | Same as Explore (read-only) |
| `general-purpose` | Default — can use all tools | All tools |
| Custom | Your own agents with defined scope | Restricted per AGENT.md |

**When to spawn a subagent vs handle it yourself:**
- Subagent: "research how authentication works in this codebase" — would flood main context
- Inline: "add a null check to this function" — trivial, no context risk

---

## 15. AGENT.md — Custom Subagent Configuration

Custom subagents live in `.claude/agents/<name>/AGENT.md`:

```
.claude/agents/
├── security-reviewer/
│   └── AGENT.md
├── db-analyst/
│   └── AGENT.md
└── frontend-specialist/
    └── AGENT.md
```

**AGENT.md structure:**

```markdown
---
name: security-reviewer
description: >
  Focused security review agent. Reads code to identify injection risks,
  hardcoded credentials, auth gaps, and dependency vulnerabilities.
  Auto-spawned when security analysis is needed.
model: claude-opus-4-7
tools:
  - Read
  - Glob
  - Grep
  - Bash(npm audit)
  - Bash(pip-audit)
permissions:
  deny:
    - Write
    - Edit
    - Bash(git *)
---

You are a focused security reviewer. Your only job is to identify security issues.

When analyzing code:
1. Check all input paths for injection risks (SQL, shell, HTML)
2. Search for hardcoded secrets: `grep -r "password\|secret\|api_key" --include="*.py"`
3. Verify authentication on all HTTP endpoints
4. Run dependency audit tools
5. Check file operation paths for traversal vulnerabilities

Report every issue with: file, line number, severity (HIGH/MEDIUM/LOW), and exact fix.
Do not suggest architectural changes — only security fixes.
```

---

### Key AGENT.md frontmatter fields

| Field | Purpose |
|-------|---------|
| `name` | Used to spawn: `Agent(subagent_type="security-reviewer")` |
| `description` | What Claude reads to decide auto-spawning |
| `model` | Model for this agent — can differ from main session |
| `tools` | Whitelist — agent can only use these tools |
| `permissions.deny` | Explicit denies (overrides tools whitelist) |
| `skills` | Skills this agent has access to |

---

## 16. Foreground vs Background Agents

**Foreground agent** — you wait for it. Use when you need the result before continuing:

```
You: "Research how the auth middleware works"
    → spawns Explore subagent
    → you wait while it searches files
    → result comes back: "Auth middleware is in src/middleware/auth.ts, uses JWT..."
    → you continue with that context
```

**Background agent** — runs while you work on something else. Use for long tasks that don't block you:

```
You: "In the background, audit all API endpoints for missing auth checks"
    → spawns background agent
    → you continue working
    → notification when complete: "Found 3 endpoints missing auth: /admin/*, ..."
```

Background agents are particularly useful with `/loop` — spawn a background checker that runs every 30 minutes while you code.

---

## 17. Task Skill vs Agent — The Core Difference

### Who executes the work

**Task skill — same Claude instance, given a script:**

When you invoke a skill, Claude Code loads the SKILL.md body and hands it to the **current Claude instance** as additional instructions. Claude then runs the steps itself using its own tool calls.

```
You: /create-pr
    Claude Code loads create-pr/SKILL.md body
    Injects it as instructions into the current Claude instance
    → Claude reads the steps
    → Claude calls Bash("git push"), mcp__github__create_pull_request, etc.
    → Output appears inline in your conversation
```

With `context: fork`, it gets a fresh context window — but it's still the same model executing the prompt. Think of it as handing Claude a checklist; Claude still does the work.

**Agent — a separate Claude instance, delegated a task:**

When Claude spawns an agent, Claude Code launches a **new Claude session** with AGENT.md as its system prompt. That new instance runs independently, makes its own tool calls in its own context, and only the final summary comes back.

```
You: "Research how auth works in this codebase"
    Main Claude calls Agent(subagent_type="Explore")
    → Claude Code spawns a new Claude session
    → New instance reads files, greps code, follows imports
       (all these tool calls stay in the agent's context — invisible to you)
    → Agent summarizes findings
    → Summary injected back into main conversation
    Main Claude continues with that summary
```

### Side-by-side comparison

| Aspect | Task Skill | Agent |
|--------|-----------|-------|
| **Executor** | Same Claude instance | Separate Claude instance |
| **Tool calls visible in main conversation** | Yes (unless `context: fork`) | No — agent's work is internal |
| **Context cost to main conversation** | Full (or zero with fork) | Only the summary |
| **Configuration** | SKILL.md body = the instructions Claude follows | AGENT.md = system prompt of the new instance |
| **Model override** | `model:` in frontmatter | `model:` in AGENT.md |
| **Can run in background** | No | Yes |
| **Tool restrictions enforced by** | Prompt instructions + `allowed-tools` | Process-level — the agent physically cannot call unlisted tools |
| **Best for** | Repeatable, well-defined procedures | Exploratory, open-ended, or context-heavy work |

### The analogy

- **Task skill** = handing someone a checklist. Same person, same desk, just following written steps.
- **Agent** = delegating to a specialist in another room. They do the work, come back with a report. You never see their scratch notes.

### When `context: fork` blurs the line

A skill with `context: fork` looks similar to an agent — both protect the main context. The difference:
- `context: fork` = same Claude instance with a clean slate; result returned inline
- Agent = different Claude instance with its own lifecycle, background capability, and process-level tool enforcement

### When to use each

| Situation | Use |
|-----------|-----|
| Reusable procedure you'll run many times | **Task skill** — `/create-pr` |
| Conventions to apply passively | **Reference skill** — auto-applied |
| One-off large research task | **Agent** (Explore type) |
| Specialized domain work (security, DB) | **Custom agent** with restricted tools |
| Step-by-step workflow with fixed steps | **Task skill** |
| "Explore and figure it out" | **Agent** |
| Fast, repeatable, well-defined | **Task skill** |
| Shouldn't pollute main context | **Task skill with `context: fork`** OR **Agent** |
| Must run in background while you work | **Agent** |
| Tool restrictions must be enforced strictly | **Agent** (process-level) not skill (prompt-level) |

---

## 18. Practical Examples

### Example 1 — PR Workflow Skill (task skill)

`.claude/skills/create-pr/SKILL.md`:

```markdown
---
name: create-pr
description: >
  Push the current branch and create a GitHub pull request. Fills the PR template,
  sets reviewers from CODEOWNERS, and labels by change type. Use when asked to
  open, create, or submit a PR.
allowed-tools: [Bash, Read, Glob, mcp__github__create_pull_request, mcp__github__list_pull_requests]
arguments:
  - name: draft
    description: Open as draft PR instead of ready for review
    required: false
    default: "false"
---

## PR Creation Steps

Current branch and recent commits:
!`git branch --show-current`
!`git log main..HEAD --oneline`

Changed files:
!`git diff --name-only main..HEAD`

1. Push branch: `git push -u origin HEAD`
2. Read PR template: check `.github/pull_request_template.md`
3. Read CODEOWNERS: identify reviewers for changed file paths
4. Determine change type from commits: feat / fix / chore / refactor
5. Create PR via GitHub MCP:
   - Title: conventional commit style based on changes
   - Body: filled template (what changed, why, testing done)
   - Reviewers: from CODEOWNERS lookup
   - Draft: {{draft}}
6. Output the PR URL
```

---

### Example 2 — Coding Standards Reference Skill

`~/.claude/skills/python-standards/SKILL.md`:

```markdown
---
name: python-standards
description: Python coding standards applied to all Python file edits in this project
---

## Python Standards

**Imports:** stdlib → third-party → local, separated by blank lines, sorted alphabetically within groups.

**Type hints:** All function signatures must have type hints. Use `Optional[X]` not `X | None` (Python 3.9 compat).

**Error handling:** Never catch bare `Exception`. Catch specific exceptions. Log before re-raising.

**Logging:** Use `structlog` not `logging`. Include `request_id` in all log calls.
```

No invocation needed — Claude applies this automatically when editing `.py` files.

---

### Example 3 — Security Reviewer Custom Agent

`.claude/agents/security-reviewer/AGENT.md`:

```markdown
---
name: security-reviewer
description: >
  Focused security analysis agent. Identifies injection risks, hardcoded secrets,
  auth gaps, and vulnerable dependencies. Read-only — never modifies files.
model: claude-opus-4-7
tools:
  - Read
  - Glob
  - Grep
  - Bash(npm audit)
  - Bash(pip-audit)
  - Bash(git diff *)
  - Bash(git log *)
permissions:
  deny: [Write, Edit]
---

You are a security-focused code reviewer. Never suggest refactoring or style changes.
Focus exclusively on: injection risks, credential exposure, broken auth, insecure dependencies, path traversal.

For each issue: file path, line number, severity (CRITICAL/HIGH/MEDIUM/LOW), and the specific fix required.
```

Spawn it with:
```
"Spawn the security-reviewer agent to audit the auth module"
```

---

## 19. Nested Skills — Can a Skill Invoke Another Skill?

No formal mechanism exists. Skills have no `invoke:` frontmatter field and no tool Claude can call internally to trigger another skill. The `/skill-name` slash command is user-facing only.

### What happens in each context mode

**Without `context: fork`** — skill runs in the main conversation where Claude still knows all skill descriptions. If the skill body says *"now run /format-imports"*, Claude might auto-invoke it — but this is emergent behavior, not a designed feature. Unreliable.

**With `context: fork`** — skill runs in an isolated context with no session state. Claude in the fork has no awareness of other skills. Nested invocation effectively impossible.

### The right pattern — agent with `skills` field

Use an agent that lists all the skills it needs to orchestrate:

```yaml
# .claude/agents/code-quality-lead/AGENT.md
---
name: code-quality-lead
description: Runs the full code quality pipeline on staged changes
tools: [Read, Glob, Grep, Bash]
skills:
  - security-review
  - format-imports
  - test-coverage
---

Run these skills in order on the staged changes:
1. /security-review — stop if any CRITICAL issues found
2. /format-imports — fix all import ordering
3. /test-coverage — report untested paths

Summarize: issues found, fixes applied, coverage gaps.
```

The agent invokes each listed skill in sequence or conditionally — clean orchestration of multiple skills.

### Summary

| Approach | Works? |
|----------|--------|
| Skill body triggers another skill (no fork) | Sometimes — emergent, unreliable |
| Skill body triggers another skill (with fork) | No — isolated context, no skill awareness |
| `invoke:` frontmatter field | Doesn't exist |
| Agent with `skills: [a, b, c]` | ✅ Designed for this |
| Main session calling skills sequentially | ✅ Always works |

---

## 20. Unloading Skill Context

There is no selective unload mechanism. Once a skill runs inline (without `context: fork`), its content and tool calls are part of the conversation history and cannot be removed.

| Goal | Mechanism |
|------|-----------|
| Prevent skill from polluting context in the first place | `context: fork` in SKILL.md — skill runs isolated, only summary returns |
| Reduce overall context after a skill ran | `/compact` — summarizes everything, cannot exclude a specific skill |
| Start completely clean | `/clear` — new session, everything gone |

**`context: fork` is the right solution** — designed exactly for this. If a skill is context-heavy (large codebase scans, deep analysis), mark it with `context: fork` so the main conversation never absorbs its internal tool calls and reasoning:

```yaml
---
name: security-review
context: fork      # skill's work stays in its own window; main context unchanged
allowed-tools: [Read, Glob, Grep]
---
```

If skills are bloating your context, add `context: fork` to the skill definition — there is no after-the-fact unload.

---

## Quick Reference

| Concept | Key Point |
|---------|-----------|
| **Reference skill** | Conventions applied passively — no special type field; Claude matches via description |
| **Task skill** | Executable workflow — invoked with `/skill-name` |
| User scope | `~/.claude/skills/` — all projects |
| Project scope | `.claude/skills/` — current project, git-tracked |
| `description` | Most important field — Claude reads to decide auto-invocation |
| `model` | Override model for this skill only; session model unchanged after |
| `effort` | `low`/`medium`/`high`/`xhigh` — reasoning depth for this skill |
| `allowed-tools` | Hard whitelist — Claude can ONLY use listed tools |
| `disable-model-invocation` | Pure shell execution, no Claude call |
| `context: fork` | Skill runs in isolated context window — result summary returned to main |
| `` !`cmd` `` | Dynamic injection — runs shell command, injects output before Claude reads skill |
| `{{arg}}` | Argument interpolation in skill body |
| `arguments` | Typed inputs — `name`, `description`, `required`, `default` |
| **Auto-invocation** | Claude matches task description to skill `description` — be specific |
| `/skills` | List all available skills |
| `/review` | Built-in: fast in-session code review |
| `/simplify` | Built-in: review + fix code quality issues |
| `/security-review` | Built-in: security audit of staged changes |
| `/ultrareview` | Built-in: multi-agent deep review, background, use before merging |
| **Subagent** | Isolated Claude instance — does focused work, returns summary |
| `Explore` type | Read-only research — keeps main context clean |
| `Plan` type | Architecture design, read-only |
| `general-purpose` | Default — all tools |
| Custom agent | `.claude/agents/<name>/AGENT.md` with tools + permissions |
| Foreground agent | You wait for result |
| Background agent | Runs while you work, notifies when done |
| Skill vs Subagent | Skill = reusable procedure; Subagent = one-off focused task |
| `context: fork` vs Subagent | Fork = skill runs isolated; Subagent = separate agent instance with own lifecycle |
