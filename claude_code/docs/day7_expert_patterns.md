# Claude Code — Expert Patterns & Anti-Patterns

**Date:** 2026-05-11
**Topic:** Expert workflows, context discipline, CLAUDE.md hygiene, common pitfalls
**Curriculum:** Module 12 (Days 40–45)

---

## 1. The Expert Cycle

Every non-trivial task follows this cycle. Skipping steps is the most common source of wasted sessions.

```
1. EXPLORE ──→ 2. PLAN ──→ 3. CODE ──→ 4. REVIEW ──→ 5. VERIFY
     ↑                                                      │
     └──────────────── if issues found ────────────────────┘
```

### Step 1 — Explore (understand before touching)

Spawn an Explore subagent or use `@file` references to understand scope first. Never start coding blind.

```
"Before we change anything — explore the auth module.
Map: which files handle it, what the request flow is,
what tests exist. Report back with file paths."
```

**What you learn:** actual scope, existing patterns to reuse, tests that define correctness, landmines to avoid.

### Step 2 — Plan (`Shift+Tab Shift+Tab`)

Enter plan mode. Claude proposes an approach — you review it before any code is touched.

```
/plan
"Refactor the auth middleware to support both JWT and API key auth side by side"
```

Review: Are the files correct? Is the approach right? Are tests included? Only approve when you'd be comfortable with the implementation.

### Step 3 — Code (implement with success criteria)

Give Claude a concrete way to verify its own work upfront:

```
"Implement the changes. Success criteria:
- All existing tests pass: npm test
- New test covers both JWT and API key paths
- The /health endpoint still returns 200 without auth"
```

Without success criteria, Claude ships when it thinks it's done. With them, Claude verifies before declaring done.

### Step 4 — Review (`/review` then `/ultrareview`)

```
/review          ← fast, in-session, during iteration
/ultrareview     ← deep, multi-agent, background — use before merging
```

`/review` catches obvious issues quickly. `/ultrareview` dispatches independent agents (security, performance, tests) and cross-verifies findings — use it once per PR, not once per commit.

### Step 5 — Verify (don't trust, check)

Run the actual verification criteria you set in Step 3:

```bash
npm test                    # tests pass?
curl localhost:3000/health  # endpoint works?
git diff --stat             # scope matches what was planned?
```

Never skip this step. Claude's confidence is not the same as correctness.

---

## 2. Permission Mode per Cycle Step

Each step in the expert cycle has a natural mode. Cycling modes with `Shift+Tab` keeps Claude appropriately constrained for each phase.

| Cycle Step | Mode | Why |
|-----------|------|-----|
| **Explore** | `plan` | Read-only — Claude reads files, greps, searches but cannot edit or run commands. No accidental changes during research. |
| **Plan** | `plan` | Claude proposes approach by reading code — cannot execute anything. You review before approving. |
| **Code** | `acceptEdits` or `auto` | `acceptEdits` — auto-accepts file edits, still prompts for Bash (good balance). `auto` — accepts everything including Bash (full trust). |
| **Review** (`/review`) | `plan` | Claude reads changed files and reports — no modifications needed. Read-only enforced. |
| **Verify** | `default` | Controlled — Claude runs your specified verification commands but prompts before anything unexpected. |

### Mode cycling in practice

```
Start session → default
    │
    ├─ Shift+Tab Shift+Tab → plan        (Explore + Plan steps)
    │   Claude reads, proposes — cannot touch anything
    │
    ├─ Shift+Tab → acceptEdits           (Code step)
    │   File edits auto-accepted; Bash still prompts
    │
    ├─ Shift+Tab Shift+Tab → plan        (Review step)
    │   /review runs read-only
    │
    └─ Shift+Tab → default               (Verify step)
        Run test commands with controlled prompting
```

### Mode reference

| Mode | File edits | Bash commands | Use for |
|------|-----------|---------------|---------|
| `plan` | ❌ blocked | ❌ blocked | Explore, Plan, Review |
| `default` | prompts | prompts | Verify, cautious implementation |
| `acceptEdits` | ✅ auto | prompts | Code step (recommended) |
| `auto` | ✅ auto | ✅ auto | Trusted bulk operations |
| `bypassPermissions` | ✅ auto | ✅ auto | Sandboxed / scripted only |

---

## 3. Context Management — The Most Underused Skill

Context is the session's working memory. Mismanaging it is the #1 cause of degraded Claude performance in long sessions.

### Monitor constantly

```
/context    ← shows % used, breakdown by category
```

Check every 30–60 minutes in long sessions. Once you're past ~70% you're paying for context that Claude is increasingly unable to use effectively.

### The /clear vs /compact vs /btw decision tree

```
Is this a new, unrelated task?
    Yes → /clear   (fresh start, no context carried)
    No  ↓

Is the session getting heavy but you're still on the same task?
    Yes → /compact "focus on X"   (summarize, keep relevant thread)
    No  ↓

Is this a quick side question that shouldn't become history?
    Yes → /btw your question   (answered without bloating history)
    No  → continue normally
```

### What `/compact` preserves

After compaction, only two things survive intact:
1. **Project-root `CLAUDE.md`** — always reloaded
2. **Auto-memory** (`~/.claude/projects/.../memory/`) — persists across sessions

Everything else (conversation history, file reads, inline tool calls) is summarized. This means:
- Rules you typed in conversation → **lost after compact**
- Rules in CLAUDE.md → **survive**

Put persistent rules in CLAUDE.md. Never in conversation turns.

### `/compact` with focus

```
/compact "focus on the auth refactor — ignore the debugging we did on the logger"
```

The focus hint guides what the compaction summary emphasizes. Without it, Claude summarizes everything equally and may drop the most relevant context.

### When to `/clear` vs `/compact`

| Situation | Action |
|-----------|--------|
| Starting a completely different task | `/clear` |
| Corrected Claude twice with no progress | `/clear` + rewrite prompt |
| Same task, context getting large | `/compact "focus on X"` |
| Need to ask a side question | `/btw question` |
| Session was productive, want to checkpoint | `/compact` before risky next step |

---

## 4. CLAUDE.md Discipline

CLAUDE.md is your persistent instruction set — but it only works if it's short enough for Claude to actually read and apply every rule.

### The size rule

**Keep under 200 lines.** Past that, Claude starts ignoring buried rules. Treat it like production code — prune ruthlessly.

### The pruning test

For every line in CLAUDE.md, ask:

> "If I deleted this line, would Claude make a mistake I'd have to correct?"

- **Yes** → keep it
- **No** → delete it
- **Maybe** → delete it and watch for one week; add back only if a mistake happens

### What belongs where

| Content type | Goes in |
|--------------|---------|
| Project conventions, build commands, architecture context | CLAUDE.md |
| Rules that must survive compaction | CLAUDE.md `## Compact Instructions` section |
| Deterministic enforcement (always run linter, never edit .env) | **Hook** — not CLAUDE.md |
| Repeatable procedures (create PR, deploy) | **Skill** — not CLAUDE.md |
| One-off context for this session | Conversation turn — not CLAUDE.md |

CLAUDE.md is for **judgment and context**. Hooks enforce deterministically. Skills encode procedures. Don't mix them.

### The "Compact Instructions" section

Rules that must survive `/compact` go in a dedicated section at the top or bottom of CLAUDE.md:

```markdown
## Compact Instructions

These rules apply at all times and must be preserved through compaction:

- Never edit `.env` or `.env.production`
- Always run `npm test` before declaring a coding task complete
- The main branch is `main`, not `master`
- Use `structlog` not `logging` for all Python logging
```

Claude is trained to preserve this section when compacting. Everything else may be summarized.

### Signs your CLAUDE.md needs pruning

- You've added a rule and Claude still violates it → rule is buried, file is too long
- CLAUDE.md is over 200 lines → prune now
- Rules reference old libraries or patterns no longer in use → delete them
- You have paragraphs explaining "why" instead of stating the rule → move to comments or delete

### Example: before and after pruning

**Before (bloated):**
```markdown
## Testing

We use Jest for testing. When you write tests, make sure to follow our testing
conventions which we established in Q3 last year after the auth incident. Tests
should be comprehensive and cover edge cases. We had a problem before where mocked
tests passed but real tests failed, so always use real database connections in tests.
Don't use mocks for the database layer. The test database is configured in .env.test.
Make sure all tests pass before submitting. We also care about test performance so
don't write tests that take more than 5 seconds each.
```

**After (pruned):**
```markdown
## Testing
- Always use real DB connections — no mocks for the database layer (.env.test)
- All tests must pass before task is complete: `npm test`
- Max 5 seconds per test
```

Same rules. 3 lines instead of 9. Claude reads and applies it every time.

---

## 5. Verification Criteria — The Most Impactful Habit

The single biggest difference between novice and expert Claude Code users: **giving Claude a way to verify its own work before calling it done.**

### Without criteria (novice)

```
"Refactor the payment service to use the new Stripe SDK"
→ Claude implements, says "done"
→ You discover edge cases missed, tests broken, old API calls remaining
→ Multiple correction rounds
```

### With criteria (expert)

```
"Refactor the payment service to use the new Stripe SDK.

Verify when done:
1. npm test — all pass, no skips
2. grep -r 'stripe.charges' src/ — returns nothing (old API gone)
3. The checkout flow still works end-to-end: node scripts/smoke-test.js
4. No new TypeScript errors: npm run type-check"
```

Claude checks each criterion before declaring done. Issues surface immediately rather than after handoff.

### Types of good verification criteria

| Type | Example |
|------|---------|
| Test suite | `npm test`, `pytest`, `go test ./...` |
| Grep absence | `grep -r 'old_pattern' src/` returns nothing |
| Grep presence | `grep -r 'new_pattern' src/` returns expected files |
| Smoke test | `curl localhost:3000/api/health` returns 200 |
| Type check | `npm run type-check` exits 0 |
| Linter | `npm run lint` exits 0 |
| Expected output | "The migration script output should end with 'Done: 42 rows'" |

---

## 6. Parallel Sessions — Writer + Reviewer

For high-stakes work, run two Claude sessions simultaneously: one implements, one reviews with fresh context.

```
Session A (Writer):
    Implements the feature
    No review bias — just builds

Session B (Reviewer):
    Fresh session — no knowledge of implementation decisions
    Reviews Session A's output as if seeing it for the first time
    Catches assumptions the writer made invisible
```

### Setup

```bash
# Terminal 1 — Writer
claude --worktree

# Terminal 2 — Reviewer (separate worktree = separate git state)
claude --worktree

# Or use /branch to fork within the same session
```

### When to use it

- Before merging any security-sensitive code
- After large refactors where bias toward your own design is high
- When you've been in the same session for 2+ hours (fresh eyes catch what you stopped seeing)

The key insight: Claude in Session B has no memory of the decisions made in Session A. It reviews the code as any outside reviewer would — which surfaces different issues than self-review.

---

## 7. Research via Subagents — Keep Main Context Clean

Every large file read, every codebase scan, every deep investigation consumes main context. Once gone, it's gone until `/compact`.

### The pattern

```
# ❌ Novice — floods main context
"Read through all the files in src/auth/ and tell me how it works"
→ Claude reads 15 files, each result in main context → 20k tokens used

# ✅ Expert — context stays clean
"Spawn an Explore agent to map the auth module.
I need: which files, the request flow, what tests exist."
→ Agent reads 15 files in its own context → main gets a 500-token summary
```

### Rule of thumb

If the research could generate more than ~2,000 tokens of tool results, delegate it to an Explore agent.

---

## 8. Common Pitfalls

### Pitfall 1 — Kitchen Sink Sessions

**What it looks like:** One long session doing unrelated tasks — fix a bug, then write docs, then refactor a module, then investigate a performance issue.

**What happens:** Context fills with irrelevant history. Claude starts connecting unrelated things. Performance degrades.

**Fix:** `/clear` between unrelated tasks. Sessions should have one clear purpose.

---

### Pitfall 2 — Correction Loops

**What it looks like:** You correct Claude. Claude tries again wrong. You correct again. Three rounds, no progress.

**What happens:** Claude is anchored to its earlier approach. Each correction patches, never replaces.

**Fix:** `/clear` + rewrite the prompt from scratch with the constraint that was being violated stated upfront. Don't correct — restart with better context.

```
# Instead of correcting:
"No, that's still wrong because X"

# Clear and restart:
/clear
"Implement Y. Important constraint upfront: X must be true.
Do not do Z (that approach doesn't work here because...)"
```

---

### Pitfall 3 — Oversized CLAUDE.md

**What it looks like:** CLAUDE.md grows to 300, 400, 500 lines. Claude starts violating rules that are "in there somewhere."

**What happens:** Claude reads CLAUDE.md but distant rules get less attention. The effective instruction set shrinks even as the file grows.

**Fix:** Prune to under 200 lines. Move deterministic rules to hooks. Move procedures to skills. Keep only judgment and context in CLAUDE.md.

---

### Pitfall 4 — Infinite Exploration

**What it looks like:** "Investigate why the payments are slow" → Claude reads 40 files, follows every import, checks every dependency → context full, no conclusion.

**What happens:** Without scope, "investigate" is infinite. Claude explores breadth-first until context runs out.

**Fix:** Always scope investigation tasks explicitly:

```
# ❌ Unscoped
"Investigate why payments are slow"

# ✅ Scoped
"Investigate payment slowness. Scope: only src/payments/ and the database queries it makes.
Time limit: stop after reading 10 files and report what you've found so far."
```

Or delegate to an Explore agent — its context runs out independently, not yours.

---

### Pitfall 5 — Trust-then-Ship Gap

**What it looks like:** Claude says "done", you read the summary, it sounds right, you merge without running anything.

**What happens:** Edge cases, broken tests, missed files — all invisible until production.

**Fix:** Always define verification criteria upfront (Section 4). Always run them yourself before accepting "done."

```
Claude says "done" → you run npm test → 3 tests fail
→ much better to catch here than in production
```

---

### Pitfall 6 — Wrong Problem Solved

**What it looks like:** You ask Claude to fix a bug. Claude fixes it — but fixes the wrong thing, or misunderstands the root cause.

**What happens:** Jumped to implementation without understanding scope. The exploration step was skipped.

**Fix:** Plan mode before implementation for anything non-trivial:

```
/plan
"The payment webhook is sometimes processing orders twice.
Here's what I know: [details]"
→ Review Claude's proposed root cause before any code is touched
```

If Claude's plan identifies a different root cause than you assumed — that's the plan mode paying off.

---

### Pitfall 7 — Instructions in Conversation Turns

**What it looks like:** "From now on, always use tabs not spaces" → Claude follows it for this session → next session it's gone.

**What happens:** Conversation instructions don't survive `/compact` or session end. You re-teach Claude the same rules every session.

**Fix:** Any rule you've stated more than once belongs in CLAUDE.md.

---

## 9. Novice vs Expert — The Full Comparison

| Situation | Novice | Expert |
|-----------|--------|--------|
| Starting a task | Jumps straight to implementation | Explore first, plan second |
| Success criteria | None — trusts Claude's "done" | Explicit tests/checks given upfront |
| Long session | One session for everything | `/clear` between unrelated tasks |
| Context at 80% | Keeps going | `/compact "focus on X"` |
| Claude gets it wrong twice | Corrects inline | `/clear` + rewrites prompt |
| Persistent rule | Types it in conversation | Writes it to CLAUDE.md |
| CLAUDE.md | Grows indefinitely | Pruned to under 200 lines, reviewed monthly |
| Large file reads | Done inline in main context | Delegated to Explore agent |
| Side question | Full question in conversation | `/btw question` |
| Code review | Reads Claude's output, looks good | `/review` then verification criteria |
| Before merging | Ships when Claude says done | `/ultrareview` + run the tests yourself |
| Deterministic rule | In CLAUDE.md as text | In a hook — enforced, not suggested |
| Repeatable workflow | Prompted each time | Written as a skill |
| Security-sensitive work | Single-pass | Parallel sessions: writer + reviewer |
| Scope unclear | Starts coding to figure it out | Plan mode first |

---

## 10. Daily Workflow Template

A repeatable structure for each working session:

```
1. START
   claude -c           ← resume last session if continuing same task
   claude              ← new session if different task
   /context            ← check where you are

2. ORIENT
   If new task → Explore agent to understand scope
   If continuing → /compact "focus on [task]" if context heavy

3. PLAN
   /plan → review → approve

4. IMPLEMENT
   State success criteria upfront
   Use @file references instead of asking Claude to read large files inline
   /btw for side questions that arise

5. REVIEW
   /review → quick in-session check
   Run verification criteria yourself

6. CLOSE
   /ultrareview before merging (not every commit — once per PR)
   /compact to checkpoint before ending session
   Update CLAUDE.md if new conventions emerged this session
```

---

## 11. CLAUDE.md — Template Structure

```markdown
# Project Name

## What This Is
One paragraph: what the project does, tech stack, key architectural decisions.

## Build & Test Commands
- Install: `npm install`
- Run: `npm run dev`
- Test: `npm test`
- Lint: `npm run lint`
- Type check: `npm run type-check`

## Conventions
- [Key language/framework conventions — 5-10 lines max]
- [Naming patterns]
- [Error handling approach]

## Architecture Notes
- [Non-obvious architectural decisions — 3-5 lines]
- [What NOT to do and why]

## Compact Instructions
These rules must be preserved through compaction:
- [Critical rules that can't be lost]
- [Usually 3-5 items max]
```

Total: ~40-60 lines. Enough to give Claude real context. Short enough that every line gets read.

---

## Quick Reference

| Concept | Key Point |
|---------|-----------|
| **Expert cycle** | Explore → Plan → Code → Review → Verify — never skip steps |
| **Verification criteria** | Give Claude a way to check its own work before calling done |
| **`/context`** | Check every 30–60 min; act before 70% |
| **`/clear`** | New unrelated task, or after 2+ failed corrections |
| **`/compact "focus on X"`** | Same task, context heavy — guide the summary |
| **`/btw`** | Side question that shouldn't bloat history |
| **CLAUDE.md size** | Under 200 lines — past that, rules get ignored |
| **Pruning test** | "Would removing this cause a mistake?" — if no, delete |
| **Compact Instructions** | Section in CLAUDE.md that survives compaction |
| **Rules in conversation** | Don't — they die at `/compact` or session end; use CLAUDE.md |
| **Hooks vs CLAUDE.md** | Hooks = deterministic enforcement; CLAUDE.md = judgment & context |
| **Skills vs CLAUDE.md** | Skills = procedures; CLAUDE.md = conventions |
| **Parallel sessions** | Writer + Reviewer with fresh context = higher quality than self-review |
| **Explore agent** | Delegate large research — keeps main context clean |
| **Kitchen sink sessions** | One session = one purpose; `/clear` between tasks |
| **Correction loop fix** | `/clear` + rewrite prompt with constraint stated upfront |
| **Infinite exploration fix** | Scope explicitly; delegate to Explore agent |
| **Trust-then-ship gap** | Always run verification criteria yourself — never trust "done" |
| **Wrong problem fix** | Plan mode before touching code |
| **`/ultrareview`** | Once per PR before merging — not during iteration |
| **Session start** | `claude -c` to resume, `claude` for fresh, `/context` to orient |
