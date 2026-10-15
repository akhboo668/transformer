# Claude Code — IDE Integration

**Date:** 2026-05-11
**Topic:** VS Code & JetBrains integration, Checkpoints, @browser
**Curriculum:** Module 6 (Days 13–15)

---

## 1. VS Code Extension vs CLI — Same Engine, Different Surface

The VS Code extension and the CLI (`claude`) run the same Claude Code engine. The difference is the interface layer:

| Capability | CLI | VS Code Extension |
|-----------|-----|-------------------|
| Conversation | Terminal text | Graphical panel |
| File edits | Inline diffs in terminal | Inline diffs in editor |
| File references | `@path/to/file` | `@`-mention with fuzzy search, `Alt+K` |
| Session history | `claude -c` / `claude -r` | Session list in panel |
| Checkpoints (rewind) | ❌ | ✅ hover any message |
| `@browser` | ❌ | ✅ |
| Multiple sessions | Multiple terminals | Multiple tabs in panel |
| Keyboard shortcut to focus | — | `Ctrl+Esc` (Win/Linux), `Cmd+Esc` (Mac) |

---

## 2. Key VS Code Shortcuts

| Action | Shortcut |
|--------|----------|
| Focus Claude Code panel | `Ctrl+Esc` / `Cmd+Esc` |
| Insert file reference | `Alt+K` / `Option+K` |
| Accept inline diff | Click ✓ in the diff gutter |
| Reject inline diff | Click ✗ in the diff gutter |
| Cycle permission modes | `Shift+Tab` (same as CLI) |

---

## 3. `@`-Mentions and File References

In the VS Code panel you can reference files, symbols, and other context using `@`:

```
@src/auth/login.ts          ← specific file
@AuthMiddleware             ← symbol (fuzzy matched across codebase)
@browser                   ← current browser state (see Section 5)
```

`Alt+K` opens a file picker with fuzzy search — faster than typing the full path. The selected file is inserted as an `@`-mention.

**Why use `@`-mentions instead of asking Claude to read a file:**
- The file content is injected directly into context — no tool call, no token cost for the Read operation
- Claude has the file immediately without a round-trip

---

## 4. Checkpoints — Rewind to Any Point

### What checkpoints are

Every message in the VS Code conversation panel is a **checkpoint** — a snapshot of your file state at the moment Claude finished responding to that message. You can rewind to any of them.

```
Message 1: "Add input validation to the login form"
    → Claude edits login.ts, form.tsx         ← Checkpoint A

Message 2: "Now extract that into a reusable hook"
    → Claude creates useValidation.ts,
      refactors login.ts, form.tsx             ← Checkpoint B

Message 3: "Refactor all forms to use the hook"
    → Claude edits 8 files                    ← Checkpoint C
    ← something went wrong here
    → Rewind to Checkpoint B → files restored to post-Message-2 state
```

### How to use

1. Hover over any message in the conversation panel
2. A **rewind** icon appears (↩) to the left of the message
3. Click it — Claude Code restores all modified files to their state at that checkpoint
4. The conversation history above the checkpoint is preserved; you can continue from there with a different instruction

### What gets restored

- **File contents** — all files Claude modified after the checkpoint are reverted
- **New files** — files Claude created after the checkpoint are deleted
- **Conversation** — the conversation above the checkpoint stays intact

### What does NOT get restored

- **Git state** — checkpoints are not git commits. If you ran `git add` or `git commit` after the checkpoint, git history is unchanged.
- **External state** — API calls, database writes, npm installs that happened after the checkpoint are not undone
- **Shell state** — environment variables, running processes are not affected

### When to use checkpoints

| Situation | Action |
|-----------|--------|
| About to do a risky multi-file refactor | Note the current message number — you can rewind to it if needed |
| Claude went in the wrong direction | Rewind to the last good state, give a better instruction |
| Want to explore two different approaches | Rewind to a branch point, try approach B |
| Claude introduced a regression | Rewind to before the regression, identify what went wrong |

### Checkpoints vs git

| | Checkpoint | git commit |
|--|-----------|-----------|
| Granularity | Every Claude message | When you commit |
| Restoring | Instant, in-editor | `git checkout`, `git reset` |
| Covers | File contents only | Full git state |
| Persists after session | No — tied to conversation | Yes |
| Shared with team | No | Yes |

**Best practice:** For anything significant, use both — checkpoint for fast iteration safety, git commit when you reach a state worth preserving.

### Checkpoints vs `/branch`

- **Checkpoint** — restores files to a past state, stays in same conversation
- **`/branch`** — forks the conversation into a new thread from any point, original thread unchanged, files not rewound

Use checkpoints when you want to undo file changes. Use `/branch` when you want to explore an alternative conversationally without losing either path.

---

## 5. `@browser` — Live UI Testing

### What `@browser` does

`@browser` gives Claude access to a live browser session inside VS Code. Claude can:
- **See** the current page — takes a screenshot and reads the DOM
- **Navigate** — go to a URL, click links, follow redirects
- **Interact** — click buttons, fill forms, submit
- **Assert** — check that elements exist, text is correct, no console errors

This is the difference between Claude reading your code and Claude actually using your UI.

### Setup

1. Start your local dev server (e.g., `npm run dev` → `localhost:3000`)
2. Open a browser to the page you want Claude to test
3. In the Claude Code panel, type `@browser` — Claude captures the current browser state

No additional configuration needed. Claude Code integrates with VS Code's built-in browser or your default browser via the extension.

### Basic usage

```
"@browser — the login form is at localhost:3000/login.
Test the happy path: fill in valid credentials and verify the dashboard loads."

"@browser — after my changes, check that the error message appears
when submitting an empty email field."

"@browser — navigate to localhost:3000/dashboard and check
for any console errors or layout issues."
```

### What Claude does with @browser

1. **Takes a screenshot** of the current browser state
2. **Reads the DOM** — visible text, form fields, interactive elements
3. **Checks the console** for JavaScript errors
4. **Interacts** — clicks, types, submits forms
5. **Takes follow-up screenshots** after each interaction
6. **Reports** what it observed — what worked, what didn't, any errors

### Use cases

| Task | @browser approach |
|------|------------------|
| Verify a UI change looks correct | `@browser` after making the change — Claude screenshots and reports |
| Test a form flow end-to-end | Give Claude the URL and steps — it fills and submits |
| Check for console errors | `@browser` — Claude reads browser console |
| Regression test after refactor | `@browser` on key pages — Claude checks for visual/functional regressions |
| Debug a layout issue | `@browser` — Claude sees what you see, reads the CSS |
| Test responsive behavior | Specify viewport; Claude navigates and screenshots |

### @browser vs writing tests

| | `@browser` | Written test (Playwright/Cypress) |
|--|------------|----------------------------------|
| Speed to run | Immediate | Requires test setup |
| Repeatability | Manual re-invocation | Automated in CI |
| Coverage | Ad-hoc, guided | Defined, consistent |
| Best for | Exploratory, one-off verification | Regression, CI gate |

`@browser` is for **exploratory verification during development** — confirming a change works before writing a formal test. It doesn't replace automated testing but reduces the cycle between "I think this works" and "I've confirmed this works."

### Limitations

- Requires VS Code extension — not available in CLI
- Works best with local dev servers (localhost)
- Cannot authenticate to external services on your behalf
- JavaScript-heavy SPAs need a moment to load before screenshotting — tell Claude to wait if needed:
  ```
  "@browser — navigate to localhost:3000/dashboard,
  wait for the data table to load, then screenshot"
  ```

---

## 6. Inline Diffs

When Claude edits a file, VS Code shows the change as an **inline diff** in the editor — the old code and new code side by side before you accept.

```
Old code (red):   const user = req.body.user
New code (green): const user = req.body.user ?? null
```

**Accept** (✓) — applies the change to the file  
**Reject** (✗) — discards the change, file unchanged

This is the VS Code equivalent of the terminal's permission prompt for file edits. With `acceptEdits` mode, diffs are accepted automatically. In `default` mode you review each one.

**`autosave` setting** — controls whether accepted diffs are also saved to disk immediately:

```json
{
  "claude": {
    "autosave": true    // save to disk when diff accepted (default: true)
  }
}
```

---

## 7. Multiple Tabs — Parallel Tasks in the IDE

The VS Code extension supports multiple Claude Code tabs — each is an independent session with its own context window.

**When to use multiple tabs:**
- Running an Explore agent in Tab 2 while implementing in Tab 1
- Writer session in Tab 1, Reviewer session in Tab 2 (parallel review pattern)
- Monitoring a background task (CI polling) in Tab 2 while coding in Tab 1

Each tab is equivalent to a separate `claude` terminal process. Files are shared — if Tab 1 edits `login.ts`, Tab 2 can read the updated version.

---

## 8. VS Code Extension Settings

These settings live in VS Code's `settings.json` (not `~/.claude/settings.json`):

```json
{
  "claude.initialPermissionMode": "acceptEdits",
  "claude.preferredLocation": "panel",
  "claude.autosave": true
}
```

| Setting | Values | Effect |
|---------|--------|--------|
| `initialPermissionMode` | `default`, `plan`, `acceptEdits`, `auto` | Permission mode when a new session starts |
| `preferredLocation` | `panel`, `editor` | Where the Claude Code UI opens |
| `autosave` | `true` / `false` | Auto-save files when diffs are accepted |

**`initialPermissionMode: "plan"`** — every session starts in read-only plan mode. You have to explicitly cycle to a writing mode before Claude can edit files. Good safety default for production codebases.

---

## 9. JetBrains Integration

JetBrains (IntelliJ, PyCharm, WebStorm, etc.) has a Claude Code plugin with similar but distinct integration:

| Feature | VS Code | JetBrains |
|---------|---------|-----------|
| Focus shortcut | `Ctrl/Cmd+Esc` | `Ctrl/Cmd+Esc` |
| File reference | `Alt+K` | `Cmd+Option+K` / `Ctrl+Alt+K` |
| Diff viewer | VS Code diff | JetBrains IDE diff (more powerful — tracks moved blocks) |
| Diagnostics | Manual | **Auto-shared** — IDE sends errors/warnings to Claude automatically |
| External terminal connect | — | `/ide` command |

**Auto-shared diagnostics** is the key JetBrains advantage: when your code has a compile error or warning, JetBrains automatically includes it in Claude's context — you don't have to copy-paste the error message. Claude sees what the IDE sees.

### `/ide` command

If you're running `claude` in an external terminal but want to connect it to your JetBrains IDE:

```bash
claude
/ide    # connects this CLI session to the running JetBrains instance
        # Claude now has access to IDE diffs, diagnostics, file references
```

---

## 10. Remote Development

### VS Code Remote (SSH, Dev Containers, WSL)

The Claude Code extension runs **on the remote host** — not on your local machine. This means:
- Claude has access to the remote filesystem, not your local one
- `@file` references resolve on the remote
- MCP servers configured for the remote run on the remote
- `@browser` requires a browser on the remote (typically not available on headless servers)

### WSL2 on Windows

If you run VS Code with the Remote WSL extension and Claude Code inside WSL2, there's a known networking quirk: Claude Code uses a local socket for extension communication, and WSL2's network bridge can interfere.

Fix: ensure VS Code is connected to WSL2 via the Remote - WSL extension (not running natively in Windows with a WSL path). The extension should show `WSL: Ubuntu` (or your distro) in the bottom-left status bar.

---

## Quick Reference

| Concept | Key Point |
|---------|-----------|
| **VS Code vs CLI** | Same engine; VS Code adds checkpoints, @browser, inline diffs, multiple tabs |
| **Focus shortcut** | `Ctrl+Esc` / `Cmd+Esc` |
| **File reference** | `@filename` fuzzy or `Alt+K` picker — injects content without a tool call |
| **Checkpoint** | Snapshot of file state at every message — hover message → rewind icon |
| **Rewind restores** | File contents and new files created after checkpoint |
| **Rewind does NOT restore** | Git state, external calls, npm installs |
| **Checkpoint vs git** | Checkpoint = fast iteration safety; git = persists, shareable |
| **Checkpoint vs /branch** | Checkpoint rewinds files; /branch forks conversation without rewinding |
| **`@browser`** | Claude screenshots, reads DOM, interacts with live browser session |
| **`@browser` use** | Exploratory verification — not a replacement for automated tests |
| **`@browser` requires** | VS Code extension + local dev server |
| **Inline diffs** | Accept (✓) or reject (✗) each file change before it's applied |
| **`acceptEdits` mode** | Auto-accepts diffs — no per-diff prompt |
| **`autosave`** | VS Code setting — saves to disk when diff accepted |
| **Multiple tabs** | Independent sessions, shared filesystem |
| **`initialPermissionMode`** | VS Code setting — mode when new session starts |
| **JetBrains diagnostics** | Auto-shared — IDE errors go to Claude automatically |
| **`/ide` command** | Connects external CLI session to JetBrains IDE |
| **Remote dev** | Extension runs on remote host — `@browser` needs remote browser |
