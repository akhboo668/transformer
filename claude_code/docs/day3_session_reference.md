# Claude Code — Day 3 Learning Session Reference

**Date:** 2026-05-10
**Topic:** Hooks System, MCP Servers & External Integrations
**Curriculum:** Module 4 (Days 7–10) + Module 5 (Days 10–13)

---

## 1. Hooks — Overview

Hooks are **deterministic shell commands** that fire automatically at specific points in Claude Code's lifecycle. Unlike Claude's responses (probabilistic), hooks always run the same way — they are your enforcement and automation layer.

### The Mental Model

```
User prompt → Claude thinks → Claude calls a tool
    → [PreToolUse hook]
    → tool runs
    → [PostToolUse hook]
    → Claude responds
    → [Stop hook]
```

Hooks sit between lifecycle events and let you: **block**, **modify**, **auto-approve**, or **react**.

---

## 2. Hook Types

| Type | What runs | Blocking? | Use for |
|------|-----------|-----------|---------|
| `command` | Shell command | Yes (via exit code) | Formatters, linters, blockers, loggers |
| `prompt` | LLM evaluates a natural language condition | Yes (LLM decides allow/deny) | Fuzzy/semantic safety checks |
| `agent` | Spawns a subagent with full tool access | Yes | Complex verification needing file reads |
| `async` | Shell command, non-blocking | No | Background tasks — tests, notifications |

---

## 3. Hook Events — Complete Reference

### The Full Lifecycle Sequence

```
Session opens
    └── SessionStart

User types prompt + hits Enter
    └── UserPromptSubmit

Claude decides to call a tool
    └── PermissionRequest  (only if permission mode would normally show a dialog)
    └── PreToolUse
    └── [tool runs]
    └── PostToolUse        (on success)
    └── PostToolUseFailure (on failure)

Claude finishes full response for this turn
    └── Stop

User runs /compact  (or auto-compact triggers)
    └── PreCompact  →  [compaction]  →  PostCompact
```

---

### Every Event — Precisely When It Fires

| Event | Fires when | Once per | Common use |
|-------|-----------|----------|-----------|
| `SessionStart` | Session process starts (including `-c` resume) | Session | Load env vars, credentials, print context |
| `UserPromptSubmit` | You press Enter on a prompt | Prompt | Log, block, or inject into prompts |
| `PermissionRequest` | Permission gate would show a dialog | Gated tool call | Auto-approve programmatically — no dialog |
| `PreToolUse` | Before any tool call, regardless of permission mode | Tool call | Validate, modify input, block |
| `PostToolUse` | After a tool call succeeds | Tool call | Auto-format, run tests, notify |
| `PostToolUseFailure` | After a tool call fails/errors | Tool call | Alert, log failures, trigger fallback |
| `Stop` | Claude finishes generating a full turn | Turn | Notify, trigger CI, post to Slack |
| `PreCompact` | Before `/compact` runs | Compaction | Re-inject context that must survive compaction |
| `PostCompact` | After compaction completes | Compaction | Receives summary — log, act on it |
| `SubagentStart` | A subagent is spawned (Agent tool) | Agent | Monitor parallel agents |
| `SubagentStop` | A subagent finishes | Agent | Collect results, enforce limits |
| `TaskCreated` | Claude creates a todo task (TodoWrite) | Task | Sync to Jira/Linear |
| `TaskCompleted` | Claude marks a task done | Task | Audit planned vs finished work |
| `FileChanged` | Any file changes on disk — including external changes | File change | React to external edits, watcher integration |
| `CwdChanged` | Working directory changes | Directory change | Load project-specific context per-directory |
| `ConfigChange` | `settings.json` modified and reloaded | Config save | Validate config changes, notify teammates |

**`PermissionRequest` vs `PreToolUse`:**
- `PermissionRequest` — fires only when the permission mode would show a dialog; lets you auto-approve instead
- `PreToolUse` — fires on every tool call, regardless of permission mode; for validation and blocking

**`FileChanged`** is the odd one out — it fires on external filesystem changes, not just Claude's tool calls.

**`PreCompact` matchers:**
```json
"matcher": "manual"    ← only user-initiated /compact
"matcher": "auto"      ← only Claude-triggered auto-compact
```

---

## 4. Exit Codes — How Hooks Control Flow

```
exit 0  →  allow / continue (success)
exit 1  →  show error message, but continue anyway
exit 2  →  BLOCK — stop the tool call entirely, show reason to Claude
```

Only `PreToolUse`, `PermissionRequest`, and `UserPromptSubmit` hooks can block (exit 2 has effect).
`PostToolUse` and `Stop` hooks run after the fact — exit 2 there only surfaces an error, doesn't undo.

---

## 5. Hook Configuration Structure

In `settings.json` at any scope (user / project / local):

```json
{
  "hooks": {
    "EVENT_NAME": [
      {
        "matcher": "ToolName|OtherTool",
        "if": { "regex": "pattern" },
        "hooks": [
          {
            "type": "command",
            "command": "your-shell-command",
            "timeout": 30,
            "statusMessage": "Running formatter..."
          }
        ]
      }
    ]
  }
}
```

---

## 6. The `matcher` Key

`matcher` selects **which tool calls** trigger the hook. The value is always a tool name or pipe-separated list of tool names.

### Category 1 — Built-in Claude Code Tools

| Matcher value | Tool it targets |
|---------------|----------------|
| `Bash` | Shell commands |
| `Read` | Reading files |
| `Write` | Writing files (full content) |
| `Edit` | Editing files (partial/targeted) |
| `Glob` | File pattern search |
| `Grep` | Content search |
| `WebFetch` | Fetching a URL |
| `WebSearch` | Web search |
| `Agent` | Spawning a subagent |
| `NotebookEdit` | Jupyter notebook edits |
| `TodoWrite` | Writing todo tasks |

### Category 2 — MCP Tool Names

```
mcp__<server-name>                    ← all tools from that server
mcp__<server-name>__<tool-name>       ← one specific tool

mcp__github                           ← any GitHub MCP tool
mcp__github__create_issue             ← only create_issue
mcp__postgres__query                  ← only postgres query tool
```

### Category 3 — Pipe `|` for Multiple Tools

```json
"matcher": "Edit|Write"         ← fires on either
"matcher": "Bash|WebFetch"
"matcher": "mcp__github|Bash"
```

### How to Discover Tool Names at Runtime

Add this temporary log hook, trigger Claude, then check the log:

```json
"PreToolUse": [{
  "matcher": "Bash",
  "hooks": [{ "type": "command", "command": "jq -r '.tool_name' >> ~/.claude/tool-log.txt" }]
}]
```

The `tool_name` in the stdin JSON is exactly what you put in `matcher`.

---

## 7. The `if` Key — Narrowing Within a Tool

`matcher` picks the tool family. `if` filters by the arguments:

```json
{ "if": { "args": "rm *" } }                    // glob on the argument string
{ "if": { "regex": "git push.*--force" } }       // regex on the argument string
{ "if": { "args": "*.env" } }                    // only calls touching .env files
```

Decision flow:
```
Tool name matches matcher?  →  No  →  skip
        ↓ Yes
if condition matches?        →  No  →  skip
        ↓ Yes
Run the hook
```

---

## 8. Hook stdin — What Your Command Receives

Every hook receives a JSON payload on stdin:

```json
{
  "session_id": "abc123",
  "tool_name": "Edit",
  "tool_input": {
    "file_path": "/src/auth.ts",
    "content": "..."
  },
  "tool_response": { "success": true }
}
```

`tool_response` is only present in `PostToolUse` and `PostToolUseFailure` hooks.

Extract with `jq`:
```bash
jq -r '.tool_input.file_path'    # file being edited
jq -r '.tool_input.command'      # bash command being run
jq -r '.tool_name'               # which tool was called
```

---

## 9. Hook JSON Output — Talking Back to Claude

Your hook can print JSON to stdout to control behaviour or inject context:

```json
{
  "systemMessage": "Message shown to user in the UI",
  "continue": false,
  "stopReason": "Shown when blocking — explains why",
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "deny",
    "permissionDecisionReason": "File is protected",
    "additionalContext": "Context injected back into the model",
    "updatedInput": { "command": "modified command here" }
  }
}
```

| Field | Effect |
|-------|--------|
| `systemMessage` | Shows a message in the UI (all hooks) |
| `continue: false` | Blocks / stops execution |
| `stopReason` | Message shown when blocking |
| `additionalContext` | Text injected into Claude's context |
| `permissionDecision` | `"allow"`, `"deny"`, or `"ask"` (PreToolUse only) |
| `updatedInput` | Modified tool input — Claude runs the modified version (PreToolUse only) |

---

## 10. Command Hook Examples

### Auto-format on Save
```json
"PostToolUse": [{
  "matcher": "Edit|Write",
  "hooks": [{
    "type": "command",
    "command": "jq -r '.tool_input.file_path' | { read -r f; prettier --write \"$f\"; } 2>/dev/null || true"
  }]
}]
```

### Block Force Push
```json
"PreToolUse": [{
  "matcher": "Bash",
  "if": { "regex": "git push.*--force" },
  "hooks": [{
    "type": "command",
    "command": "echo '{\"continue\":false,\"stopReason\":\"Force push is not allowed\"}'"
  }]
}]
```

### Re-inject Context Before Compaction
```json
"PreCompact": [{
  "hooks": [{
    "type": "command",
    "command": "cat .claude/compact-context.md"
  }]
}]
```

### Log All Bash Commands
```json
"PreToolUse": [{
  "matcher": "Bash",
  "hooks": [{
    "type": "command",
    "command": "jq -r '\"[\" + .session_id + \"] \" + .tool_input.command' >> ~/.claude/bash-log.txt"
  }]
}]
```

### Run Tests After File Changes (async — non-blocking)
```json
"PostToolUse": [{
  "matcher": "Edit|Write",
  "hooks": [{
    "type": "async",
    "command": "npm test 2>&1 | tail -5 >> ~/.claude/test-log.txt"
  }]
}]
```

### Notify on Session End
```json
"Stop": [{
  "hooks": [{
    "type": "command",
    "command": "echo '{\"systemMessage\":\"Session complete — all changes saved.\"}'"
  }]
}]
```

---

## 11. Prompt Hook Examples

Prompt hooks use an LLM to evaluate a **natural language condition** — allow or deny based on semantic understanding rather than regex. Available only on `PreToolUse`, `PostToolUse`, `PermissionRequest`.

**Variables in prompt string:** `$TOOL_INPUT`, `$TOOL_NAME`, `$ARGUMENTS`

**Cost:** Each prompt hook adds ~1–3 seconds of LLM latency per matched tool call. Don't put them on high-frequency events like every `Read`.

---

### Detect Credentials in File Writes

```json
"PreToolUse": [{
  "matcher": "Write|Edit",
  "hooks": [{
    "type": "prompt",
    "prompt": "The following file content is about to be written. Does it contain any credentials, API keys, tokens, passwords, or secrets in plaintext? If yes, deny. Tool input: $TOOL_INPUT"
  }]
}]
```

Why prompt and not regex? Secrets take hundreds of forms — regex allowlists are never complete. The LLM recognises them semantically.

---

### Scope Guard — Docs-Only Session

```json
"PreToolUse": [{
  "matcher": "Write|Edit",
  "hooks": [{
    "type": "prompt",
    "prompt": "The file path being edited is in the tool input below. Is the file path inside the 'docs/' directory? If it is NOT inside docs/, deny this edit. Tool input: $TOOL_INPUT"
  }]
}]
```

Handles edge cases a regex would miss: `./docs`, absolute paths, `docs-internal/`.

---

### Destructive Command Check (Semantic)

```json
"PreToolUse": [{
  "matcher": "Bash",
  "hooks": [{
    "type": "prompt",
    "prompt": "The following bash command is about to run. Is this command potentially destructive — meaning it could delete data, drop databases, overwrite important files, or cause irreversible changes? If yes, deny. Command: $TOOL_INPUT"
  }]
}]
```

Catches things regex misses: `truncate -s 0 file`, `> important.log`, `DROP TABLE` inside a heredoc.

---

### Prevent Scope Creep

```json
"PreToolUse": [{
  "matcher": "Write|Edit",
  "hooks": [{
    "type": "prompt",
    "prompt": "Claude should only be editing authentication-related files (auth/, middleware/, login/). The file it is trying to edit is shown below. Is this file clearly outside the scope of authentication work? If yes, deny. Tool input: $TOOL_INPUT"
  }]
}]
```

---

### Detect PII in Command Output (PostToolUse)

```json
"PostToolUse": [{
  "matcher": "Bash",
  "hooks": [{
    "type": "prompt",
    "prompt": "The following is the output of a bash command. Does it contain personally identifiable information (PII) such as names, email addresses, phone numbers, or SSNs? If yes, deny so the output is not passed to the model. Tool response: $TOOL_INPUT"
  }]
}]
```

---

### When to Use Prompt vs Command Hook

| Situation | Use |
|-----------|-----|
| Check is deterministic (regex, exit code, file exists) | `command` — faster, no LLM cost |
| Check is semantic ("is this destructive?", "contains secrets?") | `prompt` — understands intent |
| High-frequency events (every Read, every Glob) | `command` — prompt hooks are too slow here |
| Fuzzy scope or safety decisions | `prompt` — more accurate than pattern matching |
| You need to modify the tool input | `command` with `updatedInput` in JSON output |

---

## 12. `/hookify` Plugin

Rather than hand-editing JSON, the official `hookify` plugin provides slash commands for hook management:

```
/hookify                ← interactive hook builder
/hookify:list           ← see all active hooks
/hookify:configure      ← edit existing hooks
/hookify:help           ← usage reference
```

Install: add `hookify` to `enabledPlugins` in `settings.json`.

---

## 13. MCP Servers & External Integrations

### What MCP Is

**Model Context Protocol** is an open standard that lets any external system expose **tools, resources, and prompts** to Claude. It's the plug-in socket — Claude Code is the host, MCP servers are the plug-ins.

```
Claude Code  ←── MCP protocol (JSON-RPC 2.0) ───→  MCP Server  ←──→  External System
```

Without MCP: Claude uses only built-in tools (`Bash`, `Read`, `Write`…)
With MCP: Claude gets new tools — `create_issue`, `query_database`, `send_message` — defined by the server.

**Three things a server can expose:**

| Capability | What it is | Example |
|-----------|-----------|---------|
| **Tools** | Functions Claude can call | `create_pr`, `run_query` |
| **Resources** | Data Claude can read | File contents, DB rows, API responses |
| **Prompts** | Reusable prompt templates | Pre-built task instructions |

---

### Transport Types

| Transport | How it works | Use for |
|-----------|-------------|---------|
| `stdio` | Claude spawns the server as a local process; JSON-RPC over stdin/stdout | Local tools, scripts, CLIs |
| `http` | Claude calls a remote HTTP endpoint | Internal APIs, cloud services |
| `sse` | Server-Sent Events — streaming remote connection | Long-running / streaming responses |

`stdio` is the most common for custom servers — no network setup needed.

---

### How stdio Works

```
Claude Code spawns: python server.py
        ↓
Sends JSON-RPC request to server's stdin
        ↓
Server reads stdin, processes, writes JSON-RPC response to stdout
        ↓
Claude Code reads stdout, passes result to Claude
        ↓
Claude Code kills the process when done
```

**Critical rule:** stdout is the protocol channel — writing anything other than JSON-RPC messages to stdout corrupts it and breaks the server.

```python
# ❌ BREAKS THE SERVER — corrupts the JSON-RPC channel
print("Processing request")

# ✅ Safe — stderr is separate from the protocol
print("Processing request", file=sys.stderr)
import logging; logging.info("Processing")  # also safe — goes to stderr
```

```typescript
// ❌ BREAKS THE SERVER
console.log("Server started");

// ✅ Safe
console.error("Server started");  // stderr is safe
```

---

## 14. Building a Custom stdio MCP Server

### Option A — Python with FastMCP (Recommended)

FastMCP auto-generates tool schemas from type hints and docstrings — minimal boilerplate.

#### Setup

```powershell
# Windows
uv init my-server
cd my-server
uv venv
.venv\Scripts\activate
uv add "mcp[cli]" httpx
```

```bash
# macOS/Linux
uv init my-server && cd my-server
uv venv && source .venv/bin/activate
uv add "mcp[cli]" httpx
```

#### Minimal Server (`server.py`)

```python
import os
import sys
from mcp.server.fastmcp import FastMCP

# Initialize — name appears in Claude's tool list
mcp = FastMCP("my-server")

# ── Tools ────────────────────────────────────────────────

@mcp.tool()
async def get_item(item_id: str) -> str:
    """Fetch an item by ID from the internal API.

    Args:
        item_id: The unique item identifier (e.g. 'ITEM-123')
    """
    api_key = os.environ["MY_API_KEY"]   # read from env — never hardcode
    # ... call your API ...
    return f"Item {item_id}: description here"


@mcp.tool()
async def create_item(name: str, category: str, priority: int = 1) -> str:
    """Create a new item.

    Args:
        name: Item display name
        category: One of: bug, feature, task
        priority: Priority level 1-5 (default 1)
    """
    if priority < 1 or priority > 5:
        raise ValueError("Priority must be between 1 and 5")
    # ... create item ...
    return f"Created item '{name}' in {category} with priority {priority}"


# ── Entry point ──────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

FastMCP reads:
- **Parameter types** → JSON Schema types (`str` → `string`, `int` → `integer`, `float` → `number`)
- **Docstring** → tool `description` and per-parameter `description`
- **Default values** → marks fields as optional in the schema
- **Raises** → inferred as error cases

---

### Option B — Python with Low-Level SDK

Use when you need full control over the schema or server capabilities.

```python
import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent

app = Server("my-server")

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_item",
            description="Fetch an item by ID from the internal API",
            inputSchema={
                "type": "object",
                "properties": {
                    "item_id": {
                        "type": "string",
                        "description": "Unique item identifier, e.g. 'ITEM-123'"
                    }
                },
                "required": ["item_id"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "get_item":
        item_id = arguments["item_id"]
        # ... fetch ...
        return [TextContent(type="text", text=f"Item {item_id}: ...")]
    raise ValueError(f"Unknown tool: {name}")

async def main():
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
```

---

### Option C — TypeScript

```bash
mkdir my-server && cd my-server
npm init -y
npm install @modelcontextprotocol/sdk zod@3
npm install -D @types/node typescript
```

**`tsconfig.json`:**
```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "Node16",
    "moduleResolution": "Node16",
    "outDir": "./build",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true
  }
}
```

**`package.json` additions:**
```json
{
  "type": "module",
  "scripts": { "build": "tsc" }
}
```

**`src/index.ts`:**
```typescript
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const server = new McpServer({
  name: "my-server",
  version: "1.0.0"
});

// ── Tools ──────────────────────────────────────────────

server.registerTool(
  "get_item",
  {
    description: "Fetch an item by ID from the internal API",
    inputSchema: {
      item_id: z.string().describe("Unique item identifier, e.g. 'ITEM-123'")
    }
  },
  async ({ item_id }) => {
    const apiKey = process.env.MY_API_KEY;
    if (!apiKey) throw new Error("MY_API_KEY env var not set");
    // ... fetch ...
    return {
      content: [{ type: "text", text: `Item ${item_id}: description here` }]
    };
  }
);

server.registerTool(
  "create_item",
  {
    description: "Create a new item",
    inputSchema: {
      name:     z.string().describe("Item display name"),
      category: z.enum(["bug", "feature", "task"]).describe("Item category"),
      priority: z.number().min(1).max(5).default(1).describe("Priority 1-5")
    }
  },
  async ({ name, category, priority }) => {
    // ... create ...
    return {
      content: [{ type: "text", text: `Created '${name}' in ${category}` }]
    };
  }
);

// ── Entry point ────────────────────────────────────────

const transport = new StdioServerTransport();
await server.connect(transport);
```

---

## 15. Tool Schema Design — What Makes a Good Schema

Claude uses the schema to decide **whether** to call a tool and **what arguments to pass**. Poor schemas produce wrong or missing arguments.

### inputSchema Field Rules

```json
{
  "type": "object",
  "properties": {
    "status": {
      "type": "string",
      "enum": ["open", "closed", "in_progress"],
      "description": "Filter by status. Omit to return all statuses."
    },
    "limit": {
      "type": "integer",
      "minimum": 1,
      "maximum": 100,
      "description": "Max results to return. Default 20."
    },
    "tags": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Filter by tags. Example: ['bug', 'auth']"
    }
  },
  "required": ["status"]
}
```

### Description Quality Matters

| Bad | Good |
|-----|------|
| `"The ID"` | `"Issue ID in format PROJ-123, e.g. 'ENG-456'"` |
| `"Start date"` | `"Start date in ISO 8601 format, e.g. '2026-01-15'"` |
| `"The thing"` | `"Repository owner — GitHub username or org, e.g. 'anthropics'"` |
| No description | Always provide one — Claude reads it to fill the argument |

### outputSchema — Optional but Useful

Define the shape of structured responses so Claude can parse them reliably:

```json
{
  "outputSchema": {
    "type": "object",
    "properties": {
      "id":     { "type": "string" },
      "status": { "type": "string" },
      "url":    { "type": "string" }
    },
    "required": ["id", "status"]
  }
}
```

When `outputSchema` is provided, return both a text summary AND `structuredContent`:

```python
return [TextContent(type="text", text=f"Created issue #{issue_id}")]
# + structured:
# {"id": "123", "status": "open", "url": "https://..."}
```

---

## 16. Error Handling — Two Mechanisms

### Mechanism 1 — Tool Execution Error (`isError: true`)

Use for expected failures — API down, bad input, not found:

```python
@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "get_item":
        try:
            result = await fetch_item(arguments["item_id"])
            return [TextContent(type="text", text=result)]
        except ItemNotFoundError as e:
            return [TextContent(type="text", text=f"Item not found: {e}")]
            # SDK sets isError: true automatically on exception,
            # or return with isError=True explicitly
```

Claude sees `isError: true` and knows the tool failed — it can decide to retry, try a different approach, or inform the user.

### Mechanism 2 — Protocol Error (JSON-RPC error)

For unknown tools, invalid JSON, server crashes — the SDK handles these automatically. You don't need to implement them manually.

```json
{ "jsonrpc": "2.0", "id": 3, "error": { "code": -32602, "message": "Unknown tool: bad_name" } }
```

### Rule of Thumb

| Scenario | Use |
|----------|-----|
| API rate limit, item not found, bad input value | `isError: true` — expected failure Claude can handle |
| Unknown tool name, malformed request | Protocol error — SDK handles automatically |
| Unhandled exception | SDK catches and converts to protocol error |

---

## 17. Credentials & Environment Variables

**Never hardcode credentials.** Read from env vars:

```python
import os

API_KEY = os.environ["MY_API_KEY"]          # required — raises if missing
TIMEOUT = int(os.getenv("TIMEOUT", "30"))   # optional with default
```

```typescript
const apiKey = process.env.MY_API_KEY;
if (!apiKey) throw new Error("MY_API_KEY is required");
```

Pass env vars in `.mcp.json` — values can reference shell env vars with `${VAR}`:

```json
{
  "mcpServers": {
    "my-server": {
      "type": "stdio",
      "command": "python",
      "args": ["path/to/server.py"],
      "env": {
        "MY_API_KEY": "${MY_API_KEY}",
        "BASE_URL":   "https://api.internal"
      }
    }
  }
}
```

`${MY_API_KEY}` reads from the shell environment at session start — the key is never stored in the config file.

---

## 18. Configuring in Claude Code (`.mcp.json`)

Place `.mcp.json` in `.claude/` for project scope or `~/.claude/` for user scope.

### Python (uv)
```json
{
  "mcpServers": {
    "my-server": {
      "type": "stdio",
      "command": "uv",
      "args": ["--directory", "C:/absolute/path/to/my-server", "run", "server.py"],
      "env": {
        "MY_API_KEY": "${MY_API_KEY}"
      }
    }
  }
}
```

### Python (venv)
```json
{
  "mcpServers": {
    "my-server": {
      "type": "stdio",
      "command": "C:/absolute/path/to/my-server/.venv/Scripts/python.exe",
      "args": ["C:/absolute/path/to/my-server/server.py"],
      "env": {
        "MY_API_KEY": "${MY_API_KEY}"
      }
    }
  }
}
```

### TypeScript (Node)
```json
{
  "mcpServers": {
    "my-server": {
      "type": "stdio",
      "command": "node",
      "args": ["C:/absolute/path/to/my-server/build/index.js"],
      "env": {
        "MY_API_KEY": "${MY_API_KEY}"
      }
    }
  }
}
```

**Use absolute paths** — relative paths are resolved from Claude Code's working directory, which may not be your project.

---

## 19. Testing & Debugging

### MCP Inspector (Official Tool)

```bash
npx @modelcontextprotocol/inspector python server.py
# or
npx @modelcontextprotocol/inspector node build/index.js
```

Opens a browser UI where you can:
- See all tools the server exposes
- Call tools with custom inputs
- Inspect the raw JSON-RPC messages
- See stdout/stderr separately

### `mcp dev` (FastMCP)

```bash
mcp dev server.py
```

Launches the server with the Inspector attached — the fastest feedback loop.

### Smoke Test via Claude Code

After registering in `.mcp.json`:
1. Start a new session (or run `/mcp` to reload)
2. Ask Claude: `"What tools does my-server expose?"`
3. Ask Claude to call one: `"Use my-server to get item ITEM-123"`

### Debugging Checklist

| Symptom | Likely cause |
|---------|-------------|
| Server not appearing in Claude | Syntax error in `.mcp.json`; wrong path |
| `tools/list` returns empty | `list_tools` handler not registered |
| Tool call returns garbage | Something printing to stdout (use stderr!) |
| Credentials error | Env var not set in shell before Claude starts |
| Timeout | Server taking too long to start — check command path |

### Check What Claude Sees

```
/mcp                    ← shows server status and tool count
/context                ← verify MCP schemas aren't flooding context
```

---

## 20. Finding MCP Servers (Marketplace Sources)

| Source | URL | Notes |
|--------|-----|-------|
| Official reference servers | `github.com/modelcontextprotocol/servers` | 7 servers, best documented |
| Glama | `glama.ai/mcp/servers` | 23,000+ servers, filterable by category |
| Smithery | `smithery.ai` | CLI installer: `npx @smithery/cli install <name> --client claude` |
| mcpservers.org | `mcpservers.org` | Curated shortlist, 11 categories |
| npm | `npm search @modelcontextprotocol` | Package-based discovery |

**Official servers worth installing immediately:**

```bash
# GitHub — PRs, issues, files
claude mcp add github -- npx -y @modelcontextprotocol/server-github

# Filesystem — safe file ops with access controls
claude mcp add filesystem -- npx -y @modelcontextprotocol/server-filesystem /path/to/allow

# Fetch — web content retrieval
claude mcp add fetch -- npx -y @modelcontextprotocol/server-fetch

# Memory — persistent knowledge graph across sessions
claude mcp add memory -- npx -y @modelcontextprotocol/server-memory
```

---

## Quick Recap — Day 3 Key Takeaways (MCP)

| Concept | Key Point |
|---------|-----------|
| MCP purpose | Lets external systems expose tools/resources/prompts to Claude via standard protocol |
| Transport | `stdio` — Claude spawns process, talks via stdin/stdout; `http`/`sse` — remote |
| JSON-RPC 2.0 | All MCP communication uses this protocol — SDKs handle it, you don't implement manually |
| stdout rule | **Never write to stdout in stdio servers** — it corrupts the protocol; use stderr |
| `tools/list` | How Claude discovers tools and their schemas at session start |
| `tools/call` | How Claude invokes a tool — sends name + arguments conforming to inputSchema |
| Deferred loading | Full schemas loaded on demand, not at start — saves context window |
| FastMCP (Python) | Auto-generates schemas from type hints + docstrings — least boilerplate |
| Low-level SDK | `@app.list_tools()` + `@app.call_tool()` — full control over schema |
| TypeScript SDK | `server.registerTool()` with `zod` for schema validation |
| inputSchema | JSON Schema object — types, enums, ranges, required fields |
| Description quality | Claude reads descriptions to fill arguments — vague descriptions → wrong arguments |
| outputSchema | Optional — define structured response shape for reliable parsing |
| isError: true | Expected failures (API down, not found) — Claude can decide how to handle |
| Protocol error | Unknown tool, malformed request — SDK handles automatically |
| Credentials | Always env vars — `${VAR}` in `.mcp.json` reads from shell environment |
| Absolute paths | Use absolute paths in `.mcp.json` — relative paths resolve unexpectedly |
| MCP Inspector | `npx @modelcontextprotocol/inspector server.py` — browser UI to test tools |
| `mcp dev` | FastMCP shortcut — launches server + Inspector in one command |
| `/mcp` command | List, add, enable, disable servers from within Claude Code |

---

## Quick Recap — Day 3 Key Takeaways

| Concept | Key Point |
|---------|-----------|
| Hooks purpose | Deterministic enforcement layer — runs the same way every time |
| Hook types | `command` (shell), `prompt` (LLM), `agent` (subagent), `async` (non-blocking) |
| Exit codes | `0` allow, `1` error+continue, `2` block |
| `PreToolUse` | Fires before every tool call — validate, modify, block |
| `PostToolUse` | Fires after success — react, format, test |
| `PermissionRequest` | Auto-approve/deny before dialog appears |
| `Stop` | Fires once per completed turn — notify, trigger CI |
| `PreCompact` | Last chance to inject content before summarization |
| `FileChanged` | Reacts to external file changes, not just Claude's edits |
| `matcher` | Built-in tool names (`Bash`, `Edit`, `Write`…) or MCP names (`mcp__github`) |
| Pipe in matcher | `"Edit\|Write"` — fires if either tool matches |
| `if` key | Narrows matcher by `args` (glob) or `regex` |
| stdin payload | JSON with `tool_name`, `tool_input`, `tool_response` — extract with `jq` |
| JSON output | `continue:false` blocks, `additionalContext` injects to model, `updatedInput` modifies |
| Prompt hooks | LLM-evaluated; semantic checks; only on Pre/PostToolUse + PermissionRequest |
| Prompt hook cost | ~1–3s LLM latency per call — use on write/exec events, not reads |
| `/hookify` plugin | Slash-command interface to build and manage hooks without editing JSON |
