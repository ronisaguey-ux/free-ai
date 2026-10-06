# free-ai

One local endpoint in front of every free LLM tier. A ranked model list, automatic fallback, and a
linear cooldown per model. **Zero dependencies.**

```bash
node server.js          # http://127.0.0.1:8790
```

```bash
curl -s localhost:8790/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}]}'
```

No config, no key, no signup. The no-key providers serve immediately; add a key for any provider you
have and its models join the chain.

---

## The two ideas

**1. Serve strictly in rank order.** Best model first. The first one that answers wins, and
the reply is returned. The list is ordered by what has actually answered on this machine.

**2. A failure costs a cooldown that fits the failure.** A transient hiccup costs nothing (a
sibling model retries). A capacity limit is priced. Only a real "this model is gone" parks
for a long time. See `server.js` -> `penalise`.

---

## It works with no keys at all

The default chain in `data/models.json` is **keyless**. Every entry was checked with a real
completion before it was added, and each of these answers with no account and no API key:

| Provider | Endpoint | Notes |
|---|---|---|
| **Kilo Gateway** | `https://api.kilo.ai/api/gateway` | No auth header at all. Six free ids answer. |
| **LLM7.io** | `https://api.llm7.io/v1` | Keyless. `default` and `fast` are selectors that resolve server-side - they work where most concrete ids in its 66-model list need auth. |
| **AI Horde** | `https://oai.aihorde.net/v1` | Crowdsourced, slow. Its **documented** anonymous placeholder key is used. |
| **Pollinations** | `https://text.pollinations.ai/openai` | Keyless anonymous tier, budget-limited. |
| **uncloseai / unturf** | `https://hermes.ai.unturf.com/v1` | Genuinely keyless, one rotating model id, rate-limited per IP. |

Nothing else is in the default chain. **No provider of ours is baked in** - if it needs a
key it does not belong there, and it is not there.

Start it and it works:

```bash
node server.js
curl -s localhost:8790/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"freeai","messages":[{"role":"user","content":"hello"}]}'
```

---

## Adding your own keys (optional)

Create `config.json` (gitignored) and add a key. Nothing else is required:

```json
{
  "keys": {
    "groq": "gsk_...",
    "gemini": "AIza...",
    "openrouter": "sk-or-..."
  }
}
```

**Multiple keys rotate one per call** - three Gemini keys become 4,500 requests/day
instead of 1,500:

```json
{ "keys": { "gemini": ["AIza-key1", "AIza-key2", "AIza-key3"] } }
```

A provider only joins the chain **when its key is present**. `data/providers.json` is the
catalogue of what can be added - 89 providers with their base URL and where to get a key -
so adding one is a single line, not a hand-written model list.

To add a provider the catalogue does not know, declare it with its models:

```json
{
  "keys": { "myprovider": "sk-..." },
  "providers": {
    "myprovider": { "base": "https://api.myprovider.com/v1", "models": ["my-model-1"] }
  }
}
```

### Optional: prompt compression

The prompt can be shrunk before it is sent. This is **off by default** - a caller that has
not asked for it gets its bytes back untouched.

```json
{ "compression": "safe" }
```

- **`safe`** - strips trailing whitespace, collapses runs of blank lines, and elides a run
  of 4+ identical consecutive lines (a log spamming one error). It **never** touches
  line-numbered file content, template variables, URLs, or an edit contract.
- **`balanced`** - `safe`, plus comment stripping and JSON minification, applied only to
  EARLIER turns. The final user turn is exempt, because that is the text a quoted edit
  comes from.

The runtime level can be changed without a restart:
`POST /admin/compression {"level":"balanced"}`. Responses carry the saving in
`x-free-ai-compression`, and `/health` reports the running total.

### Optional: the MCP server

`mcp-server.js` exposes the router, the compressor, subagents and direct file tools as MCP
tools, so any MCP-capable client can drive them. Zero dependencies - JSON-RPC over stdio.

```json
{ "mcpServers": { "free-ai": { "command": "node", "args": ["/path/to/free-ai/mcp-server.js"] } } }
```

`freeai_subagent` runs a real multi-step loop (the model calls `read_file` / `list_dir` /
`grep` itself, then answers). It is **read-only** unless `allow_write` / `allow_bash` are
set per call. Run `freeai_selftest` to check the whole path end to end.

---

## Test

```bash
npm test
```

Starts a mock provider that fails on demand and drives the real server over HTTP: the happy path, a
429 falling through to the next model, a cooling model being skipped even though it would work, the
15→30 minute ladder, a success resetting the streak, and state surviving a restart. 17 checks.
