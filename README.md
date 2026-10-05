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

**1. Serve strictly in rank order.** Best model first. The first one that answers wins, and the reply
carries `x-free-ai-model` so you know which it was.

**2. A failure costs a LINEAR cooldown.**

| consecutive failures | cooldown |
|---|---|
| 1 | 15 min |
| 2 | 30 min |
| 3 | 45 min |
| n | n × 15 min |

A success resets the streak to zero. Cooldowns persist to `state.json`, because a rate limit does not
care that you restarted. `401` `402` `403` `404` `408` `429` `5xx`, timeouts and connection errors
all count as failures. A `400` does **not** — that is your request being wrong, and hiding it behind
a fallback would just fail identically on the next model.

That is the whole design. There is no scheduler, no scoring model, no embeddings, no database.

## Endpoints

| Route | What |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible. Streaming works. |
| `GET /v1/models` | The ranked list with live health per model. |
| `GET /status` | Cooldowns, streaks, call counts, and which keys are missing. |
| `POST /admin/reset` | `{}` clears everything; `{"model":"x"}` clears one; `{"model":"x","expire":true}` clears the cooldown but keeps the streak (so the next failure steps up the ladder). |
| `GET /health` | Liveness. |

## Adding your own keys

Copy `config.example.json` to `config.json`:

```json
{
  "keys": {
    "groq": "gsk_...",
    "gemini": "AIza...",
    "cerebras": "csk-...",
    "deepseek": "sk-..."
  }
}
```

Or via the environment: `GROQ_API_KEY=... GEMINI_API_KEY=... node server.js`.

**Multiple keys rotate one per call** — three Gemini keys become 4,500 requests/day instead of 1,500:

```json
{ "keys": { "gemini": ["AIza-a", "AIza-b", "AIza-c"] } }
```

## Adding your own models

`config.json` can carry the list outright:

```json
{
  "keys": { "my_provider": "sk-..." },
  "models": [
    { "rank": 1, "provider": "my_provider", "model": "my-model", "base": "https://api.example.com/v1", "key_env": null, "no_key": false }
  ]
}
```

`rank` is the only ordering rule. Lower is tried first.

## What is in it

`data/providers.json` — 62 providers with their OpenAI-compatible base URL and where to get a key.
`data/models.json` — 45 free models across 20 of them, ranked.

Built by surveying the free-tier projects that already exist, taking the **provider endpoints** (a
fact) rather than their code, and throwing away everything that is not the model list or the fallback
rule. Sources studied, and what was taken from each:

| Project | Licence | Taken |
|---|---|---|
| [free-claude-code](https://github.com/Alishahryar1/free-claude-code) | AGPL-3.0 | the provider catalogue: 60 services with base URLs and credential pages |
| [OmniRoute](https://github.com/diegosouzapw/OmniRoute) | MIT | the shape of the tier chain; already runs on this machine as its own gateway |
| [freellmapi](https://github.com/tashfeenahmed/freellmapi) | MIT | the failure taxonomy (what counts as a quota failure vs a bad request) and the endpoints it adds: llm7, pollinations, AI Horde |
| [free-llm-api-resources](https://github.com/jtig37/free-llm-api-resources) | — | which providers are genuinely perpetual-free vs trial credits |
| [llm7.io](https://github.com/chigwell/llm7.io) | AGPL-3.0 | used as a **provider**, not as code |
| [LiteLLM](https://github.com/BerriAI/litellm) | MIT | confirmation that a fallback chain is the right primitive |

**No code was copied from any of them.** AGPL and GPL projects are listed because their *provider
lists* are facts, not expression — the endpoints, the key pages and which tiers are free are things
you would find by reading each provider's own docs. Everything here is written from scratch against
those facts. If you are reusing this, the same reasoning applies to you.

Deliberately **not** included: a web UI, a database, embeddings, sticky sessions, prompt compression,
a scheduler, cost accounting, or a plugin system. Each of those is real in the projects above and each
one is weight this does not need.

## Test

```bash
npm test
```

Starts a mock provider that fails on demand and drives the real server over HTTP: the happy path, a
429 falling through to the next model, a cooling model being skipped even though it would work, the
15→30 minute ladder, a success resetting the streak, and state surviving a restart. 17 checks.
