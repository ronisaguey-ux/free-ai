#!/usr/bin/env node
/*
 * free-ai — one local endpoint in front of every free LLM tier.
 *
 *   POST /v1/chat/completions   OpenAI-compatible. Picks the best model that is not cooling.
 *   GET  /v1/models             The ranked list, with live health.
 *   GET  /status                Per-model cooldowns, failures and call counts.
 *   POST /admin/reset           Clear cooldowns (all, or {model:"..."}).
 *   GET  /health                Liveness.
 *
 * Zero dependencies on purpose. The whole value of this program is the model list and the
 * fallback rule; a framework around that is weight without function.
 *
 * THE FALLBACK RULE
 *   Serve strictly in rank order. The first model that answers wins.
 *   A model that fails is put on a LINEAR cooldown: 15 minutes for the first failure in a row,
 *   30 for the second, 45 for the third, 60, 75... A success resets the streak to zero.
 *   Cooldowns survive a restart, because a rate limit does not care that you rebooted.
 */

'use strict';

const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');

const ROOT = __dirname;
const PORT = Number(process.env.FREEAI_PORT || 8790);
const HOST = process.env.FREEAI_HOST || '127.0.0.1';

// ── the linear backoff ────────────────────────────────────────────────────────
// Failure N in a row costs N x 15 minutes. That is the whole policy.
const BACKOFF_STEP_MIN = Number(process.env.FREEAI_BACKOFF_MIN || 15);
const MAX_BACKOFF_MIN = Number(process.env.FREEAI_MAX_BACKOFF_MIN || 24 * 60);

// ── config ────────────────────────────────────────────────────────────────────
// Keys come from, in order: config.json, then the environment. Nothing else is needed to
// run: the no-key providers work with no configuration at all.
function loadConfig() {
  const file = process.env.FREEAI_CONFIG || path.join(ROOT, 'config.json');
  let cfg = {};
  try { cfg = JSON.parse(fs.readFileSync(file, 'utf8')); }
  catch (e) { if (e.code !== 'ENOENT') console.error(`[free-ai] config ${file}: ${e.message}`); }
  return {
    file,
    port: cfg.port || PORT,
    host: cfg.host || HOST,
    // { "groq": "gsk_...", "gemini": "AIza..." }  — or a list, which is rotated per call.
    keys: cfg.keys || {},
    // models: an explicit list to override data/models.json. See README "add your own".
    models: cfg.models || null,
    providers: cfg.providers || {},
    backoffMin: cfg.backoffMin ?? BACKOFF_STEP_MIN,
  };
}

function keyFor(provider) {
  const v = cfg.keys[provider];
  if (!v) return null;
  if (typeof v === 'string') return v;
  if (Array.isArray(v) && v.length) return v[rotate[provider] = ((rotate[provider] || 0) + 1) % v.length];
  return null;
}

const cfg = loadConfig();
const rotate = {};

/**
 * The chain, in three parts.
 *
 *   1. `data/models.json` - THE DEFAULT, every entry KEYLESS. Each was checked with a real
 *      completion before being added. This is what makes the router work with no
 *      configuration at all, and it is why no provider of ours is baked in: if it needs a
 *      key, it does not belong in the default chain.
 *   2. `cfg.providers` - providers the USER declared in their own config.json, with a key.
 *   3. `data/providers.json` - the catalogue, so adding an optional provider is one `keys`
 *      line instead of a hand-written model list. A catalogue entry joins the chain ONLY
 *      when its key is actually configured.
 */
function loadModels() {
  const keyless = JSON.parse(fs.readFileSync(path.join(ROOT, 'data', 'models.json'), 'utf8'));
  let catalogue = {};
  try { catalogue = JSON.parse(fs.readFileSync(path.join(ROOT, 'data', 'providers.json'), 'utf8')); }
  catch { catalogue = {}; }

  const out = [...keyless];
  const seen = new Set(out.map((m) => `${m.provider}\u0000${m.model}`));
  let rank = 1000;
  const add = (provider, base, models, extra = {}) => {
    for (const id of models || []) {
      const k = `${provider}\u0000${id}`;
      if (seen.has(k)) continue;
      seen.add(k);
      out.push({ rank: rank++, provider, model: id, base, key_env: `${String(provider).toUpperCase()}_API_KEY`, ...extra });
    }
  };

  for (const [pid, entry] of Object.entries(cfg.providers || {})) {
    if (!entry || typeof entry !== 'object') continue;
    if (!cfg.keys[pid] && !entry.no_key) continue;
    add(pid, entry.base, entry.models, entry.no_key ? { no_key: true } : {});
  }

  for (const [pid, entry] of Object.entries(catalogue)) {
    if (!entry || typeof entry !== 'object') continue;
    if (!(entry.models || []).length) continue;
    if (!cfg.keys[pid] && !entry.free_no_key) continue;
    add(pid, entry.base, entry.models, entry.free_no_key ? { no_key: true } : {});
  }

  return out;
}

// free-ai carries three `oculus_webchat` entries pointing at the engine's OWN deepseek
// gateways (127.0.0.1:8080/8081/8083). Measured under the engine's own load: every call to
// them died `fetch failed` / `timeout after 180s`, thousands in a row, while `curl
// /health` on those same gateways returned 200. The engine reached them directly without
// trouble - it was the router in the middle that could not.
//
// It is also redundant: the engine already holds `deepseek`, `deepseek2` and `deepseek4` as
// lanes of its own, so routing those same gateways through free-ai buys nothing and costs
// a full timeout budget per attempt (the failure path measured 300s per draw, which is
// what starved the pool).
//
// So they are dropped from the chain by default. FREEAI_WEBCHAT=1 puts them back for a
// caller that has no direct access to those gateways.
const _allModels = loadModels();
const MODELS = process.env.FREEAI_WEBCHAT === '1'
  ? _allModels
  : _allModels.filter((m) => m.provider !== 'oculus_webchat');

// ── state ─────────────────────────────────────────────────────────────────────
const STATE_FILE = process.env.FREEAI_STATE || path.join(ROOT, 'state.json');
const state = {};      // model -> { streak, until, lastError, lastOk, calls, ok, fails }

function loadState() {
  try {
    const raw = JSON.parse(fs.readFileSync(STATE_FILE, 'utf8'));
    for (const [k, v] of Object.entries(raw)) state[k] = v;
  } catch { /* first run */ }
}
let saveTimer = null;
function saveState() {
  if (saveTimer) return;
  saveTimer = setTimeout(() => {
    saveTimer = null;
    try { fs.writeFileSync(STATE_FILE + '.tmp', JSON.stringify(state)); fs.renameSync(STATE_FILE + '.tmp', STATE_FILE); }
    catch (e) { console.error('[free-ai] could not save state:', e.message); }
  }, 250);
}

const s = (m) => (state[m.model] ||= { streak: 0, until: 0, calls: 0, ok: 0, fails: 0 });
const now = () => Date.now();
const cooling = (m) => s(m).until > now();
const minsLeft = (m) => Math.max(0, Math.ceil((s(m).until - now()) / 60000));

/**
 * Price a failure before benching the model.
 *
 * 10-05 REDESIGN. The old rule was flat: every failure cooled the model for
 * `streak * 15` minutes, so a single transient hiccup benched an upstream for a quarter
 * of an hour. Measured under the engine's load: all 57 configured candidates ended up
 * cooling at once and the router answered
 *   http 503 {"message":"every model is cooling down","type":"no_model_available"}
 * while an idle probe of the same router answered 24 concurrent requests with 92% valid
 * edits. The models were fine - the bench was the outage.
 *
 * Classify, then price:
 *   TRANSIENT (timeout, 5xx, 408, connection reset, overloaded/empty)
 *       -> DO NOT bench. Count the failure and try the next candidate. A provider that is
 *          briefly overloaded is back in seconds; the chain exists to route around it.
 *   CAPACITY (429, quota, rate limit, capacity)
 *       -> a real "come back later": cool for BACKOFF_STEP_MIN x streak, capped. This is
 *          the only failure that deserves the old linear ladder.
 *   PERMANENT (401/402/403/404, model unavailable/retired, invalid key)
 *       -> park for the day. Retrying can never fix it.
 *
 * A success still resets the streak to zero.
 */
function penalise(m, why) {
  const st = s(m);
  st.streak += 1;
  st.fails += 1;
  st.lastError = why;
  const low = String(why).toLowerCase();
  const permanent = /\b(401|402|403|404)\b/.test(low)
    || /model[_ -]?(unavailable|not[_ -]?found|deprecat|retired|invalid)/.test(low)
    || /invalid api key|unauthorized|insufficient (balance|quota|credits)/.test(low);
  const capacity = /\b(429|408)\b/.test(low)
    || /rate[_ -]?limit|too frequent|quota|capacity/.test(low);
  let mins;
  if (permanent) {
    mins = MAX_BACKOFF_MIN;
  } else if (capacity) {
    mins = Math.min(st.streak * cfg.backoffMin, MAX_BACKOFF_MIN);
  } else {
    // TRANSIENT: no cooldown. Leave `until` untouched so the next request may pick it.
    st.until = 0;
    saveState();
    console.error(`[free-ai] ${m.provider}/${m.model} -> transient (streak ${st.streak}), NO cooldown: ${why}`);
    return;
  }
  st.until = now() + mins * 60000;
  saveState();
  console.error(`[free-ai] ${m.provider}/${m.model} -> cooling ${mins}m (streak ${st.streak}, `
    + `${permanent ? 'PERMANENT' : 'capacity'}): ${why}`);
}

function succeeded(m) {
  const st = s(m);
  st.calls += 1;
  st.ok += 1;
  st.streak = 0;
  st.until = 0;
  st.lastOk = new Date().toISOString();
  st.lastError = null;
  saveState();
}

/** In rank order, skipping anything on cooldown, then anything with no configured key. */
function chain({ includeCooling = false } = {}) {
  const list = [...MODELS].sort((a, b) => a.rank - b.rank);
  const ready = [], cold = [];
  for (const m of list) {
    const usable = m.no_key || keyFor(m.provider);
    if (!usable) continue;
    (cooling(m) ? cold : ready).push(m);
  }
  return includeCooling ? [...ready, ...cold] : ready;
}

// ── upstream call ─────────────────────────────────────────────────────────────
// Every model's `base` is data from data/models.json, which is tracked and therefore
// editable by a pull request. That value is concatenated into a URL that carries our
// API key, so a bad entry is a credential-leak and SSRF vector, not just a broken model.
// MEASURED 2026-10-05: no scheme or host validation existed at the call site - a base of
// file:///etc or http://169.254.169.254 would have been fetched, with the Bearer header
// attached. Validate before the fetch, and fail the model rather than the request.
function safeUpstreamUrl(base) {
  let u;
  try { u = new URL(String(base)); } catch { return null; }
  if (u.protocol !== 'http:' && u.protocol !== 'https:') return null;
  // Link-local / cloud-metadata / unspecified addresses are never a legitimate LLM
  // upstream and are the classic SSRF targets. Loopback IS legitimate here - the
  // webchat gateways we route to run on 127.0.0.1 - so it stays allowed.
  const h = u.hostname.replace(/^\[|\]$/g, '').toLowerCase();
  if (h === '169.254.169.254' || h === 'metadata.google.internal' || h === '0.0.0.0') return null;
  if (/^169\.254\./.test(h) || /^fe80:/i.test(h)) return null;
  return u.origin + u.pathname.replace(/\/$/, '') + '/chat/completions';
}

function _modelKeyOf(m, key) {
  return (m && m.anon_key) || key;
}

function upstream(m, body, key, signal) {
  const url = safeUpstreamUrl(m.base);
  if (!url) throw new Error(`refusing an unusable upstream base for ${m.provider}: ${String(m.base).slice(0, 40)}`);
  const payload = { ...body, model: m.model };
  return fetch(url, {
    method: 'POST',
    signal,
    headers: {
      'content-type': 'application/json',
      ...(key ? { authorization: `Bearer ${key}` } : {}),
      'user-agent': 'free-ai/1.0',
      ...(m.provider === 'open_router' ? { 'http-referer': 'https://github.com/ronisaguey-ux/free-ai' } : {}),
    },
    body: JSON.stringify(payload),
  });
}

/** A failure that means "this model is used up", not "your request was malformed". */
function isPenalty(status) {
  return status === 401 || status === 402 || status === 403 || status === 404 || status === 408 || status === 429 || status >= 500;
}

/**
 * A 400 is normally the caller's fault and must NOT be hidden behind a fallback. But providers
 * also use 400 for "this model is not available", which is a per-model condition and exactly
 * what the chain exists to route around. Measured live: llm7 answers 400 model_unavailable and
 * the chain stopped dead. The distinction has to be made on the body, because the status code
 * alone cannot tell the two apart.
 */
function isModelUnavailable(status, text) {
  if (status !== 400 && status !== 422) return false;
  return /model[_ -]?(unavailable|not[_ -]?found|deprecat|retired|invalid)/i.test(text)
      || /unknown model|unsupported model|model[^"]{0,40}not[^"]{0,20}(available|supported|exist)/i.test(text);
}

// ── request handling ──────────────────────────────────────────────────────────
async function chat(body, cfgRes) {
  const wantStream = !!body.stream;
  const tried = [];
  const candidates = chain();

  if (!candidates.length) {
    const soonest = [...MODELS].sort((a, b) => s(a).until - s(b).until)[0];
    return {
      status: 503,
      body: {
        error: {
          message: 'every model is cooling down',
          type: 'no_model_available',
          candidates_configured: MODELS.length,
          next_available: soonest ? { model: soonest.model, in_minutes: minsLeft(soonest) } : null,
          hint: 'configure more providers in config.json, or POST /admin/reset',
        },
      },
    };
  }

  for (const m of candidates) {
    const key = keyFor(m.provider);
    const ctl = new AbortController();
    // 10-05: a per-model timeout, because one number cannot be right for every upstream.
    // The local webchat gateways legitimately need up to 400s (they wait on a real
    // browser), while a remote API answers in seconds. One flat 180s meant the
    // oculus_webchat entries timed out on EVERY call - measured 4 in a row at exactly
    // 180.001s - which benched nothing (they are transient now) but wasted a full 3
    // minutes of the router's time on each attempt and made the router report
    // "every model is cooling down" to callers.
    // A model may carry its own `timeout_ms`; else the provider prefix decides.
    // 10-05: the REMOTE default is 45s, not 180s. Measured under the engine's load: the
    // router's own median round trip is 3.3s, but 34 draws in 45 minutes spent over 250s
    // and 66 spent over 100s - all of it waiting on a candidate that had already gone
    // quiet. A free endpoint that has not answered in 45s is not slow, it is gone, and the
    // chain should move to the next candidate while there is still budget left to use it.
    // The local webchat gateways keep their 420s, because there the browser really is
    // thinking and a long wait is the correct behaviour.
    const _toMs = Number(m.timeout_ms
      || (m.provider === 'oculus_webchat' ? 420000 : 0)
      || process.env.FREEAI_TIMEOUT_MS || 45000);
    const t = setTimeout(() => ctl.abort(), _toMs);
    const started = now();
    try {
      const res = await upstream(m, body, _modelKeyOf(m, key), ctl.signal);
      clearTimeout(t);

      if (res.ok) {
        succeeded(m);
        cfgRes.setHeader('x-free-ai-model', `${m.provider}/${m.model}`);
        cfgRes.setHeader('x-free-ai-attempts', String(tried.length + 1));
        cfgRes.setHeader('content-type', res.headers.get('content-type') || 'application/json');
        if (wantStream && res.body) {
          const reader = res.body.getReader();
          try {
            for (;;) {
              const { done, value } = await reader.read();
              if (done) break;
              cfgRes.write(Buffer.from(value));
            }
          } catch { /* client went away */ }
          cfgRes.end();
          return { status: 200, streamed: true };
        }
        return { status: 200, body: await res.json() };
      }

      const text = await res.text().catch(() => '');
      const why = `http ${res.status}: ${text.slice(0, 160)}`;
      if (isPenalty(res.status) || isModelUnavailable(res.status, text)) {
        penalise(m, why);
        tried.push(`${m.provider}/${m.model} ${why}`);
        continue;
      }
      // Not a quota problem: our request is wrong. Returning it is more useful than hiding it
      // behind a fallback that will fail identically.
      return { status: res.status, body: { error: { message: why, type: 'upstream_rejected', model: m.model } } };
    } catch (e) {
      clearTimeout(t);
      const why = e.name === 'AbortError' ? `timeout after ${(now() - started) / 1000}s` : (e.message || String(e));
      penalise(m, why);
      tried.push(`${m.provider}/${m.model} ${why}`);
    }
  }

  return {
    status: 503,
    body: { error: { message: 'every configured model failed', type: 'all_models_failed', tried } },
  };
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let n = 0;
    const chunks = [];
    req.on('data', (c) => {
      n += c.length;
      if (n > 32 * 1024 * 1024) { reject(new Error('request too large')); req.destroy(); return; }
      chunks.push(c);
    });
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

function send(res, status, obj) {
  const body = JSON.stringify(obj);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(body) });
  res.end(body);
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);
  const p = url.pathname.replace(/\/+$/, '') || '/';

  if (p === '/health') return send(res, 200, { ok: true, models: MODELS.length, ready: chain().length, version: '1.0.0' });

  if (p === '/v1/models') {
    const list = [...MODELS].sort((a, b) => a.rank - b.rank).map((m) => ({
      id: m.model, object: 'model', owned_by: m.provider, x_free_ai: {
        rank: m.rank, provider: m.provider, ready: !cooling(m) && !!(m.no_key || keyFor(m.provider)),
        cooldown_minutes: minsLeft(m), failures_in_a_row: s(m).streak, calls: s(m).calls, ok: s(m).ok,
      },
    }));
    return send(res, 200, { object: 'list', data: list });
  }

  if (p === '/status') {
    return send(res, 200, {
      config_file: cfg.file,
      backoff: `linear, ${cfg.backoffMin} min per consecutive failure, max ${MAX_BACKOFF_MIN} min`,
      ready: chain().length,
      total: MODELS.length,
      models: [...MODELS].sort((a, b) => a.rank - b.rank).map((m) => ({
        rank: m.rank, model: m.model, provider: m.provider,
        key: m.no_key ? 'none needed' : (keyFor(m.provider) ? 'set' : 'MISSING'),
        cooling_minutes: minsLeft(m), streak: s(m).streak, calls: s(m).calls, ok: s(m).ok,
        last_error: s(m).lastError || null,
      })),
    });
  }

  if (p === '/admin/reset' && req.method === 'POST') {
    const raw = await readBody(req).catch(() => '{}');
    let body = {};
    try { body = JSON.parse(raw || '{}'); } catch { /* reset all */ }
    const which = body.model || null;
    // expire:true clears the cooldown but KEEPS the failure streak, so the next failure is
    // punished at the next step of the ladder. That is how "let it try again now" is
    // expressed, and it is the only way to reach streak 2 without waiting 15 real minutes.
    const expireOnly = body.expire === true;
    let n = 0;
    for (const m of MODELS) {
      if (which && m.model !== which) continue;
      const cur = s(m);
      state[m.model] = expireOnly
        ? { ...cur, until: 0 }
        : { ...cur, streak: 0, until: 0, lastError: null };
      n++;
    }
    saveState();
    return send(res, 200, { reset: n, mode: expireOnly ? 'expired cooldown, kept streak' : 'full reset' });
  }

  if (p === '/v1/chat/completions' && req.method === 'POST') {
    let body;
    try { body = JSON.parse(await readBody(req)); }
    catch (e) { return send(res, 400, { error: { message: `invalid JSON: ${e.message}` } }); }
    try {
      const out = await chat(body, res);
      if (out.streamed) return;
      return send(res, out.status, out.body);
    } catch (e) {
      if (!res.headersSent) return send(res, 500, { error: { message: e.message } });
      return res.end();
    }
  }

  send(res, 404, { error: { message: `no route ${p}`, routes: ['POST /v1/chat/completions', 'GET /v1/models', 'GET /status', 'POST /admin/reset', 'GET /health'] } });
});

loadState();
server.listen(cfg.port, cfg.host, () => {
  const ready = chain();
  console.log(`free-ai on http://${cfg.host}:${cfg.port}`);
  console.log(`  ${MODELS.length} models across ${new Set(MODELS.map((m) => m.provider)).size} providers, ${ready.length} ready`);
  console.log(`  backoff: ${cfg.backoffMin} min x consecutive failures (linear)`);
  if (!cfg.keys || !Object.keys(cfg.keys).length) {
    console.log(`  no keys configured — ${MODELS.filter((m) => m.no_key).length} no-key models will still serve`);
  }
});
