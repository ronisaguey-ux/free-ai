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

function loadModels() {
  if (cfg.models) return cfg.models;
  return JSON.parse(fs.readFileSync(path.join(ROOT, 'data', 'models.json'), 'utf8'));
}
const MODELS = loadModels();

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

function penalise(m, why) {
  const st = s(m);
  st.streak += 1;
  st.fails += 1;
  st.lastError = why;
  st.until = now() + Math.min(st.streak * cfg.backoffMin, MAX_BACKOFF_MIN) * 60000;
  saveState();
  console.error(`[free-ai] ${m.provider}/${m.model} -> cooling ${st.streak * cfg.backoffMin}m (streak ${st.streak}): ${why}`);
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
function upstream(m, body, key, signal) {
  const url = m.base.replace(/\/$/, '') + '/chat/completions';
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
    const t = setTimeout(() => ctl.abort(), Number(process.env.FREEAI_TIMEOUT_MS || 180000));
    const started = now();
    try {
      const res = await upstream(m, body, key, ctl.signal);
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
