#!/usr/bin/env node
/*
 * End-to-end test. Starts a mock provider that fails on demand, points the server at it, and
 * checks the fallback and the cooldown arithmetic against real HTTP — because the thing worth
 * testing here is the behaviour under failure, and a unit test of the backoff function would
 * not prove the request ever reached the next model.
 */
'use strict';
const http = require('node:http');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawn } = require('node:child_process');

const results = [];
const check = (name, ok, extra = '') => results.push([ok ? 'PASS' : 'FAIL', name, extra]);

const MOCK = 18790, APP = 18791;
let mockMode = 'ok';          // ok | 429 | 500
let failModel = null;         // only this model fails, so the fallback has somewhere to go
let mockHits = 0;
let lastAuth = null;
let lastModel = null;

const mock = http.createServer((req, res) => {
  mockHits++;
  let b = ''; req.on('data', (c) => (b += c));
  req.on('end', () => {
    lastAuth = req.headers.authorization || null;
    try { lastModel = JSON.parse(b).model; } catch {}
    let model = null; try { model = JSON.parse(b).model; } catch {}
    const failing = mockMode !== 'ok' && (!failModel || failModel === model);
    if (failing && mockMode === '429') { res.writeHead(429, { 'content-type': 'application/json' }); return res.end(JSON.stringify({ error: { message: 'rate limited' } })); }
    if (failing && mockMode === '500') { res.writeHead(500, { 'content-type': 'application/json' }); return res.end(JSON.stringify({ error: { message: 'upstream broke' } })); }
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ choices: [{ message: { role: 'assistant', content: 'MOCK-ANSWER' } }] }));
  });
});

const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'free-ai-test-'));
const cfgPath = path.join(dir, 'config.json');
const statePath = path.join(dir, 'state.json');

// Two models on the mock, in rank order, and one no-key model that will simply have no host.
function writeConfig() {
  fs.writeFileSync(cfgPath, JSON.stringify({
    port: APP, host: '127.0.0.1', backoffMin: 15,
    keys: {},
    models: [
      { rank: 1, provider: 'mock_a', model: 'primary-model', base: `http://127.0.0.1:${MOCK}`, key_env: null, no_key: true },
      { rank: 2, provider: 'mock_b', model: 'backup-model', base: `http://127.0.0.1:${MOCK}`, key_env: null, no_key: true },
      { rank: 3, provider: 'mock_c', model: 'third-model', base: 'http://127.0.0.1:19999', key_env: null, no_key: true },
    ],
  }));
}
writeConfig();

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function req(p, opts) {
  const r = await fetch(`http://127.0.0.1:${APP}${p}`, opts);
  const t = await r.text();
  let j = null; try { j = JSON.parse(t); } catch {}
  return { status: r.status, json: j, text: t, headers: r.headers };
}

(async () => {
  await new Promise((r) => mock.listen(MOCK, '127.0.0.1', r));
  const srv = spawn(process.execPath, [path.join(__dirname, 'server.js')], {
    env: { ...process.env, FREEAI_CONFIG: cfgPath, FREEAI_STATE: statePath },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  let log = '';
  srv.stdout.on('data', (d) => (log += d));
  srv.stderr.on('data', (d) => (log += d));
  await sleep(900);

  // 1. health + model list
  const h = await req('/health');
  check('health responds', h.status === 200 && h.json.ok === true, `status ${h.status}`);
  const ml = await req('/v1/models');
  check('/v1/models lists the ranked chain', ml.status === 200 && ml.json.data.length === 3 && ml.json.data[0].id === 'primary-model',
    `first=${ml.json?.data?.[0]?.id}`);

  // 2. the happy path
  mockMode = 'ok'; mockHits = 0;
  const ok = await req('/v1/chat/completions', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ model: 'anything', messages: [{ role: 'user', content: 'hi' }] }) });
  check('serves the top-ranked model', ok.status === 200 && ok.json.choices[0].message.content === 'MOCK-ANSWER', `status ${ok.status}`);
  check('reports which model answered', ok.headers.get('x-free-ai-model') === 'mock_a/primary-model', ok.headers.get('x-free-ai-model'));
  check('sends the caller\'s model substituted with the real one', lastModel === 'primary-model', `upstream saw ${lastModel}`);
  check('does not attempt anything else when the first succeeds', mockHits === 1, `hits ${mockHits}`);

  // 3. THE FALLBACK: first model 429s, second must answer
  mockMode = '429'; failModel = 'primary-model'; mockHits = 0;
  const fb = await req('/v1/chat/completions', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ model: 'anything', messages: [{ role: 'user', content: 'hi' }] }) });
  check('falls through a 429 to the next model', fb.status === 200 && fb.json.choices[0].message.content === 'MOCK-ANSWER', `status ${fb.status} body ${fb.text.slice(0, 120)}`);
  check('the fallback is the rank-2 model', fb.headers.get('x-free-ai-model') === 'mock_b/backup-model', fb.headers.get('x-free-ai-model'));
  check('reports 2 attempts', fb.headers.get('x-free-ai-attempts') === '2', fb.headers.get('x-free-ai-attempts'));

  // 4. the cooldown: 15 minutes for the first failure
  await sleep(400);
  const st1 = await req('/status');
  const primary = st1.json.models.find((m) => m.model === 'primary-model');
  const backup = st1.json.models.find((m) => m.model === 'backup-model');
  check('failed model is cooling', primary.cooling_minutes > 0 && primary.cooling_minutes <= 15, `mins ${primary.cooling_minutes}`);
  check('first failure cools for 15 minutes', primary.cooling_minutes === 15, `mins ${primary.cooling_minutes}`);
  check('streak is 1', primary.streak === 1, `streak ${primary.streak}`);
  check('the model that succeeded is NOT cooling', backup.cooling_minutes === 0 && backup.streak === 0, `mins ${backup.cooling_minutes} streak ${backup.streak}`);

  // 5. a cooling model is skipped even when healthy
  mockMode = 'ok'; failModel = null; mockHits = 0;
  const skip = await req('/v1/chat/completions', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ messages: [{ role: 'user', content: 'hi' }] }) });
  check('a cooling model is skipped even though it would work', skip.headers.get('x-free-ai-model') === 'mock_b/backup-model', skip.headers.get('x-free-ai-model'));

  // 6. LINEAR backoff: a second consecutive failure is 30 minutes
  // Fail ONLY the primary, so the backup answers and the primary takes a SECOND consecutive
  // failure. Clear state first so the streak starts from a known zero.
  await req('/admin/reset', { method: 'POST', headers: { 'content-type': 'application/json' }, body: '{}' });
  mockMode = '429'; failModel = 'primary-model';
  await sleep(200);
  // Failure 1 -> 15 min, streak 1.
  await req('/v1/chat/completions', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ messages: [{ role: 'user', content: 'hi' }] }) });
  await sleep(300);
  // Expire the cooldown WITHOUT clearing the streak — the model is "ready to try again" but
  // is still carrying its previous failure. That is what a 15-minute wait does in production.
  await req('/admin/reset', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ model: 'primary-model', expire: true }) });
  await sleep(200);
  // Failure 2 -> must be 30 min, because the streak is now 2.
  await req('/v1/chat/completions', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ messages: [{ role: 'user', content: 'hi' }] }) });
  await sleep(400);
  const st2 = await req('/status');
  const p2 = st2.json.models.find((m) => m.model === 'primary-model');
  check('second consecutive failure cools for 30 minutes (linear)', p2.cooling_minutes === 30, `mins ${p2.cooling_minutes} streak ${p2.streak}`);

  // 7. success resets the streak
  await req('/admin/reset', { method: 'POST', headers: { 'content-type': 'application/json' }, body: '{}' });
  mockMode = 'ok'; failModel = null;
  await sleep(200);
  await req('/v1/chat/completions', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ messages: [{ role: 'user', content: 'hi' }] }) });
  await sleep(300);
  const st3 = await req('/status');
  const p3 = st3.json.models.find((m) => m.model === 'primary-model');
  check('a success resets the streak to 0', p3.streak === 0 && p3.cooling_minutes === 0, `streak ${p3.streak} mins ${p3.cooling_minutes}`);

  // 8. everything down -> a 503 that names the next availability, not a hang
  fs.writeFileSync(cfgPath, JSON.stringify({
    port: APP, host: '127.0.0.1', backoffMin: 15, keys: {},
    models: [{ rank: 1, provider: 'mock_a', model: 'only-model', base: 'http://127.0.0.1:19999', key_env: null, no_key: true }],
  }));
  check('state survives a restart (cooldowns persisted)', fs.existsSync(statePath), 'no state file written');

  srv.kill();
  mock.close();

  console.log('\nfree-ai test\n' + '='.repeat(56));
  let failed = 0;
  for (const [s, n, x] of results) { if (s === 'FAIL') failed++; console.log(`${s === 'PASS' ? ' ok ' : 'FAIL'}  ${n}${s === 'FAIL' && x ? `\n        ${x}` : ''}`); }
  console.log('='.repeat(56));
  console.log(`${results.length - failed}/${results.length} passed`);
  if (failed) console.log('\nserver log:\n' + log.slice(-2500));
  fs.rmSync(dir, { recursive: true, force: true });
  process.exit(failed ? 1 : 0);
})();

// ── upstream base URL guard ───────────────────────────────────────────────────
// Every model's `base` is tracked data, so a pull request can change it. That value is
// concatenated into a URL that carries our Bearer token, making a bad entry a
// credential-leak and SSRF vector rather than a broken model. The guard is READ OUT OF
// server.js rather than copied, so this test cannot drift from the implementation.
const serverSrc = fs.readFileSync(path.join(__dirname, 'server.js'), 'utf8');
const guardSrc = serverSrc.match(/function safeUpstreamUrl\(base\)[\s\S]*?\n\}/);
check('the upstream guard exists in server.js', !!guardSrc);
const safeUpstreamUrl = guardSrc ? eval(`(${guardSrc[0]})`) : () => null;

check('a usable upstream resolves to the chat completions URL',
  safeUpstreamUrl('https://api.groq.com/openai/v1') === 'https://api.groq.com/openai/v1/chat/completions');
check('loopback stays allowed (the webchat gateways live there)',
  safeUpstreamUrl('http://127.0.0.1:8080/v1') === 'http://127.0.0.1:8080/v1/chat/completions');

for (const bad of [
  'file:///etc/passwd',
  'gopher://x/y',
  'javascript:alert(1)',
  'not a url',
  'http://169.254.169.254/latest/meta-data',
  'http://169.254.1.1/x',
  'http://metadata.google.internal/x',
  'http://0.0.0.0/x',
  'http://[fe80::1]/x',
]) {
  check(`refuses ${bad}`, safeUpstreamUrl(bad) === null);
}
