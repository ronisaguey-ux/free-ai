#!/usr/bin/env node
'use strict';
//
// mcp-server.js - free-ai exposed as an MCP server.
//
// WHY THIS EXISTS. free-ai is already an OpenAI-shaped API, but an API is something an
// agent has to be TAUGHT: which port, which endpoint, what body. An MCP server turns
// the whole thing into tools the agent already knows how to call, so any MCP-capable
// client (opencode, Claude Code, Codex, ...) can put work through the free chain, run
// SUBAGENTS on it, fan a question out across several free models at once, and compress
// a prompt before sending it.
//
// SEPARATE PROCESS from the router, on purpose. It speaks MCP on stdio and reaches the
// router over HTTP, so it can report whether the router is up instead of assuming it,
// several agents can attach at once, and a crash here cannot take the lane down.
//
// ZERO DEPENDENCIES, like the rest of this repo: JSON-RPC 2.0 over newline-delimited
// stdio, Node stdlib only. No SDK, no npm install.
//

const http = require('node:http');
const https = require('node:https');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawn } = require('node:child_process');
const { URL } = require('node:url');
const { compressMessages, compressText } = require('./compressor');

// -- STDOUT IS THE WIRE. NOTHING ELSE MAY WRITE TO IT. ------------------------
//
// Any module loaded below may console.log, and a stray line between two JSON-RPC
// frames kills the connection with a parse error at the client - from a WARNING, the
// least likely thing to be suspected. So the redirect happens BEFORE anything else
// runs, and it covers every console method. Diagnostics still reach the operator's log
// because stderr is passed straight through.
// Only when THIS file is the server. As a library, hijacking the global console would
// silently swallow every console.log in the requiring process (a test's own output, for
// one) — the wire only needs protecting when we actually own the stdio channel.
if (require.main === module) {
  for (const level of ['log', 'info', 'warn', 'error', 'debug', 'trace']) {
    console[level] = (...args) => {
      try {
        process.stderr.write(args.map((x) => (typeof x === 'string' ? x : safeJson(x))).join(' ') + '\n');
      } catch { /* stderr gone */ }
    };
  }
}

function safeJson(x) {
  try { return JSON.stringify(x); } catch { return String(x); }
}

const FREEAI_URL = process.env.FREEAI_URL || 'http://127.0.0.1:8790';
const DEFAULT_ROOT = process.env.FREEAI_MCP_ROOT || process.cwd();
// Swapped around a subagent call and restored afterwards, so two concurrent subagents
// asking for different roots cannot read each other's tree.
let REPO_ROOT = DEFAULT_ROOT;

// -- the router client -------------------------------------------------------

function request(method, urlStr, body, timeoutMs) {
  return new Promise((resolve, reject) => {
    let u;
    try { u = new URL(urlStr); } catch (e) { return reject(new Error(`bad url ${urlStr}`)); }
    const lib = u.protocol === 'https:' ? https : http;
    const payload = body === undefined ? null : Buffer.from(JSON.stringify(body));
    const req = lib.request({
      method,
      hostname: u.hostname,
      port: u.port || (u.protocol === 'https:' ? 443 : 80),
      path: u.pathname + (u.search || ''),
      headers: payload
        ? { 'content-type': 'application/json', 'content-length': payload.length }
        : {},
    }, (res) => {
      const chunks = [];
      res.on('data', (c) => chunks.push(c));
      res.on('end', () => {
        const text = Buffer.concat(chunks).toString('utf8');
        let json = null;
        try { json = text ? JSON.parse(text) : null; } catch { /* non-JSON body */ }
        resolve({ status: res.statusCode, headers: res.headers, text, json });
      });
    });
    req.on('error', reject);
    if (timeoutMs) {
      req.setTimeout(timeoutMs, () => { req.destroy(new Error(`request timed out after ${timeoutMs}ms`)); });
    }
    if (payload) req.write(payload);
    req.end();
  });
}

/**
 * Call the router. `model` is passed straight through, so a caller can pin one model
 * from /v1/models or leave it as 'freeai' to let the ranked chain decide.
 */
async function chat(messages, opts = {}) {
  const {
    model = 'freeai', maxTokens = 2048, temperature, timeoutMs = 240000, extra = {},
  } = opts;
  const body = { model, messages, max_tokens: maxTokens, ...extra };
  if (typeof temperature === 'number') body.temperature = temperature;
  const r = await request('POST', `${FREEAI_URL}/v1/chat/completions`, body, timeoutMs);
  if (r.status !== 200) {
    const msg = (r.json && r.json.error && r.json.error.message) || r.text || `HTTP ${r.status}`;
    const err = new Error(msg);
    err.status = r.status;
    err.tried = r.json && r.json.error && r.json.error.tried;
    err.hint = r.json && r.json.error && r.json.error.hint;
    throw err;
  }
  const choice = (r.json.choices || [])[0] || {};
  return {
    text: (choice.message && choice.message.content) || '',
    model_used: r.headers['x-free-ai-model'] || null,
    attempts: Number(r.headers['x-free-ai-attempts'] || 1),
    compression: r.headers['x-free-ai-compression'] || null,
    finish_reason: choice.finish_reason || null,
    usage: r.json.usage || null,
  };
}

// -- subagent ----------------------------------------------------------------
//
// A subagent here is a REAL loop: the model proposes a tool call, the tool runs, the
// result goes back, repeat, until it answers or runs out of steps.
//
// The tool protocol is INSTRUCTED, not native function-calling, because the chain is
// free models of every vintage and native tool-calling is exactly the feature they
// disagree about. A model that can emit JSON can use these tools.
//
// SAFETY: read-only by default. write_file and run_bash exist but are REFUSED unless
// the caller opts in per call, so an agent cannot hand a free model the ability to
// change the machine by accident.

const READ_ONLY_TOOLS = ['read_file', 'list_dir', 'grep'];
const WRITE_TOOLS = ['write_file', 'run_bash'];

function withinRoot(p) {
  const abs = path.resolve(REPO_ROOT, p);
  const root = path.resolve(REPO_ROOT);
  return abs === root || abs.startsWith(root + path.sep);
}

function resolveIn(p) {
  if (typeof p !== 'string' || !p.trim()) throw new Error('path is required');
  if (path.isAbsolute(p)) {
    if (!withinRoot(p)) throw new Error(`refusing ${p}: outside ${REPO_ROOT}`);
    return p;
  }
  const abs = path.resolve(REPO_ROOT, p);
  if (!withinRoot(abs)) throw new Error(`refusing ${p}: resolves outside ${REPO_ROOT}`);
  return abs;
}

function toolReadFile(args) {
  const abs = resolveIn(args.path);
  const max = Math.max(200, Math.min(Number(args.max_chars) || 20000, 200000));
  const text = fs.readFileSync(abs, 'utf8');
  // `offset` is a 1-based LINE number, which is how a caller reasons about a file it has
  // already grepped. Without it a large file could only ever be read from the top, so a
  // model looking for something at line 1800 re-read 1800 lines it had already seen.
  const lines = text.split('\n');
  const from = Math.max(1, Math.min(Number(args.offset) || 1, Math.max(1, lines.length)));
  const window = lines.slice(from - 1).join('\n');
  const cut = window.slice(0, max);
  const shownLines = cut.split('\n').length;
  const more = from - 1 + shownLines < lines.length;
  return {
    path: path.relative(REPO_ROOT, abs),
    total_lines: lines.length,
    from_line: from,
    to_line: Math.min(lines.length, from + shownLines - 1),
    truncated: more,
    hint: more ? `more below — call read_file again with offset=${from + shownLines}` : 'end of file',
    content: cut,
  };
}

function toolListDir(args) {
  const abs = resolveIn(args.path || '.');
  const entries = fs.readdirSync(abs, { withFileTypes: true })
    .slice(0, 400)
    .map((d) => (d.isDirectory() ? `${d.name}/` : d.name));
  return { path: path.relative(REPO_ROOT, abs) || '.', entries };
}

function toolGrep(args) {
  const needle = String(args.pattern || '');
  if (!needle) throw new Error('pattern is required');
  const root = resolveIn(args.path || '.');
  const re = new RegExp(needle, args.ignore_case ? 'i' : '');
  const limit = Math.max(1, Math.min(Number(args.limit) || 60, 500));
  const hits = [];
  const skip = new Set(['node_modules', '.git', '.venv', '__pycache__', 'dist', 'build']);
  // A FILE path is a legitimate grep target and used to raise ENOTDIR, which reads to a
  // model as a broken tool rather than a normal search. Search just that file.
  try {
    if (fs.statSync(root).isFile()) {
      fs.readFileSync(root, 'utf8').split('\n').forEach((line, i) => {
        if (hits.length < limit && re.test(line)) {
          hits.push({ file: path.relative(REPO_ROOT, root), line: i + 1, text: line.slice(0, 200) });
        }
      });
      return { pattern: needle, hits, truncated: hits.length >= limit };
    }
  } catch { /* not a file; fall through to the directory walk */ }
  const walk = (dir, depth) => {
    if (hits.length >= limit || depth > 8) return;
    for (const d of fs.readdirSync(dir, { withFileTypes: true })) {
      if (hits.length >= limit) return;
      if (skip.has(d.name)) continue;
      const full = path.join(dir, d.name);
      if (d.isDirectory()) { walk(full, depth + 1); continue; }
      let text;
      try {
        if (fs.statSync(full).size > 2_000_000) continue;
        text = fs.readFileSync(full, 'utf8');
      } catch { continue; }
      text.split('\n').forEach((line, i) => {
        if (hits.length < limit && re.test(line)) {
          hits.push({ file: path.relative(REPO_ROOT, full), line: i + 1, text: line.slice(0, 200) });
        }
      });
    }
  };
  walk(root, 0);
  return { pattern: needle, hits, truncated: hits.length >= limit };
}

function toolWriteFile(args) {
  const abs = resolveIn(args.path);
  fs.mkdirSync(path.dirname(abs), { recursive: true });
  fs.writeFileSync(abs, String(args.content ?? ''), 'utf8');
  return { wrote: path.relative(REPO_ROOT, abs), bytes: Buffer.byteLength(String(args.content ?? '')) };
}

function toolEditFile(args) {
  const abs = resolveIn(args.path);
  const find = String(args.find ?? '');
  const replace = String(args.replace ?? '');
  if (!find) throw new Error('find is required');
  const text = fs.readFileSync(abs, 'utf8');
  const count = text.split(find).length - 1;
  if (count === 0) {
    throw new Error('find text is not present in ' + path.relative(REPO_ROOT, abs)
      + ' — read the file and copy the exact text (whitespace matters)');
  }
  if (count > 1 && !args.replace_all) {
    throw new Error(`find text appears ${count} times — pass replace_all:true or include more context to make it unique`);
  }
  const out = args.replace_all ? text.split(find).join(replace) : text.replace(find, replace);
  fs.writeFileSync(abs, out, 'utf8');
  return { edited: path.relative(REPO_ROOT, abs), replacements: args.replace_all ? count : 1, bytes: Buffer.byteLength(out) };
}

async function toolRunBash(args) {
  const cmd = String(args.command || '');
  if (!cmd.trim()) throw new Error('command is required');
  const { spawn } = require('node:child_process');
  const timeout = Math.max(1000, Math.min(Number(args.timeout_ms) || 30000, 600000));
  return await new Promise((resolve) => {
    const child = spawn('/bin/bash', ['-lc', cmd], { cwd: REPO_ROOT, timeout });
    let out = '';
    let err = '';
    child.stdout.on('data', (d) => { out += d.toString(); });
    child.stderr.on('data', (d) => { err += d.toString(); });
    child.on('close', (code) => resolve({
      command: cmd, exit_code: code, stdout: out.slice(0, 20000), stderr: err.slice(0, 8000),
    }));
    child.on('error', (e) => resolve({ command: cmd, exit_code: -1, error: String(e.message) }));
  });
}

async function runTool(name, args, allow = {}) {
  if (name === 'write_file' && !allow.write) {
    throw new Error('write_file is refused: this subagent is read-only. Re-run with allow_write to permit it.');
  }
  if (name === 'run_bash' && !allow.bash) {
    throw new Error('run_bash is refused: this subagent is read-only. Re-run with allow_bash to permit it.');
  }
  switch (name) {
    case 'read_file': return toolReadFile(args);
    case 'list_dir': return toolListDir(args);
    case 'grep': return toolGrep(args);
    case 'write_file': return toolWriteFile(args);
    case 'edit_file': return toolEditFile(args);
    case 'run_bash': return toolRunBash(args);
    default: throw new Error(`unknown tool ${name}`);
  }
}

function toolDocs(allowWrite, allowBash) {
  const lines = [
    'read_file  {"path": "...", "offset": 1, "max_chars": 20000}   read a file; offset is a 1-based LINE number',
    'list_dir   {"path": "..."}                       list a directory',
    'grep       {"pattern": "regex", "path": ".", "ignore_case": false, "limit": 60}   search files',
  ];
  if (allowWrite) {
    lines.push('write_file {"path": "...", "content": "..."}             create or overwrite a file');
    lines.push('edit_file  {"path": "...", "find": "exact text", "replace": "new text"}   surgical change to an existing file');
  }
  if (allowBash) lines.push('run_bash   {"command": "...", "timeout_ms": 30000}        run a shell command');
  return lines.join('\n');
}

/** Every balanced {...} block in the text, so a tool call after prose or inside a
 *  fence is still found — the old parser only looked at the FIRST block from the first
 *  '{', which a stray brace in narration could throw off. */
function balancedObjects(body) {
  const out = [];
  let i = 0;
  while (i < body.length) {
    if (body[i] !== '{') { i++; continue; }
    let depth = 0, inStr = false, esc = false, closed = false;
    for (let j = i; j < body.length; j++) {
      const ch = body[j];
      if (inStr) {
        if (esc) esc = false;
        else if (ch === '\\') esc = true;
        else if (ch === '"') inStr = false;
        continue;
      }
      if (ch === '"') { inStr = true; continue; }
      if (ch === '{') depth++;
      else if (ch === '}') {
        depth--;
        if (depth === 0) { out.push(body.slice(i, j + 1)); i = j + 1; closed = true; break; }
      }
    }
    if (!closed) i++; // unbalanced from here on; step past it rather than loop
  }
  return out;
}

function coerceArgs(v) {
  if (v && typeof v === 'object') return v;
  if (typeof v === 'string') {
    const o = tryParseJson(v);
    return (o && typeof o === 'object') ? o : {};
  }
  return {};
}

/** Repair backslashes that are not valid JSON escapes.
 *
   *  A model writing a shell command such as grep 'a\|b' emits `\|`, which JSON rejects
   *  ("invalid escape"), so the whole tool call failed to parse and was read as a final
   *  answer — ending the loop after a step. Walk the string and double any backslash that
   *  does not begin a legal escape, leaving real escapes (\n, \", \\, \uXXXX ...) intact. */
function repairJsonEscapes(s) {
  let out = '';
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (ch !== '\\') { out += ch; continue; }
    const nxt = s[i + 1];
    if (nxt === undefined) { out += '\\\\'; continue; }
    if (nxt === 'u' && /^[0-9a-fA-F]{4}$/.test(s.slice(i + 2, i + 6))) {
      out += s.slice(i, i + 6); i += 5; continue;
    }
    if ('"\\/bfnrt'.includes(nxt)) { out += ch + nxt; i += 1; continue; }
    out += '\\\\' + nxt; i += 1; // stray backslash: escape it
  }
  return out;
}

function tryParseJson(c) {
  try { return JSON.parse(c); } catch { /* try repaired below */ }
  try { return JSON.parse(repairJsonEscapes(c)); } catch { return undefined; }
}

/** Recognise a tool call in any of the shapes free models emit. Returns null when the
 *  object is not a tool call. */
function toolFromObject(obj) {
  if (!obj || typeof obj !== 'object') return null;
  // {"tool":"name","args|parameters|params":{...}}
  if (typeof obj.tool === 'string' && !Array.isArray(obj.tool_calls)) {
    return { kind: 'tool', tool: obj.tool, args: obj.args || obj.parameters || obj.params || {} };
  }
  // OpenAI shape: {"tool_calls":[{"function":{"name":..,"arguments":..}}]}
  if (Array.isArray(obj.tool_calls) && obj.tool_calls.length) {
    const fn = obj.tool_calls[0] && obj.tool_calls[0].function;
    if (fn && typeof fn.name === 'string') return { kind: 'tool', tool: fn.name, args: coerceArgs(fn.arguments) };
  }
  // A bare function call — what the hosted models actually emit:
  //   {"name":"read_file","parameters":{"path":"a.js"}}
  if (typeof obj.name === 'string') {
    const a = obj.parameters !== undefined ? obj.parameters
      : obj.arguments !== undefined ? obj.arguments
      : obj.args !== undefined ? obj.args : undefined;
    if (a !== undefined) return { kind: 'tool', tool: obj.name, args: coerceArgs(a) };
  }
  // {"function":{"name":..,"arguments":..}} with no wrapper.
  if (obj.function && typeof obj.function.name === 'string') {
    return { kind: 'tool', tool: obj.function.name, args: coerceArgs(obj.function.arguments) };
  }
  return null;
}

/** Pull the FIRST tool call out of a reply, or null if the reply is a final answer. */
function parseAgentReply(text) {
  const t = String(text || '');
  const fence = t.match(/```(?:json)?\s*([\s\S]*?)```/);
  // Try the fenced body first (that is where a well-behaved model puts its call), then
  // the whole reply — a model that narrates and then calls a tool must still be read.
  const bodies = fence ? [fence[1], t] : [t];
  for (const body of bodies) {
    for (const c of balancedObjects(body)) {
      const obj = tryParseJson(c);
      if (obj === undefined) continue;
      const call = toolFromObject(obj);
      if (call) return call;
      if (obj && typeof obj.final === 'string') return { kind: 'final', text: obj.final, explicit: true };
      if (obj && typeof obj.answer === 'string') return { kind: 'final', text: obj.answer, explicit: true };
    }
    const mk = parseMarkupToolCall(body);
    if (mk) return mk;
  }
  // Not an explicit final. Callers that drive tools treat this as UNPARSEABLE rather than
  // a completion, so a garbled tool attempt is not mistaken for a finished answer.
  return { kind: 'final', text: t, explicit: false };
}

/** Free models frequently answer with `<function=list_dir><parameter=path>.</parameter></function>`
 *  instead of the JSON the prompt asks for. Reading that as a final answer ends the loop after one
 *  step, so accept the two common markup shapes too. */
function parseMarkupToolCall(body) {
  const fn = body.match(/<function\s*[=:]\s*["']?([A-Za-z_][\w.-]*)/) ||
             body.match(/<function\s+name\s*=\s*["']([A-Za-z_][\w.-]*)["']/);
  if (!fn) return null;
  const args = {};
  const paramRe = /<parameter\s*[=:]\s*["']?([A-Za-z_][\w.-]*)["']?\s*>([\s\S]*?)<\/parameter>/g;
  let m;
  while ((m = paramRe.exec(body))) {
    const raw = m[2].trim();
    args[m[1]] = /^-?\d+$/.test(raw) ? Number(raw)
      : /^(true|false)$/i.test(raw) ? raw.toLowerCase() === 'true'
      : raw;
  }
  return { kind: 'tool', tool: fn[1], args };
}

async function runSubagent(prompt, opts = {}) {
  const {
    maxSteps = 24, model = 'freeai', allowWrite = false, allowBash = false,
    timeoutMs = 240000, temperature, system,
  } = opts;

  // Progress must be observable. A subagent can run for many minutes; without a signal
  // the caller cannot tell "working" from "wedged", and a slow run looks identical to a
  // hang. Steps go to stderr (stdout is the MCP protocol and must stay clean) and, when
  // a path is given, to that file — so a detached run can be watched with `tail -f`.
  const logFile = opts.logFile || process.env.FREEAI_SUBAGENT_LOG || '';
  const onStep = typeof opts.onStep === 'function' ? opts.onStep : null;
  const emit = (line) => {
    const text = `[subagent ${new Date().toISOString()}] ${line}`;
    try { process.stderr.write(text + '\n'); } catch { /* stderr closed */ }
    if (logFile) { try { fs.appendFileSync(logFile, text + '\n'); } catch { /* ignore */ } }
    if (onStep) { try { onStep(line); } catch { /* caller callback must not break the run */ } }
  };

  const sys = system || [
    'You are a subagent with tools. Work step by step until the task is done.',
    '',
    'RULES — follow these exactly:',
    '1. Act; do not narrate. Reply with a tool call, not a description of one.',
    '2. Read before you write, but read EFFICIENTLY: use grep to find the exact line range,',
    '   then read_file with offset to read only that range. Never read a large file from the top twice.',
    '3. Prefer edit_file for a change to an existing file; use write_file to create a new one.',
    '4. Target the file named in the task. Do not grep the whole repo for something you were told.',
    '5. Do not repeat a tool call you already made with the same arguments — you already have its result.',
    '6. When the task is done (and, if it said to, the tests pass), stop and answer.',
    '',
    'To use a tool, reply with ONLY this JSON and nothing else:',
    '{"tool":"<name>","args":{...}}',
    'A bare {"name":"...","parameters":{...}} is also accepted.',
    '',
    'Available tools:',
    toolDocs(allowWrite, allowBash),
    '',
    'When you are finished, reply with ONLY:',
    '{"final":"<what you changed, file:line, and the exact verification you ran and its result>"}',
    '',
    'Never invent a tool result. If the task cannot be done, say so plainly in the final.',
  ].join('\n');

  const messages = [
    { role: 'system', content: sys },
    { role: 'user', content: String(prompt) },
  ];

  const trace = [];
  let finalNudges = 0;         // bounded re-asks when a reply is neither a tool call nor a final
  const seenCalls = new Map(); // "tool\u0000args" -> count, to catch a model spinning
  const _prevRoot = REPO_ROOT;
  if (opts.root) REPO_ROOT = path.resolve(String(opts.root));
  emit(`start model=${model} max_steps=${maxSteps} write=${allowWrite} bash=${allowBash} root=${REPO_ROOT}`);
  try {
    for (let step = 1; step <= maxSteps; step++) {
      const r = await chat(messages, { model, timeoutMs, temperature, extra: { messages_meta: { step } } });
      const parsed = parseAgentReply(r.text);

      if (parsed.kind === 'final') {
        // A reply that did no work and is not an explicit {"final":...} is almost always a
        // garbled tool call (measured: the model emitted <dots_function_call> markup that
        // does not parse). Accepting it ends the run having changed nothing, so nudge for a
        // real tool call first. Bounded to 2, so a genuine no-tool answer still gets through.
        const didWork = trace.length > 0;
        if (!parsed.explicit && !didWork && finalNudges < 2) {
          finalNudges += 1;
          emit(`step ${step}: reply was neither a tool call nor a final — nudge ${finalNudges}/2`);
          messages.push({ role: 'assistant', content: r.text });
          messages.push({
            role: 'user',
            content: 'That was not a usable action and the task is NOT complete. Reply with ONLY one tool call: '
              + '{"tool":"<name>","args":{...}}  — valid names are read_file, list_dir, grep'
              + (allowWrite ? ', write_file, edit_file' : '') + (allowBash ? ', run_bash' : '') + '.',
          });
          continue;
        }
        emit(`step ${step}: FINAL (${String(parsed.text || '').length} chars) via ${r.model_used}`);
        return {
          answer: parsed.text,
          steps: step,
          model_used: r.model_used,
          compression: r.compression,
          trace,
          stopped: 'final',
        };
      }

      // A reply that is neither a tool call nor a final is a format miss. Nudge once in
      // the exact shape; without this the loop silently repeats the same broken reply.
      if (parsed.kind !== 'tool' || !parsed.tool) {
        emit(`step ${step}: unparseable reply — nudging for a tool call or a final`);
        messages.push({ role: 'assistant', content: r.text });
        messages.push({
          role: 'user',
          content: 'That reply was not a tool call and not a final. Reply with ONLY '
            + '{"tool":"<name>","args":{...}} or {"final":"..."}.',
        });
        continue;
      }

      const sig = parsed.tool + '\u0000' + safeJson(parsed.args);
      const times = (seenCalls.get(sig) || 0) + 1;
      seenCalls.set(sig, times);

      let result;
      try {
        result = await runTool(parsed.tool, parsed.args, { write: allowWrite, bash: allowBash });
      } catch (e) {
        result = { error: String(e.message || e) };
      }
      const err = result && result.error ? ' ERROR: ' + result.error : '';
      emit(`step ${step}: ${parsed.tool}${err}${times > 1 ? ` (repeat x${times})` : ''}`);
      trace.push({ step, tool: parsed.tool, args: parsed.args, result_summary: summarise(result) });

      messages.push({ role: 'assistant', content: r.text });

      // The same call with the same arguments a third time means the model is stuck, not
      // progressing; the result is already in the transcript. Say so, and require a
      // different action or a final — otherwise it burns the whole step budget spinning.
      if (times >= 3) {
        messages.push({
          role: 'user',
          content: `TOOL RESULT for ${parsed.tool}:\n${JSON.stringify(result).slice(0, 20000)}\n\n`
            + 'You have now called this exact tool with these exact arguments ' + times
            + ' times. The result will not change. Take a DIFFERENT action, or reply now with {"final":"..."}.',
        });
        continue;
      }

      messages.push({
        role: 'user',
        content: `TOOL RESULT for ${parsed.tool}:\n${JSON.stringify(result).slice(0, 20000)}\n\n`
          + 'Continue. Reply with the next tool call, or {"final":"..."} when done.',
      });
    }

    // Out of steps: ask once for a plain answer so the caller gets something usable.
    emit(`out of steps (${maxSteps}) — demanding a final answer`);
    messages.push({ role: 'user', content: 'You are out of tool steps. Reply now with {"final":"..."} summarising what you found.' });
    const last = await chat(messages, { model, timeoutMs, temperature });
    return {
      answer: parseAgentReply(last.text).text || last.text,
      steps: maxSteps,
      model_used: last.model_used,
      compression: last.compression,
      trace,
      stopped: 'max_steps',
    };
  } finally {
    REPO_ROOT = _prevRoot;
  }
}

// ── background subagents ─────────────────────────────────────────────────────
//
// freeai_subagent blocks until the run finishes, so a caller that wants to do anything
// else — or simply not hold a tool call open for twenty minutes — had to detach the
// process itself and then poll files by hand. That is not a capability an MCP should
// push onto its caller. These tools run a subagent in its own process and expose it as
// a job: spawn returns an id, and status/log/result/list read it back. A job outlives
// the calling tool call and the MCP process itself.
const JOBS_DIR = process.env.FREEAI_JOBS_DIR
  || path.join(os.homedir(), '.cache', 'free-ai', 'subagents');

const RUNNER_SRC = `'use strict';
const fs = require('fs');
const payload = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const { runSubagent } = require(payload.module);
const { meta, result } = payload.paths;
const writeMeta = (patch) => {
  let m = {};
  try { m = JSON.parse(fs.readFileSync(meta, 'utf8')); } catch { /* first write */ }
  fs.writeFileSync(meta, JSON.stringify({ ...m, ...patch }, null, 2));
};
// stdio (below) already points at the log file, so passing logFile here too would
// write every progress line twice. Let the redirect be the single sink.
runSubagent(payload.prompt, { ...payload.opts })
  .then((r) => {
    fs.writeFileSync(result, JSON.stringify(r, null, 2));
    writeMeta({ state: 'done', finished: Date.now(), steps: r.steps, model_used: r.model_used, stopped: r.stopped, answer: r.answer });
  })
  .catch((e) => {
    const msg = String((e && e.stack) || e);
    fs.writeFileSync(result, JSON.stringify({ error: msg }, null, 2));
    writeMeta({ state: 'error', finished: Date.now(), error: msg });
    process.exit(1);
  });
`;

function _jobsRoot() {
  fs.mkdirSync(JOBS_DIR, { recursive: true });
  return JOBS_DIR;
}

function _jobDir(id) {
  // Reject anything that is not a plain job id, so a caller cannot walk the tree.
  if (!/^sub_[A-Za-z0-9_]+$/.test(String(id || ''))) throw new Error('invalid job id');
  return path.join(_jobsRoot(), String(id));
}

function _readJson(f, fallback = null) {
  try { return JSON.parse(fs.readFileSync(f, 'utf8')); } catch { return fallback; }
}

/** Start a subagent in its own detached process. Returns immediately with a job id. */
function spawnSubagent(prompt, opts = {}) {
  if (!String(prompt || '').trim()) throw new Error('prompt is required');
  const id = 'sub_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 8);
  const dir = path.join(_jobsRoot(), id);
  fs.mkdirSync(dir, { recursive: true });
  const paths = { meta: path.join(dir, 'job.json'), log: path.join(dir, 'log.txt'), result: path.join(dir, 'result.json') };
  const payloadFile = path.join(dir, 'payload.json');
  const runnerFile = path.join(dir, 'runner.js');

  fs.writeFileSync(payloadFile, JSON.stringify({ prompt: String(prompt), opts, module: __filename, paths }, null, 2));
  fs.writeFileSync(runnerFile, RUNNER_SRC);
  fs.writeFileSync(paths.meta, JSON.stringify({
    id, state: 'running', started: Date.now(), root: opts.root || REPO_ROOT,
    max_steps: opts.maxSteps ?? 24, model: opts.model || 'freeai',
    allow_write: !!opts.allowWrite, allow_bash: !!opts.allowBash,
    prompt: String(prompt).slice(0, 500), pid: null,
  }, null, 2));

  const out = fs.openSync(paths.log, 'a');
  const child = spawn(process.execPath, [runnerFile, payloadFile], {
    detached: true,
    stdio: ['ignore', out, out],
    env: { ...process.env, FREEAI_SUBAGENT_JOB: id },
    cwd: opts.root || REPO_ROOT,
  });
  child.unref();
  fs.closeSync(out);
  const meta = _readJson(paths.meta, {});
  fs.writeFileSync(paths.meta, JSON.stringify({ ...meta, pid: child.pid }, null, 2));
  return { job_id: id, state: 'running', pid: child.pid, log_file: paths.log, result_file: paths.result };
}

function subagentStatus(id) {
  const dir = _jobDir(id);
  const meta = _readJson(path.join(dir, 'job.json'));
  if (!meta) throw new Error(`no such job ${id}`);
  // A pid that is gone with no finished timestamp means the process died without
  // writing a result (killed, OOM). Say so rather than reporting "running" forever.
  if (meta.state === 'running' && meta.pid) {
    let alive = true;
    try { process.kill(meta.pid, 0); } catch { alive = false; }
    if (!alive) meta.state = 'lost';
  }
  const hasResult = fs.existsSync(path.join(dir, 'result.json'));
  return { job_id: id, state: meta.state, started: meta.started, finished: meta.finished || null,
    steps: meta.steps ?? null, model_used: meta.model_used || null, stopped: meta.stopped || null,
    pid: meta.pid || null, result_ready: hasResult, log_file: path.join(dir, 'log.txt'),
    result_file: path.join(dir, 'result.json') };
}

function subagentLog(id, lines = 40) {
  const dir = _jobDir(id);
  const log = path.join(dir, 'log.txt');
  if (!fs.existsSync(log)) return { job_id: id, lines: [], note: 'no log yet' };
  const all = fs.readFileSync(log, 'utf8').split('\n').filter(Boolean);
  const n = Math.max(1, Math.min(Number(lines) || 40, 2000));
  return { job_id: id, total_lines: all.length, lines: all.slice(-n) };
}

function subagentResult(id) {
  const dir = _jobDir(id);
  const meta = _readJson(path.join(dir, 'job.json')) || {};
  const result = _readJson(path.join(dir, 'result.json'));
  if (!result) return { job_id: id, state: meta.state || 'unknown', ready: false, hint: 'still running — poll freeai_subagent_status' };
  if (result.error) return { job_id: id, state: 'error', ready: true, error: result.error };
  return { job_id: id, state: 'done', ready: true, answer: result.answer, steps: result.steps,
    model_used: result.model_used, stopped: result.stopped, trace: result.trace || [] };
}

function subagentList() {
  const root = _jobsRoot();
  const jobs = [];
  for (const name of fs.readdirSync(root)) {
    if (!name.startsWith('sub_')) continue;
    const meta = _readJson(path.join(root, name, 'job.json'));
    if (!meta) continue;
    jobs.push({ job_id: name, state: meta.state, started: meta.started,
      root: meta.root, has_result: fs.existsSync(path.join(root, name, 'result.json')) });
  }
  jobs.sort((a, b) => (b.started || 0) - (a.started || 0));
  return { jobs, jobs_dir: root };
}

function subagentKill(id) {
  const meta = _readJson(path.join(_jobDir(id), 'job.json'));
  if (!meta) throw new Error(`no such job ${id}`);
  if (!meta.pid) return { job_id: id, killed: false, reason: 'no pid recorded' };
  try { process.kill(meta.pid, 'SIGKILL'); }
  catch (e) { return { job_id: id, killed: false, reason: String(e.message) }; }
  const mpath = path.join(_jobDir(id), 'job.json');
  fs.writeFileSync(mpath, JSON.stringify({ ...meta, state: 'killed', finished: Date.now() }, null, 2));
  return { job_id: id, killed: true };
}

function summarise(v) {
  const s = safeJson(v);
  return s.length > 300 ? s.slice(0, 300) + '...' : s;
}

// -- MCP plumbing ------------------------------------------------------------

// -- the tool catalogue ------------------------------------------------------
//
// Declared as a compact spec and expanded below, because 35 hand-written JSON Schema
// literals drift from their handlers. One line per tool, `args` is a name -> type map,
// `req` lists the required ones.
const TOOL_SPEC = [
  // chain / routing
  ['freeai_ask', 'Send one prompt through the free chain and get the answer back.', { prompt: 'string', system: 'string', model: 'string', max_tokens: 'number', temperature: 'number' }, ['prompt']],
  ['freeai_chat', 'Multi-turn chat: pass the full message list and get the next reply.', { messages: 'array', model: 'string', max_tokens: 'number', temperature: 'number' }, ['messages']],
  ['freeai_models', 'The ranked free-model chain with live health, cooldowns and per-model stats.', {}, []],
  ['freeai_route', 'Which model the chain would serve NEXT, without sending anything.', {}, []],
  ['freeai_health', 'Router liveness, how many models are ready, and compression savings since boot.', {}, []],
  ['freeai_pick', 'Pick the best live model for a stated need (e.g. "code", "long context", "fast").', { need: 'string' }, ['need']],
  ['freeai_probe', 'Probe ONE model with a tiny prompt: does it answer, how fast, what does it return.', { model: 'string', prompt: 'string', timeout_ms: 'number' }, ['model']],
  ['freeai_reset', 'Clear every cooldown so the whole chain is eligible again.', {}, []],
  ['freeai_cooldowns', 'List exactly what is cooling right now, and for how many minutes.', {}, []],
  ['freeai_stats', 'Per-model call / ok / fail counters and failure streaks.', {}, []],
  ['freeai_config', 'The router config it is actually running with, keys redacted.', {}, []],
  ['freeai_log_tail', 'Recent lines from the router log.', { lines: 'number' }, []],

  // compression
  ['freeai_compress', 'Compress a prompt and report the saving, WITHOUT sending it.', { text: 'string', level: ['off', 'safe', 'balanced'] }, ['text']],
  ['freeai_compress_compare', 'Run the same text through every compression level and show all three.', { text: 'string' }, ['text']],
  ['freeai_compression_stats', 'What prompt compression has saved since the router started.', {}, []],
  ['freeai_set_compression', 'Change the router compression level at runtime.', { level: ['off', 'safe', 'balanced'] }, ['level']],

  // subagents and fan-out
  ['freeai_subagent', 'Run a multi-step SUBAGENT on a free model that reads/edits files and runs commands, then answers. Read-only unless allow_write/allow_bash. Pass log_file to watch its steps live with tail -f.', { prompt: 'string', root: 'string', max_steps: 'number', model: 'string', allow_write: 'boolean', allow_bash: 'boolean', log_file: 'string' }, ['prompt']],
  ['freeai_subagent_spawn', 'Start a subagent in the BACKGROUND and return a job id immediately. Poll it with freeai_subagent_status / _log / _result. This is the way to run a long task without holding a tool call open.', { prompt: 'string', root: 'string', max_steps: 'number', model: 'string', allow_write: 'boolean', allow_bash: 'boolean' }, ['prompt']],
  ['freeai_subagent_status', 'State of a background subagent: running / done / error / killed / lost, plus steps and where its log and result are.', { job_id: 'string' }, ['job_id']],
  ['freeai_subagent_log', 'The last N progress lines of a background subagent (its step-by-step trace).', { job_id: 'string', lines: 'number' }, ['job_id']],
  ['freeai_subagent_result', 'The answer (and trace) of a background subagent. Reports not-ready while it still runs.', { job_id: 'string' }, ['job_id']],
  ['freeai_subagent_list', 'Every background subagent this router has spawned, newest first.', {}, []],
  ['freeai_subagent_kill', 'Stop a running background subagent.', { job_id: 'string' }, ['job_id']],
  ['freeai_swarm', 'Run several subagents IN PARALLEL on the free chain and collect every answer.', { prompts: 'array', max_steps: 'number', max_parallel: 'number', allow_write: 'boolean', allow_bash: 'boolean' }, ['prompts']],
  ['freeai_compare', 'Send the SAME prompt to several named models and return the answers side by side.', { prompt: 'string', models: 'array', max_tokens: 'number' }, ['prompt', 'models']],
  ['freeai_batch', 'Send many INDEPENDENT prompts (no tools) and collect the answers in parallel.', { prompts: 'array', model: 'string', max_tokens: 'number', max_parallel: 'number' }, ['prompts']],
  ['freeai_debate', 'Two models argue a question for N rounds, then a third judges which was stronger.', { question: 'string', rounds: 'number', judge_model: 'string' }, ['question']],
  ['freeai_review', 'One model drafts, a second reviews and scores it. Returns both.', { task: 'string', draft_model: 'string', review_model: 'string' }, ['task']],

  // repo and code work
  ['freeai_explain_file', 'Subagent reads a file and explains what it does, its risks and its interfaces.', { path: 'string', root: 'string' }, ['path']],
  ['freeai_review_diff', 'Review a git diff for bugs, security issues and missing tests.', { diff: 'string', repo: 'string', max_tokens: 'number' }, ['diff']],
  ['freeai_summarise_dir', 'Summarise what a directory contains and what each file is for.', { path: 'string', root: 'string', max_files: 'number' }, []],
  ['freeai_generate_tests', 'Read a source file and propose unit tests for it. Returns test code, does not write it.', { path: 'string', root: 'string', framework: 'string' }, ['path']],
  ['freeai_docs', 'Generate reference documentation for a file or a named symbol.', { path: 'string', symbol: 'string', root: 'string' }, ['path']],
  ['freeai_find', 'Grep for a pattern, then have a model explain what the hits mean.', { pattern: 'string', path: 'string', root: 'string', limit: 'number' }, ['pattern']],

  // structured output
  ['freeai_json', 'Force a JSON answer matching a schema you supply. Returns parsed JSON.', { prompt: 'string', schema: 'object', model: 'string' }, ['prompt', 'schema']],
  ['freeai_classify', 'Classify text into one of the labels you give.', { text: 'string', labels: 'array', model: 'string' }, ['text', 'labels']],
  ['freeai_extract', 'Extract named fields from unstructured text.', { text: 'string', fields: 'array', model: 'string' }, ['text', 'fields']],
  ['freeai_translate', 'Translate text into a target language.', { text: 'string', language: 'string', model: 'string' }, ['text', 'language']],

  // files and search, directly (no model in the loop)
  ['freeai_read_file', 'Read a file directly, sandboxed to the configured root. No model call.', { path: 'string', max_chars: 'number' }, ['path']],
  ['freeai_grep', 'Search files directly with a regex. No model call.', { pattern: 'string', path: 'string', ignore_case: 'boolean', limit: 'number' }, ['pattern']],
  ['freeai_list_dir', 'List a directory directly. No model call.', { path: 'string' }, []],

  // ops
  ['freeai_estimate', 'Rough token and character estimate for a prompt, before you send it.', { text: 'string' }, ['text']],
  ['freeai_state_export', 'Dump the router state file (cooldowns and counters).', {}, []],
  ['freeai_selftest', 'Check the router, the compressor and the tool loop all work end to end.', {}, []],
];

function schemaFor(argsSpec = {}) {
  const properties = {};
  for (const [k, t] of Object.entries(argsSpec)) {
    if (Array.isArray(t)) properties[k] = { type: 'string', enum: t };
    else if (t === 'array') properties[k] = { type: 'array', items: { type: 'string' } };
    else if (t === 'object') properties[k] = { type: 'object' };
    else properties[k] = { type: t };
  }
  return properties;
}

const TOOLS = TOOL_SPEC.map(([name, description, argsSpec, required]) => ({
  name,
  description,
  inputSchema: {
    type: 'object',
    properties: schemaFor(argsSpec),
    ...(required && required.length ? { required } : {}),
  },
}));

// -- helpers -----------------------------------------------------------------

/** Router REST helpers, so every tool reads the same way. */
async function getJson(pathname, timeoutMs = 15000) {
  const r = await request('GET', `${FREEAI_URL}${pathname}`, undefined, timeoutMs);
  if (r.status !== 200) throw new Error(`router ${pathname} -> HTTP ${r.status}`);
  return r.json;
}

async function postJson(pathname, body, timeoutMs = 20000) {
  const r = await request('POST', `${FREEAI_URL}${pathname}`, body || {}, timeoutMs);
  if (r.status !== 200) {
    const m = (r.json && r.json.error && r.json.error.message) || r.text || `HTTP ${r.status}`;
    throw new Error(`router ${pathname} -> ${m}`);
  }
  return r.json;
}

async function askText(prompt, opts = {}) {
  const messages = [];
  if (opts.system) messages.push({ role: 'system', content: String(opts.system) });
  messages.push({ role: 'user', content: String(prompt) });
  const r = await chat(messages, opts);
  return r.text;
}

/** Rough token estimate. Deliberately crude and LABELLED as an estimate: free models
 *  have different tokenizers, so a precise number would be a false precision. */
function estimateTokens(text) {
  const chars = String(text || '').length;
  return { chars, tokens_approx: Math.ceil(chars / 4) };
}

/** First balanced {...} or [...] in a string. Free models wrap JSON in prose. */
function firstJson(text) {
  const t = String(text || '');
  const fence = t.match(/```(?:json)?\s*([\s\S]*?)```/);
  const body = fence ? fence[1] : t;
  for (const [open, close] of [['{', '}'], ['[', ']']]) {
    const start = body.indexOf(open);
    if (start === -1) continue;
    let depth = 0;
    let inStr = false;
    let esc = false;
    for (let i = start; i < body.length; i++) {
      const c = body[i];
      if (esc) { esc = false; continue; }
      if (c === '\\') { esc = true; continue; }
      if (c === '"') { inStr = !inStr; continue; }
      if (inStr) continue;
      if (c === open) depth += 1;
      else if (c === close) {
        depth -= 1;
        if (depth === 0) {
          try { return JSON.parse(body.slice(start, i + 1)); } catch { return null; }
        }
      }
    }
  }
  return null;
}

/** Ask for JSON and give the model ONE corrective retry, because free models wrap
 *  their JSON in prose often enough that a single miss is not a real failure. */
async function askJson(prompt, opts = {}) {
  const first = await askText(prompt, opts);
  let parsed = firstJson(first);
  if (parsed !== null) return { json: parsed, raw: first, retried: false };
  const second = await askText(
    `${prompt}\n\nYour previous reply was not parseable JSON. Reply with ONLY the JSON value, no prose, no code fence.`,
    opts,
  );
  parsed = firstJson(second);
  return { json: parsed, raw: second, retried: true, first_attempt: first };
}

function readStateFile() {
  const f = process.env.FREEAI_STATE || path.join(__dirname, 'state.json');
  try { return JSON.parse(fs.readFileSync(f, 'utf8')); } catch (e) { return { error: `cannot read ${f}: ${e.message}` }; }
}

function logTail(lines) {
  const f = process.env.FREEAI_LOG || path.join(__dirname, 'freeai.log');
  try {
    const all = fs.readFileSync(f, 'utf8').split('\n').filter(Boolean);
    return { file: f, lines: all.slice(-Math.max(1, Math.min(Number(lines) || 40, 1000))) };
  } catch (e) {
    return { file: f, error: String(e.message) };
  }
}

// -- tool handlers -----------------------------------------------------------

const HANDLERS = {
  // ── chain / routing ───────────────────────────────────────────────────────
  freeai_ask: async (a) => {
    const r = await chat(
      [...(a.system ? [{ role: 'system', content: String(a.system) }] : []),
        { role: 'user', content: String(a.prompt) }],
      { model: a.model || 'freeai', maxTokens: a.max_tokens || 2048, temperature: a.temperature },
    );
    return { answer: r.text, model_used: r.model_used, attempts: r.attempts, compression: r.compression };
  },

  freeai_chat: async (a) => {
    const msgs = Array.isArray(a.messages) ? a.messages : [];
    if (!msgs.length) throw new Error('messages must be a non-empty array');
    const r = await chat(msgs, { model: a.model || 'freeai', maxTokens: a.max_tokens || 2048, temperature: a.temperature });
    return { reply: r.text, model_used: r.model_used, compression: r.compression };
  },

  freeai_models: async () => getJson('/v1/models'),

  freeai_route: async () => {
    const d = await getJson('/v1/models');
    const ready = (d.data || []).filter((m) => m.x_free_ai && m.x_free_ai.ready)
      .sort((a, b) => a.x_free_ai.rank - b.x_free_ai.rank);
    const first = ready[0];
    return {
      next_model: first ? first.id : null,
      provider: first ? first.x_free_ai.provider : null,
      ready_count: ready.length,
      total: (d.data || []).length,
      next_few: ready.slice(0, 5).map((m) => `${m.id} (rank ${m.x_free_ai.rank})`),
      note: first ? null : 'nothing is ready — every model is cooling or unkeyed',
    };
  },

  freeai_health: async () => getJson('/health'),

  freeai_pick: async (a) => {
    const need = String(a.need || '').toLowerCase();
    const d = await getJson('/v1/models');
    const ready = (d.data || []).filter((m) => m.x_free_ai && m.x_free_ai.ready);
    if (!ready.length) return { picked: null, reason: 'nothing is ready', candidates: [] };
    // Score by the words the caller used. Deliberately simple and explainable: a
    // route an operator cannot reason about is worse than a plain rank order.
    const scored = ready.map((m) => {
      const id = m.id.toLowerCase();
      let score = 1000 - m.x_free_ai.rank;             // rank is the baseline
      if (need.includes('code') && /code|coder|deepseek|qwen|laguna/.test(id)) score += 500;
      if (need.includes('reason') && /reason|think|nemotron|n2\.5/.test(id)) score += 400;
      if (need.includes('fast') && /mini|small|flash|lightning|nano|2\.6b/.test(id)) score += 400;
      if (need.includes('long') && /nemotron-3-ultra|super|pro|120b/.test(id)) score += 300;
      if (need.includes('cheap') || need.includes('free')) score += 100;
      return { model: m.id, provider: m.x_free_ai.provider, rank: m.x_free_ai.rank, score };
    }).sort((x, y) => y.score - x.score);
    return { need: a.need, picked: scored[0].model, candidates: scored.slice(0, 5) };
  },

  freeai_probe: async (a) => {
    const model = String(a.model || '').trim();
    if (!model) throw new Error('model is required');
    const started = Date.now();
    try {
      const r = await chat(
        [{ role: 'user', content: a.prompt || 'Reply with exactly: PONG' }],
        { model, maxTokens: 64, timeoutMs: a.timeout_ms || 60000 },
      );
      return {
        model, ok: true, ms: Date.now() - started,
        reply: r.text.slice(0, 500), served_by: r.model_used,
      };
    } catch (e) {
      return { model, ok: false, ms: Date.now() - started, error: String(e.message || e) };
    }
  },

  freeai_reset: async () => postJson('/admin/reset', {}),

  freeai_cooldowns: async () => {
    const d = await getJson('/v1/models');
    const cooling = (d.data || [])
      .filter((m) => m.x_free_ai && m.x_free_ai.cooldown_minutes > 0)
      .map((m) => ({ model: m.id, provider: m.x_free_ai.provider, minutes_left: m.x_free_ai.cooldown_minutes, streak: m.x_free_ai.failures_in_a_row }))
      .sort((x, y) => y.minutes_left - x.minutes_left);
    return { cooling_count: cooling.length, total: (d.data || []).length, cooling };
  },

  freeai_stats: async () => {
    const d = await getJson('/v1/models');
    const rows = (d.data || []).map((m) => ({
      model: m.id, provider: m.x_free_ai.provider, calls: m.x_free_ai.calls,
      ok: m.x_free_ai.ok, fails: m.x_free_ai.calls - m.x_free_ai.ok,
      streak: m.x_free_ai.failures_in_a_row, ready: m.x_free_ai.ready,
    }));
    const tot = rows.reduce((acc, r) => ({ calls: acc.calls + r.calls, ok: acc.ok + r.ok }), { calls: 0, ok: 0 });
    return {
      totals: { ...tot, yield_pct: tot.calls ? Math.round((tot.ok / tot.calls) * 1000) / 10 : 0 },
      models: rows.sort((a, b) => b.calls - a.calls),
    };
  },

  freeai_config: async (a) => {
    const st = await getJson('/status');
    return st;                                   // provider keys are reported as set/MISSING, never printed
  },

  freeai_log_tail: async (a) => logTail(a.lines),

  // ── compression ───────────────────────────────────────────────────────────
  freeai_compress: async (a) => {
    const level = a.level || 'safe';
    const r = compressMessages([{ role: 'user', content: String(a.text) }], level);
    return {
      level, chars_before: r.stats.before, chars_after: r.stats.after,
      chars_saved: r.stats.saved, percent_saved: r.stats.pct,
      compressed: r.messages[0].content,
    };
  },

  freeai_compress_compare: async (a) => {
    const text = String(a.text);
    return {
      chars_before: text.length,
      levels: ['off', 'safe', 'balanced'].map((level) => {
        const r = compressMessages([{ role: 'user', content: text }], level);
        return { level, chars_after: r.stats.after, percent_saved: r.stats.pct };
      }),
    };
  },

  freeai_compression_stats: async () => {
    const h = await getJson('/health');
    return h.compression || { level: 'off', note: 'the running router does not report an x-free-ai-compression block' };
  },

  freeai_set_compression: async (a) => postJson('/admin/compression', { level: String(a.level || '') }),

  // ── subagents and fan-out ─────────────────────────────────────────────────
  freeai_subagent_spawn: async (a) => spawnSubagent(String(a.prompt), {
    root: a.root, allowWrite: !!a.allow_write, allowBash: !!a.allow_bash,
    model: a.model, maxSteps: a.max_steps,
  }),
  freeai_subagent_status: async (a) => subagentStatus(String(a.job_id)),
  freeai_subagent_log: async (a) => subagentLog(String(a.job_id), a.lines),
  freeai_subagent_result: async (a) => subagentResult(String(a.job_id)),
  freeai_subagent_list: async () => subagentList(),
  freeai_subagent_kill: async (a) => subagentKill(String(a.job_id)),

  freeai_subagent: async (a) => runSubagent(String(a.prompt), {
    maxSteps: a.max_steps || 8, model: a.model || 'freeai',
    root: a.root, allowWrite: !!a.allow_write, allowBash: !!a.allow_bash,
    model: a.model, logFile: a.log_file,
  }),

  freeai_swarm: async (a) => {
    const prompts = Array.isArray(a.prompts) ? a.prompts : [];
    if (!prompts.length) throw new Error('prompts must be a non-empty array');
    const parallel = Math.max(1, Math.min(Number(a.max_parallel) || 4, 12));
    const results = new Array(prompts.length);
    let next = 0;
    const worker = async () => {
      for (;;) {
        const i = next++;
        if (i >= prompts.length) return;
        try {
          results[i] = await runSubagent(String(prompts[i]), {
            maxSteps: a.max_steps || 4,
            allowWrite: !!a.allow_write, allowBash: !!a.allow_bash,
          });
        } catch (e) {
          results[i] = { answer: null, error: String(e.message || e) };
        }
      }
    };
    await Promise.all(Array.from({ length: Math.min(parallel, prompts.length) }, worker));
    return { count: prompts.length, results };
  },

  freeai_compare: async (a) => {
    const models = Array.isArray(a.models) ? a.models : [];
    if (!models.length) throw new Error('models must be a non-empty array');
    const results = await Promise.all(models.map(async (m) => {
      try {
        const r = await chat([{ role: 'user', content: String(a.prompt) }], { model: m, maxTokens: a.max_tokens || 2048 });
        return { model: m, ok: true, answer: r.text, served_by: r.model_used };
      } catch (e) {
        return { model: m, ok: false, error: String(e.message || e) };
      }
    }));
    return { prompt: a.prompt, results };
  },

  freeai_batch: async (a) => {
    const prompts = Array.isArray(a.prompts) ? a.prompts : [];
    if (!prompts.length) throw new Error('prompts must be a non-empty array');
    const parallel = Math.max(1, Math.min(Number(a.max_parallel) || 4, 12));
    const out = new Array(prompts.length);
    let next = 0;
    const worker = async () => {
      for (;;) {
        const i = next++;
        if (i >= prompts.length) return;
        try {
          const r = await chat([{ role: 'user', content: String(prompts[i]) }], {
            model: a.model || 'freeai', maxTokens: a.max_tokens || 1024,
          });
          out[i] = { ok: true, answer: r.text, model_used: r.model_used };
        } catch (e) {
          out[i] = { ok: false, error: String(e.message || e) };
        }
      }
    };
    await Promise.all(Array.from({ length: Math.min(parallel, prompts.length) }, worker));
    return { count: prompts.length, results: out };
  },

  freeai_debate: async (a) => {
    const rounds = Math.max(1, Math.min(Number(a.rounds) || 2, 5));
    const q = String(a.question);
    let pro = '';
    let con = '';
    const transcript = [];
    for (let i = 1; i <= rounds; i++) {
      pro = await askText(
        `Question: ${q}\n\nYou are arguing FOR. ${con ? `The opponent said:\n${con}\n\n` : ''}Give your strongest case in under 150 words.`,
      );
      con = await askText(
        `Question: ${q}\n\nYou are arguing AGAINST. The proponent said:\n${pro}\n\nGive your strongest rebuttal in under 150 words.`,
      );
      transcript.push({ round: i, pro, con });
    }
    const verdict = await askText(
      `Question: ${q}\n\nDebate transcript:\n${transcript.map((t) => `ROUND ${t.round}\nFOR: ${t.pro}\nAGAINST: ${t.con}`).join('\n\n')}\n\n`
      + 'Which side argued better, and why? Be decisive. Under 150 words.',
      { model: a.judge_model },
    );
    return { question: q, rounds, transcript, verdict };
  },

  freeai_review: async (a) => {
    const task = String(a.task);
    const draft = await askText(`Complete this task:\n${task}`, { model: a.draft_model });
    const critique = await askJson(
      `Task:\n${task}\n\nProposed answer:\n${draft}\n\n`
      + 'Review the answer. Reply with ONLY JSON: {"score": 0-10, "problems": ["..."], "improved": "<your better answer>"}',
      { model: a.review_model },
    );
    return { task, draft, review: critique.json, review_raw: critique.json ? null : critique.raw };
  },

  // ── repo and code work ────────────────────────────────────────────────────
  freeai_explain_file: async (a) => {
    const f = toolReadFile({ path: a.path, max_chars: 24000 });
    const answer = await askText(
      `File: ${f.path}\n\n\`\`\`\n${f.content}\n\`\`\`\n\n`
      + 'Explain in plain language: what this file does, its public interface, '
      + 'anything risky or surprising, and what would break if it changed.',
    );
    return { path: f.path, truncated: f.truncated, explanation: answer };
  },

  freeai_review_diff: async (a) => {
    const diff = String(a.diff).slice(0, 40000);
    const r = await askJson(
      `Review this git diff.\n\n\`\`\`diff\n${diff}\n\`\`\`\n\n`
      + 'Reply with ONLY JSON: {"verdict":"approve"|"comment"|"request_changes",'
      + '"bugs":["..."],"security":["..."],"missing_tests":["..."],"summary":"..."}',
      { maxTokens: a.max_tokens || 2048 },
    );
    return { review: r.json, raw: r.json ? null : r.raw };
  },

  freeai_summarise_dir: async (a) => {
    const dir = resolveIn(a.path || '.');
    const maxFiles = Math.max(1, Math.min(Number(a.max_files) || 40, 200));
    const out = [];
    const walk = (d, depth, prefix) => {
      if (out.length >= maxFiles || depth > 3) return;
      let entries;
      try { entries = fs.readdirSync(d, { withFileTypes: true }); } catch { return; }
      for (const e of entries) {
        if (out.length >= maxFiles) return;
        if (['node_modules', '.git', '.venv', '__pycache__', 'dist', 'build'].includes(e.name)) continue;
        const full = path.join(d, e.name);
        if (e.isDirectory()) walk(full, depth + 1, prefix + e.name + '/');
        else {
          let head = '';
          try { head = fs.readFileSync(full, 'utf8').slice(0, 400).replace(/\s+/g, ' '); } catch { /* binary */ }
          out.push({ file: prefix + e.name, head });
        }
      }
    };
    walk(dir, 0, '');
    const answer = await askText(
      `Directory listing with a snippet of each file:\n${JSON.stringify(out, null, 1)}\n\n`
      + 'Summarise: what this directory is for, what each notable file does, and where to look first.',
    );
    return { path: path.relative(REPO_ROOT, dir) || '.', files_seen: out.length, summary: answer };
  },

  freeai_generate_tests: async (a) => {
    const f = toolReadFile({ path: a.path, max_chars: 20000 });
    const fw = String(a.framework || '').trim();
    const answer = await askText(
      `Source file ${f.path}:\n\n\`\`\`\n${f.content}\n\`\`\`\n\n`
      + `Write unit tests for the behaviour above${fw ? ` using ${fw}` : ''}. `
      + 'Cover the normal path and the edge cases you can actually see in the code. '
      + 'Return ONLY the test code.',
    );
    return { path: f.path, framework: fw || null, tests: answer, note: 'tests are returned as text; nothing was written to disk' };
  },

  freeai_docs: async (a) => {
    const f = toolReadFile({ path: a.path, max_chars: 24000 });
    const symbol = a.symbol ? `Focus on the symbol: ${a.symbol}` : 'Cover the whole file.';
    const answer = await askText(
      `File ${f.path}:\n\n\`\`\`\n${f.content}\n\`\`\`\n\n${symbol} `
      + 'Write reference documentation: purpose, how to use it, parameters, return values, and gotchas.',
    );
    return { path: f.path, symbol: a.symbol || null, documentation: answer };
  },

  freeai_find: async (a) => {
    const g = toolGrep({ pattern: a.pattern, path: a.path || '.', limit: a.limit || 60 });
    if (!g.hits.length) return { pattern: a.pattern, hits: [], explanation: 'no matches' };
    const explanation = await askText(
      `A search for /${a.pattern}/ found:\n${JSON.stringify(g.hits, null, 1)}\n\n`
      + 'In plain language: what do these hits have in common, and what do they tell us about the code?',
    );
    return { pattern: a.pattern, hit_count: g.hits.length, explanation };
  },

  // ── structured output ─────────────────────────────────────────────────────
  freeai_json: async (a) => {
    const schema = a.schema && typeof a.schema === 'object' ? a.schema : {};
    const r = await askJson(
      `${a.prompt}\n\nYour reply MUST be JSON matching this shape:\n${JSON.stringify(schema)}\n`
      + 'Reply with ONLY the JSON value.',
      { model: a.model },
    );
    return { json: r.json, parse_ok: r.json !== null, retried: !!r.retried, raw: r.json ? null : r.raw };
  },

  freeai_classify: async (a) => {
    const labels = (Array.isArray(a.labels) ? a.labels : []).map(String);
    if (!labels.length) throw new Error('labels must be a non-empty array');
    const r = await askJson(
      `Classify the text below into exactly ONE of these labels: ${JSON.stringify(labels)}\n\n`
      + `TEXT:\n${a.text}\n\n`
      + 'Reply with ONLY JSON: {"label":"<one of the labels>","confidence":0.0-1.0,"reason":"<short>"}',
      { model: a.model },
    );
    const j = r.json || {};
    return {
      label: labels.includes(j.label) ? j.label : null,
      confidence: typeof j.confidence === 'number' ? j.confidence : null,
      reason: j.reason || null,
      labels_allowed: labels,
      parse_ok: !!r.json,
      raw: r.json ? null : r.raw,
    };
  },

  freeai_extract: async (a) => {
    const fields = (Array.isArray(a.fields) ? a.fields : []).map(String);
    if (!fields.length) throw new Error('fields must be a non-empty array');
    const r = await askJson(
      `Extract these fields from the text: ${JSON.stringify(fields)}\n\nTEXT:\n${a.text}\n\n`
      + 'Reply with ONLY JSON containing those keys. Use null for anything not present. Do not guess.',
      { model: a.model },
    );
    return { fields, extracted: r.json, parse_ok: !!r.json, raw: r.json ? null : r.raw };
  },

  freeai_translate: async (a) => {
    const answer = await askText(
      `Translate the following into ${a.language}. Preserve meaning and tone; do not explain, just translate.\n\n${a.text}`,
      { model: a.model },
    );
    return { language: a.language, translation: answer };
  },

  // ── files and search, directly (no model in the loop) ─────────────────────
  freeai_read_file: async (a) => toolReadFile({ path: a.path, max_chars: a.max_chars }),
  freeai_grep: async (a) => toolGrep({ pattern: a.pattern, path: a.path || '.', ignore_case: a.ignore_case, limit: a.limit }),
  freeai_list_dir: async (a) => toolListDir({ path: a.path || '.' }),

  // ── ops ───────────────────────────────────────────────────────────────────
  freeai_estimate: async (a) => {
    const e = estimateTokens(a.text);
    const c = compressMessages([{ role: 'user', content: String(a.text) }], 'safe');
    const cPct = c.stats.pct;
    return {
      ...e,
      after_safe_compression: { chars: c.stats.after, tokens_approx: Math.ceil(c.stats.after / 4) },
      compression_saving_pct: cPct,
      note: 'tokens are a chars/4 estimate — free models tokenize differently, so treat it as a ballpark',
    };
  },

  freeai_state_export: async () => {
    const st = readStateFile();
    return { file: process.env.FREEAI_STATE || path.join(__dirname, 'state.json'), state: st };
  },

  freeai_selftest: async () => {
    const checks = [];
    const add = (name, ok, detail) => checks.push({ name, ok, ...(detail ? { detail } : {}) });

    // 1. the router is up and knows its models
    let models = null;
    try { models = await getJson('/v1/models', 10000); add('router reachable', true); } catch (e) { add('router reachable', false, String(e.message)); }
    add('chain has models', !!(models && (models.data || []).length), models ? `${(models.data || []).length} models` : undefined);

    // 2. the compressor actually shrinks something
    const spam = Array.from({ length: 100 }, () => 'ERROR: same line').join('\n');
    const c = compressMessages([{ role: 'user', content: spam }], 'safe');
    add('compressor shrinks repeated lines', c.stats.after < c.stats.before / 2, `${c.stats.before} -> ${c.stats.after}`);

    // 3. the compressor does NOT touch numbered file content
    const numbered = ['  1: a', '', '', '  2: b', '  3: c'].join('\n');
    const nc = compressMessages([{ role: 'user', content: numbered }], 'safe');
    add('compressor leaves numbered content alone', nc.messages[0].content === numbered);

    // 4. the agent reply parser reads both shapes
    add('parser reads a tool call', parseAgentReply('{"tool":"read_file","args":{"path":"x"}}').kind === 'tool');
    add('parser reads a final answer', parseAgentReply('{"final":"done"}').kind === 'final');

    // 5. the sandbox refuses an escape
    let escaped = false;
    try { resolveIn('../../etc/passwd'); escaped = true; } catch { /* refused, correct */ }
    add('sandbox refuses a path escape', !escaped);

    // 6. a real model answers
    try {
      const r = await chat([{ role: 'user', content: 'Reply with exactly: PONG' }], { maxTokens: 16, timeoutMs: 60000 });
      add('a model answers', /pong/i.test(r.text), `${r.model_used || 'unknown'} -> ${JSON.stringify(r.text.slice(0, 40))}`);
    } catch (e) {
      add('a model answers', false, String(e.message));
    }

    const failed = checks.filter((x) => !x.ok).length;
    return { ok: failed === 0, failed, checks };
  },
};

async function callTool(name, args = {}) {
  const h = HANDLERS[name];
  if (!h) throw new Error(`unknown tool ${name}`);
  return h(args || {});
}
// -- JSON-RPC over stdio -----------------------------------------------------

let buffer = '';
const pendingWrites = [];

function send(msg) {
  process.stdout.write(JSON.stringify(msg) + '\n');
}

function ok(id, result) {
  send({ jsonrpc: '2.0', id, result });
}

function fail(id, code, message, data) {
  send({ jsonrpc: '2.0', id, error: { code, message, ...(data ? { data } : {}) } });
}

async function handle(msg) {
  const { id, method, params } = msg;
  try {
    switch (method) {
      case 'initialize':
        return ok(id, {
          protocolVersion: '2024-11-05',
          capabilities: { tools: {} },
          serverInfo: { name: 'free-ai', version: '1.0.0' },
        });
      case 'notifications/initialized':
        return;                                  // a notification: no reply
      case 'tools/list':
        return ok(id, { tools: TOOLS });
      case 'tools/call': {
        const name = params && params.name;
        const args = (params && params.arguments) || {};
        const tool = TOOLS.find((t) => t.name === name);
        if (!tool) return fail(id, -32602, `unknown tool ${name}`);
        try {
          const result = await callTool(name, args);
          return ok(id, { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] });
        } catch (e) {
          // A tool that failed is a RESULT, not a protocol error - the agent needs to
          // read the reason and decide what to do about it.
          return ok(id, {
            content: [{ type: 'text', text: `ERROR: ${String(e.message || e)}` }],
            isError: true,
          });
        }
      }
      case 'ping':
        return ok(id, {});
      default:
        return fail(id, -32601, `method not found: ${method}`);
    }
  } catch (e) {
    fail(id, -32603, String(e.message || e));
  }
}

// The stdio MCP loop is attached ONLY when this file is the entry point. Requiring it (a
// test, a detached runner, another module) previously hijacked stdin and exited the whole
// process on its EOF — that is why a detached subagent runner had to be wrapped in
// `tail -f /dev/null |`. As a library it must leave stdin alone.
if (require.main === module) {
  process.stdin.on('data', (chunk) => {
    buffer += chunk.toString('utf8');
    let nl;
    while ((nl = buffer.indexOf('\n')) !== -1) {
      const line = buffer.slice(0, nl).trim();
      buffer = buffer.slice(nl + 1);
      if (!line) continue;
      let msg;
      try { msg = JSON.parse(line); } catch { continue; }   // ignore a malformed frame
      handle(msg);
    }
  });

  process.stdin.on('end', () => process.exit(0));
}

module.exports = {
  callTool, runSubagent, parseAgentReply, TOOLS, chat,
  // Exported so the tool layer and the reply parser can be tested without a model in the
  // loop — the parser and the edit/read/grep tools are where the subagent's failures lived.
  __tools: { toolReadFile, toolListDir, toolGrep, toolWriteFile, toolEditFile, runTool, toolDocs, balancedObjects, repairJsonEscapes },
  __jobs: { spawnSubagent, subagentStatus, subagentLog, subagentResult, subagentList, subagentKill, JOBS_DIR },
};
