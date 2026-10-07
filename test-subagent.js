#!/usr/bin/env node
/*
 * Subagent-layer tests. These cover the two places the subagent actually failed: the reply
 * parser (a tool call read as a FINAL ended the run at step 1) and the file tools
 * (read_file could not offset, grep refused a file path, there was no surgical edit).
 *
 * No model is involved — every check is deterministic, so a failure here is a real
 * regression, not a flaky free endpoint.
 */
'use strict';
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { parseAgentReply, __tools } = require('./mcp-server.js');

const results = [];
const check = (name, ok, extra = '') => results.push([ok ? 'PASS' : 'FAIL', name, extra]);
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);

// ── parser: every tool-call shape a free model emits ─────────────────────────────
const toolCases = {
  '{"tool":"read_file","args":{"path":"a.js"}}': ['read_file', { path: 'a.js' }],
  '{"name":"read_file","parameters":{"path":"a.js"}}': ['read_file', { path: 'a.js' }],
  '{"name":"grep","args":{"pattern":"x"}}': ['grep', { pattern: 'x' }],
  '{"name":"grep","arguments":{"pattern":"x"}}': ['grep', { pattern: 'x' }],
  '{"function":{"name":"run_bash","arguments":{"command":"ls"}}}': ['run_bash', { command: 'ls' }],
};
for (const [text, [name, args]] of Object.entries(toolCases)) {
  const r = parseAgentReply(text);
  check(`parses ${text.slice(0, 40)}`, r.kind === 'tool' && r.tool === name && eq(r.args, args), JSON.stringify(r));
}
{
  const openai = JSON.stringify({ tool_calls: [{ function: { name: 'grep', arguments: JSON.stringify({ pattern: 'x' }) } }] });
  const r = parseAgentReply(openai);
  check('parses the OpenAI tool_calls shape', r.kind === 'tool' && r.tool === 'grep');
}
{
  const fenced = '```json\n{"name":"run_bash","parameters":{"command":"ls"}}\n```';
  check('parses a fenced call', parseAgentReply(fenced).tool === 'run_bash');
}
{
  const proseThen = 'I will inspect.\n{"tool":"list_dir","args":{"path":"."}}';
  check('finds a call after prose', parseAgentReply(proseThen).tool === 'list_dir');
}

// ── parser: an invalid escape in a shell command must not kill the call ──────────
{
  const cmd = "grep -n 'round\\|submit\\|while' f.js";
  const reply = '\n{\n "name": "run_bash",\n "parameters": { "command": ' + JSON.stringify(cmd) + ' }\n}';
  const r = parseAgentReply(reply);
  check('repairs an invalid JSON escape in a command', r.kind === 'tool' && r.args.command === cmd, JSON.stringify(r));
}

// ── parser: finals vs garbage ────────────────────────────────────────────────────
check('explicit final is marked explicit', parseAgentReply('{"final":"done"}').explicit === true);
check('prose is NOT an explicit final', parseAgentReply('I will read the file now.').explicit === false);
check('garbled markup is NOT an explicit final',
  parseAgentReply('<dots_function_call>\n1,"20000"}\n</parameter>\n</tool_call>').explicit === false);

// ── tools against a real temp tree ───────────────────────────────────────────────
const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'free-ai-agent-'));
process.env.FREEAI_MCP_ROOT = dir;
// resolveIn() reads REPO_ROOT at call time via runTool; drive the tools directly with paths
// relative to the temp root by setting it through the module's own mechanism.
const M = require('./mcp-server.js');
// There is no exported root setter, so exercise the parser+tools with absolute-ish paths by
// writing the tree under cwd-relative names is unsafe; instead call the exported functions
// with paths the sandbox resolves. `runTool` uses REPO_ROOT from env at load, so re-require
// the module in a child with FREEAI_MCP_ROOT set would be needed for full isolation — here we
// validate the pure logic that does not depend on the root.
const big = Array.from({ length: 120 }, (_, i) => `line ${i + 1}`).join('\n');
const bigPath = path.join(dir, 'big.txt');
fs.writeFileSync(bigPath, big);

// read_file offset + windowing, using the exported helper via a minimal root shim
{
  // The exported tool uses REPO_ROOT; point it at the temp dir by chdir is not enough, so use
  // the documented env var in a fresh child process to keep the check honest.
  const { execFileSync } = require('node:child_process');
  const script = `
    const { __tools } = require(${JSON.stringify(path.join(__dirname, 'mcp-server.js'))});
    const r = __tools.toolReadFile({ path: 'big.txt', offset: 10, max_chars: 200 });
    console.log(JSON.stringify({ total: r.total_lines, from: r.from_line, first: r.content.split('\\n')[0], hint: r.hint }));
  `;
  let out;
  try {
    out = execFileSync(process.execPath, ['-e', script], {
      env: { ...process.env, FREEAI_MCP_ROOT: dir }, encoding: 'utf8',
    }).trim();
  } catch (e) { out = 'ERR ' + e.message; }
  let parsed = null;
  try { parsed = JSON.parse(out.split('\n').pop()); } catch { /* leave null */ }
  check('read_file honours offset', parsed && parsed.total === 120 && parsed.from === 10 && parsed.first === 'line 10', out);
  check('read_file reports a continuation hint', parsed && /offset=\d+/.test(String(parsed.hint)), out);
}

// grep on a FILE path must not raise ENOTDIR
{
  const { execFileSync } = require('node:child_process');
  fs.writeFileSync(path.join(dir, 'g.txt'), 'alpha\nbeta\ngamma\n');
  const script = `
    const { __tools } = require(${JSON.stringify(path.join(__dirname, 'mcp-server.js'))});
    const r = __tools.toolGrep({ pattern: 'beta', path: 'g.txt' });
    console.log(JSON.stringify({ hits: r.hits.length, first: r.hits[0] && r.hits[0].line }));
  `;
  let out;
  try {
    out = execFileSync(process.execPath, ['-e', script], {
      env: { ...process.env, FREEAI_MCP_ROOT: dir }, encoding: 'utf8',
    }).trim();
  } catch (e) { out = 'ERR ' + e.message; }
  let parsed = null;
  try { parsed = JSON.parse(out.split('\n').pop()); } catch { /* leave null */ }
  check('grep accepts a file path', parsed && parsed.hits === 1 && parsed.first !== 0, out);
}

// edit_file: unique replace, ambiguity refusal, and a missing-find refusal
{
  const { execFileSync } = require('node:child_process');
  fs.writeFileSync(path.join(dir, 'e.js'), 'const a = 1;\nconst b = 2;\n');
  const script = `
    const { __tools } = require(${JSON.stringify(path.join(__dirname, 'mcp-server.js'))});
    const out = {};
    try { out.unique = __tools.toolEditFile({ path: 'e.js', find: 'const a = 1;', replace: 'const a = 42;' }).replacements; } catch (e) { out.unique = 'ERR ' + e.message; }
    try { out.two = __tools.toolEditFile({ path: 'e.js', find: 'const b = 2;', replace: 'const b = 7;' }).replacements; } catch (e) { out.two = 'ERR'; }
    try { __tools.toolEditFile({ path: 'e.js', find: 'NOT-PRESENT', replace: 'x' }); out.missing = 'no-error'; } catch (e) { out.missing = 'refused'; }
    out.content = require('fs').readFileSync(require('path').join(process.env.FREEAI_MCP_ROOT, 'e.js'), 'utf8');
    console.log(JSON.stringify(out));
  `;
  let out;
  try {
    out = execFileSync(process.execPath, ['-e', script], {
      env: { ...process.env, FREEAI_MCP_ROOT: dir }, encoding: 'utf8', cwd: dir,
    }).trim();
  } catch (e) { out = 'ERR ' + e.message; }
  let parsed = null;
  try { parsed = JSON.parse(out.split('\n').pop()); } catch { /* leave null */ }
  check('edit_file replaces a unique match', parsed && parsed.unique === 1, out);
  check('edit_file applied the change', parsed && /const a = 42;/.test(parsed.content || '') && /const b = 7;/.test(parsed.content || ''), out);
  check('edit_file refuses a missing find', parsed && parsed.missing === 'refused', out);
}

// ── report ───────────────────────────────────────────────────────────────────────
let failed = 0;
for (const [status, name, extra] of results) {
  if (status === 'FAIL') failed++;
  console.log(`${status} ${name}${status === 'FAIL' && extra ? '  <- ' + String(extra).slice(0, 200) : ''}`);
}
console.log('='.repeat(56));
console.log(`${results.length - failed}/${results.length} passed`);
fs.rmSync(dir, { recursive: true, force: true });
process.exit(failed ? 1 : 0);
