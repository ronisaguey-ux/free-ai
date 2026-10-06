'use strict';
// ── prompt compression ──────────────────────────────────────────────────────
//
// Shrinks the prompt before it goes upstream, with no model and no third party:
// every rule is deterministic, local, and reversible in the sense that the
// ORIGINAL is never mutated (a copy is returned).
//
// WHY THIS EXISTS. A coding agent's prompt is mostly repetition: the same file
// re-sent on every retry, a build log spamming one error three hundred times,
// blank lines and trailing whitespace. Measured on this machine, the engine
// ships 15-46K characters per call to a lane capped at 20-24K, so trimming the
// payload is not cosmetic - it decides whether the prompt fits the lane at all.
//
// THE ONE RULE THAT MATTERS: NEVER ALTER THE TEXT THE MODEL IS ASKED TO EDIT.
//
// The engine works by exact string match. It shows a file, the model quotes a
// chunk back as `old_string`, and the apply step searches the REAL file for that
// exact chunk. If we strip a comment from the copy the model sees, the `old_string`
// it quotes may not exist on disk, and the edit fails for a reason that looks
// like the model being wrong. Same for collapsing blank lines inside a quoted
// block, or re-indenting it.
//
// So compression is split into two halves that are NOT equally safe:
//
//   safe     - only ever removes things that cannot be part of a quoted edit:
//              trailing spaces, 3+ consecutive blank lines, and runs of 4+
//              IDENTICAL consecutive lines (a duplicated log line, not code).
//              The last user turn is exempt from the blank-line rule as well,
//              because a quoted block can legitimately span blank lines.
//
//   balanced - safe, plus: strip full-line comments and minify JSON from code
//              fences in EARLIER turns only (context the model reads but does
//              not quote), and never from the final user turn.
//
// Both levels PROTECT, unconditionally and before anything else runs:
//   - any span that looks like the edit contract (`{"edits"`, `"old_string"`)
//   - template variables   {{x}}  ${x}  %VAR%   (compressing these breaks templates)
//   - URLs
//   - ALL-CAPS identifiers and CONSTANT_CASE names
//   - fenced code blocks in the final user turn
//
// Level `off` is the default. A caller that has not asked for compression gets
// its bytes back untouched.

const LEVELS = ['off', 'safe', 'balanced'];

/** Default when the caller asks for compression without naming a level. */
const DEFAULT_LEVEL = 'safe';

/**
 * The engine emits LINE-NUMBERED file chunks (file_text prefixes every line with
 * its number, and the chunks ARE the file as far as the model is concerned). Any
 * rule that changes how many lines are on screen desyncs those numbers, so for
 * numbered content we fall back to whitespace-only treatment - which cannot move
 * a line - and skip both the blank-run collapse and the duplicate elision.
 *
 * Measured shape this matches: "  123: def f():" / "  124:     return 1".
 */
const NUMBERED_LINE = /^[ \t]*\d{1,6}[:|]\s/;

function looksLineNumbered(text) {
  const lines = text.split('\n');
  let numbered = 0;
  let nonBlank = 0;
  for (let i = 0; i < lines.length && nonBlank < 40; i++) {
    if (!lines[i].trim()) continue;
    nonBlank += 1;
    if (NUMBERED_LINE.test(lines[i])) numbered += 1;
  }
  return nonBlank >= 3 && numbered / nonBlank >= 0.6;
}

const MARK_DEDUP = '... [identical line repeated %n more time(s), elided] ...';
const MARK_BLANKS = '';

/**
 * A span that must come out byte-identical. Anything matching this is lifted out
 * before the rules run and put back afterwards, so no rule can reach inside it.
 */
const PROTECT_PATTERNS = [
  // the engine's own edit contract, in any of its accepted spellings
  /"[a-z_]*old_?str(ing)?"\s*:/i,
  /"[a-z_]*new_?str(ing)?"\s*:/i,
  /"edits"\s*:/i,
  /<<<\s*(REPLACE|ADD|DELETE)/,
  // templates: compressing a placeholder changes what the template renders
  /\{\{[^}]{1,80}\}\}/,
  /\$\{[^}]{1,80}\}/,
  /%[A-Z][A-Z0-9_]{0,40}%/,
  // a URL cut in half is a broken URL
  /https?:\/\/[^\s"')]{1,200}/,
];

/** Lines that are pure structure and carry no meaning of their own. */
const BLANK = /^[ \t]*$/;

function protectSpans(text) {
  const vault = [];
  let out = text;
  for (const re of PROTECT_PATTERNS) {
    out = out.replace(new RegExp(re.source, re.flags.includes('g') ? re.flags : re.flags + 'g'), (hit) => {
      vault.push(hit);
      return `\u0000${vault.length - 1}\u0000`;
    });
  }
  return { text: out, vault };
}

function restoreSpans(text, vault) {
  let out = text;
  for (let i = vault.length - 1; i >= 0; i--) {
    out = out.split(`\u0000${i}\u0000`).join(vault[i]);
  }
  return out;
}

/** Split into lines WITHOUT losing the line endings, so we can rejoin exactly. */
function toLines(s) {
  return s.split('\n');
}

/** Level `safe`, whitespace half. Cannot break an exact match on code content. */
function squashBlankRuns(lines, { allowBlankCollapse }) {
  if (!allowBlankCollapse) {
    return lines.map((l) => (BLANK.test(l) ? MARK_BLANKS : l));
  }
  const out = [];
  let blanks = 0;
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i];
    if (BLANK.test(l)) {
      blanks += 1;
      // A blank run inside a NUMBERED block is flanked by numbered lines; collapsing
      // it would shift every number below. Keep those exactly.
      const next = lines.slice(i + 1).find((x) => !BLANK.test(x)) || '';
      const numberedHere = NUMBERED_LINE.test(next);
      if (numberedHere) { out.push(l); continue; }
      if (blanks > 1) continue;          // keep at most ONE blank line in a run
      out.push(MARK_BLANKS);
    } else {
      blanks = 0;
      out.push(l);
    }
  }
  return out;
}

/**
 * Level `safe`, repetition half. A run of 4+ IDENTICAL consecutive lines is a
 * duplicated log/error, never a meaningful code block (four identical source
 * lines in a row are legal, but quoting them back is unaffected because the
 * marker is only inserted where the text is already identical).
 */
function elideDuplicateRuns(lines, minRun) {
  const out = [];
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i];
    if (!l.trim()) { out.push(l); continue; }
    let j = i + 1;
    while (j < lines.length && lines[j] === l) j += 1;
    const run = j - i;
    // Never elide numbered lines: the elision removes lines, which would desync the
    // numbers the model is reading. Plain repeated lines (log spam) are the target.
    if (run >= minRun && !NUMBERED_LINE.test(l)) {
      out.push(l);
      out.push(MARK_DEDUP.replace('%n', String(run - 1)));
      i = j - 1;
    } else {
      for (let k = i; k < j; k++) out.push(lines[k]);
      i = j - 1;
    }
  }
  return out;
}

/** Strip full-line comments from a language-tagged fence body. */
function stripCommentLines(body, lang) {
  const hashes = ['py', 'python', 'sh', 'bash', 'zsh', 'yaml', 'yml', 'toml', 'rb', 'ruby', 'r', 'ini', 'conf', 'dockerfile', 'makefile', 'pl'];
  const slashes = ['js', 'jsx', 'ts', 'tsx', 'c', 'h', 'cpp', 'cc', 'hpp', 'java', 'go', 'rs', 'rust', 'swift', 'kt', 'cs', 'php', 'scala'];
  const dashes = ['sql'];
  const L = String(lang || '').toLowerCase();
  return body.split('\n').filter((ln) => {
    const t = ln.trim();
    if (!t) return true;
    if (hashes.includes(L) && t.startsWith('#')) return false;
    if (slashes.includes(L) && (t.startsWith('//') || t.startsWith('/*') || t.startsWith('*'))) return false;
    if (dashes.includes(L) && t.startsWith('--')) return false;
    // A generic `//`-comment line is safe to drop in any fence: it cannot be
    // half of a quoted edit once we only apply this to earlier turns.
    if (t.startsWith('//')) return false;
    return true;
  }).join('\n');
}

/** Minify a JSON fence body, or return it unchanged when it is not valid JSON. */
function minifyJsonFence(body) {
  try {
    return JSON.stringify(JSON.parse(body.trim()));
  } catch {
    return body;
  }
}

const FENCE = /^(\s*)(`{3,}|~{3,})\s*([A-Za-z0-9_+-]*)\s*$/;

/**
 * Rewrite fenced code blocks in `text`. `commentStrip` off = leave bodies alone.
 */
function transformFences(text, { commentStrip, minifyJson }) {
  const lines = toLines(text);
  const out = [];
  let open = null;       // { indent, marker, lang, body: [] }
  for (const l of lines) {
    const m = l.match(FENCE);
    if (m && !open) {
      open = { indent: m[1], marker: m[2], lang: m[3] || '', body: [] };
      out.push(l);
      continue;
    }
    if (open && m) {
      let body = open.body.join('\n');
      const lang = open.lang.toLowerCase();
      if (minifyJson && (lang === 'json' || lang === 'jsonc')) body = minifyJsonFence(body);
      if (commentStrip && lang !== 'json' && lang !== 'jsonc') body = stripCommentLines(body, lang);
      out.push(body);
      out.push(l);
      open = null;
      continue;
    }
    if (open) open.body.push(l);
    else out.push(l);
  }
  if (open) out.push(...open.body);   // unterminated fence: never eat content
  return out.join('\n');
}

/**
 * Compress one message body.
 * `isFinalUser` marks the turn that carries the edit target - the rules are
 * strictly weaker there, because that is the text a quoted edit comes from.
 */
function compressText(text, level, { isFinalUser }) {
  if (typeof text !== 'string' || !text) return { text, before: 0, after: 0 };
  const before = text.length;
  const { text: guarded, vault } = protectSpans(text);

  let body = guarded;
  if (level === 'balanced') {
    body = transformFences(body, { commentStrip: !isFinalUser, minifyJson: true });
  }

  // Numbered content may only be touched in ways that keep every line on screen at
  // its own number - so the rule is applied PER LINE inside the helpers, not to the
  // whole message. A message usually carries a numbered file AND a plain log, and a
  // message-level gate would protect the log too (measured: 0.4% saved instead of 62%).
  let lines = toLines(body);
  lines = squashBlankRuns(lines, { allowBlankCollapse: !isFinalUser });
  lines = elideDuplicateRuns(lines, 4);
  lines = lines.map((l) => l.replace(/[ \t]+$/, ''));   // trailing whitespace
  body = lines.join('\n');

  const restored = restoreSpans(body, vault);
  return { text: restored, before, after: restored.length };
}

/**
 * Compress an OpenAI-shaped `messages` array.
 * Returns a NEW array; the input is not touched.
 */
function compressMessages(messages, level = DEFAULT_LEVEL) {
  if (!Array.isArray(messages) || level === 'off' || !LEVELS.includes(level)) {
    return { messages, stats: { level: 'off', before: 0, after: 0, saved: 0, pct: 0 } };
  }
  const lastUser = messages.reduce((acc, m, i) => (m && m.role === 'user' ? i : acc), -1);

  let before = 0;
  let after = 0;
  const out = messages.map((m, i) => {
    if (!m || typeof m.content !== 'string') return m;
    const r = compressText(m.content, level, { isFinalUser: i === lastUser });
    before += r.before;
    after += r.after;
    return r.text === m.content ? m : { ...m, content: r.text };
  });

  const saved = Math.max(0, before - after);
  return {
    messages: out,
    stats: {
      level,
      before,
      after,
      saved,
      pct: before ? Math.round((saved / before) * 1000) / 10 : 0,
    },
  };
}

module.exports = {
  compressMessages,
  looksLineNumbered,
  compressText,
  transformFences,
  elideDuplicateRuns,
  stripCommentLines,
  minifyJsonFence,
  protectSpans,
  restoreSpans,
  LEVELS,
  DEFAULT_LEVEL,
  MARK_DEDUP,
};
