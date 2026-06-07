// Tiny grammar engine for acrostic-style constraints.
//
// Primitives:
//   - Atoms: `{kind: 'lit', allowed: Set<string>}` (consumes exactly one char
//     from `allowed`) or `{kind: 'body', max: number}` (consumes 0..max
//     non-newline chars).
//   - Atom sequences are concatenation-only; with the body/newline structure
//     we use, transitions are deterministic, so state packs into one int:
//     `atomIdx * stride + count`.
//
// Builders:
//   - `compileAcrostic(secret, opts)` — list-mode or prose-mode acrostic.
//   - `compileLiteral(text)` — exact-text matcher (used by the classifier).
//   - `unionGrammars([g1, g2, ...])` — accept if any branch is alive.

function compileFromAtoms(atoms) {
  let maxBodyMax = 0;
  for (const a of atoms) if (a.kind === 'body' && a.max > maxBodyMax) maxBodyMax = a.max;
  const stride = maxBodyMax + 2;
  const PAST_END = atoms.length * stride;
  const stateCount = PAST_END + 1;

  // Precompute accepting states: (a, c) accepts iff atom `a` can be
  // epsilon-skipped at count `c` AND (a+1, 0) is accepting.
  const accepting = new Uint8Array(stateCount);
  accepting[PAST_END] = 1;
  let nextAccepting = true;
  for (let a = atoms.length - 1; a >= 0; a--) {
    const atom = atoms[a];
    if (nextAccepting) {
      const min = atom.kind === 'lit' ? 1 : 0;
      const max = atom.kind === 'lit' ? 1 : atom.max;
      for (let c = min; c <= max; c++) accepting[a * stride + c] = 1;
    }
    nextAccepting = accepting[a * stride + 0] === 1;
  }

  function consumeAt(a, ch) {
    while (a < atoms.length) {
      const atom = atoms[a];
      if (atom.kind === 'lit') {
        return atom.allowed.has(ch)
          ? (a + 1 >= atoms.length ? PAST_END : (a + 1) * stride)
          : -1;
      }
      if (ch !== '\n') return a * stride + 1;
      a++;
    }
    return -1;
  }

  function advance(state, str) {
    let s = state;
    for (let i = 0; i < str.length; i++) {
      if (s === PAST_END) return -1;
      const a = (s / stride) | 0;
      const c = s - a * stride;
      const atom = atoms[a];
      const ch = str[i];
      let next;
      if (atom.kind === 'lit') {
        if (c < 1 && atom.allowed.has(ch)) {
          next = (a + 1 >= atoms.length) ? PAST_END : (a + 1) * stride;
        } else return -1;
      } else {
        if (c < atom.max && ch !== '\n') next = a * stride + (c + 1);
        else next = consumeAt(a + 1, ch);
      }
      if (next === -1) return -1;
      s = next;
    }
    return s;
  }

  return {
    initial: 0,
    advance,
    accepts: (s) => s >= 0 && s < stateCount && accepting[s] === 1,
    stateCount,
  };
}

// Spaces in the secret are treated as "word breaks" — they don't pin the line
// to any particular letter, but they still produce a line, and the line must
// start with a punctuation character so the acrostic reads naturally
// ("HI WORLD" → H… / I… / <punct>… / W… / O… / R… / L… / D…).
const PUNCT_FOR_SPACE = new Set([
  '.', ',', ';', ':', '!', '?', '-',
  '(', ')', '[', ']', '{', '}',
  '~', '<', '>',
  '"', "'", '`',
  '@', '#', '$', '%', '&', '+', '=', '/', '\\', '|', '_', '^',
]);

export function compileAcrostic(secret, {
  listPrefix = ' * ',
  maxLine = 80,
  caseInsensitive = false,
  firstLinePrefix = true,
} = {}) {
  if (!secret || !secret.length) throw new Error('secret must be non-empty');
  const atoms = [];
  for (let i = 0; i < secret.length; i++) {
    const wantPrefix = i > 0 || firstLinePrefix;
    if (wantPrefix) {
      for (const c of listPrefix) atoms.push({ kind: 'lit', allowed: new Set([c]) });
    }
    const letter = secret[i];
    let allowed;
    if (letter === ' ') {
      allowed = new Set(PUNCT_FOR_SPACE);
    } else if (caseInsensitive) {
      allowed = new Set([letter.toUpperCase(), letter.toLowerCase()]);
    } else {
      allowed = new Set([letter]);
    }
    atoms.push({ kind: 'lit', allowed });
    atoms.push({ kind: 'body', max: maxLine });
    if (i < secret.length - 1) atoms.push({ kind: 'lit', allowed: new Set(['\n']) });
  }
  return compileFromAtoms(atoms);
}

export function compileLiteral(text) {
  if (!text || !text.length) throw new Error('literal must be non-empty');
  const atoms = [];
  for (const c of text) atoms.push({ kind: 'lit', allowed: new Set([c]) });
  return compileFromAtoms(atoms);
}

// Run several grammars in parallel; a token is alive iff at least one branch
// is alive. State is stored as an array of per-branch ints (-1 = dead branch).
// When every branch is dead, `advance` returns -1 (matches the single-grammar
// dead sentinel).
export function unionGrammars(grammars) {
  const k = grammars.length;
  return {
    initial: grammars.map((g) => g.initial),
    advance(state, str) {
      const next = new Array(k);
      let anyLive = false;
      for (let i = 0; i < k; i++) {
        if (state[i] === -1) { next[i] = -1; continue; }
        const s = grammars[i].advance(state[i], str);
        next[i] = s;
        if (s !== -1) anyLive = true;
      }
      return anyLive ? next : -1;
    },
    accepts(state) {
      if (state === -1) return false;
      for (let i = 0; i < k; i++) {
        if (state[i] !== -1 && grammars[i].accepts(state[i])) return true;
      }
      return false;
    },
  };
}
