// Local-crossing-objective search for acrostics.
// See ACROSTIC_DECODING_SEARCH.md for the journey here. The lesson from
// every prior A/B: the *mechanism* (branch near the cliff, roll out, score,
// pick) wasn't the problem — the *objective* was. Every variant scored a whole
// line by length-normalised log-prob + a 1-token peek, which (a) prefers lines
// that run to the maxLine wall (the forced newline there is "free" under the
// mask) and (b) only checks whether the forced letter is cheap, not whether the
// next line *continues* well.
//
// This module changes only the objective, per the diagnosis:
//   1. Score a SHORT fixed window straddling the crossing — the last `k`
//      content tokens before the line break, plus the forced letter and the
//      next `j` content tokens. Length-neutral; the structural newline is never
//      scored, so there's no wall bias.
//   2. Look `j` tokens PAST the forced letter (continuation), not just at it.
//   3. Make the break point a search variable, snapped to word boundaries:
//      generate the line greedily, then consider ending it 0..R tokens earlier
//      (only at a token that starts a new word/punctuation). r=0 (the natural
//      break) is always a candidate, so this can only match or beat greedy by
//      the window metric.
//
// Greedy (the control) is just this with R=0. Public-API only; committed text
// is always a valid token-prefix of a grammar-legal line, so there's no
// gobbledygook risk even if a window score is slightly mis-estimated.

import { LogitsProcessorList, StoppingCriteriaList, TextStreamer } from '@huggingface/transformers';
import { LineMaskScore, NewlineStop } from './surprisalLookahead.js';

function oneProcessor(proc) {
  const lp = new LogitsProcessorList();
  lp.push(proc);
  return lp;
}

const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);

export async function generateCrossingSearch(ctx, {
  grammar,
  secret,
  maxLine,
  prompt,
  systemPrompt = 'You are a helpful assistant.',
  k = 4,           // window: content tokens before the break
  j = 3,           // window: content tokens after the forced letter
  R = 4,           // max tokens to trim back from the line's natural end
  onToken = null,  // live token stream of the line being generated (lagged by R upstream)
  onLine = null,   // a line was committed (authoritative; replaces the streamed tail)
} = {}) {
  const { generator, tokenText, eosTokenIds } = ctx;
  const tokenizer = generator.tokenizer;

  const messages = [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: prompt },
  ];
  const promptString = tokenizer.apply_chat_template(messages, { tokenize: false, add_generation_prompt: true });
  const encIds = (text) => Array.from(tokenizer(text, { add_special_tokens: false }).input_ids.data, Number);
  // Mid-sentence iff the text so far doesn't end a sentence — then the next
  // forced letter should be lowercase (grammatically right, and keeps the
  // acrostic hidden). Empty prefix (line 0) is a sentence start → capital ok.
  const midSentence = (t) => {
    const s = (t || '').replace(/\s+$/, '');
    return s.length > 0 && !/[.!?]["'”’)\]]?$/.test(s);
  };

  // Greedy line from `prefixText` (acrostic text so far). Returns the line text
  // (incl. trailing newline for non-last lines), the line's token ids, and the
  // per-token log-probs.
  const genLine = async (prefixText, isLast) => {
    const startState = grammar.advance(grammar.initial, prefixText);
    const ctxStr = promptString + prefixText;
    const proc = new LineMaskScore({ grammar, startState, tokenizer, tokenText, eosTokenIds, forceLowerFirst: midSentence(prefixText) });
    const stops = new StoppingCriteriaList();
    if (!isLast) stops.push(new NewlineStop(tokenizer, encIds(ctxStr).length));
    const streamer = onToken
      ? new TextStreamer(tokenizer, { skip_prompt: true, skip_special_tokens: true, callback_function: onToken })
      : undefined;
    const out = await generator(ctxStr, {
      max_new_tokens: maxLine + 8, do_sample: false, return_full_text: false, add_special_tokens: false,
      logits_processor: oneProcessor(proc), stopping_criteria: stops, streamer,
    });
    let text = out[0].generated_text;
    if (!isLast) { const nl = text.indexOf('\n'); if (nl !== -1) text = text.slice(0, nl + 1); }
    const baseN = encIds(ctxStr).length;
    const lineIds = encIds(ctxStr + text).slice(baseN);
    return { text, lineIds, logps: proc.stepLogprobs };
  };

  // Roll the NEXT line's opening from `prefixText` (which ends in a newline):
  // the forced letter + up to `n-1` content tokens. Returns their log-probs.
  const rollOpen = async (prefixText, n) => {
    const startState = grammar.advance(grammar.initial, prefixText);
    if (startState === -1) return [];
    const proc = new LineMaskScore({ grammar, startState, tokenizer, tokenText, eosTokenIds, forceLowerFirst: midSentence(prefixText) });
    const stops = new StoppingCriteriaList();
    stops.push(new NewlineStop(tokenizer, encIds(promptString + prefixText).length));
    await generator(promptString + prefixText, {
      max_new_tokens: n, do_sample: false, return_full_text: false, add_special_tokens: false,
      logits_processor: oneProcessor(proc), stopping_criteria: stops,
    });
    return proc.stepLogprobs;
  };

  const nLines = secret.length;
  let committed = '';
  const perLine = [];

  for (let i = 0; i < nLines; i++) {
    const isLast = i === nLines - 1;
    const { text, lineIds, logps } = await genLine(committed, isLast);

    // Last line, or no break search: commit the greedy line as-is.
    if (isLast || R <= 0) {
      committed += text;
      perLine.push({ line: i, chosen: text, r: 0, candidates: null });
      if (onLine) onLine(text, { line: i });
      continue;
    }

    // End-align token ids with their log-probs (tokenisation at the prompt
    // boundary can drift by a token; the window only looks at the tail).
    const m = Math.min(lineIds.length, logps.length);
    const ids = lineIds.slice(-m);
    const lps = logps.slice(-m);
    const hasNl = text.endsWith('\n');
    const lineStartState = grammar.advance(grammar.initial, committed);

    const candidates = [];
    for (let r = 0; r <= Math.min(R, m - 1); r++) {
      // r tokens trimmed → break after (m-r) tokens. Require the first trimmed
      // token to begin a new word / punctuation (clean boundary). r=0 always ok.
      if (r > 0) {
        const firstTrimmed = tokenText[ids[m - r]];
        if (!firstTrimmed || !/^[\s.,;:!?)\]"'’”]/.test(firstTrimmed)) continue;
      }
      const keptIds = ids.slice(0, m - r);
      if (!keptIds.length) continue;
      const prefixText = tokenizer.decode(keptIds, { skip_special_tokens: true }).replace(/\n+$/, '');
      const brokeLine = prefixText + '\n';
      // The trimmed line must still be grammar-legal — i.e. still contain its
      // forced letter. Trimming from the end can strip the letter in LIST mode
      // (where it sits after the " * " bullet), which would break the acrostic
      // and cascade into empty lines. r=0 (the full line) is always legal.
      if (grammar.advance(lineStartState, brokeLine) === -1) continue;

      // "before" window: last k content log-probs (drop the trailing newline's).
      let beforeLps = lps.slice(0, m - r);
      if (r === 0 && hasNl) beforeLps = beforeLps.slice(0, -1);
      beforeLps = beforeLps.slice(-k);

      // "through": forced letter + j content tokens of the next line.
      const afterLps = await rollOpen(committed + brokeLine, 1 + j);

      candidates.push({
        r, brokeLine,
        score: mean(beforeLps.concat(afterLps)),
        nBefore: beforeLps.length, nAfter: afterLps.length,
        preview: brokeLine.slice(-28),
      });
    }

    let chosen = text, r = 0;
    if (candidates.length) {
      candidates.sort((a, b) => b.score - a.score);
      chosen = candidates[0].brokeLine;
      r = candidates[0].r;
    }
    committed += chosen;
    perLine.push({ line: i, chosen, r, candidates });
    if (onLine) onLine(chosen, { line: i });
  }

  return { text: committed, perLine };
}
