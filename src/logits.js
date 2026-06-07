// Grammar-constrained LogitsProcessor.  At each generation step:
//   1. Decode the generated suffix back to text.
//   2. Advance the grammar NFA by that text.
//   3. For every candidate token id, check whether appending its decoded text
//      keeps the NFA alive; mask losers to -Infinity.
//   4. EOS is allowed only once the NFA has reached an accept state.
//
// Per-token decoding can disagree with BPE sequence-decoding in edge cases
// (merged punctuation, etc.); for the acrostic patterns we care about this
// approximation is fine.  Revisit if outputs go sideways.
import { LogitsProcessor } from '@huggingface/transformers';

export class GrammarLogitsProcessor extends LogitsProcessor {
  constructor({ grammar, tokenizer, tokenText, eosTokenIds = [] }) {
    super();
    this.grammar = grammar;
    this.tokenizer = tokenizer;
    this.tokenText = tokenText;
    this.eosTokenIds = new Set(eosTokenIds.map(Number));
    this.promptLength = undefined;
    this.stats = freshStats();
  }

  reset() {
    this.promptLength = undefined;
    this._prevExit = undefined;
    this.stats = freshStats();
  }

  _call(input_ids, logits) {
    const tEntry = performance.now();
    // Gap since the previous _call's exit ≈ time the model+sampler+streamer
    // spent between steps (dominated by the ONNX forward pass). If the KV
    // cache is being re-used, this should be roughly constant as generation
    // proceeds; if not, it grows with sequence length.
    const priorModelMs = this._prevExit === undefined ? 0 : tEntry - this._prevExit;
    if (this.promptLength === undefined) {
      this.promptLength = input_ids[0].length;
    }
    const ids = input_ids[0];
    const generated = [];
    for (let j = this.promptLength; j < ids.length; j++) generated.push(Number(ids[j]));

    const tDec0 = performance.now();
    const text = generated.length
      ? this.tokenizer.decode(generated, { skip_special_tokens: true })
      : '';
    const tDec = performance.now() - tDec0;

    const tPref0 = performance.now();
    const state = this.grammar.advance(this.grammar.initial, text);
    const tPref = performance.now() - tPref0;

    const data = logits[0].data;

    if (state === -1) {
      // Already violated; nothing useful we can do without rewinding. Let the
      // original logits through so generation at least terminates.
      const tTotal = performance.now() - tEntry;
      this._prevExit = tEntry + tTotal;
      this._record({ tEntry, tTotal, tDec, tPref, tScan: 0, survivors: -1, violated: true, priorModelMs, seqLen: input_ids[0].length });
      return logits;
    }

    const atAccept = this.grammar.accepts(state);

    const tScan0 = performance.now();
    let survivors = 0;
    let advanceCalls = 0;
    for (let i = 0; i < this.tokenText.length; i++) {
      if (this.eosTokenIds.has(i)) {
        if (!atAccept) data[i] = -Infinity;
        else survivors++;
        continue;
      }
      const tok = this.tokenText[i];
      if (!tok) { data[i] = -Infinity; continue; }
      advanceCalls++;
      if (this.grammar.advance(state, tok) === -1) data[i] = -Infinity;
      else survivors++;
    }
    const tScan = performance.now() - tScan0;
    const tTotal = performance.now() - tEntry;
    this._prevExit = tEntry + tTotal;
    this._record({ tEntry, tTotal, tDec, tPref, tScan, survivors, advanceCalls, violated: false, priorModelMs, seqLen: input_ids[0].length });
    return logits;
  }

  _record(s) {
    const st = this.stats;
    st.calls++;
    st.totalMs += s.tTotal;
    st.decodeMs += s.tDec;
    st.prefixMs += s.tPref;
    st.scanMs += s.tScan;
    if (s.advanceCalls) st.advanceCalls += s.advanceCalls;
    st.perStep.push(s);
  }
}

function freshStats() {
  return { calls: 0, totalMs: 0, decodeMs: 0, prefixMs: 0, scanMs: 0, advanceCalls: 0, perStep: [] };
}

// One-shot build of tokenId → text, using per-token decode. Special tokens
// decode to '' under skip_special_tokens: true, which we treat as "disallowed".
export function buildTokenTextTable(tokenizer, vocabSize) {
  const arr = new Array(vocabSize);
  for (let i = 0; i < vocabSize; i++) {
    try {
      arr[i] = tokenizer.decode([i], { skip_special_tokens: true }) || '';
    } catch {
      arr[i] = '';
    }
  }
  return arr;
}
