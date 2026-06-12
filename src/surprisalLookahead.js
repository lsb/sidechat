// Surprisal-triggered lookahead decoding for acrostics.
// See ACROSTIC_DECODING_SEARCH.md for the design and the literature it
// rests on (AdaDec, NeuroLogic A*esque, Grammar-Aligned Decoding).
//
// Decode greedily under the acrostic grammar by default. At each line boundary
// (a known constraint cliff — the forced first letter), measure how *surprised*
// the model is by the constraint: surprise = -log( max_{legal t} P_unmasked(t) ),
// i.e. how little probability the model's own distribution puts on the best
// grammar-legal token. Only when surprise ≥ threshold do we branch the top-B
// legal openings, roll each line out, and rank by a future-aware score
// (length-normalised log-prob + the next cliff's cost). Otherwise we just take
// the greedy line. Search budget is spent only where the constraint bites.
//
// Public-API only (no KV-cache surgery): each line is a fresh generator()
// continuation of the chat-templated prompt + committed text fed back as a raw
// string. Greedy by construction can only match greedy; deviation happens only
// at high-surprise cliffs.

import {
  LogitsProcessor,
  LogitsProcessorList,
  StoppingCriteria,
  StoppingCriteriaList,
} from '@huggingface/transformers';

// Grammar-masking processor that also (a) accumulates the greedy chosen-token
// log-prob per step, and — at its first step, when captureTopN > 0 — (b) the
// top-N legal tokens and (c) the surprise signal (unmasked −log p_best_legal).
export class LineMaskScore extends LogitsProcessor {
  constructor({ grammar, startState, tokenizer, tokenText, eosTokenIds = [], captureTopN = 0 }) {
    super();
    this.grammar = grammar;
    this.startState = startState;
    this.tokenizer = tokenizer;
    this.tokenText = tokenText;
    this.eosTokenIds = new Set(eosTokenIds.map(Number));
    this.captureTopN = captureTopN;
    this.promptLength = undefined;
    this.stepLogprobs = [];   // chosen (argmax) log-prob, one per generated step
    this.topN = null;         // [{id, logit, logprob}] from the first step
    this.surprise = null;     // −log p_best_legal under the UNMASKED dist, first step
  }

  _call(input_ids, logits) {
    const ids = input_ids[0];
    if (this.promptLength === undefined) this.promptLength = ids.length;
    const generated = [];
    for (let j = this.promptLength; j < ids.length; j++) generated.push(Number(ids[j]));
    const gen = generated.length
      ? this.tokenizer.decode(generated, { skip_special_tokens: true })
      : '';

    const state = this.grammar.advance(this.startState, gen);
    const data = logits[0].data;
    if (state === -1) { this.stepLogprobs.push(0); return logits; }

    const atAccept = this.grammar.accepts(state);
    const firstStep = this.topN === null && this.surprise === null;
    const wantSignal = this.captureTopN && firstStep;

    // Unmasked logsumexp — only needed at the first step for the surprise signal.
    let lseAll = null;
    if (wantSignal) {
      let maxAll = -Infinity;
      for (let i = 0; i < data.length; i++) if (data[i] > maxAll) maxAll = data[i];
      let s = 0;
      for (let i = 0; i < data.length; i++) s += Math.exp(data[i] - maxAll);
      lseAll = maxAll + Math.log(s);
    }

    // Mask illegal tokens; track the best *legal* (original) logit and survivors.
    let maxLegal = -Infinity;
    const survivors = wantSignal ? [] : null;
    for (let i = 0; i < this.tokenText.length; i++) {
      let legal;
      if (this.eosTokenIds.has(i)) {
        legal = atAccept;
      } else {
        const tok = this.tokenText[i];
        legal = !!tok && this.grammar.advance(state, tok) !== -1;
      }
      if (!legal) { data[i] = -Infinity; continue; }
      if (data[i] > maxLegal) maxLegal = data[i];
      if (survivors) survivors.push([i, data[i]]);
    }
    if (maxLegal === -Infinity) { this.stepLogprobs.push(0); return logits; }

    // Chosen (= argmax-legal) log-prob under the masked/renormalised distribution.
    let sumExp = 0;
    for (let i = 0; i < data.length; i++) { const d = data[i]; if (d !== -Infinity) sumExp += Math.exp(d - maxLegal); }
    const lseMasked = maxLegal + Math.log(sumExp);
    this.stepLogprobs.push(maxLegal - lseMasked);

    if (wantSignal) {
      this.surprise = lseAll - maxLegal;  // = −log p_best_legal (unmasked)
      survivors.sort((a, b) => b[1] - a[1]);
      this.topN = survivors.slice(0, this.captureTopN).map(([id, logit]) => ({ id, logit, logprob: logit - lseMasked }));
    }
    return logits;
  }
}

// Stop a rollout as soon as the newly generated token contains a newline.
export class NewlineStop extends StoppingCriteria {
  constructor(tokenizer, promptLength) {
    super();
    this.tokenizer = tokenizer;
    this.promptLength = promptLength;
  }

  _call(input_ids) {
    return input_ids.map((ids) => {
      if (ids.length <= this.promptLength) return false;
      const last = Number(ids[ids.length - 1]);
      return this.tokenizer.decode([last], { skip_special_tokens: true }).includes('\n');
    });
  }
}

function oneProcessor(proc) {
  const lp = new LogitsProcessorList();
  lp.push(proc);
  return lp;
}

export async function generateSurprisalLookahead(ctx, {
  grammar,
  secret,
  maxLine,
  prompt,
  B = 4,                  // legal openings to try when a cliff triggers
  surpriseThreshold = 2.0,  // nats; trigger lookahead when surprise ≥ this
  peek = true,            // future-cost term: cost of the next cliff
  systemPrompt = 'You are a helpful assistant.',
  onLine = null,          // (lineText, info) callback as each line is committed
} = {}) {
  const { generator, tokenText, eosTokenIds } = ctx;
  const tokenizer = generator.tokenizer;

  const messages = [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: prompt },
  ];
  const promptString = tokenizer.apply_chat_template(messages, {
    tokenize: false,
    add_generation_prompt: true,
  });
  const encLen = (text) => tokenizer(text, { add_special_tokens: false }).input_ids.dims.at(-1);

  // One cheap forward at `prefixText`: returns the top-M legal next tokens, the
  // surprise signal (−log p_best_legal, unmasked), and the grammar state.
  const discover = async (prefixText, M) => {
    const startState = grammar.advance(grammar.initial, prefixText);
    if (startState === -1) return { startState, openings: [], surprise: Infinity };
    const proc = new LineMaskScore({ grammar, startState, tokenizer, tokenText, eosTokenIds, captureTopN: M });
    await generator(promptString + prefixText, {
      max_new_tokens: 1, do_sample: false, return_full_text: false, add_special_tokens: false,
      logits_processor: oneProcessor(proc),
    });
    return { startState, openings: proc.topN || [], surprise: proc.surprise ?? Infinity };
  };

  // Greedily roll a line out from `prefixText` + `openText`, grammar-masked,
  // to the newline (or EOS for the last line).
  const rollout = async (prefixText, openText, rollStart, isLast) => {
    const ctxStr = promptString + prefixText + openText;
    const roll = new LineMaskScore({ grammar, startState: rollStart, tokenizer, tokenText, eosTokenIds });
    const stops = new StoppingCriteriaList();
    if (!isLast) stops.push(new NewlineStop(tokenizer, encLen(ctxStr)));
    const out = await generator(ctxStr, {
      max_new_tokens: maxLine + 8, do_sample: false, return_full_text: false, add_special_tokens: false,
      logits_processor: oneProcessor(roll),
      stopping_criteria: stops,
    });
    let cont = out[0].generated_text;
    if (!isLast) {
      const nl = cont.indexOf('\n');
      if (nl !== -1) cont = cont.slice(0, nl + 1);
    }
    return { text: openText + cont, bodyLP: roll.stepLogprobs.reduce((a, b) => a + b, 0), nBody: roll.stepLogprobs.length };
  };

  // Future-cost (A*esque): log p_best_legal of the NEXT line's forced letter
  // given everything so far. Higher (closer to 0) = the next cliff is cheap.
  const peekCost = async (prefixText) => -(await discover(prefixText, 1)).surprise;

  const nLines = secret.length;
  let committed = '';
  const perLine = [];

  for (let i = 0; i < nLines; i++) {
    const isLast = i === nLines - 1;
    const before = committed;

    const { startState, openings, surprise } = await discover(before, B);
    if (!openings.length) break;  // dead end

    const triggered = surprise >= surpriseThreshold;
    let chosen, candidates = null;

    if (!triggered) {
      // Easy cliff — greedy: top-1 legal opening, roll the line out.
      const op = openings[0];
      const rs = grammar.advance(startState, tokenText[op.id]);
      chosen = (await rollout(before, tokenText[op.id], rs, isLast)).text;
    } else {
      // Hard cliff — look ahead over the top-B openings.
      candidates = [];
      for (const op of openings) {
        const openText = tokenText[op.id];
        const rs = grammar.advance(startState, openText);
        if (rs === -1) continue;
        const { text: lineText, bodyLP, nBody } = await rollout(before, openText, rs, isLast);
        let peekLP = null;
        let nTok = 1 + nBody;
        if (peek && !isLast) { peekLP = await peekCost(before + lineText); nTok += 1; }
        candidates.push({
          openText, lineText,
          openLP: op.logprob, bodyLP, peekLP,
          nTok, score: (op.logprob + bodyLP + (peekLP ?? 0)) / Math.max(1, nTok),
        });
      }
      if (!candidates.length) break;
      candidates.sort((a, b) => b.score - a.score);
      chosen = candidates[0].lineText;
    }

    committed = before + chosen;
    perLine.push({ line: i, surprise, triggered, chosen, candidates });
    if (onLine) onLine(chosen, { line: i, surprise, triggered });
  }

  return { text: committed, perLine };
}
