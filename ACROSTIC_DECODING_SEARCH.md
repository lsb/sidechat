# Acrostic decoding: search strategies (and the road to them)

How we decode text whose lines spell a secret word, and the long road of
dead-ends that taught us what actually matters. Two things shipped from this:
a content-shape fix (markdown suppression, commit `a9772ea`) and a prose-only
search (`src/crossingSearch.js`). The honest headline: **greedy is a strong
baseline, and search only beats it when the *objective* is right — not when you
search harder or in a different place.**

## The problem

Each output line must start with the next letter of the secret (`grammar.js`
compiles this; `logits.js` masks every token that would break it). Two render
modes:

- **list** — ` * ` bullet + forced letter + item, one per line.
- **prose** — forced letter + up to `maxLine` chars + newline.

The forced first letter of each line is a **cliff**: where the constraint
fights what the model wants to say. A second, sneakier cliff is the **maxLine
wall** — the forced newline when a line hits the character cap.

## Baseline: grammar-masked greedy

One `generator()` call + `GrammarLogitsProcessor`, streamed token-by-token. It's
good. Every search idea below had to clear "better than greedy," and most didn't.

## The journey (what we tried, in order)

1. **Per-line openings lookahead.** Try the top-N grammar-legal openings per
   line, roll each line out, score by whole-line length-normalised log-prob + a
   one-token peek at the next forced letter, keep the best. Result: **≈ greedy**,
   and slower (searches every line).

2. **Rewind-reflow** (vary the line *ending*). Rewind R tokens at the cliff, try
   M endings, score the same way. Result: a **desync bug** — a candidate whose
   first token was the line-ending newline rolled *past* it into the next
   acrostic line, merging two lines (`" * ural\n * bral\n"`) and cascading into
   gobbledygook. Even with that fixed, it only hurt. Removed.

3. **Surprisal-triggered lookahead.** Only search where the model is *surprised*
   by the forced letter — `surprise = −log p_best_legal` (how little probability
   the model's own distribution puts on the best grammar-legal token). Greedy on
   easy cliffs, search on hard ones; ~4–12% of cliffs trigger on lists, all of
   them on prose. Result: **≈ greedy, slightly worse on prose.** But it produced
   the key realisation:

   > We kept changing *where* we search (openings vs endings) and *whether* to
   > gate it (surprise), but never *what we optimise*. Every variant scored
   > whole-line average log-prob + a one-token peek. And that objective is the
   > problem: it prefers lines that run to the maxLine wall (the forced newline
   > there is "free" — the only legal token, so log-prob ≈ 0), and its peek only
   > checks whether the forced letter is *cheap*, not whether the next line
   > *reads well*.

4. **Markdown suppression** (a system-prompt change, not a search change — the
   actual prose win). Asked "what is the history of X", the model answers in
   markdown (`1. **Ancient Origins**…`), and the per-letter line-chopper shreds
   it into `1. **Anc` / `uve**` fragments. Telling it to write plain prose (and,
   in list mode, a plain bulleted list — `systemPrompt` is mode-tailored in
   `main.js`) fixed prose **under plain greedy**. Decoder-agnostic; shipped
   alone as `a9772ea`.

5. **Local-crossing objective** (the search win, prose only). Change *only* the
   objective, exactly as the realisation in (3) demanded:
   - Score a **short fixed window straddling the crossing** — the last `k`
     content tokens before the break + the forced letter + the next `j` tokens.
     Length-neutral, so no wall bias.
   - **Never score the structural newline** — that's what removes the
     run-to-the-wall preference.
   - **Look `j` tokens past the forced letter** (does the next line *continue*?),
     not just at it.
   - Make the **break point a search variable**, snapped to word boundaries:
     generate the line greedily, then consider ending it 0..R tokens earlier.

   Result on "history of the potato": smooth prose where forced letters become
   the *natural next word* — `…it was` → **used** ("it was used by"), `…poor
   soil` → **led** ("poor soil led to"). The line breaks stop mattering to the
   meaning. It deliberately moved 4/6 breaks. First search variant that's
   plausibly *better* than greedy, not just equal.

   Two bugs surfaced and were fixed: (a) trimming could strip the forced letter
   (in list mode it sits after the ` * ` bullet) → empty-line cascade → added a
   **grammar-validity guard** (reject any break whose trimmed line is no longer
   grammar-legal); (b) lists have no flow across breaks, so optimising the
   crossing just chops items into fragments (`* ural`, `* br`) → **gated to
   prose only**; list mode stays plain greedy.

## The final algorithm (`src/crossingSearch.js`)

```
committed = ""
for each secret letter (line i):
    generate line i greedily (grammar-masked) → text, token ids, per-token logprobs
    if last line or list mode: commit text; continue        # prose-only, greedy floor

    lineStartState = grammar state at start of line i
    candidates = []
    for r in 0..R:                                          # trim r tokens from the end
        if r>0 and the first trimmed token doesn't start a new word: skip
        brokeLine = (line minus last r tokens) + "\n"
        if grammar.advance(lineStartState, brokeLine) == -1: skip   # must keep the forced letter
        before = last k content logprobs of brokeLine (newline excluded)
        after  = rollout of the next line's [forced letter + j tokens] after brokeLine
        candidates.push({ r, brokeLine, score: mean(before ++ after) })
    commit the highest-scoring candidate (r=0 = greedy is always a candidate)
```

Knobs (UI): `k` (window before), `j` (window after), `R` (max trim). `R=0` is
exactly greedy. Public-API only — each line/rollout is a fresh `generator()`
continuation of the chat-templated prompt + committed text fed back as a raw
string; no KV-cache surgery. Committed text is always a valid token-prefix of a
grammar-legal line, so there's no gobbledygook risk even if a score is slightly
mis-estimated.

## Core lessons

- **Greedy is the floor and it's high.** Every "search harder" idea matched or
  lost to it until the objective changed.
- **Masking distorts the objective.** Renormalising over legal tokens isn't the
  model-conditioned-on-grammar distribution (Grammar-Aligned Decoding). Whole-line
  average log-prob then prefers wall-lines and degenerate repetition (the
  likelihood trap; Holtzman et al.).
- **The lever was never where/whether to search — it was what to optimise.** A
  short, newline-free, look-past-the-constraint window is the whole trick.
- **Two independent wins, both prose:** content shape (markdown suppression) and
  the local-crossing objective. Lists want plain greedy.

## Literature this rests on

- **AdaDec — Uncertainty-Guided Adaptive Decoding** (2025): entropy-gated
  lookahead; spend search only where the model is surprised. <https://arxiv.org/html/2506.08980v1>
- **NeuroLogic A\*esque Decoding** (Lu et al., NAACL 2022): score partial decodes
  by an A\*-style estimate of future constraint cost. <https://aclanthology.org/2022.naacl-main.57/>
- **Grammar-Aligned Decoding / ASAp** (Park et al., NeurIPS 2024): masking
  distorts the distribution; reweight toward the grammar-conditioned one.
- **The Curious Case of Neural Text Degeneration** (Holtzman et al.): why
  maximising probability yields degenerate text.

## Resolved along the way

- **Streaming (done).** The per-line search can't reveal a token the break might
  trim, but the first `n−R` tokens of a line are guaranteed kept. So we stream
  each line token-by-token *lagged by R* (the trim zone) via a `TextStreamer`
  whose chunks `main.js` buffers; `onLine` then commits the authoritative
  (possibly-trimmed) line and the tail fills in — no retraction. Inter-line
  pauses (the crossing rollouts) remain inherent to search.
- **Acrostic casing / stealth (done).** The capitalised forced letters weren't
  the model's style — greedy keeps them lowercase (it breaks mid-word at the
  wall, so continuations are clearly mid-sentence), while the crossing search's
  *clean* breaks land at clause/sentence boundaries, after which the model
  capitalises (its "new line → capital" prior) — even after a comma
  (`underbrush, Surprised`), which is wrong AND makes the hidden word legible
  down the margin (`SUBTLE`). Fix: at a line's first token, if the previous line
  didn't end in `.?!`, force the forced letter **lowercase** (the grammar already
  allows it; we just mask the uppercase variant when a lowercase one exists).
  Result: `Subtle`, not `SUBTLE` — the acrostic hides in flowing prose. This is
  the steganacrostics goal, so it's on by default in the crossing path.

## Open questions / next

- **Breadth + tuning.** Does this hold over ~20 prompts × several secret words,
  and what `(k, j, R)` / prompt config wins most often? Wants a generation-eval
  harness like the classifier eval in `eval.js` — the hard part is an automatic
  quality metric beyond "acrostic correct". (Trying prompts by hand first.)
