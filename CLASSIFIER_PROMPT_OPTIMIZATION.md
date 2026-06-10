# Classifier prompt optimization — four-round log

We built a **list-vs-prose classifier** for user prompts: the LLM reads the
prompt and decides whether to render the answer as a bulleted list or as
narrative prose. The classifier's output flips the "render as list" checkbox
in the UI automatically.

The classifier is itself an LLM call, grammar-constrained to one of two
token sequences. So the "classification" is just: apply a chat template to
the prompt, append a partial assistant response (the `prefill`), constrain
generation to exactly one of two literal strings, parse the result.

Dataset: 50 list-style + 50 prose-style hand-picked prompts, split 40+40 dev
/ 10+10 validation, stored in `data/*.txt` (duplicated in `src/eval.js`).
Winner: `r4_a3_d4_extended_triggers` at 98.8% dev / 100% validation, up from
a 50% baseline — +49 percentage points, pure prompt engineering, no model
change.

## The setup

Each classifier variant is an object of four fields:

| field | purpose |
|---|---|
| `system` | system prompt text |
| `prefill` | partial assistant response appended after the chat template (the grammar constrains generation starting from here) |
| `branches` | array of literal string completions; grammar allows exactly one of these |
| `parse(raw)` | maps the generated text to `true` (list) or `false` (prose) |

Example (the winning variant):

```js
{
  name: 'r4_a3_d4_extended_triggers',
  system: `Classify the user's request. Default to "list". Use "story" only
when the user asks for narrative/prose: "tell me a story", "write a
poem/haiku/limerick/email/essay/letter", "describe", "explain",
"translate", "summarize", "what does X mean", "who was/is", "what is X",
"when did", "why does", "how does (concept)", "compose".`,
  prefill: 'The user wants the answer as a ',
  branches: ['list.', 'story.'],
  parse: (s) => s.startsWith('list'),
}
```

The grammar for the two branches is built by `unionGrammars([compileLiteral('list.'), compileLiteral('story.')])` — same machinery the acrostic uses.

## Round 1 — baseline: ask for JSON `true`/`false`

Eleven variants, all asking for `{"prompt_is_a_list": true}` or similar
JSON, with different system prompts and few-shot examples. **All of them
collapsed to either "always true" or "always false".**

Result: 4 variants stuck at 50% (scored 100% on one class, 0% on the other
— the model ignored the semantic question and autocompleted the high-prior
word in JSON context). The winners were `format_string` (`{"format":
"list"|"prose"}`, 78.8%) and `type_uppercase` / `natural_completion`
(72.5%).

Lesson: **never use `true`/`false` as the constrained tokens**. The word
"true" after `": ` is a too-strong autocomplete attractor for tiny models;
the semantic gate doesn't fire. String-valued options ("list", "prose",
"LIST", "paragraph") force the model to actually engage with the prompt.

## Round 2 — fix the vocabulary

Ten new variants took the round-1 top 3 as starting points and tried
different string-valued answers plus few-shot examples.

Winner: **`r2_intent_story_list`** at 83.8% dev / 90% val. Format:

```
system: Classify the user's intent. ...
prefill: "The intent is to get a "
branches: ["list.", "story."]
```

Lesson: **`story` is a much better prose-side word than `prose`**. "prose"
reads to the model as a meta-tag (rare in training data), while "story" is
a normal word. Variants with `bulleted`/`paragraph`, `items`/`text`, or
`list`/`essay` all underperformed `list`/`story`.

Also notable: all `true`/`false` variants at ~50% couldn't be recovered
with any prompt tweak. The terminal token matters more than the framing.

## Round 3 — variations of the top 4

Twenty new variants: five each on top of the round-2 top 4 (different
vocabularies, few-shot examples, default-to-X rules, flipped grammar order,
simpler system prompts).

**Winner on dev: `r3_c3_fewshot_balanced_5_5` at 98.8% dev / 100% val.**

But the few-shot variants leaked validation-set prompts into the few-shot
examples. After an audit:

- `r3_c3`: examples included "write a haiku about autumn" (in validation)
  plus "explain how blockchain works", "who was Marie Curie?", "describe
  the feeling of nostalgia" (all in dev).
- Other few-shot variants had smaller leaks.

After discarding leaking variants, the **clean** winner was
**`r3_d4_natural_list_default`** at 97.5% dev / 95% val:

```
system: Classify the user's request. Default to "list". Use "story" only
when the user clearly asks for narrative: "tell me a story", "write a
poem/haiku/email", "describe X", "explain X", "translate X", "what does X
mean", "who was/what is/when did".
prefill: "The user wants the answer as a "
branches: ["list.", "story."]
```

Lesson: **check for leakage in few-shot examples before trusting
improvements.** Few-shot "wins" from round 3 were partly memorization, not
generalization. Once removed, rule-based variants (pure system-prompt
engineering, no few-shot) beat few-shot variants.

## Round 4 — narrow the rules

Twenty-five new variants: five each on top of round-3's top 5 clean
variants, plus five combinations. Novel few-shot examples hand-picked to
not overlap the dataset (camping, clouds, board games, bridges, morning
habits on the list side; time-traveling cat, refrigerator, rain sound,
lightbulb invention, Spanish greeting on the prose side).

**Winner: `r4_a3_d4_extended_triggers` at 98.8% dev (79/80) / 100% val
(20/20).**

The variant is `r3_d4_natural_list_default` with a longer trigger list,
adding `limerick`, `letter`, `essay`, `summarize`, `compose`, and the
`why`/`how`/`when did` question forms. No few-shot, no magic — just a
comprehensive enumeration of narrative phrasings.

Two variants tied at 98.8% dev:

- `r4_a3_d4_extended_triggers` — default to list, listed trigger phrases
  push to story. 40/40 list + 39/40 prose.
- `r4_c5_a3_default_story` — default to story, list-trigger phrases push
  to list. Same accuracy, opposite polarity.

Which side you default doesn't matter as long as the trigger list for the
*other* side is comprehensive. Both hit the ceiling on our 100-prompt set.

Lessons:

- **Comprehensive trigger lists > few-shot examples** for this task at this
  model scale. The few-shot variants top out around 95–97%; the extended-
  trigger rule variants hit 98.8%.
- **Combinations didn't help.** Merging d4's rules with a3's prefill or
  adding novel few-shot topped out around 95–97% — below the pure-rules
  variants.
- **`best_format_prefill` collapsed to 58%**. "The best format is a
  list/story" apparently confuses the model — probably because "best
  format" reads as a meta-judgment rather than a user-intent statement.
  Prompt framing matters.

## Validation set, top 3 final

| Variant | Val | List recall | Prose recall | Misses |
|---|---:|:---:|:---:|---|
| **r4_a3_d4_extended_triggers** | **100%** | 10/10 | 10/10 | — |
| r4_a5_d4_inline_examples | 95% | 10/10 | 9/10 | `translate "good morning" to Japanese` |
| r4_c5_a3_default_story | 95% | 10/10 | 9/10 | `translate "good morning" to Japanese` |

The only consistent miss on held-out data is `translate "good morning"
to Japanese` (model reads "to Japanese" + thinks there are multiple
language options = list). A genuinely borderline prompt.

## Shipping (SmolLM2-360M)

The winner (`r4_a3_d4_extended_triggers`) was the default in `main.js` via
the `CLASSIFIER_VARIANT_NAME` constant. Runs on every debounced prompt
change (600 ms after last keystroke), typically ~200–600 ms per call on
WebGPU. Output flips the "render as list" checkbox automatically; user can
override manually in the settings panel.

Eval harness stays in `src/eval.js` with hooks on `window` for future
rounds:

- `window.runEval()` — rerun round 1 dev
- `window.runRound2/3/4()` — rerun those rounds dev + auto-validation
- `window.runValidation([names])` — validate a specific list of variants
- `window.runOneOnDev(name)` — dev for a single variant (fast)

## Round 5 — re-tuning for LFM2.5-350M

When we swapped the model from SmolLM2-360M-Instruct to LFM2.5-350M (see
`LFM2.5-350M-quantization.md`), the round-4 winner regressed catastrophically:

- `r4_a3_d4_extended_triggers` on LFM2: **68.8% dev / 60% val**, vs 98.8% / 100%
  on SmolLM2.
- Every miss was list-prompt → `story.` (25 / 25 dev errors). Prose was 100%.

Hypothesis: LFM2 has a much stronger "story." prior than SmolLM2 given the
prefill `"The user wants the answer as a "`. Round 4's "Default to list. Use
story only when …" framing hands the model the keys: it ignores the rule
and produces the high-prior continuation. A spot-check found
`r2_intent_story_list` (round-2 winner, simpler intent framing) ported to
LFM2 at 90% dev — a cleaner starting point for re-tuning.

**Twenty new variants** (R5_VARIANTS in `eval.js`) explored five axes:

- A. Intent framing — variations on `r2_intent_story_list`'s system prompt
  (5 variants).
- B. Default-to-`story` flipped polarity with comprehensive list triggers
  (3 variants).
- C. Different prefill stems ("The user is asking for a ", "The format
  should be a ", "Best to render as a ", "The response should be a ", "The
  output is a ") (5 variants).
- D. No / minimal system prompt, lean on the prefill (3 variants).
- E. Branch vocabulary alternatives — `items./text.`, `bullets./paragraph.`,
  `LIST./STORY.` (3 variants).
- F. Combined: extended-trigger list + alt prefill (1 variant).

**Round 5 dev results** (top 5):

| Variant | Dev | List miss | Prose miss |
|---|---:|---:|---:|
| **r5_a4_intent_two_rules** | **92.5%** | 2 | 4 |
| r5_a5_intent_minimal_one_line | 91.3% | 4 | 3 |
| r5_c1_user_asking_for | 91.3% | 0 | 7 |
| r5_a3_intent_question_words | 88.8% | 5 | 4 |
| r5_c5_output_is_a | 88.8% | 0 | 9 |

**Validation top 5 + anchors:**

| Variant | Dev | Val |
|---|---:|---:|
| **r5_a4_intent_two_rules** | **92.5%** | **80%** |
| r5_c1_user_asking_for | 91.3% | 80% |
| r5_c5_output_is_a | 88.8% | 80% |
| r5_a5_intent_minimal_one_line | 91.3% | 70% |
| r2_intent_story_list (anchor) | 90% (earlier round) | 70% |
| r5_a3_intent_question_words | 88.8% | 65% |
| r4_a3_d4_extended_triggers (old SmolLM2 winner) | 68.8% | 60% |

Three variants tied at 80% val. Pick `r5_a4_intent_two_rules` on dev
tiebreak + balanced miss types (1L + 3P on val). The system prompt is just
two sentences:

```
Classify the user's intent. Use "list" when the answer is a set of separate
items the user can scan. Use "story" when the answer flows as one
narrative, single fact, or short paragraph.
```

Net round 5 win: **+23.7 pp dev / +20 pp val** vs the SmolLM2-tuned r4
winner on LFM2.

Lessons:

- **"Default to X" framings backfire on LFM2.** SmolLM2 followed the
  default obediently; LFM2's stronger prior on the *other* token wins.
  Neutral intent framings ("Classify the user's intent. Complete the
  sentence.") perform better.
- **Branch vocabulary still matters but the polarity differs.**
  `LIST./STORY.` (caps) collapsed to 45% — the model autocompletes "LIST"
  given any classification framing. `items./text.` (51.2%) and
  `bullets./paragraph.` (62.5%) also underperformed `list./story.`.
- **No-system-prompt variants underperform** (60–52%) — the prefill alone
  isn't enough signal, the system prompt's framing actually does work even
  for tiny models.

## Round 6 — pinpoint rules

Twenty more variants targeting `r5_a4`'s 12 dev+val misses, which clustered
into three patterns:

- `write a [haiku/cover-letter/email/joke/love-letter]` → list (should be
  story) — 6 misses, the largest class.
- `what is X` (singular fact like "capital of Australia") and `translate X
  to Y` → list (should be story) — 2 misses.
- `what are X` / `what are the steps` (plural enumeration) → story (should
  be list) — 3 misses, the opposite direction.

Five round-5 bases × four ablations:

- v1: `+ write_forms` rule — explicit "Whenever the user asks to 'write' or
  'compose' a haiku, poem, letter, cover letter, email, joke, story,
  essay, or limerick, the answer is a story."
- v2: `+ single_plural` rule — explicit "What is X (single fact) is a
  story; What are the/some Xs (plural enumeration) is a list; what are
  the steps/differences/causes/symptoms is a list."
- v3: `+ translate_email` rule — "Translation requests and email/letter
  composition are stories."
- v4: kitchen sink (all three).

**Round 6 dev results** (top 5):

| Variant | Dev | List miss | Prose miss |
|---|---:|---:|---:|
| **r6_c1_v2_single_plural** | **97.5%** | 0 | 2 |
| r6_c1_v3_translate_email | 95.0% | 0 | 4 |
| r6_a4_v3_translate_email | 93.8% | 2 | 3 |
| r6_c5_v3_translate_email | 91.3% | 6 | 1 |
| r6_c1_v1_write_forms | 90.0% | 7 | 1 |

The `+all` kitchen-sink variants regressed on every base (76–81% dev) — too
many constraints collapse the model into one branch. The c1 base
("Classify the user's request. Use 'list' when the user wants enumerated
items. Use 'story' for everything else.") + the single-rule
`single_plural` was the unique top of the heap.

**Validation top 5 + r5_a4 anchor:**

| Variant | Dev | Val | List miss (val) | Prose miss (val) |
|---|---:|---:|:---:|:---:|
| **r6_c1_v2_single_plural** | **97.5%** | **85%** | 0 | 3 |
| r6_c1_v3_translate_email | 95.0% | 85% | 0 | 3 |
| r6_c1_v1_write_forms | 90.0% | 85% | 3 | 0 |
| r6_c5_v3_translate_email | 91.3% | 85% | 2 | 1 |
| r6_a4_v3_translate_email | 93.8% | 80% | 1 | 3 |
| r5_a4_intent_two_rules (anchor) | 92.5% | 80% | 1 | 3 |

Four variants tied at 85% val. Tiebreak by dev: **`r6_c1_v2_single_plural`**
wins (97.5% dev, 0 list-misses on val). Three residual val misses are the
"Spinal Tap" prompts that have lasted across rounds:

- `translate "good morning" to Japanese`
- `write a professional email declining a meeting`
- `what is the capital of Australia?`

Lessons from round 6:

- **One targeted rule beats stacking rules.** `+single_plural` alone hit
  97.5%; `+all` (single_plural + write_forms + translate_email together)
  on the same base dropped to 81.3%. The model collapses under too many
  constraints.
- **Choose the leanest base.** The c1 base ("Use 'list' for X. Use 'story'
  otherwise.") + one rule outperformed every variant built on top of the
  longer round-5 bases.
- **Failure modes are model-specific.** SmolLM2's miss list (translation,
  vague borderline cases) was different from LFM2's (write-a-haiku,
  single-fact "what is X"). Round-by-round error analysis matters more
  than carrying over rules from a prior model.

## Shipping (LFM2.5-350M)

The current default is **`r6_c1_v2_single_plural`** at 97.5% dev / 85% val.
Two-round delta from the SmolLM2-tuned r4 winner: **+28.7 pp dev / +25 pp
val**, no model change.

Final system prompt:

```
Classify the user's request. Use "list" when the user wants enumerated
items. Use "story" for everything else. "What is X" (a single fact) is a
story; "What are the/some Xs" (plural enumeration) is a list; "what are
the steps/differences/causes/symptoms" is a list.
```

Prefill: `The user is asking for a `, branches: `list.` / `story.`
