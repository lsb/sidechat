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

## Shipping

The winner (`r4_a3_d4_extended_triggers`) is the default in `main.js` via
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
