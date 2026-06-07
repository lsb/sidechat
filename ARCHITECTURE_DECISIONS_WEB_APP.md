# Steganacrostics — web-app architecture

A browser app that generates LLM text constrained to spell an **acrostic**
secret (first letter of each line spells the secret), optionally wrapped in
bullet-list format. Runs entirely client-side; the model is SmolLM2-360M-
Instruct quantized to q4f16, loaded by transformers.js with the WebGPU
backend.

## Tech stack

| Piece | Choice | Why |
|---|---|---|
| Build tool | Vite 6 | Minimal config; dev server with HMR; clean static build |
| LLM runtime | `@huggingface/transformers` 4.1 (transformers.js) | Stable API, browser-native, supports WebGPU backend |
| Backend | WebGPU (`device: 'webgpu'`) | 19× faster than WASM for this model; TTFT ~0.2s vs ~4.8s |
| Model | SmolLM2-360M-Instruct q4f16 | Small enough to download (~195 MB), good enough quality for demo; custom-quantized from fp32 source (see `SmolLM2-360M-quantization.md`) |
| Grammar engine | Hand-rolled DFA in `src/grammar.js` | Full control, <100 lines, no dependencies, tailored to the narrow language we need |
| Logits masking | Custom `LogitsProcessor` subclass | transformers.js supports this via `LogitsProcessorList` in generation options |
| Eval harness | Pure JS, browser-side | 100 labelled prompts, 37 classifier variants, runs in ~3 min |

## Layout

```
.
├── index.html                    -- UI shell + inline CSS
├── src/
│   ├── main.js                   -- boot, DOM wiring, generate loop, eval hooks
│   ├── grammar.js                -- acrostic → atoms → packed-int DFA; unionGrammars
│   └── logits.js                 -- GrammarLogitsProcessor (mask dead tokens per step)
├── src/eval.js                   -- prompts, variants, runVariantOn / runAllOnDev
├── data/
│   ├── list_dev.txt              -- 40 list-style prompts
│   ├── list_validation.txt       -- 10 list-style prompts (held out)
│   ├── prose_dev.txt             -- 40 prose-style prompts
│   └── prose_validation.txt      -- 10 prose-style prompts (held out)
├── public/
│   ├── (ort helpers served at /ort/*, see vite.config.js)
│   └── local-models/
│       └── HuggingFaceTB/SmolLM2-360M-Instruct/
│           ├── config.json, tokenizer.json, …
│           └── onnx/
│               ├── model_q4f16.onnx          -- graph + small initializers (~0.3 MB)
│               ├── model_q4f16.onnx_data     -- chunk 0, 49.7 MB (includes shared embedding blob)
│               ├── model_q4f16.onnx_data_1   -- chunk 1, 49.6 MB
│               ├── model_q4f16.onnx_data_2   -- chunk 2, 49.6 MB
│               └── model_q4f16.onnx_data_3   -- chunk 3, 46.2 MB
├── vite.config.js                -- inline plugin serves ORT wasm helpers
├── SmolLM2-360M-quantization.md  -- quant pipeline + attempts log
└── ARCHITECTURE_DECISIONS_WEB_APP.md  -- this file
```

## Runtime pipeline, end to end

1. **Boot** (`main.js`):
   - ORT and local-model paths are computed **at runtime** from
     `window.location.href` so the same build works whether served at `/`
     or a subpath like `/dist/`:
     ```js
     const pageBasePath = new URL('./', window.location.href).pathname;
     env.backends.onnx.wasm.wasmPaths = pageBasePath + 'ort/';
     env.localModelPath = pageBasePath + 'local-models/';
     ```
   - `env.backends.onnx.wasm.proxy` is **left off**. With WebGPU we don't
     need it, and turning it on broke static deploys: the proxy worker's
     URL is constructed from bundler metadata that vite's dev server
     populates but a bare static server doesn't.
   - `env.allowRemoteModels = false`, `env.allowLocalModels = true`,
     `env.localModelPath = <computed>` — we never fall back to the 273 MB
     HF reference; our 195 MB custom quant is both smaller and the one
     we have eval data for.
   - `pipeline('text-generation', 'HuggingFaceTB/SmolLM2-360M-Instruct',
     { dtype: 'q4f16', device: 'webgpu', use_external_data_format: 4 })`
     — the last option tells transformers.js to mount all 4
     `.onnx_data[_N]` chunks.
   - Builds `tokenText[i] = tokenizer.decode([i])` — a one-shot table so
     the grammar processor doesn't pay `decode()` cost per step per
     candidate.
   - Scans token texts for ones containing `'\n'` (logged to console;
     informational — 129 of 49,152 on SmolLM's tokenizer).

2. **Compile grammar** (`grammar.js`):
   - User enters a secret (e.g., `"HI BOB"`) and toggles list mode.
   - `compileAcrostic(secret, {listPrefix: ' * ', maxLine: 80,
     caseInsensitive: true, firstLinePrefix})` builds a list of **atoms**:
     - `{kind: 'lit', allowed: Set<string>}` — match exactly one char from
       `allowed`
     - `{kind: 'body', max: N}` — match 0..N non-newline chars
   - Secret letters become case-insensitive lit atoms
     (`{allowed: {'H', 'h'}}`). A space in the secret maps to a fixed
     punctuation class `{'.', ',', ';', ':', '!', '?', '-', '(', ')', …}`
     — so "HI BOB" becomes H-I-{punct}-B-O-B line starts.
   - List mode with a bulleted prefix is compiled as the **union** of two
     DFAs (one with the first `* ` prefix required, one with it optional)
     via `unionGrammars([...])`. Each branch gets an independent int state;
     the union is alive if any branch is alive.
   - The pattern is concatenation-only with body atoms always followed by
     newline literals, so the NFA is deterministic: state can be packed
     into one integer `atomIdx * stride + count` (stride = `maxLine + 2`).
   - Exposes `{initial, advance(state, str) → state|-1, accepts(state),
     stateCount}`.

3. **Install logits processor** (`logits.js`):
   - `GrammarLogitsProcessor extends LogitsProcessor`:
     - `_call(input_ids, logits)`:
       1. Decode the generated suffix with `tokenizer.decode(...)`.
       2. Advance the grammar from `initial` through that text.
       3. For every candidate token id `i` in vocabulary: if
          `grammar.advance(state, tokenText[i])` is dead (-1), set
          `logits[0].data[i] = -Infinity`.
       4. Special-case EOS ids: live only if the grammar is currently in
          an accept state.
   - Tracks detailed per-step timing (`tDec`, `tPref`, `tScan`,
     `priorModelMs`, `seqLen`) so we can attribute slowness correctly.

4. **Classifier** — list vs prose auto-detection:
   - Debounced on `$prompt` input changes (600 ms).
   - Winner variant `r4_a3_d4_extended_triggers` runs via
     `classifyWithVariant(ctx, variant, userPrompt)` in `eval.js`:
     - Apply chat template to `[system, user]` messages with
       `add_generation_prompt: true`, then append the variant's
       `prefill` (e.g., `"The user wants the answer as a "`).
     - Constrain generation to one of `variant.branches`
       (e.g., `["list.", "story."]`) via a small union-grammar
       (`unionGrammars(branches.map(compileLiteral))`).
     - `max_new_tokens: 16`, `do_sample: false`, `return_full_text: false`.
     - Parse the generated text with `variant.parse` → boolean.
   - Flips the `list mode` checkbox if the classifier's answer differs.
   - Guarded by a global `busy` flag so the classifier and the main
     generator don't collide.

5. **Main generation** — acrostic:
   - On Generate click: `compileAcrostic(secret, ...)`, build a
     `GrammarLogitsProcessor`, push into `LogitsProcessorList`.
   - `generator(messages, {max_new_tokens: 400, do_sample: false,
     logits_processor, streamer: new TextStreamer(..., skip_prompt: true,
     skip_special_tokens: true, callback_function})`.
   - `TextStreamer`'s callback updates the UI per token; metrics display
     TTFT, tok/s, per-step processor breakdown.

## Eval harness (`eval.js`)

- `LIST_PROMPTS` (50) and `PROSE_PROMPTS` (50) — first 10 of each held out
  as validation; rest used as dev. Kept in sync with `data/*.txt`.
- 37 classifier variants across 4 rounds, all defined in
  `eval.js::VARIANTS`. Each variant carries its own `system` prompt,
  `prefill` string, grammar `branches`, and `parse(raw) → boolean`.
- `classify(ctx, variant, userPrompt)` runs a single classification:
  applies chat template manually, appends prefill, constrains with
  `unionGrammars(branches.map(compileLiteral))`, returns
  `{prediction, raw}`.
- `runVariantOn(ctx, variant, labelledPrompts)` / `runVariantsOnDev` /
  `runOnValidation` — orchestrate the sweep. Exposed on `window` as
  `runEval`, `runRound2/3/4`, `runValidation`, `runOneOnDev` for quick
  console-driven iteration.
- Winning classifier: `r4_a3_d4_extended_triggers` — "default to list,
  use story only for these narrative phrasings" rule, no few-shot, no
  leakage. 98.8% dev / 100% validation.

## Grammar-constrained-decoding details

The token-mask-via-DFA is the standard technique (XGrammar, Outlines,
llguidance, et al.). Our specific choices:

- **Per-token decoding** via `tokenizer.decode([i])` once at boot.
  Approximation: BPE's `decode(seq + [i])` isn't always
  `decode(seq) + decode([i])`, but SmolLM's BPE tokenizer is well-behaved
  for our prompts (verified by scanning the 129/49k tokens containing
  `\n`; no surprise merges).
- **Deterministic DFA** (no subset construction needed for pure
  concatenation with body-then-newline structure).
- **Packed-int state** (~4k states, ~1 ns per `advance` step) instead of
  Map-of-objects NFA states.
- **No per-state allow-mask caching** — the vocab scan is ~3 ms/step which
  is <1% of wall clock on WebGPU; caching would be XGrammar territory.

## Classifier prompt engineering

Full details in `CLASSIFIER_PROMPT_OPTIMIZATION.md` — four rounds, 67
variants total, +49 percentage points over baseline (50% → 98.8% dev,
100% validation). Short punch line:

- **Do not** ask a 360M model to answer `{"key": true|false}` — it
  autocompletes "true" almost uniformly (baseline: 50% accuracy).
- **Do** ask it to fill a natural sentence with a meaningful terminal
  word: `"The user wants the answer as a list./story."` clears 98%.
- **Do** include an explicit list of narrative-triggering phrasings in
  the system prompt (tell me a story, write a poem, describe, explain,
  translate, what does X mean, who was/is, when did, …).
- **`story` beats `prose` and `paragraph`** as the prose-side vocabulary.
  "story" is an everyday word; "prose" reads as a meta-tag.
- **Few-shot examples didn't help beyond the rules** at 360M scale and
  are easy to leak validation set prompts into. Pure rule variants hit
  the ceiling cleanly.

## ORT helpers via vite plugin

ORT web fetches WASM helper files at runtime using
`env.backends.onnx.wasm.wasmPaths`. We serve them from `<base>/ort/*` using
an inline vite plugin in `vite.config.js` that:

- During dev: intercepts `/ort/*` requests and streams the matching file
  from `node_modules/onnxruntime-web/dist/`.
- During build: emits the matching files to `dist/ort/*`.
- After build (`closeBundle`): removes duplicate copies that vite's default
  asset pipeline emitted into `dist/assets/` from `new URL(…, import.meta.url)`
  references inside onnxruntime-web. Those are never fetched at runtime
  (ORT uses `wasmPaths`, not the hashed URLs) — ~22 MB of dead weight
  otherwise.

We emit **only the asyncify variant** (`ort-wasm-simd-threaded.asyncify.{mjs,wasm}`).
The onnxruntime-web **webgpu bundle** only references asyncify, so the
three other variants (base, jsep, jspi) that ORT ships (~51 MB combined)
are unused dead weight for this build and excluded by the plugin's
`MATCH` regex. If you ever switch to a non-WebGPU backend or a different
bundle, widen the regex to include the variant(s) you need.

## Local-model hosting

The custom-quantized model lives under `public/local-models/...`. Vite
serves everything under `public/` at the site root during dev, and copies
it into `dist/` at build time. At runtime, transformers.js fetches:

- `…/onnx/model_q4f16.onnx` — graph + small initializers
- `…/onnx/model_q4f16.onnx_data`, `_data_1`, `_data_2`, `_data_3` — four
  external-data chunks carrying the weights.

The chunks are split to target **4 files, each under 50 MB (decimal)**
for two reasons: parallel fetches by the browser, and fitting within
per-file size limits on some static hosts. The `ChunkWriter` in
`model_work/quantize.py` does a two-pass run: it first sums the total
external-data bytes, then picks
`chunk_cap = ceil(total / 4) × 1.02` (hard-capped at 50 MB) so the last
chunk doesn't shrink into a tiny straggler.

Naming must match transformers.js's expectations: the file for chunk `i`
is `<base>.onnx_data` for `i=0` and `<base>.onnx_data_<i>` for `i > 0`
(underscore, not dot). Initializer protos' `external_data.location`
fields point to these same names so ORT can resolve them after
transformers.js mounts them. The shared 23.59 MB embedding blob is
written exactly once at the start of chunk 0, and both the
`GatherBlockQuantized` uint4 view and the `MatMulNBits` uint8 view
reference that same `(location, offset, length)` range.

The pipeline call must pass `use_external_data_format: 4` so
transformers.js fetches all four chunks. Without it, transformers.js
defaults to 0 external chunks and ORT fails with
"Module.MountedFiles is not available".

## Static build

`npm run build` → outputs everything to `dist/`:

- `dist/index.html`, `dist/assets/*.js` (bundled app; JS is ~580 KB, gzips
  to ~170 KB)
- `dist/ort/*` (WASM/JSEP helpers, via the inline plugin)
- `dist/local-models/...` (model files copied from `public/`)

Total ~285 MB. No server logic; everything (tokenizer, model, grammar,
inference, classifier) is client-side. Deploy by copying `dist/` to any
static host.

### Path-portability

`vite.config.js` sets `base: './'` so the built `index.html` references
assets via relative paths (`./assets/…`, `./ort/…`). Combined with the
runtime path computation in `main.js` described above, the same build
works at the site root *and* inside any subdirectory — e.g. both of
these work with no rebuild:

- `http://host:8000/` (python `http.server` run from `dist/`)
- `http://host:8000/dist/` (python `http.server` run from the repo root)

### Cache gotcha

`python -m http.server` sets no `Cache-Control` headers, so browsers
heuristically cache the large `.onnx_data*` files. If you rebuild in
place (same URLs, new content), clear the browser's IndexedDB
`transformers-cache` and the HTTP cache, or add a version query string to
the page. In production, give the data files long max-age and bust them
with content-hashed filenames if you plan to rotate model versions.

### What *not* to do

- Don't re-enable `env.backends.onnx.wasm.proxy = true` — it works on
  vite dev but breaks under static serving (its worker spawn relies on
  bundler-injected module metadata).
- Don't gzip the `.onnx_data*` chunks; they're already dense int4+fp16,
  gzip rarely helps and can delay streaming. Do gzip HTML/JS.
- Don't quantize with `keep_io_types=True` in the fp16 conversion step:
  transformers.js at `dtype: 'q4f16'` feeds fp16 tensors, and if the
  model keeps its graph I/O as fp32 you get runtime
  `Unexpected input data type. Actual: (tensor(float16)) , expected: (tensor(float))`.
  Use `keep_io_types=False` so the model's inputs/outputs match.

## Current known quirks

- **Browser freezes with WASM backend and small grammar work**: solved by
  moving to WebGPU. The logits processor's ~3 ms/step is tiny; the old
  WASM problem was that 124 ms/step (pre-DFA) held the main thread long
  enough to block repaints. DFA + WebGPU together eliminates this.
- **Classifier is a bit slow on first prompt edit**: expected, it's a full
  forward pass with prefill. ~1 s for a typical prompt; not in the hot
  path of the main generation.
- **Tok/s variance across reloads**: normal WebGPU shader-compile caching;
  subsequent runs within the same session are stable (~15-17 tok/s).
