import { env, pipeline, TextStreamer, LogitsProcessorList } from '@huggingface/transformers';
import { compileAcrostic, compileLiteral, unionGrammars } from './grammar.js';
import { GrammarLogitsProcessor, buildTokenTextTable } from './logits.js';
import { VARIANTS, classify as classifyWithVariant, runAllOnDev, runOnValidation, variantsWithPrefix, runVariantsOnDev } from './eval.js';

// --- Backend setup ---------------------------------------------------------
// Compute `ort/` and `local-models/` paths relative to the page so the
// same build works whether it's served at the site root (http://host/) or
// inside a subdirectory (http://host/dist/). `new URL('./', location.href)`
// gives us the directory that index.html lives in; we append `ort/` and
// `local-models/` to it and use the pathname so transformers.js / ORT treat
// them as same-origin absolute paths.
const pageBaseURL = new URL('./', window.location.href);
const pageBasePath = pageBaseURL.pathname;

// Serve ORT runtime files (inc. jsep variants used by WebGPU) from <base>/ort/*.
env.backends.onnx.wasm.wasmPaths = pageBasePath + 'ort/';
env.backends.onnx.wasm.numThreads = 1;
// Only matters for the pure-WASM backend; WebGPU already runs GPU work off
// the main thread. We leave this off because (a) with the WebGPU backend we
// don't need it and (b) spinning up the WASM proxy worker fails in plain
// static-file deploys (the worker URL construction relies on bundler
// metadata that vite's dev server provides but a bare http-server does not).
// env.backends.onnx.wasm.proxy = true;

// Always load the model from our locally-hosted quantized copy — smaller
// (195 MB) and known-good via our eval. We never want the remote 273 MB
// HF reference as a fallback, so remote models are disabled outright.
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = pageBasePath + 'local-models/';

// --- DOM -------------------------------------------------------------------
const $status   = document.getElementById('status');
const $output   = document.getElementById('output');
const $metrics  = document.getElementById('metrics');
const $secret   = document.getElementById('secret');
const $listmode = document.getElementById('listmode');
const $maxline  = document.getElementById('maxline');
const $prompt   = document.getElementById('prompt');
const $run      = document.getElementById('run');
const $subtitle = document.getElementById('subtitle');

const SUBTITLE_DOWNLOADING = 'Downloading over 208 MB (one-shot transfer)';
const SUBTITLE_PRETEST     = 'Available. Chat requires one small test. Investigating correctness.';
const SUBTITLE_READY       = 'Available. Chat ready; output satisfies text input commands.';

// Aggregate download progress across all files reported by transformers.js.
// Each progress event gives us loaded/total for a single file; we sum across
// files to compute the overall fraction. `setSubtitleProgress(f)` maps that
// fraction to the subtitle background: fully red at 0, transparent at 1.
const fileBytes = new Map();  // file → { loaded, total }
function setSubtitleProgress(fraction) {
  const redness = Math.max(0, Math.min(1, 1 - fraction));
  $subtitle.style.backgroundColor = `rgba(220, 50, 50, ${redness.toFixed(3)})`;
}
function clearSubtitleProgress() {
  $subtitle.style.backgroundColor = '';
}

const setStatus   = (s) => { $status.textContent = s; };
const appendOutput = (s) => { $output.textContent += s; };
const clearOutput  = () => { $output.textContent = ''; };

// The Generate button doubles as progress indicator while the model is
// loading. `setButton(state)` controls its enabled/label without touching the
// status line (which lives in the settings panel).
const setButton = (label, disabled = false) => {
  $run.textContent = label;
  $run.disabled = disabled;
};

// --- Single-threaded async lock -------------------------------------------
// Inference (classifier, generation, self-test, eval) is serialized through
// `busyLock`. The lock carries a human-readable reason that gets mirrored to
// the Generate button label so the user always sees why the button is
// disabled. `acquire(reason, fn)` throws if someone else holds the lock; use
// `waitFor(reason, fn)` to queue instead. For our event-driven UI (click,
// debounced prompt change, startup self-test), we just drop events that
// arrive while busy — they'll re-fire if they're still relevant.
let _busyReason = 'loading model';  // starts busy until boot() finishes

function _refreshButton() {
  if (_busyReason) {
    setButton(_busyReason === 'loading model'
              ? 'Loading model…'
              : _busyReason + '…', true);
  } else {
    setButton('Generate', false);
  }
}

async function acquireLock(reason, fn) {
  if (_busyReason) return undefined; // dropped
  _busyReason = reason;
  _refreshButton();
  try {
    return await fn();
  } finally {
    _busyReason = null;
    _refreshButton();
  }
}

function isBusy() {
  return _busyReason !== null && _busyReason !== undefined;
}

// Case-insensitive acrostic; in list mode the very first `* ` prefix is
// optional (some models start with a preamble-free letter, others don't).
function buildGrammar(secret, listMode, maxLine) {
  const caseInsensitive = true;
  if (!listMode) {
    return compileAcrostic(secret, { listPrefix: '', maxLine, caseInsensitive });
  }
  const withPrefix    = compileAcrostic(secret, { listPrefix: ' * ', maxLine, caseInsensitive, firstLinePrefix: true });
  const withoutPrefix = compileAcrostic(secret, { listPrefix: ' * ', maxLine, caseInsensitive, firstLinePrefix: false });
  return unionGrammars([withPrefix, withoutPrefix]);
}

// --- Boot: grammar sanity check, then load model --------------------------
async function boot() {
  // _busyReason starts as 'loading model'; button already shows "Loading
  // model…". We update the button label live as the download progresses,
  // but we don't release the lock until the self-test has finished below.
  setStatus('loading LFM2.5-350M @ q4f16 (local MatMulNBits/GatherBlockQuantized build)…');

  const generator = await pipeline(
    'text-generation',
    'LiquidAI/LFM2.5-350M',
    {
      dtype: 'q4f16',
      device: 'webgpu',
      // Our model ships its weights in 6 external-data chunks named
      // model_q4f16.onnx_data, _data_1, … _data_5.
      use_external_data_format: 6,
      progress_callback: (p) => {
        if (p.status === 'progress' && p.file) {
          if ($subtitle.textContent !== SUBTITLE_DOWNLOADING) {
            $subtitle.textContent = SUBTITLE_DOWNLOADING;
          }
          if (typeof p.loaded === 'number' && typeof p.total === 'number' && p.total > 0) {
            fileBytes.set(p.file, { loaded: p.loaded, total: p.total });
            let sumLoaded = 0, sumTotal = 0;
            for (const v of fileBytes.values()) { sumLoaded += v.loaded; sumTotal += v.total; }
            setSubtitleProgress(sumTotal > 0 ? sumLoaded / sumTotal : 0);
          } else if (typeof p.progress === 'number') {
            setSubtitleProgress(p.progress / 100);
          }
          const pct = typeof p.progress === 'number' ? p.progress.toFixed(1) : '?';
          setStatus(`downloading ${p.file}: ${pct}%`);
        } else if (p.status === 'ready') {
          setStatus('model ready.');
          $subtitle.textContent = SUBTITLE_PRETEST;
          clearSubtitleProgress();
        }
      }
    }
  );
  window.__generator = generator;
  $subtitle.textContent = SUBTITLE_PRETEST;
  clearSubtitleProgress();

  const vocabSize = generator.model.config.vocab_size;
  setStatus(`building token→text table for ${vocabSize} tokens…`);
  _busyReason = 'indexing tokens';
  _refreshButton();
  const tTab = performance.now();
  const tokenText = buildTokenTextTable(generator.tokenizer, vocabSize);
  console.log(`tokenText built in ${((performance.now() - tTab) / 1000).toFixed(2)}s`);

  // One-shot scan: which tokens contain a newline, and which are pure "\n"?
  const newlineTokens = [];
  for (let i = 0; i < tokenText.length; i++) {
    if (tokenText[i] && tokenText[i].includes('\n')) newlineTokens.push(i);
  }
  console.log(`[scan] ${newlineTokens.length}/${tokenText.length} tokens contain '\\n'`);
  const sample = newlineTokens.slice(0, 20).map((i) => [i, JSON.stringify(tokenText[i])]);
  console.log('[scan] sample newline tokens:', sample);
  window.__newlineTokens = newlineTokens;
  window.__tokenText = tokenText;

  const eosTokenIds = new Set();
  const addEos = (x) => {
    if (x == null) return;
    if (Array.isArray(x)) x.forEach(addEos);
    else eosTokenIds.add(Number(x));
  };
  addEos(generator.tokenizer.eos_token_id);
  addEos(generator.model.generation_config?.eos_token_id);

  setStatus('model ready — running startup self-test…');
  _busyReason = 'running self-test';
  _refreshButton();

  const ctx = { generator, tokenText, eosTokenIds: [...eosTokenIds] };
  window.__ctx = ctx;

  // Startup smoke test runs under the init lock — Generate stays disabled
  // until it finishes. Failures are surfaced but don't keep the lock held.
  try {
    await runStartupSelfTest(ctx);
  } catch (e) {
    console.error('self-test:', e);
    const body = document.getElementById('self-test-body');
    if (body) body.innerHTML = `<span class="failed">self-test failed: ${e.message}</span>`;
  }

  setStatus('model ready. edit the secret and/or prompt, then click Generate.');
  $subtitle.textContent = SUBTITLE_READY;
  _busyReason = null;
  _refreshButton();

  // Wire up the "Run full eval" button in the settings panel now that ctx
  // is ready. Uses the same acquireLock so it serializes with everything else.
  const $fullEvalBtn = document.getElementById('run-full-eval');
  const $fullEvalBody = document.getElementById('full-eval-body');
  if ($fullEvalBtn) {
    $fullEvalBtn.disabled = false;
    $fullEvalBtn.addEventListener('click', () => {
      acquireLock('running 100-prompt eval', () => runFullEval(ctx, $fullEvalBody))
        .catch((e) => {
          console.error('full eval:', e);
          if ($fullEvalBody) $fullEvalBody.innerHTML = `<span class="failed">eval failed: ${escapeHtml(e.message)}</span>`;
        });
    });
  }

  $run.addEventListener('click', () => {
    acquireLock('generating', () => runGeneration(ctx)).catch((e) => {
      setStatus(`error: ${e.message}`);
      console.error(e);
    });
  });

  // Debounced classifier: whenever the prompt changes, disable Generate,
  // ask the LLM whether the user wants a bulleted list, flip the checkbox,
  // re-enable Generate. If another prompt edit or a Generate click fires
  // while the classifier is running, it's dropped (acquireLock returns
  // undefined); the user can edit again to retry.
  let classifyTimer = null;
  $prompt.addEventListener('input', () => {
    clearTimeout(classifyTimer);
    classifyTimer = setTimeout(() => {
      acquireLock('classifying prompt', () => classifyListness(ctx, $prompt.value))
        .catch((e) => console.error('classifier:', e));
    }, 600);
  });

  // Eval hooks (run from devtools): window.runEval() runs all 11 variants on
  // the 80-prompt dev set, logging progress and a final summary table.
  window.runEval = async () => acquireLock('running eval', async () => {
    console.log('[eval] starting — 11 variants × 80 prompts = 880 calls');
    const tStart = performance.now();
    const summaries = await runAllOnDev(ctx, ({ variantIdx, variantName, variantTotal, done, total, last }) => {
      if (done === total || done % 10 === 0) {
        setStatus(`eval: variant ${variantIdx + 1}/${variantTotal} (${variantName}) · ${done}/${total}`);
      }
    });
    const totalMs = performance.now() - tStart;
    console.log(`[eval] dev done in ${(totalMs / 1000).toFixed(1)}s`);
    console.table(summaries.map((s) => ({
      variant: s.variant,
      accuracy: +(s.accuracy * 100).toFixed(1),
      correct: s.correct,
      total: s.total,
      wall_s: +(s.wallMs / 1000).toFixed(1),
    })));
    window.__evalSummaries = summaries;
    setStatus('eval complete — see console.table for ranking');
    return summaries;
  });

  window.runRound4 = async (topK = 5) => acquireLock('running eval', async () => {
    const variants = variantsWithPrefix('r4_');
    console.log(`[eval] round 4: ${variants.length} variants × 80 prompts`);
    const tStart = performance.now();
    const summaries = await runVariantsOnDev(ctx, variants, ({ variantIdx, variantName, variantTotal, done, total }) => {
      if (done === total || done % 20 === 0) {
        setStatus(`eval r4: variant ${variantIdx + 1}/${variantTotal} (${variantName}) · ${done}/${total}`);
      }
    });
    window.__evalR4Summaries = summaries;
    const totalMs = performance.now() - tStart;
    console.log(`[eval] r4 dev done in ${(totalMs / 1000).toFixed(1)}s`);
    const topR4 = [...summaries].sort((a, b) => b.accuracy - a.accuracy).slice(0, topK).map((s) => s.variant);
    const valSet = Array.from(new Set([...topR4, 'r3_d4_natural_list_default']));
    setStatus(`validating top ${topK} on held-out set…`);
    const valSummaries = await runOnValidation(ctx, valSet);
    window.__evalR4ValSummaries = valSummaries;
    setStatus('r4 eval complete');
    return { dev: summaries, validation: valSummaries };
  });

  window.runRound3 = async (topK = 4) => acquireLock('running eval', async () => {
    const variants = variantsWithPrefix('r3_');
    console.log(`[eval] round 3: ${variants.length} variants × 80 prompts`);
    const tStart = performance.now();
    const summaries = await runVariantsOnDev(ctx, variants, ({ variantIdx, variantName, variantTotal, done, total }) => {
      if (done === total || done % 20 === 0) {
        setStatus(`eval r3: variant ${variantIdx + 1}/${variantTotal} (${variantName}) · ${done}/${total}`);
      }
    });
    window.__evalR3Summaries = summaries;
    const totalMs = performance.now() - tStart;
    console.log(`[eval] r3 dev done in ${(totalMs / 1000).toFixed(1)}s`);
    console.log('[eval] r3 dev ranked:');
    console.table([...summaries].sort((a, b) => b.accuracy - a.accuracy).map((s) => ({
      variant: s.variant,
      accuracy: +(s.accuracy * 100).toFixed(1),
      correct: s.correct,
      total: s.total,
      wall_s: +(s.wallMs / 1000).toFixed(1),
    })));

    // Then run validation on top-K r3 variants plus r2 winner as anchor.
    const topR3 = [...summaries].sort((a, b) => b.accuracy - a.accuracy).slice(0, topK).map((s) => s.variant);
    const valSet = Array.from(new Set([...topR3, 'r2_intent_story_list']));
    console.log(`[eval] running validation on:`, valSet);
    setStatus(`validating top ${topK} on held-out set…`);
    const valSummaries = await runOnValidation(ctx, valSet);
    window.__evalR3ValSummaries = valSummaries;
    console.log('[eval] r3 validation:');
    console.table(valSummaries.map((s) => {
      const lists = s.results.filter((r) => r.expected).length;
      const proses = s.results.filter((r) => !r.expected).length;
      const listHits = s.results.filter((r) => r.expected && r.correct).length;
      const proseHits = s.results.filter((r) => !r.expected && r.correct).length;
      return {
        variant: s.variant,
        accuracy: +(s.accuracy * 100).toFixed(1),
        list: `${listHits}/${lists}`,
        prose: `${proseHits}/${proses}`,
      };
    }));
    setStatus('r3 eval complete');
    return { dev: summaries, validation: valSummaries };
  });

  window.runRound2 = async () => acquireLock('running eval', async () => {
    const variants = variantsWithPrefix('r2_');
    console.log(`[eval] round 2: ${variants.length} variants × 80 prompts`);
    const tStart = performance.now();
    const summaries = await runVariantsOnDev(ctx, variants, ({ variantIdx, variantName, variantTotal, done, total }) => {
      if (done === total || done % 10 === 0) {
        setStatus(`eval r2: variant ${variantIdx + 1}/${variantTotal} (${variantName}) · ${done}/${total}`);
      }
    });
    const totalMs = performance.now() - tStart;
    console.log(`[eval] r2 dev done in ${(totalMs / 1000).toFixed(1)}s`);
    console.table(summaries.map((s) => ({
      variant: s.variant,
      accuracy: +(s.accuracy * 100).toFixed(1),
      correct: s.correct,
      total: s.total,
      wall_s: +(s.wallMs / 1000).toFixed(1),
    })));
    window.__evalR2Summaries = summaries;
    setStatus('r2 eval complete');
    return summaries;
  });

  window.runOneOnDev = async (variantName) => acquireLock('running eval', async () => {
    const variant = VARIANTS.find((v) => v.name === variantName);
    if (!variant) throw new Error(`no variant named ${variantName}`);
    const summaries = await runVariantsOnDev(ctx, [variant]);
    console.table(summaries.map((s) => ({ variant: s.variant, accuracy: +(s.accuracy * 100).toFixed(1), correct: s.correct, total: s.total })));
    return summaries;
  });

  window.runValidation = async (variantNames) => acquireLock('running eval', async () => {
    const summaries = await runOnValidation(ctx, variantNames);
    console.table(summaries.map((s) => ({
      variant: s.variant,
      accuracy: +(s.accuracy * 100).toFixed(1),
      correct: s.correct,
      total: s.total,
    })));
    window.__validationSummaries = summaries;
    return summaries;
  });
}

// Startup self-test: classify one list prompt and one prose prompt, run them
// through the full acrostic pipeline with a short fixed secret, and confirm
// (a) the classifier called each one correctly and (b) each output line
// starts with the matching secret letter. Renders inside the settings panel.
const SELF_TEST_CASES = [
  { kind: 'list',  prompt: 'give me five reasons to learn Rust', secret: 'AB', listMode: true },
  { kind: 'prose', prompt: 'tell me a short story about a lighthouse keeper', secret: 'AB', listMode: false },
];

function checkAcrostic(output, secret, caseInsensitive = true) {
  // Skip leading preamble: find the first non-empty line that starts with a
  // letter matching secret[0]. Then walk the following lines.
  const lines = output.split('\n').map((l) => l.trim()).filter(Boolean);
  const cmp = (a, b) => caseInsensitive ? a.toLowerCase() === b.toLowerCase() : a === b;
  // Strip leading ` * ` or `* ` bullets if present.
  const strip = (l) => l.replace(/^\*?\s*/, '');
  const firsts = lines.map((l) => strip(l)[0] || '');
  // Find the window starting at some i where firsts[i..i+len-1] matches secret.
  for (let i = 0; i + secret.length <= firsts.length; i++) {
    let ok = true;
    for (let j = 0; j < secret.length; j++) {
      if (!cmp(firsts[i + j], secret[j])) { ok = false; break; }
    }
    if (ok) return { ok: true, firsts: firsts.slice(i, i + secret.length).join('') };
  }
  return { ok: false, firsts: firsts.join('') };
}

async function runStartupSelfTest(ctx) {
  const body = document.getElementById('self-test-body');
  if (!body) return;
  body.innerHTML = 'running…';
  const rows = [];
  for (const tc of SELF_TEST_CASES) {
    const { prediction, raw } = await classifyWithVariant(ctx, VARIANTS.find((v) => v.name === CLASSIFIER_VARIANT_NAME), tc.prompt);
    const classifierOk = prediction === tc.listMode;
    const grammar = buildGrammar(tc.secret, tc.listMode, 80);
    const processor = new GrammarLogitsProcessor({
      grammar, tokenizer: ctx.generator.tokenizer, tokenText: ctx.tokenText, eosTokenIds: ctx.eosTokenIds,
    });
    const processors = new LogitsProcessorList();
    processors.push(processor);
    let text = '';
    const streamer = new TextStreamer(ctx.generator.tokenizer, {
      skip_prompt: true, skip_special_tokens: true, callback_function: (t) => { text += t; },
    });
    await ctx.generator(
      [{ role: 'system', content: 'You are a helpful assistant.' }, { role: 'user', content: tc.prompt }],
      { max_new_tokens: 180, do_sample: false, logits_processor: processors, streamer }
    );
    const acro = checkAcrostic(text, tc.secret);
    rows.push({ tc, prediction, raw, classifierOk, text, acro });
  }
  body.innerHTML = rows.map((r) => `
    <div class="row">
      <strong>${r.tc.kind}</strong> prompt: <code>${escapeHtml(r.tc.prompt)}</code><br>
      classifier: <span class="${r.classifierOk ? 'passed' : 'failed'}">${r.classifierOk ? 'OK' : 'WRONG'}</span>
      (wanted ${r.tc.listMode ? 'list' : 'prose'}, got ${r.prediction ? 'list' : 'prose'} · raw ${JSON.stringify(r.raw)})<br>
      acrostic "${r.tc.secret}" on output: <span class="${r.acro.ok ? 'passed' : 'failed'}">${r.acro.ok ? 'OK' : 'MISS'}</span>
      (first letters: <code>${escapeHtml(r.acro.firsts)}</code>)
      <pre>${escapeHtml(r.text)}</pre>
    </div>
  `).join('');
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

// Run the winning classifier variant over the full 80-dev + 20-val set and
// render a summary table inside the settings panel. Caller must hold the
// busy lock. Shows live progress in the panel while it runs.
async function runFullEval(ctx, body) {
  const { DEV_LIST, DEV_PROSE, VALIDATION_LIST, VALIDATION_PROSE, makeLabelled, runVariantOn } = await import('./eval.js');
  const variant = VARIANTS.find((v) => v.name === CLASSIFIER_VARIANT_NAME);
  if (!variant) throw new Error(`no variant ${CLASSIFIER_VARIANT_NAME}`);

  const prev = $status.textContent;
  const dev = makeLabelled(DEV_LIST, DEV_PROSE);
  const val = makeLabelled(VALIDATION_LIST, VALIDATION_PROSE);

  const rows = [];
  const render = (sections) => {
    body.innerHTML = sections.map(([label, r]) => {
      if (!r) return `<div>${label}: running…</div>`;
      const lists = r.results.filter((x) => x.expected).length;
      const proses = r.results.filter((x) => !x.expected).length;
      const listHit = r.results.filter((x) => x.expected && x.correct).length;
      const proseHit = r.results.filter((x) => !x.expected && x.correct).length;
      const cls = r.accuracy === 1 ? 'passed' : '';
      return `<div><strong>${label}:</strong> <span class="${cls}">${r.correct}/${r.total} = ${(r.accuracy * 100).toFixed(1)}%</span>
        (list ${listHit}/${lists}, prose ${proseHit}/${proses})</div>`;
    }).join('');
  };

  body.innerHTML = '';
  render([['dev (80)', null], ['validation (20)', null]]);

  const tStart = performance.now();
  const devRes = await runVariantOn(ctx, variant, dev, ({ done, total }) => {
    setStatus(`eval dev: ${done}/${total}`);
    if (done % 10 === 0) {
      render([['dev (80)', { correct: 0, total, accuracy: 0, results: [] }], ['validation (20)', null]]);
      body.firstElementChild.innerHTML = `<strong>dev (80):</strong> ${done}/${total} running…`;
    }
  });
  rows.push(['dev (80)', devRes]);
  render(rows.concat([['validation (20)', null]]));

  const valRes = await runVariantOn(ctx, variant, val, ({ done, total }) => {
    setStatus(`eval val: ${done}/${total}`);
  });
  rows.push(['validation (20)', valRes]);
  render(rows);

  // Miss summary
  const misses = [...devRes.results, ...valRes.results].filter((r) => !r.correct);
  if (misses.length) {
    body.innerHTML += `<h4 style="margin-top:0.5rem">Misses (${misses.length})</h4><table>
      <tr><th>expected</th><th>got</th><th>raw</th><th>prompt</th></tr>
      ${misses.map((r) => `<tr>
        <td>${r.expected ? 'list' : 'prose'}</td>
        <td>${r.prediction ? 'list' : 'prose'}</td>
        <td>${escapeHtml(JSON.stringify(r.raw))}</td>
        <td>${escapeHtml(r.prompt)}</td>
      </tr>`).join('')}
    </table>`;
  }

  const elapsed = ((performance.now() - tStart) / 1000).toFixed(1);
  body.innerHTML += `<div style="margin-top:0.4rem;color:#888">variant <code>${variant.name}</code> · ${elapsed}s</div>`;
  setStatus(prev);
}

// Winner from the prompt-optimization sweep: 98.8% dev / 100% validation on an
// 80+20-prompt eval, up from 50% for the baseline JSON prompt. Uses a pure
// "default to list; use story only for <long explicit trigger list>" rule, no
// few-shot. See src/eval.js for the variant definition and runAllOnDev /
// runRound{2,3,4} on window for reproducing the sweep.
const CLASSIFIER_VARIANT_NAME = 'r6_c1_v2_single_plural';

// Callers hold the busy lock (reason='classifying prompt'); this just runs
// the inference and flips the checkbox.
async function classifyListness(ctx, prompt) {
  if (!prompt.trim()) return;
  const prev = $status.textContent;
  setStatus('classifying prompt (list vs. prose)…');
  const variant = VARIANTS.find((v) => v.name === CLASSIFIER_VARIANT_NAME) ?? VARIANTS[0];
  const { prediction, raw } = await classifyWithVariant(ctx, variant, prompt);
  $listmode.checked = prediction;
  console.log(`[classifier:${variant.name}] prompt=${JSON.stringify(prompt)} → ${JSON.stringify(raw)} (listMode=${prediction})`);
  setStatus(prev);
}

async function runGeneration({ generator, tokenText, eosTokenIds }) {
  // Caller holds the busy lock (reason='generating'); button is already
  // disabled with label "generating…". We don't touch the button here.
  clearOutput();
  $metrics.textContent = '';

  const secret = $secret.value.trim();
  const listMode = $listmode.checked;
  const maxLine = Math.max(1, parseInt($maxline.value, 10) || 80);
  const prompt = $prompt.value;

  // Nudge the content shape to fit the acrostic line structure: plain text, no
  // markdown/bold/headings/numbered lists — those strand "**"/"#"/"1." fragments
  // when the per-letter line chopping cuts through them. Tailored per mode so
  // list mode still gets list-style items.
  const systemPrompt = listMode
    ? 'You are a helpful assistant. Answer as a plain bulleted list — one short item per line. Do not use markdown, bold text, headings, code, or numbered lists.'
    : 'You are a helpful assistant. Answer in plain prose. Do not use markdown, bold text, headings, code, or bulleted/numbered lists.';

  let grammar;
  try {
    grammar = buildGrammar(secret, listMode, maxLine);
  } catch (e) {
    setStatus(`grammar build error: ${e.message}`);
    return;
  }
  console.log(`[grammar] secret=${JSON.stringify(secret)} listMode=${listMode} maxLine=${maxLine}`);

  const processor = new GrammarLogitsProcessor({
    grammar,
    tokenizer: generator.tokenizer,
    tokenText,
    eosTokenIds
  });
  const processors = new LogitsProcessorList();
  processors.push(processor);

  setStatus('generating (grammar-constrained)…');

  const messages = [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: prompt }
  ];

  const tStart = performance.now();
  let tFirst = null;
  let tokens = 0;
  let chars = 0;

  const updateMetrics = () => {
    const now = performance.now();
    if (tFirst === null) {
      $metrics.textContent = `awaiting first token… (${((now - tStart) / 1000).toFixed(2)}s elapsed)`;
      return;
    }
    const ttft = ((tFirst - tStart) / 1000).toFixed(2);
    const genSec = Math.max(0.001, (now - tFirst) / 1000);
    const tps = (tokens / genSec).toFixed(2);
    $metrics.textContent = `TTFT ${ttft}s · ~${tps} tok/s · ${tokens} tokens · ${chars} chars`;
  };

  const streamer = new TextStreamer(generator.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (text) => {
      if (tFirst === null) tFirst = performance.now();
      tokens += 1;
      chars += text.length;
      appendOutput(text);
      updateMetrics();
    }
  });

  await generator(messages, {
    max_new_tokens: 400,
    do_sample: false,
    logits_processor: processors,
    streamer
  });
  const tEnd = performance.now();
  updateMetrics();

  const s = processor.stats;
  const genMs = tEnd - tStart;
  const procMs = s.totalMs;
  const modelMs = Math.max(0, genMs - procMs);
  const avg = (x) => s.calls ? (x / s.calls).toFixed(1) : '0';
  console.log(
    `[timing] wall=${genMs.toFixed(0)}ms · processor=${procMs.toFixed(0)}ms ` +
    `(${((procMs / genMs) * 100).toFixed(1)}%) · model+other=${modelMs.toFixed(0)}ms · ` +
    `${s.calls} steps · per-step avg: total=${avg(s.totalMs)}ms decode=${avg(s.decodeMs)}ms ` +
    `prefix=${avg(s.prefixMs)}ms scan=${avg(s.scanMs)}ms · scan advances=${s.advanceCalls}`
  );
  console.table(
    s.perStep.map((p, i) => ({
      step: i,
      total_ms: +p.tTotal.toFixed(2),
      decode_ms: +p.tDec.toFixed(2),
      prefix_ms: +p.tPref.toFixed(2),
      scan_ms: +p.tScan.toFixed(2),
      survivors: p.survivors
    }))
  );
  $metrics.textContent +=
    ` · proc ${procMs.toFixed(0)}ms (${((procMs / genMs) * 100).toFixed(0)}%) ` +
    `[dec ${s.decodeMs.toFixed(0)} · pref ${s.prefixMs.toFixed(0)} · scan ${s.scanMs.toFixed(0)}]`;

  setStatus('done. edit the secret and/or prompt and click Generate again.');
  // Button is re-enabled by acquireLock's finally block in the click handler.
}

boot().catch((e) => {
  setStatus(`error: ${e.message}`);
  console.error(e);
});
