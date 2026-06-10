# LFM2.5-350M — custom quantization log

## Goal

Replace the SmolLM2-360M-Instruct model (see `SmolLM2-360M-quantization.md`)
with LiquidAI/LFM2.5-350M, reusing our existing `model_work/quantize.py`
pipeline (fp16 → bake `Transpose(embed)` → int4 weight-only with shared
bytes between `MatMulNBits` and `GatherBlockQuantized`). Ship a 4-bit
quantized model that loads in the browser via transformers.js + WebGPU.

Baseline: LiquidAI's own pre-built `onnx/model_q4f16.onnx_data` is **255 MB**
(at `LiquidAI/LFM2.5-350M-ONNX`). Our pipeline produced **207 MB** — same
dynamic-range strategy that took SmolLM2 from 273 MB → 195 MB.

## Source

LiquidAI ships seven pre-quantized variants in `onnx/`:

| File | Size |
|---|---:|
| `model.onnx_data` | 1.45 GB (fp32) |
| `model_fp16.onnx_data` | 725 MB |
| `model_q4f16.onnx_data` | 255 MB |
| `model_q4.onnx_data` | 294 MB |
| `model_q4f32.onnx_data` | 481 MB |
| `model_q8.onnx_data` | 634 MB |
| `model_quantized.onnx_data` | 510 MB |

We start from **`model_fp16.onnx`** and skip our usual fp32→fp16 conversion
step (saves the 725 MB download vs the 1.45 GB fp32 source, and skips a
slow but mostly-loss-free pass through `onnxconverter_common.float16`).
Quantization picks up at the bake-Transpose step.

## Architecture

`config.json`:

- `model_type: lfm2`, `architectures: [Lfm2ForCausalLM]`
- `vocab_size: 65536`, `hidden_size: 1024`, `num_hidden_layers: 16`
- `tie_word_embeddings: true` (the embed-Transpose-MatMul tied LM head
  pattern works, same as SmolLM2)
- `layer_types`: 10 `conv` (LIV double-gated convolution blocks) + 6
  `full_attention` (GQA blocks). Conv layers use small kernels
  (`conv_L_cache: 3`); the bulk of parameters are still in MatMul
  (FFN gate/up/down projections, attention QKV, embedding).
- `transformers.js_config.use_external_data_format: true` and
  `kv_cache_dtype.q4f16: float16` — already configured for our load path.

The fp16 ONNX graph (140 KB without external data) confirms:

- `Transpose(model.embed_tokens.weight) → MatMul(hidden, ·)` is present —
  our `bake_transpose_into_initializer` step applies cleanly.
- Opset 21 already (the `int4`/`uint4` dtypes our pipeline needs).
- Input/output dtypes are fp16 already (no `keep_io_types=False` shim
  needed).
- 25 named inputs (input_ids, attention_mask, num_logits_to_keep, then
  alternating `past_conv.N` and `past_key_values.N.{key,value}` per layer
  type). transformers.js 4.1.0 has `Lfm2ForCausalLM` registered and
  handles this layout via the base `PreTrainedModel`.

Transformers.js: 4.1.0 ships with `src/models/lfm2/modeling_lfm2.js`, just
five lines extending `PreTrainedModel`. All LFM2-specific past-cache
plumbing comes from the base class via the `layer_types` config field.

## Pipeline changes

The existing `quantize.py` was hardcoded for SmolLM2's dimensions
(vocab=49152, hidden=960). Three changes:

### 1. Parametrized config block

```python
SRC          = Path(__file__).parent / "lfm2" / "onnx" / "model_fp16.onnx"
SRC_IS_FP16  = True
VOCAB        = 65536
HIDDEN       = 1024
BLOCK_SIZE   = 64                  # 1024/64 = 16 blocks, exact fit
TARGET_N_CHUNKS = 4
CHUNK_HARD_CAP  = 50 * 1000 * 1000  # GitHub Pages 100 MB cap with headroom
```

Old hardcoded constants (`49152`, `960`, padded reshape `[49152, 480]`,
etc.) became `VOCAB`, `HIDDEN`, `N_BLOCKS = HIDDEN // BLOCK_SIZE`,
`BLOB_SIZE = BLOCK_SIZE // 2`. The `SRC_IS_FP16` flag skips the
`float16.convert_float_to_float16(...)` call.

### 2. Deterministic naming in `bake_transpose_into_initializer`

The original implementation named the new initializer after the Transpose
node's *output name* (`node.output[0]`). For SmolLM2 that happened to be
`model.embed_tokens.weight_transposed` — exactly the name our
`MatMulNBitsQuantizer` postprocessing expected. For LFM2 the output name
is `/lm_head/Transpose/output_0`, which broke the lookup
`inits['model.embed_tokens.weight_transposed_scales']`.

Fix: name the new initializer `f"{input}_transposed"` deterministically,
and rewrite all consumers' input references to match.

```python
def bake_transpose_into_initializer(model):
    inits = {i.name: i for i in model.graph.initializer}
    new_inits, to_remove, rename = [], [], {}
    for node in list(model.graph.node):
        if node.op_type != "Transpose" or len(node.input) != 1 or node.input[0] not in inits:
            continue
        arr = onnx.numpy_helper.to_array(inits[node.input[0]])
        perm = next((a.ints for a in node.attribute if a.name == "perm"),
                    list(reversed(range(arr.ndim))))
        new_name = f"{node.input[0]}_transposed"
        new_inits.append(onnx.numpy_helper.from_array(
            np.transpose(arr, list(perm)).copy(), name=new_name))
        rename[node.output[0]] = new_name
        to_remove.append(node)
    for n in to_remove:
        model.graph.node.remove(n)
    model.graph.initializer.extend(new_inits)
    for node in model.graph.node:
        for i, inp in enumerate(node.input):
            if inp in rename:
                node.input[i] = rename[inp]
    return len(to_remove)
```

### 3. Dynamic chunk count

The shipping pipeline writes external data into chunks ≤ 50 MB each (the
GitHub Pages serve cap with headroom). SmolLM2 fit in 4 chunks; LFM2's
larger embedding (33.6 MB shared blob — 1.42× SmolLM2's 23.6 MB because
65536/49152 × 1024/960) plus 173.8 MB of layer weights pushed total
external data to 207.4 MB, which 4 chunks couldn't accept under the cap.

Fix: bump the chunk count if the configured target won't fit:

```python
n_chunks = max(TARGET_N_CHUNKS, (total_external + CHUNK_HARD_CAP - 1) // CHUNK_HARD_CAP)
ideal_per_chunk = (total_external + n_chunks - 1) // n_chunks
chunk_cap = int(ideal_per_chunk * 1.02)  # 2% slack for greedy packing
```

Resulted in 6 chunks (5 × ~41 MB + 1 × 2.6 MB tail). The 2 % slack let one
2.6 MB blob spill into a 6th chunk; not worth tuning further.

## Result

| | SmolLM2-360M | LFM2.5-350M |
|---|---:|---:|
| Source dtype / size | fp32 / 1.45 GB | fp16 / 725 MB |
| Quantized total | 195 MB | 207 MB |
| Chunks | 4 | 6 |
| Shared embedding blob | 23.6 MB (49152 × 960) | 33.6 MB (65536 × 1024) |
| Block size | 64 (15 blocks/row) | 64 (16 blocks/row) |
| Byte agreement Gather↔MatMulNBits | ~99.7% | 99.71% |
| HF/LiquidAI shipped q4f16 reference | 273 MB | 255 MB |
| Reduction vs reference | -28% | -19% |

The smaller relative reduction vs reference is mostly because LFM2 has 10
unquantized conv blocks (small kernels, but stay fp16) — a larger fraction
of model weight is non-MatMul than for SmolLM2's pure-Llama architecture.

Byte agreement at 99.71% means ~0.29% of int4 values drift by one step
between the two quantizer codepaths (`MatMulNBitsQuantizer` for the LM head
side and the `GatherBlockQuantized` codepath). MatMulNBits's bytes are
canonical; the GBQ initializer is rewritten to match. Undetectable in
output quality, same as for SmolLM2.

## Verification

End-to-end load + generation in the browser:

- Page subtitle: "Downloading over 208 MB (one-shot transfer)"
  during the chunked external-data fetch; clears to "Available. Chat
  ready; output satisfies text input commands." on ready.
- Self-test (in `src/main.js`): two prompts (one list, one prose),
  acrostic on `AB`. Both classify correctly under the round-6 winner
  (`r6_c1_v2_single_plural`) and produce grammar-constrained acrostic
  output starting with `A` then `B`.
- Classifier accuracy on the 100-prompt eval (after re-tuning, see
  `CLASSIFIER_PROMPT_OPTIMIZATION.md`): 97.5% dev / 85% val, vs the
  ported-from-SmolLM2 baseline of 68.8% / 60%.

## File layout

```
public/local-models/LiquidAI/LFM2.5-350M/
  config.json
  generation_config.json
  tokenizer.json
  tokenizer_config.json
  chat_template.jinja
  onnx/
    model_q4f16.onnx          (~170 KB graph)
    model_q4f16.onnx_data     (~42 MB, includes 33.6 MB shared embedding)
    model_q4f16.onnx_data_1   (~41 MB)
    model_q4f16.onnx_data_2   (~41 MB)
    model_q4f16.onnx_data_3   (~40 MB)
    model_q4f16.onnx_data_4   (~40 MB)
    model_q4f16.onnx_data_5   (~3 MB tail)
```

`src/main.js` loads it via:

```js
const generator = await pipeline('text-generation', 'LiquidAI/LFM2.5-350M', {
  dtype: 'q4f16',
  device: 'webgpu',
  use_external_data_format: 6,  // 6 chunks
  progress_callback: ...,
});
```
