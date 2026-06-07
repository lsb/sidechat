# SmolLM2-360M-Instruct — custom quantization log

## Goal

Ship a browser-loadable quantized ONNX of SmolLM2-360M-Instruct that is **as
small as possible** without losing accuracy on our 100-prompt list/prose
classifier eval. Baseline: HuggingFaceTB's pre-built `onnx/model_q4f16.onnx`
(273 MB).

Success metric: ≥ baseline accuracy (98.8% dev / 100% validation) with smaller
file size.

## Environment

- `model_work/venv`: Python 3.13, `onnxruntime==1.25.0`, `onnx`, `onnx_ir`,
  `onnxconverter_common`, `sympy`, `huggingface_hub`, `onnx-simplifier`.
- Pipeline script: `model_work/quantize.py`.
- Output goes to `public/local-models/HuggingFaceTB/SmolLM2-360M-Instruct/onnx/model_q4f16.onnx`
  and is served via vite when the browser URL has `?custom`.
- Eval: `src/eval.js` defines 80 dev + 20 validation prompts and the 37
  classifier variants across rounds 1–4. Winner: `r4_a3_d4_extended_triggers`.

## Attempts

### 1. fp32 → int4 directly (block_size=128, MatMul + Gather)

- Script: initial `DefaultWeightOnlyQuantConfig(block_size=128, is_symmetric=True, accuracy_level=4, quant_format=QOperator, op_types_to_quantize=("MatMul","Gather"))` on the fp32 model.
- Result: **392 MB**. Bigger than the reference (273 MB).
- Cause: weights quantized but activations/norms still fp32; also block_size=128 padded 960 → 1024 inside each MatMulNBits B tensor.
- Accuracy: 98.8% dev / 100% val. Matched reference.

### 2. fp32 → fp16 → int4 (block_size=128, MatMul + Gather)

- Added `onnxconverter_common.float16.convert_float_to_float16(model_fp32, keep_io_types=True, disable_shape_infer=True, check_fp16_ready=False)` before quantization.
- Bug: `float16.sort_topology` fails with `AttributeError: 'NoneType' object has no attribute 'output'` on this graph. Worked around with `float16.sort_topology = lambda g: None` (the model is already topologically sorted).
- Result: **291 MB** (still 18 MB bigger than HF). Accuracy unchanged.

### 3. Diffed against HF to find the bloat

Key findings:

- **Duplicate embedding copies**: our output has a 94 MB fp16 `model.embed_tokens.weight` AND a 47 MB int4 `model.embed_tokens.weight_Q4`. The int4 version is used by `GatherBlockQuantized` (embedding lookup); the fp16 version is consumed by `Transpose → MatMul` (the tied-weights LM head). Both stay on disk.
- **block_size=128 padding**: our MatMulNBits B tensors are `[2560, 8, 64]` (padding to 1024 cols). HF uses `[2560, 30, 16]` (block_size=32, matching 960 exactly).
- **128 extra `/Cast` ops** we left behind from the fp16 conversion (HF has 3).

### 4. block_size=32 + drop Gather quantization

- To match HF: quantize only MatMul ops, leave the embedding fp16 (so it can be shared between `Gather` and the LM head MatMul with no duplication).
- Result: **273 MB**, essentially identical to HF. Accuracy unchanged.
- But user wanted the embedding quantized.

### 5. Graph surgery: bake `Transpose` → two int4 copies

- In the fp16 model, before quantization, replace `Transpose(embed_tokens.weight) → MatMul(hidden, ·)` with a standalone `model.embed_tokens.weight_transposed` initializer (literally `np.transpose(embed)`) feeding a `MatMul`. Constant fold; bit-equivalent.
- Now MatMulNBits quantizes the LM head (`weight_transposed_Q4`), and GatherBlockQuantized quantizes the embedding lookup (`weight_Q4`). No fp16 embedding left.
- Result: **231 MB**, 98.8% dev / 100% val. Two int4 copies of the embedding (47 MB unpacked each), since `GatherBlockQuantized` and `MatMulNBits` need different layouts.

### 6. Dequantize-on-the-fly

- Replace the LM head's `MatMulNBits` node + its private transposed int4 weight + scales with:
  - `DequantizeLinear(embed_Q4, embed_scales, axis=1, block_size=32) → fp16 [vocab, hidden]`
  - `Transpose` → `[hidden, vocab]`
  - `MatMul(hidden_state, transposed)` → `[batch, seq, vocab]` logits
- Bumped opset to 21 so int4 dtype is valid.
- Result: **205 MB** (**-25% vs HF**), 98.8% dev / 100% val, 16.1 tok/s, bit-identical generation on the grocery-list prompt.
- Single int4 copy of the embedding on disk, shared between `GatherBlockQuantized` and the new dequant subgraph.
- Caveat: at session load ORT constant-folds the `DequantizeLinear → Transpose → MatMul` (verified empirically: 364 forward passes, per-step model time 37.9–51.5 ms, centered at 41 ms, no per-step dequant overhead visible). That leaves a materialized 94 MB fp16 LM-head weight in GPU memory, which is the "wasted" side of the tradeoff vs variant 7.

### 7. Shared-bytes via external-data aliasing (final)

- Observation: `MatMulNBits.B` (uint8 `[vocab, n_blocks, blob_size]`) and
  `GatherBlockQuantized.data` (int4/uint4 `[vocab, hidden]`) encode the same
  weight with identical *bytes* if we use uint4 with zero_point=8 on both
  sides. The ONNX type system forbids a single initializer with two
  `(shape, dtype)` declarations — but nothing forbids two initializer protos
  whose `external_data` entries point at the same `(location, offset, length)`
  in the companion `.onnx_data` file.
- Pipeline: steps 5 + 6 as before → quantize → take the MatMulNBits uint8
  bytes as canonical → declare a uint4 `[vocab, hidden]` initializer with
  the same bytes for `GatherBlockQuantized`, and set `zero_points = 8` on
  that node → write both initializers' `external_data` to point at the same
  23.59 MB blob.
- ~0.28% of int4 values drift by one step between the two quantizer
  codepaths (independent FP rounding order); MatMulNBits's copy is chosen
  as canonical. Undetectable in accuracy.
- Result: **208 MB** total (0.3 MB `.onnx` + 208 MB `.onnx_data`),
  98.8% dev / 100% val (same single miss as variants 5/6), 15.9 tok/s,
  bit-identical generation on the grocery-list prompt.
- GPU memory: ~47 MB int4 (potentially ~23 MB if ORT dedupes
  same-external-data-offset initializers), vs ~117 MB for variant 6.
- **Final winner**: within 3 MB of the smallest on disk, smallest in GPU
  memory, no fp16 intermediate anywhere.

### 8. block_size=64 on top of the shared-bytes variant (current winner)

- Same pipeline as step 7 but with `block_size=64` everywhere (both in
  `DefaultWeightOnlyQuantConfig` and in the hand-rolled zero_point tensor
  for GatherBlockQuantized). 960 / 64 = 15 blocks exactly — same clean
  fit as block_size=32, no padding.
- Halves the scales tensor: fp16 `[49152, 30]` → `[49152, 15]`; similar
  fractional savings across the 224 layer MatMulNBits scale tensors.
- Result: **195 MB** total (0.3 MB `.onnx` + 195 MB `.onnx_data`),
  **100% val / 98.8% dev** (same single miss as variants 5-7 — "explain
  the difference between empathy and sympathy" → list), 17.8 tok/s,
  bit-similar generation.
- Disk reduction from block_size=32: ~13 MB (above the pure scales-count
  savings — ORT's MatMulNBits quantization produces slightly different
  packing overhead at different block sizes).
- Note on packing: with 15 blocks per row in the uint4 zero_point tensor,
  per-row byte parity is odd; ONNX packs uint4 over the flat element order
  so row boundaries don't align to byte boundaries. Have to build the
  packed bytes over the whole flat array, not row-by-row.

## File format note

All "shared-bytes" variants (7 and 8) produce two files — `model_q4f16.onnx`
(~0.3 MB graph + metadata) plus `model_q4f16.onnx_data` (the big blob).
ONNX external data is the only way to make two initializers share bytes on
disk: the protobuf `raw_data` field is per-tensor and gets serialized
separately for each initializer, so without external data two identical
tensors would each carry their own copy of the bytes inside the `.onnx`
file.

If single-file simplicity matters more than the dedupe savings, variant 6
(dequant-on-the-fly, ~203 MB at block_size=64) is a single `.onnx`. Variant
8's extra complexity (two files) buys ~8 MB on disk and ~70 MB in GPU
memory.

## Structure of the 205 MB "dequant-on-load" variant

| Component | Bytes | Notes |
|---|---|---|
| 224 `MatMulNBits` (transformer layer matmuls) | ~157 MB uint8 + ~3 MB fp16 scales | int4 weights, block_size=32 |
| 1 `GatherBlockQuantized` (embedding lookup) | ~23.6 MB int4 + 3 MB fp16 scales | shared with LM head |
| 1 `DequantizeLinear` + 1 `Transpose` + 1 `MatMul` (LM head) | 0 MB private | reads shared int4 embedding |
| 131 `/Cast` ops | small | residue from fp16 conversion; not worth fighting |

Total: 205 MB on disk, single int4 copy of the embedding.

## Open question: does the LM head waste ~100 MB per pass?

The concern: `DequantizeLinear(int4 [49152, 960], scales) → fp16 [49152, 960]`
materializes a **94 MB** fp16 tensor, then `Transpose` + `MatMul(h, transposed)`
computes logits for **all 49152 vocab tokens**, of which greedy decoding picks
one. Most of the output is discarded.

Two sub-questions:

1. **Is the 94 MB intermediate materialized once at load, or every pass?**
   - Both inputs to `DequantizeLinear` (`embed_Q4`, `embed_scales`) are
     initializers → the whole subgraph is a pure constant computation. ORT's
     graph optimizer can (and typically does) constant-fold at session
     creation.
   - **Verified empirically** (364 forward passes on the 205 MB model,
     instrumenting `decoder.run`):
     - First call: 123.5 ms (full prefill of ~15 input tokens)
     - Second call: 37.9 ms
     - Rest avg: **41 ms/step** (p50 41, p95 48, range 34.5–51.5)
     - If dequant-on-the-fly were running per step, every call would carry a
       consistent extra ~10–30 ms; instead the distribution is tight and
       centered at 41 ms, with the second call actually faster than average.
   - **Conclusion: ORT is folding the `DequantizeLinear(initializer) →
     Transpose → MatMul` chain at session creation**, leaving a plain fp16
     LM-head MatMul in memory. On disk we still have only int4 (205 MB); in
     GPU memory there's a 94 MB fp16 weight materialized once.

2. **~99.998% of the logits matmul output is discarded (still true).**
   - The classifier reads the full 49152-wide logit vector, picks the max
     among tokens the grammar allows.
   - For acrostic generation, the grammar masks most tokens to `-Infinity`
     *after* the matmul; no savings.
   - This is independent of the quant story — HF's reference model has the
     same waste. Addressing it is (c) below, a separate project.

## Alternative approaches worth exploring

Listed rough-to-implement first:

### (a) Verify ORT folding + accept current state

- Just measure: patch the decoder session to print session-creation time, post-optimization graph info. If DequantizeLinear is folded, the current 205 MB model is probably the pragmatic end point for pure size work.

### (b) Skip Transpose by storing the transposed int4 directly

- Precompute `Transpose(embed)` once in Python and store THAT as a `[hidden, vocab]` int4 initializer. Feed it directly into `MatMul` (after DequantizeLinear).
- Saves nothing on disk (same number of int4 values) but removes one runtime op (if not folded).
- Low priority if (a) shows folding is happening.

### (c) Sparse logits MatMul driven by the grammar

- During constrained decoding, our logits processor already knows which tokens are live (for most states, a small fraction of the 49152).
- Replace the dense LM head `MatMul` with `Gather(transposed_embed, live_ids) → MatMul(h, gathered)` → `[batch, seq, K]` where K ≪ vocab.
- Requires passing `live_ids` as a runtime input to the model. Transformers.js doesn't expose a clean extension point for this; would need a custom forward wrapper.
- Potential savings: if average K = 1000 (for mid-line body states), we'd compute ~2% of the current MatMul.
- Tradeoff: grammar-coupled model (only useful when constrained); complex to integrate with the existing pipeline.

### (d) Custom op that reads `GatherBlockQuantized` layout for the LM head

- Write a WebGPU/WASM kernel that does `hidden @ W^T` directly on the int4 row-major layout used by `GatherBlockQuantized`, skipping the intermediate fp16 materialization.
- Eliminates the 94 MB GPU memory usage (streams int4 blocks directly).
- Significant work; essentially a new ORT contrib op.

### (d') Share the int4 bytes between `GatherBlockQuantized` and `MatMulNBits`

- Observation: the byte layouts are identical for the same logical weight.
  - `GatherBlockQuantized.input_data`: int4 `[vocab, hidden]`, blocks along
    axis=1, block_size=32. Packed 2 int4/byte, row-major by vocab. 23.6 MB.
  - `MatMulNBits.B` (representing the same weight): uint8
    `[vocab, n_blocks=30, blob_size=16]`. Each row v still 480 bytes
    `[int4(v,0)|int4(v,1), int4(v,2)|int4(v,3), …, int4(v,958)|int4(v,959)]`.
    Identical bytes.
- Problem: ONNX initializers carry exactly one `(shape, dtype)`. The two ops
  declare different shapes AND different dtypes (int4 vs uint8) for the
  same bytes. No standard ONNX op reinterprets bytes across dtypes:
  - `Reshape` keeps dtype (int4 → int4), can't morph to uint8.
  - `Cast(int4, uint8)` does value conversion (each int4 becomes a uint8),
    producing an 8× larger tensor. Not what we want.
  - No `BitcastView` / `InterpretAs` op in the standard set.
- Feasible paths:
  1. Custom ORT op `AsUint8View` that takes an int4 tensor + new shape and
     returns a uint8 tensor sharing the same memory. Tiny op, but a new
     contrib op — requires maintaining a patched ORT build.
  2. Pre-session graph pass that collapses two initializer declarations onto
     the same underlying memory allocation. Requires hooking ORT's session
     builder.
- Savings: 0 disk (we already share via DequantizeLinear + fold) but ~94 MB
  of GPU memory (would stream int4 blocks directly in the MatMulNBits kernel
  instead of materializing the fp16 weight).

### (e) Strip the 128 `/Cast` ops

- `onnx-simplifier` fails on Microsoft-custom ops (`SimplifiedLayerNormalization`) even after upgrading. Could hand-roll a targeted Cast-of-initializer constant-fold pass.
- Probably saves <1 MB; not a priority for size but would clean up the graph.

## Next step

205 MB dequant-on-the-fly model is folded at load → call quant work done.
Remaining file-size wins are diminishing returns ((e) strip 128 Cast ops for
<1 MB). The bigger next lever is (c) sparse logits driven by the grammar,
which is a runtime/quality optimization rather than a size one.

## Addendum: disk size vs. GPU memory tradeoff

Three final variants, all same accuracy (98.8% dev / 100% val) and same
bit-identical generation:

| Variant | Disk | GPU memory (embed + LM head) | tok/s | Notes |
|---|---:|---:|---:|---|
| 205 MB "dequant-on-load" | 205 MB (single file) | ~117 MB (23 int4 + 94 fp16) | 16.1 | LM head uses `DequantizeLinear → Transpose → MatMul`; ORT folds at session load, materializing a 94 MB fp16 weight. |
| 231 MB "dual int4" | 231 MB (single file) | ~47 MB (two int4 copies) | ~15.9 | `MatMulNBits` for LM head has its own int4 copy of the transposed embedding. No fp16 intermediate; bytes are independent between the two ops. |
| **208 MB "shared bytes"** (current winner) | **208 MB** (0.3 MB + 208 MB external) | **~47 MB** (int4 only — potentially 23 MB if ORT dedupes) | 15.9 | Post-process the int4 Gather weight into uint4 (values += 8), set `zero_points=8` on `GatherBlockQuantized`, then write the ONNX file with both the uint4 Gather weight and the uint8 MatMulNBits weight pointing at the *same* 23.59 MB byte range in the external data file. ~0.3% of embedding positions drift by one int4 step (independent FP rounding between the two quantizer codepaths); undetectable in accuracy. |

The only ops in stable ORT that consume int4/uint4 bytes *directly* (no
whole-tensor dequant to fp16 first):

- **`MatMulNBits`** — int4 weight × fp16 activation → fp16 output, fused.
- **`GatherBlockQuantized`** — dequantizes only the *selected rows*.

There is no `GatherNBits + MatMulNBits` super-fusion in standard ORT; at
the ONNX level we had to keep two initializer declarations (different shape
and dtype) that *happen to point at the same bytes on disk* via
`external_data`.

### Recommendation

- If the deployment target is bandwidth-constrained: ship **208 MB** (tiny
  net benefit over 205 MB; does require shipping two files — .onnx and
  .onnx_data — together).
- If the deployment target is GPU-memory-constrained: ship **208 MB**
  — same runtime footprint as the 231 MB dual-int4 variant but 23 MB
  smaller on disk, with ORT potentially deduping the two initializers into
  a single 23.59 MB allocation if the external-data offsets are detected as
  shared (need to verify empirically).

### Process, step by step

1. fp32 → fp16 (`onnxconverter_common.float16.convert_float_to_float16`,
   with `sort_topology = lambda g: None` monkey-patch).
2. Bake `Transpose(embed_tokens.weight)` into a standalone initializer so
   the LM head's MatMul becomes a `MatMul(hidden, initializer)` and is
   thus quantizable.
3. `MatMulNBitsQuantizer` with `DefaultWeightOnlyQuantConfig(block_size=32,
   is_symmetric=True, accuracy_level=4, quant_format=QOperator,
   op_types_to_quantize=("MatMul", "Gather"))`.
4. Post-process: take the MatMulNBits uint8 bytes for the LM head and
   declare a *second* initializer as uint4 `[vocab, hidden]` holding the
   same bytes (same file offset) for the `GatherBlockQuantized` node. Set
   `zero_points = 8` on GatherBlockQuantized to account for uint4
   symmetric quant.
5. Write the ONNX file with external data, with both initializers
   referencing the same (offset, length) in the `.onnx_data` file.
