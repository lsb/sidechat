"""Re-run the first half of quantize.py (fp16 → bake transpose → MatMulNBits +
GatherBlockQuantized) and dump int4-level statistics on how the two quantizer
codepaths differ on the same tied embedding weight."""
from pathlib import Path
import time
import onnx
import numpy as np
from onnxconverter_common import float16
from onnxruntime.quantization import matmul_nbits_quantizer, quant_utils

float16.sort_topology = lambda g: None

SRC = Path(__file__).parent.parent / "lfm2" / "onnx" / "model_fp16.onnx"
VOCAB, HIDDEN, BLOCK_SIZE = 65536, 1024, 64
N_BLOCKS = HIDDEN // BLOCK_SIZE


def bake_transpose_into_initializer(model):
    inits = {i.name: i for i in model.graph.initializer}
    new_inits, to_remove, rename = [], [], {}
    for node in list(model.graph.node):
        if node.op_type != "Transpose" or len(node.input) != 1 or node.input[0] not in inits:
            continue
        arr = onnx.numpy_helper.to_array(inits[node.input[0]])
        perm = next((a.ints for a in node.attribute if a.name == "perm"), list(reversed(range(arr.ndim))))
        new_name = f"{node.input[0]}_transposed"
        new_inits.append(onnx.numpy_helper.from_array(np.transpose(arr, list(perm)).copy(), name=new_name))
        rename[node.output[0]] = new_name
        to_remove.append(node)
    for n in to_remove:
        model.graph.node.remove(n)
    model.graph.initializer.extend(new_inits)
    for node in model.graph.node:
        for i, inp in enumerate(node.input):
            if inp in rename:
                node.input[i] = rename[inp]


t0 = time.time()
print(f"loading {SRC}…")
model = onnx.load(str(SRC))
print(f"  loaded in {time.time()-t0:.1f}s")

t0 = time.time()
print("baking transpose + quantizing…")
bake_transpose_into_initializer(model)
quantizer = matmul_nbits_quantizer.MatMulNBitsQuantizer(
    model,
    algo_config=matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
        block_size=BLOCK_SIZE, is_symmetric=True, accuracy_level=4,
        quant_format=quant_utils.QuantFormat.QOperator,
        op_types_to_quantize=("MatMul", "Gather"),
    ),
)
quantizer.process()
print(f"  quantized in {time.time()-t0:.1f}s")

m = quantizer.model.model
inits = {i.name: i for i in m.graph.initializer}
GATHER_W = "model.embed_tokens.weight_Q4"          # int4 [VOCAB, HIDDEN] signed
GATHER_S = "model.embed_tokens.weight_scales"      # fp16 [VOCAB, N_BLOCKS]
MNB_W    = "model.embed_tokens.weight_transposed_Q4"     # uint8 packed
MNB_S    = "model.embed_tokens.weight_transposed_scales"

# Decode GBQ side: signed int4, one value per slot, range [-8, 7]
g_int4 = onnx.numpy_helper.to_array(inits[GATHER_W]).astype(np.int32)  # [VOCAB, HIDDEN]
print(f"GBQ int4 array: shape={g_int4.shape}, range=[{g_int4.min()}, {g_int4.max()}]")

# Decode MNB side: uint8 packed [VOCAB, N_BLOCKS, BLOCK_SIZE//2], two int4/byte
mnb_raw = inits[MNB_W].raw_data or onnx.numpy_helper.to_array(inits[MNB_W]).tobytes()
mnb_bytes = np.frombuffer(mnb_raw, dtype=np.uint8).reshape(VOCAB, HIDDEN // 2)
# Unpack: low nibble first, then high. MNB stores values as uint4 with implicit
# zero_point=8 in symmetric mode, so the "real" int4 value is byte_nibble - 8.
mnb_lo = (mnb_bytes & 0x0F).astype(np.int32)
mnb_hi = (mnb_bytes >> 4).astype(np.int32)
mnb_uint4 = np.empty((VOCAB, HIDDEN), dtype=np.int32)
mnb_uint4[:, 0::2] = mnb_lo
mnb_uint4[:, 1::2] = mnb_hi
mnb_int4 = mnb_uint4 - 8  # symmetric uint4 → signed int4

# Sanity: scales must agree (the dedup pass asserts this)
g_scales = onnx.numpy_helper.to_array(inits[GATHER_S])
m_scales = onnx.numpy_helper.to_array(inits[MNB_S])
print(f"scales identical: {np.array_equal(g_scales, m_scales)}")

# === Int4-level comparison ==================================================
diff = mnb_int4 - g_int4   # signed difference, expected in {-1, 0, +1} mostly
total = g_int4.size
n_diff = int((diff != 0).sum())
print(f"\nTotal int4 weights in tied embedding: {total:,} ({total/1e6:.1f}M)")
print(f"Weights that differ between MNB and GBQ codepaths: {n_diff:,} ({n_diff/total*100:.4f}%)")

# Histogram of differences
print("\nDifference histogram (mnb_int4 - gbq_int4):")
unique, counts = np.unique(diff, return_counts=True)
for u, c in zip(unique.tolist(), counts.tolist()):
    pct = c / total * 100
    print(f"  {u:+d}: {c:>12,}  ({pct:7.4f}%)")

# Of the differing positions, what were the GBQ values and what did MNB pick?
mask = diff != 0
print("\nAmong the differing positions only:")
print(f"  GBQ int4 distribution: {dict(zip(*[a.tolist() for a in np.unique(g_int4[mask], return_counts=True)]))}")
print(f"  MNB int4 distribution: {dict(zip(*[a.tolist() for a in np.unique(mnb_int4[mask], return_counts=True)]))}")

# Reconstructed-fp16 magnitude of the change: diff * scale at that block
# scales shape: [VOCAB, N_BLOCKS]; broadcast to [VOCAB, HIDDEN] by block index
block_idx = np.arange(HIDDEN) // BLOCK_SIZE
scales_per_elem = g_scales[:, block_idx].astype(np.float32)
fp_delta = diff.astype(np.float32) * scales_per_elem
# Stats only over differing positions to avoid the 0-mass spike
nz = fp_delta[mask]
print(f"\nReconstructed fp16-space delta on differing positions (n={nz.size:,}):")
print(f"  |delta|  mean={np.abs(nz).mean():.6f}  max={np.abs(nz).max():.6f}  median={np.median(np.abs(nz)):.6f}")
print(f"  scale range (any block): [{g_scales.min():.4g}, {g_scales.max():.4g}]")

# Per-row impact: how many rows (tokens) are affected at all?
rows_affected = int((mask.any(axis=1)).sum())
rows_unchanged = VOCAB - rows_affected
print(f"\nPer-token-row: {rows_affected:,} of {VOCAB:,} rows have ≥1 changed weight ({rows_affected/VOCAB*100:.2f}%)")
print(f"  median changes per affected row: {int(np.median(mask.sum(axis=1)[mask.any(axis=1)]))}")
print(f"  max changes in any single row: {int(mask.sum(axis=1).max())}")
