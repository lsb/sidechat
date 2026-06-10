"""Adversarial test: construct fp32 weights where (almost) every value lands
on an even-half boundary in `w/scale` space — the worst case for banker vs
half-up disagreement. Then verify:
  Phase 1: ~98.4% of int4 values disagree at conversion time
  Phase 2: after dedup, runtime identity-op extraction produces bit-identical
           outputs anyway — i.e., the dedup trick is safe even when the
           underlying disagreement is maximal."""

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from onnxruntime.quantization import matmul_nbits_quantizer, quant_utils
import onnxruntime as ort

VOCAB, HIDDEN = 128, 128       # 2 blocks per row
BLOCK_SIZE = 64
N_BLOCKS = HIDDEN // BLOCK_SIZE
BLOB_SIZE = BLOCK_SIZE // 2

# === Build the adversarial weight ===========================================
# ORT's symmetric int4 quantizer: scale = -max_abs / 8 (yes, negative — it's
# an ORT convention: dequant(q) = q * scale, so q=-8 maps to +max_abs).
# We want every weight w (other than the scale-setter) to satisfy
# `w/scale = even.5` exactly, where banker→even and half-up→odd disagree.
# Disagreement boundaries in `w/scale` space: {±0.5, ±1.5, ±2.5, ..., ±7.5},
# specifically the ones where banker rounds to an EVEN integer:
#   w/scale ∈ {+0.5, +2.5, +4.5, +6.5, -1.5, -3.5, -5.5, -7.5}
# With scale = -0.875 (max_abs=7.0), w = (w/scale) * scale gives:
#   w/scale = -1.5  →  w = +1.3125
#   w/scale = -3.5  →  w = +3.0625
#   w/scale = -5.5  →  w = +4.8125
#   w/scale = -7.5  →  w = +6.5625
#   w/scale = +0.5  →  w = -0.4375
#   w/scale = +2.5  →  w = -2.1875
#   w/scale = +4.5  →  w = -3.9375
#   w/scale = +6.5  →  w = -5.6875
# All 8 have |w| ≤ 6.5625 < 7.0, so the scale-setter (w=7.0) holds.
disagreement_vals = np.array(
    [1.3125, 3.0625, 4.8125, 6.5625, -0.4375, -2.1875, -3.9375, -5.6875],
    dtype=np.float32,
)

def make_adversarial_block():
    block = np.empty(BLOCK_SIZE, dtype=np.float32)
    block[0] = 7.0
    block[1:] = disagreement_vals[np.arange(BLOCK_SIZE - 1) % len(disagreement_vals)]
    return block

W = np.tile(make_adversarial_block(), (VOCAB, N_BLOCKS)).astype(np.float32)
print(f"adversarial weight: shape={W.shape}, range=[{W.min():.4f}, {W.max():.4f}]")
print(f"  per-block: 1 scale-setter (w=7.0 → q=-8) + 63 even-half-tie values")
print(f"  expected scale per block: -0.875  (= -max_abs/8 in ORT convention)")
print(f"  theoretical max disagreement: 63/64 = {63/64*100:.2f}%")


# === Phase 1: conversion-time ================================================
def build_tied(weight):
    init = numpy_helper.from_array(weight, name="W")
    iid = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [None])
    out = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, VOCAB])
    nodes = [
        helper.make_node("Gather", ["W", "input_ids"], ["hidden"], axis=0),
        helper.make_node("Transpose", ["W"], ["wT"], perm=[1, 0]),
        helper.make_node("MatMul", ["hidden", "wT"], ["logits"]),
    ]
    g = helper.make_graph(nodes, "g", [iid], [out], [init])
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", 21)])


def bake_transpose(model):
    inits = {i.name: i for i in model.graph.initializer}
    new_inits, to_remove, rename = [], [], {}
    for node in list(model.graph.node):
        if node.op_type != "Transpose" or node.input[0] not in inits:
            continue
        arr = numpy_helper.to_array(inits[node.input[0]])
        perm = next((a.ints for a in node.attribute if a.name == "perm"), list(reversed(range(arr.ndim))))
        new_name = f"{node.input[0]}_transposed"
        new_inits.append(numpy_helper.from_array(np.transpose(arr, list(perm)).copy(), name=new_name))
        rename[node.output[0]] = new_name
        to_remove.append(node)
    for n in to_remove:
        model.graph.node.remove(n)
    model.graph.initializer.extend(new_inits)
    for node in model.graph.node:
        for i, inp in enumerate(node.input):
            if inp in rename:
                node.input[i] = rename[inp]


m = build_tied(W)
bake_transpose(m)
quantizer = matmul_nbits_quantizer.MatMulNBitsQuantizer(
    m,
    algo_config=matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
        block_size=BLOCK_SIZE, is_symmetric=True, accuracy_level=4,
        quant_format=quant_utils.QuantFormat.QOperator,
        op_types_to_quantize=("MatMul", "Gather"),
    ),
)
quantizer.process()
mq = quantizer.model.model
inits = {i.name: i for i in mq.graph.initializer}

g_int4 = numpy_helper.to_array(inits["W_Q4"]).astype(np.int32)
mnb_raw = inits["W_transposed_Q4"].raw_data \
    or numpy_helper.to_array(inits["W_transposed_Q4"]).tobytes()
mnb_bytes = np.frombuffer(mnb_raw, dtype=np.uint8).reshape(VOCAB, HIDDEN // 2)
mnb_uint4 = np.empty((VOCAB, HIDDEN), dtype=np.int32)
mnb_uint4[:, 0::2] = mnb_bytes & 0x0F
mnb_uint4[:, 1::2] = mnb_bytes >> 4
mnb_int4 = mnb_uint4 - 8

g_s = numpy_helper.to_array(inits["W_scales"])
m_s = numpy_helper.to_array(inits["W_transposed_scales"])
print(f"\n=== Phase 1: conversion-time int4 disagreement ===")
print(f"  scales identical: {np.array_equal(g_s, m_s)}  (all should be 1.0: range [{g_s.min()}, {g_s.max()}])")

diff = mnb_int4 - g_int4
n_diff = int((diff != 0).sum())
n_total = g_int4.size
pct = n_diff / n_total * 100
print(f"  total int4 values: {n_total:,}")
print(f"  disagreements:     {n_diff:,}  ({pct:.2f}%)")
print(f"  +1 / -1 split:     +1: {(diff==1).sum():,}  -1: {(diff==-1).sum():,}")
print(f"  GBQ values at differing positions: {dict(zip(*[a.tolist() for a in np.unique(g_int4[diff!=0], return_counts=True)]))}")
print(f"  MNB values at differing positions: {dict(zip(*[a.tolist() for a in np.unique(mnb_int4[diff!=0], return_counts=True)]))}")


# === Phase 2: dedup + runtime extraction =====================================
print("\n=== Phase 2: runtime equivalence after dedup (bit-identical?) ===")
shared_bytes = mnb_raw

def make_uint4_tensor(name, shape, packed_bytes):
    t = TensorProto()
    t.name = name; t.data_type = TensorProto.UINT4
    t.dims.extend(shape); t.raw_data = bytes(packed_bytes)
    return t

total_zp = VOCAB * N_BLOCKS
zp_flat = np.full(total_zp, 8, dtype=np.uint8)
zp_packed = np.zeros((total_zp + 1) // 2, dtype=np.uint8)
zp_packed[:total_zp // 2] = zp_flat[0:total_zp - (total_zp % 2):2] | (zp_flat[1::2] << 4)
zp_init = make_uint4_tensor("zp_uint4", [VOCAB, N_BLOCKS], zp_packed.tobytes())
gbq_weight = make_uint4_tensor("gbq_w", [VOCAB, HIDDEN], shared_bytes)

mnb_weight = TensorProto()
mnb_weight.name = "mnb_w"; mnb_weight.data_type = TensorProto.UINT8
mnb_weight.dims.extend([VOCAB, N_BLOCKS, BLOB_SIZE])
mnb_weight.raw_data = shared_bytes

scales_fp32 = numpy_helper.from_array(g_s.astype(np.float32), name="scales_fp32")

# --- GBQ extraction graph ---
indices_init = numpy_helper.from_array(np.arange(VOCAB, dtype=np.int64), name="all_indices")
out_g = helper.make_tensor_value_info("out", TensorProto.FLOAT, [VOCAB, HIDDEN])
gbq_node = helper.make_node(
    "GatherBlockQuantized",
    inputs=["gbq_w", "all_indices", "scales_fp32", "zp_uint4"],
    outputs=["out"], domain="com.microsoft",
    bits=4, block_size=BLOCK_SIZE, gather_axis=0, quantize_axis=1,
)
gbq_graph = helper.make_graph(
    [gbq_node], "gbq_extract", [], [out_g],
    initializer=[gbq_weight, indices_init, scales_fp32, zp_init],
)
gbq_model = helper.make_model(
    gbq_graph,
    opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
)

# --- MNB extraction graph ---
A_init = numpy_helper.from_array(np.eye(HIDDEN, dtype=np.float32), name="A_eye")
out_m = helper.make_tensor_value_info("out", TensorProto.FLOAT, [HIDDEN, VOCAB])
mnb_node = helper.make_node(
    "MatMulNBits",
    inputs=["A_eye", "mnb_w", "scales_fp32"],
    outputs=["out"], domain="com.microsoft",
    K=HIDDEN, N=VOCAB, bits=4, block_size=BLOCK_SIZE, accuracy_level=0,
)
mnb_graph = helper.make_graph(
    [mnb_node], "mnb_extract", [], [out_m],
    initializer=[A_init, mnb_weight, scales_fp32],
)
mnb_model = helper.make_model(
    mnb_graph,
    opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
)

gbq_sess = ort.InferenceSession(gbq_model.SerializeToString(), providers=["CPUExecutionProvider"])
mnb_sess = ort.InferenceSession(mnb_model.SerializeToString(), providers=["CPUExecutionProvider"])
gbq_out = gbq_sess.run(None, {})[0]
mnb_out = mnb_sess.run(None, {})[0].T

n_eq = int((gbq_out == mnb_out).sum())
abs_diff = np.abs(gbq_out - mnb_out)
print(f"  GBQ output shape={gbq_out.shape}, MNB output (transposed) shape={mnb_out.shape}")
print(f"  bit-identical: {n_eq == gbq_out.size}  ({n_eq}/{gbq_out.size} elements equal)")
print(f"  max |diff|: {abs_diff.max():.6e}")

# Sanity: also check the dequantized values vs the ORIGINAL adversarial W,
# choosing the MNB int4 (which is what BOTH ops now read after dedup).
dequant_via_mnb_bytes = (mnb_int4.astype(np.float32) * g_s.repeat(BLOCK_SIZE, axis=1))
print(f"\n  dequant = mnb_int4 * scale; vs original W:")
print(f"    max |dequant - W|: {np.abs(dequant_via_mnb_bytes - W).max():.4f}")
print(f"  the GBQ-int4 alternative would have given:")
dequant_via_gbq_bytes = (g_int4.astype(np.float32) * g_s.repeat(BLOCK_SIZE, axis=1))
print(f"    max |gbq_dequant - W|: {np.abs(dequant_via_gbq_bytes - W).max():.4f}")
print(f"  (both are ≤ 1 step = 1.0; both are equally-valid quantizations; the dedup picks MNB's)")
