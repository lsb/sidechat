"""End-to-end test on synthetic random fp32 data:

  Phase 1 (conversion time):
    - Build a tiny tied-embedding ONNX graph with a random fp32 weight
    - Run MatMulNBitsQuantizer (the production conversion path)
    - Compare GBQ int4 storage vs MNB int4 storage
    - Hypothesis: with truly random fp32 input, the two codepaths should
      almost never disagree (vs. 96k disagreements on the fp16 LFM2 weight)

  Phase 2 (runtime):
    - Apply our shared-bytes dedup pass on the same model
    - Build two minimal extraction graphs against the dedup'd bytes:
        A) GatherBlockQuantized + identity indices [0..VOCAB-1]
        B) MatMulNBits + identity matrix I_HIDDEN
    - Run both via onnxruntime.InferenceSession
    - Verify the dequantized fp16 matrices are bit-identical
      (the actual safety claim of the dedup trick at runtime)"""

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from onnxruntime.quantization import matmul_nbits_quantizer, quant_utils
import onnxruntime as ort

# Small enough to be fast, large enough to exercise both ops realistically
VOCAB, HIDDEN = 256, 128
BLOCK_SIZE = 64
N_BLOCKS = HIDDEN // BLOCK_SIZE          # 2
BLOB_SIZE = BLOCK_SIZE // 2              # 32 bytes per block


# ============================================================================
# Phase 1: build a minimal tied-embedding graph and run the production quantizer
# ============================================================================
def build_tied_embed_model(weight_fp32):
    """Tiny graph:
        input_ids: int64[seq]
        hidden = Gather(W, input_ids)              # embedding lookup
        wT = Transpose(W, perm=[1,0])              # tied LM head
        logits = MatMul(hidden, wT)
    The quantizer needs both a Gather and a MatMul over the SAME initializer."""
    W_init = numpy_helper.from_array(weight_fp32, name="W")

    input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [None])
    logits = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, VOCAB])

    nodes = [
        helper.make_node("Gather", ["W", "input_ids"], ["hidden"], axis=0),
        helper.make_node("Transpose", ["W"], ["wT"], perm=[1, 0]),
        helper.make_node("MatMul", ["hidden", "wT"], ["logits"]),
    ]
    graph = helper.make_graph(nodes, "tied", [input_ids], [logits], [W_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    onnx.checker.check_model(model)
    return model


def bake_transpose_into_initializer(model):
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


rng = np.random.default_rng(42)
W = rng.standard_normal((VOCAB, HIDDEN)).astype(np.float32) * 0.05
print(f"random weight: shape={W.shape} dtype={W.dtype} range=[{W.min():.4f}, {W.max():.4f}]")

print("\n=== Phase 1: conversion-time disagreement on random fp32 ===")
m = build_tied_embed_model(W)
bake_transpose_into_initializer(m)

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
print(f"  scales identical: {np.array_equal(g_s, m_s)}")

diff = mnb_int4 - g_int4
n_diff = int((diff != 0).sum())
print(f"  total int4 weights: {g_int4.size:,}")
print(f"  disagreements:      {n_diff:,}  (+1: {(diff==1).sum()}, -1: {(diff==-1).sum()})")
print(f"  contrast: LFM2 fp16 input had 96,015 / 67,108,864 (0.143%) — fp32 random should have far fewer ties")


# ============================================================================
# Phase 2: apply dedup pass + build extraction graphs + run via InferenceSession
# ============================================================================
print("\n=== Phase 2: runtime equivalence on dedup'd bytes ===")

# --- 2a: dedup pass (mirrors quantize.py) ----------------------------------
shared_bytes = mnb_raw  # canonical bytes both ops will read

# Build uint4 [VOCAB, HIDDEN] tensor for GBQ pointing at the SAME bytes
def make_uint4_tensor(name, shape, packed_bytes):
    t = TensorProto()
    t.name = name; t.data_type = TensorProto.UINT4
    t.dims.extend(shape); t.raw_data = bytes(packed_bytes)
    return t

# Zero-point uint4 = 8 per block
total_zp = VOCAB * N_BLOCKS
zp_flat = np.full(total_zp, 8, dtype=np.uint8)
zp_packed = np.zeros((total_zp + 1) // 2, dtype=np.uint8)
zp_packed[:total_zp // 2] = zp_flat[0:total_zp - (total_zp % 2):2] | (zp_flat[1::2] << 4)
zp_init = make_uint4_tensor("zp_uint4", [VOCAB, N_BLOCKS], zp_packed.tobytes())
gbq_weight = make_uint4_tensor("gbq_w", [VOCAB, HIDDEN], shared_bytes)

# MNB needs uint8 view [VOCAB, N_BLOCKS, BLOB_SIZE] of the SAME bytes
mnb_weight = TensorProto()
mnb_weight.name = "mnb_w"; mnb_weight.data_type = TensorProto.UINT8
mnb_weight.dims.extend([VOCAB, N_BLOCKS, BLOB_SIZE])
mnb_weight.raw_data = shared_bytes

scales_init = numpy_helper.from_array(g_s, name="scales")  # shape [VOCAB, N_BLOCKS]

# --- 2b: minimal extraction graph A — GatherBlockQuantized identity ---------
def build_gbq_extraction_model():
    indices_init = numpy_helper.from_array(
        np.arange(VOCAB, dtype=np.int64), name="all_indices"
    )
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [VOCAB, HIDDEN])
    # GBQ kernel returns scales' dtype; cast scales to fp32 so output is fp32
    scales_fp32 = numpy_helper.from_array(g_s.astype(np.float32), name="scales_fp32")
    node = helper.make_node(
        "GatherBlockQuantized",
        inputs=["gbq_w", "all_indices", "scales_fp32", "zp_uint4"],
        outputs=["out"],
        domain="com.microsoft",
        bits=4, block_size=BLOCK_SIZE,
        gather_axis=0, quantize_axis=1,
    )
    graph = helper.make_graph(
        [node], "gbq_extract", [], [out],
        initializer=[gbq_weight, indices_init, scales_fp32, zp_init],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
    )
    return model


# --- 2c: minimal extraction graph B — MatMulNBits identity ------------------
def build_mnb_extraction_model():
    # A = identity matrix [HIDDEN, HIDDEN], fp16 (matches scales dtype expectation)
    # MNB: Y[m, n] = sum_k A[m, k] * dequant(B)[n, k]
    # With A=I:    Y[m, n] = dequant(B)[n, m] = B_dequant.T
    A_init = numpy_helper.from_array(
        np.eye(HIDDEN, dtype=np.float32), name="A_eye"
    )
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [HIDDEN, VOCAB])
    # Use fp32 scales so output is fp32 (matches GBQ extraction)
    scales_fp16 = numpy_helper.from_array(g_s.astype(np.float32), name="scales_fp16")
    # In symmetric mode MNB has implicit zp=8; don't pass an explicit zp.
    # accuracy_level=0 lets the kernel pick; non-zero with int8 would coerce A.
    node = helper.make_node(
        "MatMulNBits",
        inputs=["A_eye", "mnb_w", "scales_fp16"],
        outputs=["out"],
        domain="com.microsoft",
        K=HIDDEN, N=VOCAB, bits=4, block_size=BLOCK_SIZE, accuracy_level=0,
    )
    graph = helper.make_graph(
        [node], "mnb_extract", [], [out],
        initializer=[A_init, mnb_weight, scales_fp16],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
    )
    return model


# --- 2d: run both, compare --------------------------------------------------
gbq_model = build_gbq_extraction_model()
mnb_model = build_mnb_extraction_model()

gbq_sess = ort.InferenceSession(gbq_model.SerializeToString(), providers=["CPUExecutionProvider"])
mnb_sess = ort.InferenceSession(mnb_model.SerializeToString(), providers=["CPUExecutionProvider"])

gbq_out = gbq_sess.run(None, {})[0]    # [VOCAB, HIDDEN] fp16
mnb_out = mnb_sess.run(None, {})[0]    # [HIDDEN, VOCAB] fp16
mnb_aligned = mnb_out.T                # → [VOCAB, HIDDEN]

print(f"  GBQ dequant output: shape={gbq_out.shape} dtype={gbq_out.dtype}")
print(f"  MNB dequant output (transposed): shape={mnb_aligned.shape} dtype={mnb_aligned.dtype}")

# Bit-identical check
n_eq = int((gbq_out == mnb_aligned).sum())
n_total = gbq_out.size
exact_match = (n_eq == n_total)
abs_diff = np.abs(gbq_out.astype(np.float32) - mnb_aligned.astype(np.float32))
print(f"  bit-identical: {exact_match}  ({n_eq}/{n_total} elements equal)")
print(f"  max |abs diff|: {abs_diff.max():.6e}")
print(f"  mean |abs diff|: {abs_diff.mean():.6e}")

# Compare both dequant outputs to the original W (sanity: dequant should be close to W,
# scaled to int4 precision)
abs_w_diff = np.abs(gbq_out.astype(np.float32) - W).max()
print(f"\n  sanity: max |gbq_dequant - W|: {abs_w_diff:.4f}  (expected ~one int4 step in scale = max_scale/2)")
print(f"          max scale: {g_s.max():.4f}, so step size ≈ {g_s.max():.4f}, half-step ≈ {g_s.max()/2:.4f}")
