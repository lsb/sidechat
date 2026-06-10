"""Two-part test:
  Part A (numpy): does fp32 + tiny perturbation drop banker-vs-half-up
    disagreement to ~0?
  Part B (real ORT): replace the embedding initializer with fp32 + tiny
    perturbation, run the actual MatMulNBitsQuantizer, and compare GBQ vs MNB
    int4 outputs. This is the definitive test against the actual codepaths
    (whose rounding rules we don't know exactly)."""
from pathlib import Path
import time
import onnx
import numpy as np
from onnx import TensorProto
from onnxconverter_common import float16
from onnxruntime.quantization import matmul_nbits_quantizer, quant_utils

float16.sort_topology = lambda g: None
SRC = Path(__file__).parent.parent / "lfm2" / "onnx" / "model_fp16.onnx"
VOCAB, HIDDEN, BLOCK_SIZE = 65536, 1024, 64
N_BLOCKS = HIDDEN // BLOCK_SIZE


# ============================================================================
# Part A: numpy banker vs half-up on fp32 + perturbation
# ============================================================================
def block_quantize_np(w, round_mode):
    blocked = w.reshape(VOCAB, N_BLOCKS, BLOCK_SIZE).astype(np.float32)
    max_abs = np.abs(blocked).max(axis=2, keepdims=True)
    scale = np.where(max_abs == 0, 1, max_abs) / 7.0
    ratio = blocked / scale
    if round_mode == "banker":
        q = np.round(ratio)
    elif round_mode == "half_up":
        q = np.floor(ratio + 0.5)
    return q.clip(-8, 7).astype(np.int32).reshape(VOCAB, HIDDEN)


print("loading…")
m = onnx.load(str(SRC))
embed = next(i for i in m.graph.initializer if i.name == "model.embed_tokens.weight")
w_fp16 = onnx.numpy_helper.to_array(embed)
w_fp32 = w_fp16.astype(np.float32)
print(f"  embedding {w_fp32.shape} dtype={w_fp32.dtype}")

print("\n=== Part A: numpy banker vs half-up ===")
print(f"{'condition':<45} {'banker vs half-up':>20}")
for label, w in [
    ("fp16 input (orig)", w_fp16),
    ("fp16 → fp32, no perturbation", w_fp32),
    ("fp16 → fp32, +random 1e-7 perturbation",
     (w_fp32 + np.random.default_rng(0).uniform(-1, 1, w_fp32.shape).astype(np.float32) * 1e-7)),
    ("fp16 → fp32, +random 1e-5 perturbation",
     (w_fp32 + np.random.default_rng(0).uniform(-1, 1, w_fp32.shape).astype(np.float32) * 1e-5)),
]:
    qa = block_quantize_np(w, "banker")
    qb = block_quantize_np(w, "half_up")
    n = int((qa != qb).sum())
    print(f"  {label:<43} {n:>20,}")


# ============================================================================
# Part B: real ORT pipeline. Replace the embedding initializer with fp32 +
# perturbation, then run the existing pipeline up through the GBQ vs MNB diff.
# ============================================================================
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


def run_real_ort(perturbation):
    """Run the real quantizer with embedding cast to fp32 + perturbation.
    Returns (n_disagreements, n_plus, n_minus)."""
    print(f"\n--- Part B: ORT with fp32 + perturbation={perturbation} ---")
    t = time.time()
    model = onnx.load(str(SRC))
    # Replace the embedding initializer with fp32 + perturbation.
    rng = np.random.default_rng(0)
    new_w = w_fp32.copy()
    if perturbation > 0:
        new_w = new_w + (rng.uniform(-1, 1, new_w.shape).astype(np.float32) * perturbation)
    new_init = onnx.numpy_helper.from_array(new_w.astype(np.float32), name="model.embed_tokens.weight")
    for i, init in enumerate(model.graph.initializer):
        if init.name == "model.embed_tokens.weight":
            model.graph.initializer.remove(init)
            model.graph.initializer.append(new_init)
            break
    # The graph's value_info for this initializer was fp16; not strictly required
    # to update because initializers carry their own dtype, but the consumer
    # nodes (Gather, Transpose→MatMul) must accept fp32 input. The Transpose op
    # passes through dtype, so the LM head MatMul will see fp32 input — but the
    # other operand (hidden states) is fp16, which would create an op-mismatch.
    # For our purpose we only care about the quantizer's int4 OUTPUT, not graph
    # correctness, so we skip the dtype propagation fix.
    print(f"  loaded + replaced in {time.time()-t:.1f}s")

    t = time.time()
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
    print(f"  quantized in {time.time()-t:.1f}s")
    m_out = quantizer.model.model
    inits = {i.name: i for i in m_out.graph.initializer}

    g_int4 = onnx.numpy_helper.to_array(inits["model.embed_tokens.weight_Q4"]).astype(np.int32)
    mnb_raw = inits["model.embed_tokens.weight_transposed_Q4"].raw_data \
        or onnx.numpy_helper.to_array(inits["model.embed_tokens.weight_transposed_Q4"]).tobytes()
    mnb_bytes = np.frombuffer(mnb_raw, dtype=np.uint8).reshape(VOCAB, HIDDEN // 2)
    mnb_uint4 = np.empty((VOCAB, HIDDEN), dtype=np.int32)
    mnb_uint4[:, 0::2] = mnb_bytes & 0x0F
    mnb_uint4[:, 1::2] = mnb_bytes >> 4
    mnb_int4 = mnb_uint4 - 8

    # Sanity: scales should still match
    g_s = onnx.numpy_helper.to_array(inits["model.embed_tokens.weight_scales"])
    m_s = onnx.numpy_helper.to_array(inits["model.embed_tokens.weight_transposed_scales"])
    print(f"  scales identical: {np.array_equal(g_s, m_s)}")

    diff = mnb_int4 - g_int4
    n = int((diff != 0).sum())
    p = int((diff == 1).sum())
    minus = int((diff == -1).sum())
    other = int(((diff != 0) & (np.abs(diff) > 1)).sum())
    print(f"  disagreements: {n:,}  +1: {p:,}  -1: {minus:,}  |Δ|>1: {other:,}")
    return n, p, minus


print("\n=== Part B: real ORT (fp32 input) ===")
n0, p0, m0 = run_real_ort(perturbation=0)
n1, p1, m1 = run_real_ort(perturbation=1e-7)
n2, p2, m2 = run_real_ort(perturbation=1e-5)

print("\n=== Summary ===")
print(f"  empirical (fp16 input):              96,015 disagreements (+1: 92,429, -1: 3,586)")
print(f"  ORT with fp32 input, no perturb:     {n0:,} disagreements (+1: {p0:,}, -1: {m0:,})")
print(f"  ORT with fp32 input + 1e-7 perturb:  {n1:,} disagreements (+1: {p1:,}, -1: {m1:,})")
