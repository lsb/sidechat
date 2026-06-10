"""KL divergence: dedup'd LFM2.5-350M vs un-dedup'd, on random tokens.

Both variants have identical structure and identical bytes for everything
EXCEPT the embedding lookup. The deduped variant has GBQ reading MNB's int4
bytes; the un-deduped variant has GBQ reading its own (banker-rounded) int4
bytes. We translate both into the same uint4 + zero_point=8 layout so the
graph topology is identical between the two — the only difference is the
content of one initializer."""

from pathlib import Path
import time, os, gc
import onnx
import numpy as np
from onnx import TensorProto, numpy_helper
from onnxconverter_common import float16
from onnxruntime.quantization import matmul_nbits_quantizer, quant_utils
import onnxruntime as ort

float16.sort_topology = lambda g: None

SRC = Path(__file__).parent.parent / "lfm2" / "onnx" / "model_fp16.onnx"
OUT_DEDUPED   = Path(__file__).parent.parent / "kl_test_deduped"
OUT_UNDEDUPED = Path(__file__).parent.parent / "kl_test_undeduped"
VOCAB, HIDDEN, BLOCK_SIZE = 65536, 1024, 64
N_BLOCKS = HIDDEN // BLOCK_SIZE
BLOB_SIZE = BLOCK_SIZE // 2

N_SEQS = 100        # random token sequences
SEQ_LEN = 256

OUT_DEDUPED.mkdir(exist_ok=True)
OUT_UNDEDUPED.mkdir(exist_ok=True)


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


def make_uint4(name, shape, packed_bytes):
    t = TensorProto()
    t.name = name; t.data_type = TensorProto.UINT4
    t.dims.extend(shape); t.raw_data = bytes(packed_bytes)
    return t


def build_variant(quantized_model_proto, gbq_bytes_uint4_packed, mnb_bytes_uint8_packed, out_dir):
    """Take the quantizer's output model, rewrite the embedding init pair
    so GBQ is uint4 with zp=8 reading `gbq_bytes_uint4_packed`, and
    MNB is uint8 packed reading `mnb_bytes_uint8_packed`. For the deduped
    variant, pass the SAME bytes for both. For the un-deduped variant,
    pass GBQ-quantized bytes (g_packed) for GBQ and MNB-quantized bytes
    (mnb_raw) for MNB."""
    GATHER_W = "model.embed_tokens.weight_Q4"
    MNB_W    = "model.embed_tokens.weight_transposed_Q4"

    inits = {i.name: i for i in quantized_model_proto.graph.initializer}
    total_zp = VOCAB * N_BLOCKS
    zp_flat = np.full(total_zp, 8, dtype=np.uint8)
    zp_packed = np.zeros((total_zp + 1) // 2, dtype=np.uint8)
    zp_packed[:total_zp // 2] = zp_flat[0:total_zp - (total_zp % 2):2] | (zp_flat[1::2] << 4)
    zp_init = make_uint4("model.embed_tokens.weight_zero_point", [VOCAB, N_BLOCKS], zp_packed.tobytes())

    new_gbq = make_uint4(GATHER_W, [VOCAB, HIDDEN], gbq_bytes_uint4_packed)
    new_mnb = TensorProto()
    new_mnb.name = MNB_W; new_mnb.data_type = TensorProto.UINT8
    new_mnb.dims.extend([VOCAB, N_BLOCKS, BLOB_SIZE])
    new_mnb.raw_data = bytes(mnb_bytes_uint8_packed)

    # Replace initializers
    new_inits = []
    drop = {GATHER_W, MNB_W}
    for init in quantized_model_proto.graph.initializer:
        if init.name in drop:
            continue
        new_inits.append(init)
    new_inits.append(new_gbq)
    new_inits.append(new_mnb)
    new_inits.append(zp_init)
    del quantized_model_proto.graph.initializer[:]
    quantized_model_proto.graph.initializer.extend(new_inits)

    # Wire up zero_point input on the GatherBlockQuantized node
    for node in quantized_model_proto.graph.node:
        if node.op_type == "GatherBlockQuantized":
            while len(node.input) < 4:
                node.input.append("")
            node.input[3] = zp_init.name
            break

    onnx.save(
        quantized_model_proto,
        str(out_dir / "model.onnx"),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.onnx_data",
        size_threshold=1024,
    )


# === Step 1: run quantizer once, capture both int4 layouts ==================
print(f"loading {SRC}…")
model = onnx.load(str(SRC))
print("baking transpose + quantizing…")
t0 = time.time()
bake_transpose(model)
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

mq = quantizer.model.model
inits = {i.name: i for i in mq.graph.initializer}
g_int4 = numpy_helper.to_array(inits["model.embed_tokens.weight_Q4"]).astype(np.int32)
mnb_raw = inits["model.embed_tokens.weight_transposed_Q4"].raw_data \
    or numpy_helper.to_array(inits["model.embed_tokens.weight_transposed_Q4"]).tobytes()
print(f"  GBQ int4 captured: shape={g_int4.shape}")
print(f"  MNB bytes captured: {len(mnb_raw):,} bytes")

# Build the GBQ-as-uint4 packed bytes: (g_int4 + 8) packed two-per-byte
g_uint4 = ((g_int4 + 8) % 16).astype(np.uint8)
g_packed = (g_uint4[:, 0::2] | (g_uint4[:, 1::2] << 4)).astype(np.uint8).tobytes()
print(f"  GBQ uint4 packed: {len(g_packed):,} bytes (== MNB bytes? {g_packed == mnb_raw})")
diff = (np.frombuffer(g_packed, dtype=np.uint8) != np.frombuffer(mnb_raw, dtype=np.uint8)).sum()
print(f"  byte differences between the two: {diff:,}")

# === Step 2: write both variants ============================================
import copy
print("\nwriting deduped variant (GBQ→mnb_raw, MNB→mnb_raw)…")
build_variant(copy.deepcopy(mq), mnb_raw, mnb_raw, OUT_DEDUPED)
print("writing un-deduped variant (GBQ→g_packed, MNB→mnb_raw)…")
build_variant(copy.deepcopy(mq), g_packed, mnb_raw, OUT_UNDEDUPED)


# === Step 3: load both, run on random tokens, compute KL ====================
def make_kv_inputs(batch_size):
    """Build the full input dict for one forward pass with no past context."""
    # past_conv: [B, 1024, 3] of zeros
    # past_key_values: [B, 8, 0, 64] zero-length on seq axis
    inputs = {}
    # 16 layers, layer_types: 10 conv + 6 attention
    # From the inspection: conv layers at 0,1,3,4,6,7,9,11,13,15; attention at 2,5,8,10,12,14
    conv_layers = {0, 1, 3, 4, 6, 7, 9, 11, 13, 15}
    attn_layers = {2, 5, 8, 10, 12, 14}
    for i in range(16):
        if i in conv_layers:
            inputs[f"past_conv.{i}"] = np.zeros((batch_size, 1024, 3), dtype=np.float16)
        else:
            inputs[f"past_key_values.{i}.key"] = np.zeros((batch_size, 8, 0, 64), dtype=np.float16)
            inputs[f"past_key_values.{i}.value"] = np.zeros((batch_size, 8, 0, 64), dtype=np.float16)
    return inputs


def softmax_log_fp32(logits):
    """Numerically stable log-softmax in fp32. logits: [..., V] fp16/fp32."""
    x = logits.astype(np.float32)
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


print("\nloading inference sessions…")
sopt = ort.SessionOptions()
sopt.log_severity_level = 3
t0 = time.time()
sess_fp16 = ort.InferenceSession(str(SRC), sopt, providers=["CPUExecutionProvider"])
sess_d    = ort.InferenceSession(str(OUT_DEDUPED / "model.onnx"), sopt, providers=["CPUExecutionProvider"])
sess_u    = ort.InferenceSession(str(OUT_UNDEDUPED / "model.onnx"), sopt, providers=["CPUExecutionProvider"])
print(f"  loaded in {time.time()-t0:.1f}s")

rng = np.random.default_rng(0)
# Per-pair containers
pairs = {
    "dedup_vs_undedup": {"kls": [], "max_logit_diff": 0.0, "top1_disagree": 0},
    "dedup_vs_fp16":    {"kls": [], "max_logit_diff": 0.0, "top1_disagree": 0},
    "undedup_vs_fp16":  {"kls": [], "max_logit_diff": 0.0, "top1_disagree": 0},
}
total_positions = 0

def update(pair_name, p_logits, q_logits):
    log_p = softmax_log_fp32(p_logits)
    log_q = softmax_log_fp32(q_logits)
    p = np.exp(log_p)
    kl = (p * (log_p - log_q)).sum(axis=-1)
    pairs[pair_name]["kls"].append(kl)
    diff = float(np.abs(p_logits.astype(np.float32) - q_logits.astype(np.float32)).max())
    pairs[pair_name]["max_logit_diff"] = max(pairs[pair_name]["max_logit_diff"], diff)
    pairs[pair_name]["top1_disagree"] += int((p_logits.argmax(-1) != q_logits.argmax(-1)).sum())

t_start = time.time()
print(f"\nrunning {N_SEQS} sequences × {SEQ_LEN} tokens…")
for seq_i in range(N_SEQS):
    input_ids = rng.integers(0, VOCAB, size=(1, SEQ_LEN), dtype=np.int64)
    attn_mask = np.ones((1, SEQ_LEN), dtype=np.int64)
    base_inputs = {
        "input_ids": input_ids,
        "attention_mask": attn_mask,
        "num_logits_to_keep": np.array(0, dtype=np.int64),
        **make_kv_inputs(1),
    }
    t = time.time()
    out_fp16 = sess_fp16.run(["logits"], base_inputs)[0][0]    # [S, V]
    out_d    = sess_d.run(["logits"], base_inputs)[0][0]
    out_u    = sess_u.run(["logits"], base_inputs)[0][0]
    elapsed = time.time() - t

    update("dedup_vs_undedup", out_d, out_u)
    update("dedup_vs_fp16",    out_d, out_fp16)
    update("undedup_vs_fp16",  out_u, out_fp16)
    total_positions += SEQ_LEN

    if seq_i < 3 or seq_i == N_SEQS - 1 or seq_i % 25 == 0:
        kl_dvf = pairs["dedup_vs_fp16"]["kls"][-1].mean()
        kl_uvf = pairs["undedup_vs_fp16"]["kls"][-1].mean()
        kl_duv = pairs["dedup_vs_undedup"]["kls"][-1].mean()
        print(f"  seq {seq_i+1:>3}/{N_SEQS}: {elapsed:.1f}s  KL(d||fp16)={kl_dvf:.3e}  KL(u||fp16)={kl_uvf:.3e}  KL(d||u)={kl_duv:.3e}")

# === Aggregate ==============================================================
print(f"\n=== Results across {total_positions:,} token positions ===")
for name, info in pairs.items():
    all_kl = np.concatenate(info["kls"])
    print(f"\n  {name}:")
    print(f"    KL  mean={all_kl.mean():.4e}  median={np.median(all_kl):.4e}  p99={np.quantile(all_kl, 0.99):.4e}  max={all_kl.max():.4e}")
    print(f"    max |Δlogit|: {info['max_logit_diff']:.4e}")
    print(f"    top-1 disagreements: {info['top1_disagree']:,}/{total_positions:,}  ({info['top1_disagree']/total_positions*100:.3f}%)")

print(f"\n  total wall time: {time.time()-t_start:.1f}s")
