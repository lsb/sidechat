"""[fp32 → ] fp16 → bake Transpose(embed) → int4 weight-only quantize (MatMulNBits
+ GatherBlockQuantized, block_size=32, symmetric, accuracy_level=4) → rewrite
the GatherBlockQuantized int4 weight into uint4 (add 8 to values, zero_point=8)
so its packed bytes match the MatMulNBits uint8 layout exactly → save with
external data, both initializers pointing at the SAME byte range. On-disk
weight stored once, ORT can still decode it through both interpretations at
load.

Model-specific knobs are in the CONFIG block below — change those to retarget
the script at a different model. SmolLM2-360M and LFM2.5-350M ship with
`tie_word_embeddings: true` and use the same `model.embed_tokens.weight`
initializer name, so the only differences are dimensions and whether we have
to do the fp32→fp16 conversion ourselves."""

from pathlib import Path
import os
import onnx
import numpy as np
from onnx import TensorProto, helper
from onnx.external_data_helper import _is_valid_filename  # side-effect: loaded
from onnxconverter_common import float16
from onnxruntime.quantization import matmul_nbits_quantizer, quant_utils
from collections import Counter

float16.sort_topology = lambda g: None

# === CONFIG ================================================================
# LFM2.5-350M: vocab=65536, hidden=1024; LiquidAI ships fp16 directly so we
# can skip the float16 conversion. SmolLM2-360M was vocab=49152, hidden=960,
# SRC=model.onnx (fp32), SRC_IS_FP16=False.
SRC          = Path(__file__).parent / "lfm2" / "onnx" / "model_fp16.onnx"
DST          = Path(__file__).parent / "model_q4f16.onnx"
DATA         = Path(__file__).parent / "model_q4f16.onnx_data"
SRC_IS_FP16  = True
VOCAB        = 65536
HIDDEN       = 1024
BLOCK_SIZE   = 64                  # HIDDEN must be divisible (LFM2: 1024/64=16)
TARGET_N_CHUNKS = 4
CHUNK_HARD_CAP  = 50 * 1000 * 1000  # GitHub Pages 100MB cap with headroom
assert HIDDEN % BLOCK_SIZE == 0, "BLOCK_SIZE must divide HIDDEN evenly"
N_BLOCKS = HIDDEN // BLOCK_SIZE
BLOB_SIZE = BLOCK_SIZE // 2        # uint8 bytes per block (2 int4 vals/byte)
# ===========================================================================


def bake_transpose_into_initializer(model):
    """Replace `Transpose(initializer)` with a precomputed transposed
    initializer named `<input>_transposed`, and rewire consumers of the old
    Transpose output to read the new initializer. Deterministic naming matters
    because the downstream MatMulNBits quantization derives the int4 weight
    name from the initializer name (`<name>_Q4`/`<name>_scales`)."""
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
    return len(to_remove)


def summarize(path):
    m = onnx.load(str(path), load_external_data=False)
    return f"file={os.path.getsize(path)/1e6:.1f}MB, nodes={len(m.graph.node)}, init={len(m.graph.initializer)}"


# === 1. Quantize as normal (produces separate bytes for GBQ int4 and MNB uint8)
print(f"loading {SRC}…")
model_fp16 = onnx.load(str(SRC))
if not SRC_IS_FP16:
    print("fp32 → fp16…")
    model_fp16 = float16.convert_float_to_float16(
        # keep_io_types=False makes the model's own inputs/outputs fp16 too, so
        # transformers.js at `dtype: 'q4f16'` (which feeds fp16 tensors through
        # the generate loop) matches the graph signature exactly. With True we'd
        # get "Unexpected input data type. Actual: float16, expected: float" at
        # inference time on the fp32 input ports.
        model_fp16, keep_io_types=False, disable_shape_infer=True, check_fp16_ready=False,
    )
print("bake Transpose → quantize…")
bake_transpose_into_initializer(model_fp16)
quantizer = matmul_nbits_quantizer.MatMulNBitsQuantizer(
    model_fp16,
    algo_config=matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
        block_size=BLOCK_SIZE, is_symmetric=True, accuracy_level=4,
        quant_format=quant_utils.QuantFormat.QOperator,
        op_types_to_quantize=("MatMul", "Gather"),
    ),
)
quantizer.process()
model = quantizer.model.model
for o in model.opset_import:
    if o.domain == "" and o.version < 21:
        o.version = 21

inits = {i.name: i for i in model.graph.initializer}
GATHER_W = "model.embed_tokens.weight_Q4"            # int4  [VOCAB, HIDDEN]
GATHER_S = "model.embed_tokens.weight_scales"        # fp16  [VOCAB, N_BLOCKS]
MNB_W    = "model.embed_tokens.weight_transposed_Q4"  # uint8 [VOCAB, N_BLOCKS, BLOB_SIZE]
MNB_S    = "model.embed_tokens.weight_transposed_scales"  # fp16 [VOCAB, N_BLOCKS]

# === 2. Verify scales are identical; measure how close the two quantizers
#        came to bit-for-bit agreement on the packed bytes, then use the
#        MatMulNBits bytes for both (nudges ~0.3% of embedding positions by
#        ±1 int4 step; well within noise).
g_int4 = onnx.numpy_helper.to_array(inits[GATHER_W]).astype(np.int32)  # [VOCAB, HIDDEN] signed
g_scales = onnx.numpy_helper.to_array(inits[GATHER_S])
m_scales = onnx.numpy_helper.to_array(inits[MNB_S])
assert np.array_equal(g_scales, m_scales), "scales differ; dedupe unsafe"
u4 = ((g_int4 + 8) % 16).astype(np.uint8)
packed = (u4[:, 0::2] | (u4[:, 1::2] << 4)).astype(np.uint8)
mnb_bytes = bytes(inits[MNB_W].raw_data) if inits[MNB_W].raw_data else onnx.numpy_helper.to_array(inits[MNB_W]).tobytes()
mnb_flat = np.frombuffer(mnb_bytes, dtype=np.uint8).reshape(VOCAB, HIDDEN // 2)
agree = (packed == mnb_flat).mean()
print(f"gather(+8) vs matmulnbits packed bytes agree on {agree*100:.2f}% of {mnb_flat.size/1e6:.1f}M bytes "
      f"(remainder differs by ≤ one int4 step, FP rounding in independent codepaths)")
shared_bytes = mnb_bytes  # use the MatMulNBits bytes as the canonical shared copy

# === 3. Rewrite Gather's weight as uint4 with zero_point=8, pointing at the
#        same bytes as MatMulNBits's B.
# Delete the old MNB_W, MNB_S initializers; GBQ and MNB will both read from
# a single new initializer.

# Create a zero_point initializer for GBQ: uint4 value 8 per block (symmetric
# uint4). Shape matches scales: [49152, n_blocks]. ONNX packs uint4 as two
# values per byte in *flat* element order, so for odd n_blocks the row
# boundaries don't align to byte boundaries — must pack over the flat array.
n_blocks = N_BLOCKS
total_zp = VOCAB * n_blocks
zp_flat = np.full(total_zp, 8, dtype=np.uint8)
# Pack 2 uint4 values per byte, low nibble first
zp_packed = np.zeros((total_zp + 1) // 2, dtype=np.uint8)
zp_packed[:total_zp // 2] = zp_flat[0:total_zp - (total_zp % 2):2] | (zp_flat[1::2] << 4)
if total_zp % 2 == 1:
    zp_packed[-1] = zp_flat[-1]  # odd trailing value in low nibble
# Create a UINT4 tensor; ONNX helper doesn't directly support building uint4
# from numpy, so we construct the TensorProto manually.
def make_uint4_tensor(name, shape, packed_bytes):
    t = TensorProto()
    t.name = name
    t.data_type = TensorProto.UINT4
    t.dims.extend(shape)
    t.raw_data = bytes(packed_bytes)
    return t

zp_init = make_uint4_tensor("model.embed_tokens.weight_zero_point", [VOCAB, n_blocks], zp_packed.tobytes())

# Replace the int4 Gather weight with a uint4 tensor holding the +8-shifted bytes
shared_bytes = mnb_bytes  # the bytes both ops will reference
new_gather_weight = make_uint4_tensor(GATHER_W, [VOCAB, HIDDEN], shared_bytes)

# Rebuild the initializer list: drop old Gather int4 + old MNB uint8 + add the new uint4 + zero_point
new_inits = []
drop_names = {GATHER_W, MNB_W}
for init in model.graph.initializer:
    if init.name in drop_names:
        continue
    new_inits.append(init)
new_inits.append(new_gather_weight)  # uint4 shared with MNB
new_inits.append(zp_init)

# We ALSO need MatMulNBits to reference the shared uint4 bytes. MatMulNBits
# expects B with shape [N, n_blocks, blob_size] and dtype uint8. ONNX treats
# each initializer as a distinct named tensor; we cannot reuse GATHER_W (a
# uint4 [VOCAB, HIDDEN] tensor) as B (a uint8 [VOCAB, n_blocks, blob_size]
# tensor) because the (shape, dtype) differ. BUT: when we later write the
# external data file below, we point both initializer protos at the SAME
# byte offset in the file — disk-level dedupe.
# Recreate MNB_W as a uint8 [VOCAB, n_blocks, BLOB_SIZE] tensor holding the same bytes:
mnb_tensor = TensorProto()
mnb_tensor.name = MNB_W
mnb_tensor.data_type = TensorProto.UINT8
mnb_tensor.dims.extend([VOCAB, n_blocks, BLOB_SIZE])
mnb_tensor.raw_data = shared_bytes
new_inits.append(mnb_tensor)

del model.graph.initializer[:]
model.graph.initializer.extend(new_inits)

# Update the GatherBlockQuantized node to: (a) use our new uint4 weight,
# (b) include the zero_point input.
for node in model.graph.node:
    if node.op_type == "GatherBlockQuantized":
        # Inputs are typically [data, indices, scales] and optional zero_points.
        # Append zero_point if absent; otherwise replace.
        while len(node.input) < 4:
            node.input.append("")
        node.input[3] = zp_init.name
        # Ensure the GBQ node knows which axis/block — these are attributes, not inputs.
        break

print("rewrote Gather's int4 weight → uint4 (values +=8 mod 16, zero_point=8); MatMulNBits weight unchanged")

# === 4. Save with external data split into multiple files of ≤ CHUNK_TARGET
#        bytes each. Both shared-bytes initializers (GATHER_W and MNB_W)
#        reference the same (location, offset, length) in the first chunk —
#        the 23.59 MB embedding is stored exactly once on disk.

BASE_NAME = "model_q4f16.onnx_data"

class ChunkWriter:
    """Writes binary blobs to a sequence of chunk files, opening a new chunk
    *before* any write that would push the current chunk over the cap.
    Assumes no individual blob exceeds the cap."""
    def __init__(self, out_dir, base_name, cap):
        self.out_dir = out_dir
        self.base_name = base_name
        self.cap = cap
        self.chunks = []  # list of [name, file_handle, size_so_far]

    def _open_new(self):
        idx = len(self.chunks)
        # transformers.js expects "<name>_data", "<name>_data_1", "<name>_data_2", ...
        # (underscore, not dot). base_name already ends with "_data".
        name = self.base_name if idx == 0 else f"{self.base_name}_{idx}"
        path = self.out_dir / name
        self.chunks.append([name, open(path, "wb"), 0])

    def _ensure_room(self, need):
        assert need <= self.cap, f"single blob ({need} B) exceeds chunk cap ({self.cap} B)"
        if not self.chunks or self.chunks[-1][2] + need > self.cap:
            self._open_new()
        return self.chunks[-1]

    def write(self, data):
        entry = self._ensure_room(len(data))
        name = entry[0]
        offset = entry[2]
        entry[1].write(data)
        entry[2] += len(data)
        return name, offset, len(data)

    def close(self):
        sizes = []
        for name, fh, size in self.chunks:
            fh.close()
            sizes.append((name, size))
        return sizes


# Clean up any previous data files (including chunked ones)
for p in Path(__file__).parent.glob(BASE_NAME + "*"):
    p.unlink()

# Pass 1: figure out how many bytes of external data we have total, so we can
# choose a chunk cap that spreads them across exactly TARGET_N_CHUNKS files.
EXTERNAL_THRESHOLD = 1024
total_external = len(shared_bytes)
for init in model.graph.initializer:
    if init.name in (GATHER_W, MNB_W):
        continue  # already accounted for by shared_bytes
    if init.raw_data and len(init.raw_data) >= EXTERNAL_THRESHOLD:
        total_external += len(init.raw_data)

# Bump the chunk count if the configured target won't fit under the hard cap.
n_chunks = max(TARGET_N_CHUNKS, (total_external + CHUNK_HARD_CAP - 1) // CHUNK_HARD_CAP)
ideal_per_chunk = (total_external + n_chunks - 1) // n_chunks
# Add a small slack (2%) so greedy packing doesn't spill into an extra chunk.
chunk_cap = int(ideal_per_chunk * 1.02)
assert chunk_cap <= CHUNK_HARD_CAP, "chunk cap exceeded hard cap after sizing"
print(f"external data total {total_external/1e6:.2f} MB → chunk cap {chunk_cap/1e6:.2f} MB "
      f"({n_chunks} chunks)")

writer = ChunkWriter(Path(__file__).parent, BASE_NAME, chunk_cap)

# Write the shared bytes first (into chunk 0) so both aliases fit in the first file.
shared_loc, shared_off, shared_len = writer.write(shared_bytes)

for init in model.graph.initializer:
    if init.name in (GATHER_W, MNB_W):
        init.ClearField("raw_data")
        init.data_location = TensorProto.EXTERNAL
        del init.external_data[:]
        for k, v in [("location", shared_loc), ("offset", str(shared_off)), ("length", str(shared_len))]:
            e = init.external_data.add()
            e.key = k
            e.value = v
        continue
    if init.raw_data and len(init.raw_data) >= EXTERNAL_THRESHOLD:
        b = init.raw_data
        init.ClearField("raw_data")
        init.data_location = TensorProto.EXTERNAL
        del init.external_data[:]
        loc, off, ln = writer.write(b)
        for k, v in [("location", loc), ("offset", str(off)), ("length", str(ln))]:
            e = init.external_data.add()
            e.key = k
            e.value = v

chunk_sizes = writer.close()
onnx.save(model, str(DST))
print(f"done — {summarize(DST)}")
print(f"external data split across {len(chunk_sizes)} files:")
for name, size in chunk_sizes:
    print(f"  {name}: {size/1e6:.2f} MB")
print(f"shared embedding blob ({shared_len/1e6:.2f} MB) in {shared_loc}, referenced by both {GATHER_W} and {MNB_W}")
