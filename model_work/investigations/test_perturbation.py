"""Definitive perturbation test: nudge each fp16 weight by exactly 1 fp16 ULP
in a random direction. This GUARANTEES the perturbation survives any fp16 cast,
and any value that was sitting exactly on a half-integer boundary in `w/scale`
space gets pushed off it. If the rounding-mode hypothesis is right,
disagreements should drop to ~zero."""
from pathlib import Path
import onnx
import numpy as np

SRC = Path(__file__).parent.parent / "lfm2" / "onnx" / "model_fp16.onnx"
VOCAB, HIDDEN, BLOCK_SIZE = 65536, 1024, 64
N_BLOCKS = HIDDEN // BLOCK_SIZE


def block_quantize(w_fp16, round_mode):
    w = w_fp16.astype(np.float32)
    blocked = w.reshape(VOCAB, N_BLOCKS, BLOCK_SIZE)
    max_abs = np.abs(blocked).max(axis=2, keepdims=True)
    scale = np.where(max_abs == 0, 1, max_abs) / 7.0
    ratio = blocked / scale
    if round_mode == "banker":
        q = np.round(ratio)
    elif round_mode == "half_up":
        q = np.floor(ratio + 0.5)
    q = q.clip(-8, 7).astype(np.int32)
    return q.reshape(VOCAB, HIDDEN)


print("loading…")
m = onnx.load(str(SRC))
embed = next(i for i in m.graph.initializer if i.name == "model.embed_tokens.weight")
w_fp16 = onnx.numpy_helper.to_array(embed)

# Baseline: banker vs half-up on the original fp16 weight
qa = block_quantize(w_fp16, "banker")
qb = block_quantize(w_fp16, "half_up")
n_orig = int((qa != qb).sum())
print(f"\noriginal: {n_orig:,} disagreements (banker vs half-up, fp16 input)")

# Perturb by 1 fp16 ULP in a random direction per element
rng = np.random.default_rng(42)
direction = rng.choice([-1, 1], size=w_fp16.shape).astype(np.int8)
# np.nextafter shifts by exactly 1 ULP toward the given target.
target = np.where(direction > 0, np.float16(np.inf), np.float16(-np.inf))
w_perturbed = np.nextafter(w_fp16, target)
# Sanity: every value should have changed (except where w==0, nextafter still moves)
n_unchanged = int((w_perturbed == w_fp16).sum())
print(f"  fp16 values unchanged after 1-ULP nudge: {n_unchanged} (should be 0 or near-0)")

qa_p = block_quantize(w_perturbed, "banker")
qb_p = block_quantize(w_perturbed, "half_up")
n_perturbed = int((qa_p != qb_p).sum())
print(f"\nafter 1-ULP perturbation: {n_perturbed:,} disagreements")
print(f"  drop: {n_orig:,} → {n_perturbed:,}  ({(n_orig - n_perturbed)/n_orig*100:.2f}% of disagreements gone)")

# How does the perturbation affect any individual weight's quantization?
# (Did rounding-stable weights stay put under banker, or did the perturbation
# move some across non-tie boundaries?)
banker_orig = qa
banker_pert = qa_p
moved_under_banker = int((banker_orig != banker_pert).sum())
print(f"\nsanity: 1-ULP fp16 perturbation moved {moved_under_banker:,} banker-quantized weights ({moved_under_banker/qa.size*100:.4f}%)")
print("  (these are values that were already very close to an integer-step boundary;")
print("   the perturbation was enough to push them across — expected to be small.)")
