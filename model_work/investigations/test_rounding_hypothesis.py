"""Test the hypothesis that MNB-vs-GBQ disagreement is purely a rounding-mode
tie-breaker. We bypass ORT entirely: replicate the symmetric int4 block
quantization in numpy with TWO rounding modes on the SAME fp16 weight, and
see if (a) the disagreement count matches the empirical 96,015, and (b) a
tiny input perturbation that eliminates exact half-integer ties drops the
disagreement to near zero."""

from pathlib import Path
import onnx
import numpy as np

SRC = Path(__file__).parent.parent / "lfm2" / "onnx" / "model_fp16.onnx"
VOCAB, HIDDEN, BLOCK_SIZE = 65536, 1024, 64
N_BLOCKS = HIDDEN // BLOCK_SIZE


def block_quantize(w_fp16, round_mode):
    """Symmetric int4 block quantization, axis=1, block_size=64.
    round_mode: 'banker' (numpy half-to-even) or 'half_up' (floor(x + 0.5))."""
    w = w_fp16.astype(np.float32)
    blocked = w.reshape(VOCAB, N_BLOCKS, BLOCK_SIZE)
    max_abs = np.abs(blocked).max(axis=2, keepdims=True)        # [V, B, 1]
    scale = np.where(max_abs == 0, 1, max_abs) / 7.0            # match ORT
    ratio = blocked / scale                                     # [V, B, 64]
    if round_mode == "banker":
        q = np.round(ratio)                                     # half-to-even
    elif round_mode == "half_up":
        # round-half-up: floor(x + 0.5) for x>=0; for x<0 the C++ idiom
        # `(int)(x + 0.5)` truncates toward zero AFTER adding 0.5, giving
        # half-up for positive and half-down (toward -inf) for negative.
        # Use the symmetric "half-away-from-zero" variant which is what
        # std::round and many C++ quant kernels actually do:
        q = np.where(ratio >= 0, np.floor(ratio + 0.5), -np.floor(-ratio + 0.5))
    else:
        raise ValueError(round_mode)
    q = q.clip(-8, 7).astype(np.int32)
    return q.reshape(VOCAB, HIDDEN), scale.squeeze(-1)


# Load the embedding weight only (skip the rest of the graph).
print(f"loading embedding from {SRC.name}…")
m = onnx.load(str(SRC))
embed = next(i for i in m.graph.initializer if i.name == "model.embed_tokens.weight")
w_fp16 = onnx.numpy_helper.to_array(embed)
print(f"  shape={w_fp16.shape}, dtype={w_fp16.dtype}")

# === Experiment 1: same weight, two rounding modes ==========================
print("\n=== Experiment 1: original weight, banker vs half-away-from-zero ===")
q_banker, s1 = block_quantize(w_fp16, "banker")
q_halfup,  s2 = block_quantize(w_fp16, "half_up")
assert np.allclose(s1, s2), "scales must match"
diff = q_halfup - q_banker
total = diff.size
n_diff = int((diff != 0).sum())
n_plus = int((diff == 1).sum())
n_minus = int((diff == -1).sum())
print(f"  total weights: {total:,}")
print(f"  disagreements: {n_diff:,} ({n_diff/total*100:.4f}%)  +1: {n_plus:,}  -1: {n_minus:,}")
print(f"  empirical (from MNB-vs-GBQ run earlier): 96,015 disagreements (+1: 92,429, -1: 3,586)")

# === Experiment 2: perturb input by sub-ULP noise to break exact ties =======
print("\n=== Experiment 2: perturbed weight (random ±1 fp32 ULP-ish noise) ===")
rng = np.random.default_rng(0)
# Add a tiny non-half-integer offset to each weight in fp32 space. The smallest
# nonzero perturbation that reliably nudges every value off a half-integer
# boundary — in `w/scale` space — is something like 1e-4 * scale_typical.
# Empirically the scales here are ~1e-2, so a fp32 noise of 1e-6 should be
# more than enough to break ties without changing any non-tied rounding.
noise = (rng.uniform(-1, 1, w_fp16.shape) * 1e-6).astype(np.float32)
w_perturbed = (w_fp16.astype(np.float32) + noise).astype(np.float16)
q_banker_p, _ = block_quantize(w_perturbed, "banker")
q_halfup_p, _ = block_quantize(w_perturbed, "half_up")
diff_p = q_halfup_p - q_banker_p
n_diff_p = int((diff_p != 0).sum())
print(f"  disagreements after perturbation: {n_diff_p:,} ({n_diff_p/total*100:.4f}%)")
print(f"  drop: {n_diff} → {n_diff_p}  ({(n_diff - n_diff_p)/n_diff*100:.2f}% of original disagreements were ties)")

# === Experiment 3: probe at the individual half-integer boundaries ===========
# Among differing positions, what was the original `ratio` value?
print("\n=== Experiment 3: where were the disagreements before perturbation? ===")
# Recompute ratio (fp32 division) for inspection
blocked = w_fp16.astype(np.float32).reshape(VOCAB, N_BLOCKS, BLOCK_SIZE)
max_abs = np.abs(blocked).max(axis=2, keepdims=True)
scale = np.where(max_abs == 0, 1, max_abs) / 7.0
ratio = (blocked / scale).reshape(VOCAB, HIDDEN)
mask = (q_halfup != q_banker)
# Distance from nearest half-integer for differing positions
ratio_diff = ratio[mask]
nearest_half = np.round(ratio_diff * 2) / 2     # snap to nearest 0.5
dist = np.abs(ratio_diff - nearest_half)
print(f"  on differing positions (n={mask.sum():,}):")
print(f"    distance from nearest half-integer  mean={dist.mean():.2e}  max={dist.max():.2e}")
print(f"    fraction within 1e-5 of a half-integer: {(dist < 1e-5).mean()*100:.2f}%")

# Quick check of the half-integer values that show disagreement
print(f"\n  histogram of `floor(ratio_at_disagreement)` (i.e. which boundary):")
unique, counts = np.unique(np.floor(ratio_diff).astype(int), return_counts=True)
for u, c in zip(unique.tolist(), counts.tolist()):
    print(f"    [{u:+d}.5, between {u} and {u+1}): {c:>10,}")
