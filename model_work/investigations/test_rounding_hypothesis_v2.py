"""Refined rounding hypothesis test. Two questions:
  1. Which exact rounding rule does the MNB C++ kernel use? (half-up toward +inf
     would fully explain the 96%/4% asymmetry; half-away-from-zero would not.)
  2. Does the empirical 96k vs my numpy 48k gap come from doing the divide in
     fp16 vs fp32?
"""
from pathlib import Path
import onnx
import numpy as np

SRC = Path(__file__).parent.parent / "lfm2" / "onnx" / "model_fp16.onnx"
VOCAB, HIDDEN, BLOCK_SIZE = 65536, 1024, 64
N_BLOCKS = HIDDEN // BLOCK_SIZE


def block_quantize(w_fp16, round_mode, divide_in_fp16=False, perturb=0.0, seed=0):
    w = w_fp16.astype(np.float32)
    if perturb:
        rng = np.random.default_rng(seed)
        w = (w + (rng.uniform(-1, 1, w.shape).astype(np.float32) * perturb)).astype(np.float16).astype(np.float32)
    blocked = w.reshape(VOCAB, N_BLOCKS, BLOCK_SIZE)
    max_abs = np.abs(blocked).max(axis=2, keepdims=True)
    scale = np.where(max_abs == 0, 1, max_abs) / 7.0
    if divide_in_fp16:
        # Match C++ kernel hypothesis: convert ratio to fp16 then back, simulating
        # a fast path that holds the divided value in fp16 before rounding.
        ratio = (blocked.astype(np.float16) / scale.astype(np.float16)).astype(np.float32)
    else:
        ratio = blocked / scale
    if round_mode == "banker":
        q = np.round(ratio)
    elif round_mode == "half_away":
        q = np.where(ratio >= 0, np.floor(ratio + 0.5), -np.floor(-ratio + 0.5))
    elif round_mode == "half_up":
        q = np.floor(ratio + 0.5)        # always toward +inf
    elif round_mode == "half_down":
        q = np.ceil(ratio - 0.5)         # always toward -inf
    else:
        raise ValueError(round_mode)
    q = q.clip(-8, 7).astype(np.int32)
    return q.reshape(VOCAB, HIDDEN)


def compare(w, round_a, round_b, **kwargs):
    qa = block_quantize(w, round_a, **kwargs)
    qb = block_quantize(w, round_b, **kwargs)
    diff = qb - qa
    return int((diff != 0).sum()), int((diff == 1).sum()), int((diff == -1).sum())


print("loading…")
m = onnx.load(str(SRC))
embed = next(i for i in m.graph.initializer if i.name == "model.embed_tokens.weight")
w = onnx.numpy_helper.to_array(embed)

print("\nempirical (from real ORT run): 96,015 disagreements  +1: 92,429  -1: 3,586")
print()
print(f"{'rule_a vs rule_b':<35} {'div':<5} {'total':>10} {'+1':>10} {'-1':>10}")
print("-" * 75)

cases = [
    ("banker", "half_away"),
    ("banker", "half_up"),
    ("banker", "half_down"),
]
for ra, rb in cases:
    for div_fp16 in (False, True):
        n, p, m_ = compare(w, ra, rb, divide_in_fp16=div_fp16)
        tag = f"{ra} vs {rb}"
        div = "fp16" if div_fp16 else "fp32"
        print(f"{tag:<35} {div:<5} {n:>10,} {p:>10,} {m_:>10,}")

print("\nperturbation test (use the rule that matches empirical best):")
# Find the closest match and re-run perturbed
best = None
best_score = None
for ra, rb in cases:
    for div_fp16 in (False, True):
        n, p, m_ = compare(w, ra, rb, divide_in_fp16=div_fp16)
        # score: closeness to (96015, 92429, 3586)
        score = abs(n - 96015) + abs(p - 92429) + abs(m_ - 3586)
        if best_score is None or score < best_score:
            best_score = score
            best = (ra, rb, div_fp16, n, p, m_)
ra, rb, div_fp16, n, p, m_ = best
print(f"  closest to empirical: {ra} vs {rb}, div={'fp16' if div_fp16 else 'fp32'}  →  {n} ({p}/{m_})")

n2, p2, m2 = compare(w, ra, rb, divide_in_fp16=div_fp16, perturb=1e-5, seed=0)
print(f"  same rules, with 1e-5 random fp32 perturbation: {n2} ({p2}/{m2})")
print(f"  drop: {n} → {n2}  ({(n - n2)/n*100:.2f}% of disagreements eliminated)")
