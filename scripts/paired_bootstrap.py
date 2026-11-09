"""Paired bootstrap for the difference in AUC between Min-K% and Zlib (Pythia-2.8B, WikiMIA).

Both methods score the same passages, so overlapping per-method intervals do not test their difference.
This resamples passages within each length split, recomputes both AUCs on the same resample, and averages
the four per-length differences. AUC follows eval.py's convention: roc_curve(labels, -score).
Usage:  python scripts/paired_bootstrap.py      (writes results/replication/paired_delta.json)
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
LENGTHS, B, SEED = [32, 64, 128, 256], 1000, 0
A, Z = "Min_20.0% Prob", "ppl/zlib"

splits = []
for L in LENGTHS:
    rows = [json.loads(l) for l in (ROOT / f"results/replication/scores/pythia-2.8b_len{L}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    y = np.array([r["label"] for r in rows])
    splits.append((y, -np.array([r["pred"][A] for r in rows]), -np.array([r["pred"][Z] for r in rows])))

point = np.mean([roc_auc_score(y, a) - roc_auc_score(y, z) for y, a, z in splits])
rng, deltas = np.random.default_rng(SEED), []
for _ in range(B):
    d = []
    for y, a, z in splits:
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], z[i]))
    deltas.append(np.mean(d))
lo, hi = np.percentile(deltas, [2.5, 97.5])
out = {"minK_auc_mean": float(np.mean([roc_auc_score(y, a) for y, a, _ in splits])),
       "zlib_auc_mean": float(np.mean([roc_auc_score(y, z) for y, _, z in splits])),
       "delta_mean": float(point), "ci95": [float(lo), float(hi)], "share_delta_le_0": float(np.mean(np.array(deltas) <= 0)),
       "resamples": B, "seed": SEED}
(ROOT / "results/replication/paired_delta.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
