"""Shift-controlled evaluation (ND-2): does Min-K% still separate members from non-members once the passages
are made indistinguishable from their text alone?

For every passage we have (a) its Min-K% score from Pythia-2.8B (notebooks 01/02) and (b) the out-of-fold
probability of being a member from the blind TF-IDF classifier, which never sees the model
(scripts/blind_baseline.py). The blind probability is a "propensity score" for the time shift. Two controls:

  stratified : AUC computed only over (member, non-member) pairs that fall in the same quintile of the
               blind score, so pairs the text alone already separates are excluded.
  matched    : greedy 1:1 nearest-neighbour matching of members to non-members on logit(blind score),
               caliper 0.2 SD, without replacement. On the matched set the blind classifier is near chance by
               construction; the question is whether Min-K% is too.

If Min-K% keeps its AUC under both controls it carries signal the text does not; if it falls to ~0.5, its
benchmark AUC is explained by the same shift the blind classifier exploits.

Outputs results/shift_controlled/summary.csv (one row per dataset x length x control x method) and
data/wikimia2026_matched/ (the matched WikiMIA-2026 subset at each length, in the original JSONL schema).
Usage:  python scripts/shift_controlled.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
LENGTHS = [32, 64, 128, 256]
DATA = {"WikiMIA": ("data/external/wikimia/WikiMIA_length{L}.jsonl", "results/replication/scores/pythia-2.8b_len{L}.jsonl"),
        "WikiMIA-2026": ("data/wikimia2026/WikiMIA2026_length{L}.jsonl", "results/wikimia2026/scores/pythia-2.8b_len{L}.jsonl")}
METHODS = {"Min-20%": "Min_20.0% Prob", "PPL": "ppl", "Zlib": "ppl/zlib"}
CONFOUNDS = {"tfidf_lr": "blind TF-IDF (raw text)", "tfidf_lr_noyr": "blind TF-IDF (years masked)"}


def read(path):
    return [json.loads(l) for l in (ROOT / path).read_text(encoding="utf-8").splitlines() if l.strip()]


def load(name, L):
    """Rows in dataset-file order with label, Min-K%/PPL/Zlib scores (higher = member) and blind scores.
    Scores are joined by passage TEXT: the original eval.load_jsonl shuffles rows, so positions differ."""
    data_path, score_path = DATA[name]
    rows = read(data_path.format(L=L))
    scores = {r["input"]: r["pred"] for r in read(score_path.format(L=L))}
    blind = read(f"results/blind_baseline/scores/{name}_len{L}.jsonl")
    assert len(blind) == len(rows) and all(b["label"] == int(r["label"]) for b, r in zip(blind, rows))
    out = pd.DataFrame({"label": [int(r["label"]) for r in rows]})
    for m, key in METHODS.items():
        out[m] = [-scores[r["input"]][key] for r in rows]          # run.py stores the negated score
    for c in CONFOUNDS:
        out[c] = [b[c] for b in blind]
    return rows, out


def pair_auc(y, s, groups):
    """AUC over (member, non-member) pairs within the same group only."""
    num = den = 0.0
    for g in np.unique(groups):
        i = groups == g
        n1, n0 = (y[i] == 1).sum(), (y[i] == 0).sum()
        if n1 and n0:
            num += roc_auc_score(y[i], s[i]) * n1 * n0
            den += n1 * n0
    return num / den if den else np.nan, den


def match(y, c, caliper_sd=0.2, seed=0):
    """Greedy 1:1 nearest-neighbour matching on logit(c), members in random order, without replacement."""
    z = np.log(np.clip(c, 1e-6, 1 - 1e-6) / np.clip(1 - c, 1e-6, 1 - 1e-6))
    caliper = caliper_sd * z.std()
    members = np.flatnonzero(y == 1)
    free = list(np.flatnonzero(y == 0))
    pairs = []
    for i in np.random.default_rng(seed).permutation(members):
        if not free:
            break
        d = np.abs(z[free] - z[i])
        j = int(np.argmin(d))
        if d[j] <= caliper:
            pairs.append((i, free.pop(j)))
    return pairs


def boot(y, s, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = [roc_auc_score(y[i], s[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))
            if 0 < y[i].sum() < len(i)]
    return np.percentile(vals, [2.5, 97.5])


def main():
    out_dir = ROOT / "results/shift_controlled"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = []
    for name in DATA:
        for L in LENGTHS:
            rows, df = load(name, L)
            y = df.label.to_numpy()
            for m in METHODS:
                s = df[m].to_numpy()
                lo, hi = boot(y, s)
                res.append(dict(dataset=name, length=L, control="none", confound="", n=len(y),
                                method=m, auc=roc_auc_score(y, s), auc_lo=lo, auc_hi=hi))
            for c in CONFOUNDS:
                cv = df[c].to_numpy()
                strata = pd.qcut(cv, 5, labels=False, duplicates="drop")
                for m in list(METHODS) + [c]:
                    a, npairs = pair_auc(y, df[m].to_numpy(), strata)
                    res.append(dict(dataset=name, length=L, control="stratified", confound=c, n=int(npairs),
                                    method=m if m in METHODS else "blind (itself)", auc=a, auc_lo=np.nan, auc_hi=np.nan))
                pairs = match(y, cv)
                idx = np.array([k for p in pairs for k in p], dtype=int)
                for m in list(METHODS) + [c]:
                    s = df[m].to_numpy()[idx]
                    lo, hi = boot(y[idx], s) if len(pairs) >= 10 else (np.nan, np.nan)
                    res.append(dict(dataset=name, length=L, control="matched", confound=c, n=len(idx),
                                    method=m if m in METHODS else "blind (itself)",
                                    auc=roc_auc_score(y[idx], s) if len(pairs) else np.nan, auc_lo=lo, auc_hi=hi))
                if name == "WikiMIA-2026" and c == "tfidf_lr":     # ship the matched subset as a dataset
                    d = ROOT / "data/wikimia2026_matched"
                    d.mkdir(parents=True, exist_ok=True)
                    keep = sorted(idx.tolist())
                    (d / f"WikiMIA2026matched_length{L}.jsonl").write_text(
                        "".join(json.dumps(rows[k], ensure_ascii=False) + "\n" for k in keep), encoding="utf-8")
    df = pd.DataFrame(res)
    df.to_csv(out_dir / "summary.csv", index=False)
    show = df[df.method.isin(["Min-20%", "blind (itself)"])]
    print(show.pivot_table(index=["dataset", "control", "confound", "method"], columns="length",
                           values="auc").round(3).to_string())
    print("\nmatched set sizes (passages):")
    print(df[(df.control == "matched") & (df.method == "Min-20%")].pivot_table(
        index=["dataset", "confound"], columns="length", values="n").to_string())


if __name__ == "__main__":
    main()
