"""Blind-baseline audit (after Das et al., 2024): can membership be predicted WITHOUT the model?

A blind classifier never queries the language model. It only reads the passage. If it separates
members from non-members, the benchmark's labels leak through the text itself (dates, tense, topics),
and an attack's AUC on that benchmark partly measures the leak rather than memorisation.

Three blind scorers, all evaluated per length bucket:
  * max_year      : no learning at all. Score = the largest 4-digit year mentioned (newer => non-member).
  * tfidf_lr      : TF-IDF (1-2 grams) + logistic regression, 5-fold stratified CV inside a Pipeline,
                    so the vectoriser is refit on each training fold (no test leakage).
  * tfidf_lr_noyr : the same, after masking every year (and season, e.g. 2016-17) as "YEAR", to see what
                    leaks beyond dates. The masking is make_annotation.mask_years, the one the human
                    annotators, the LLM judge and the year-masked Min-K% run (notebook 04) all use.

Per-passage out-of-fold scores are saved in results/blind_baseline/scores/ (indices and labels only, no text),
in the same order as the datasets, so they can be joined with the Min-K% scores (scripts/shift_controlled.py).

Usage:  python scripts/blind_baseline.py            (WikiMIA is downloaded once and cached in data/external)
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from make_annotation import mask_years

LENGTHS = [32, 64, 128, 256]
YEAR = re.compile(r"\b(1[5-9]\d\d|20\d\d)\b")
ROWS = "https://datasets-server.huggingface.co/rows"


def load_wikimia(cache):
    """Original WikiMIA (MIT licence) via the Hugging Face datasets-server API, cached locally."""
    data = {}
    for L in LENGTHS:
        path = cache / f"WikiMIA_length{L}.jsonl"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            rows, offset = [], 0
            while True:
                r = requests.get(ROWS, params={"dataset": "swj0419/WikiMIA", "config": "default",
                                               "split": f"WikiMIA_length{L}", "offset": offset,
                                               "length": 100}, timeout=60).json()
                rows += [x["row"] for x in r["rows"]]
                offset += 100
                if offset >= r["num_rows_total"]:
                    break
            path.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
        data[L] = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    return data


def load_arxivtection(cache):
    """arXivTection (DE-COP, MIT licence). Each row holds the real excerpt plus three paraphrases;
    'Answer' names the real one. Label 1 = paper published before 2022, 0 = published in 2023."""
    path = cache / "arXivTection.jsonl"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        rows, offset = [], 0
        while True:
            r = requests.get(ROWS, params={"dataset": "avduarte333/arXivTection", "config": "default",
                                           "split": "train", "offset": offset, "length": 100}, timeout=60).json()
            rows += [x["row"] for x in r["rows"]]
            offset += 100
            if offset >= r["num_rows_total"]:
                break
        out = [{"input": x[f"Example_{x['Answer']}"], "label": int(x["Label"]), "paper": x["ID"]} for x in rows]
        path.write_text("".join(json.dumps(x) + "\n" for x in out), encoding="utf-8")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]


def load_mimir(folder):
    """MIMIR Wikipedia split (Duan et al., COLM 2024; gated on Hugging Face, so it is saved locally by hand
    and never committed). Members = Pile train, non-members = Pile test: same source, same time period.
    Each line is a JSON string of 100-200 words; we cut to the WikiMIA lengths that every passage reaches."""
    files = {1: "wikipedia_en_ngram_13_0.8_train_members.jsonl", 0: "wikipedia_en_ngram_13_0.8_test_nonmembers.jsonl"}
    if not all((folder / f).exists() for f in files.values()):
        return None
    texts = {lab: [json.loads(l) for l in (folder / f).read_text(encoding="utf-8").splitlines() if l.strip()]
             for lab, f in files.items()}
    return {L: [{"input": " ".join(t.split()[:L]), "label": lab} for lab in (1, 0) for t in texts[lab]]
            for L in (32, 64, 128)}


def load_jsonl_dir(folder, prefix):
    return {L: [json.loads(l) for l in (folder / f"{prefix}_length{L}.jsonl").read_text(encoding="utf-8").splitlines()]
            for L in LENGTHS}


def max_year_score(texts):
    years = [[int(y) for y in YEAR.findall(t)] for t in texts]
    return np.array([max(ys) if ys else 0 for ys in years], dtype=float)


def cv_scores(texts, y, groups=None, seed=0):
    """Out-of-fold P(member) from TF-IDF + LR; the vectoriser is fit inside each training fold.
    With `groups` (e.g. the source paper), all passages of a group stay in the same fold, so the
    classifier cannot score well just by recognising a paper it saw in training."""
    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                         LogisticRegression(max_iter=2000, C=4.0))
    if groups is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    else:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    return cross_val_predict(pipe, texts, y, cv=cv, groups=groups, method="predict_proba")[:, 1]


def boot_ci(y, s, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            vals.append(roc_auc_score(y[i], s[i]))
    return np.percentile(vals, [2.5, 97.5])


def top_features(texts, y, k=12):
    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                         LogisticRegression(max_iter=2000, C=4.0)).fit(texts, y)
    vocab = pipe[0].get_feature_names_out()
    coef = pipe[1].coef_[0]
    order = np.argsort(coef)
    return {"non-member": list(vocab[order[:k]]), "member": list(vocab[order[::-1][:k]])}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="results/blind_baseline")
    ap.add_argument("--new-data", default="data/wikimia2026")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    datasets = {"WikiMIA": load_wikimia(Path("data/external/wikimia"))}
    if Path(args.new_data, "WikiMIA2026_length32.jsonl").exists():
        datasets["WikiMIA-2026"] = load_jsonl_dir(Path(args.new_data), "WikiMIA2026")
    datasets["arXivTection"] = {"all": load_arxivtection(Path("data/external/arxivtection"))}
    mimir = load_mimir(Path("data/external/mimir"))
    if mimir:
        datasets["MIMIR-Wikipedia"] = mimir

    rows, feats = [], {}
    for name, data in datasets.items():
        for L in data:
            texts = [x["input"] for x in data[L]]
            y = np.array([int(x["label"]) for x in data[L]])
            groups = [x["paper"] for x in data[L]] if "paper" in data[L][0] else None
            masked = [mask_years(t) for t in texts]
            scorers = {"max_year": -max_year_score(texts),          # newer year => less likely a member
                       "tfidf_lr": cv_scores(texts, y, groups),
                       "tfidf_lr_noyr": cv_scores(masked, y, groups)}
            if groups is not None:                                  # the leaky version, for comparison
                scorers["tfidf_lr_ungrouped_LEAKY"] = cv_scores(texts, y)
            (out / "scores").mkdir(exist_ok=True)
            with open(out / "scores" / f"{name}_len{L}.jsonl", "w", encoding="utf-8") as f:
                for i in range(len(y)):
                    f.write(json.dumps({"idx": i, "label": int(y[i]),
                                        **{m: float(s[i]) for m, s in scorers.items()}}) + "\n")
            for method, s in scorers.items():
                lo, hi = boot_ci(y, s)
                rows.append(dict(dataset=name, length=L, n=len(y), method=method,
                                 auc=roc_auc_score(y, s), auc_lo=lo, auc_hi=hi))
            share = lambda lab: np.mean([bool(YEAR.search(t)) for t, l in zip(texts, y) if l == lab])
            rows.append(dict(dataset=name, length=L, n=len(y), method="share_with_a_year_members",
                             auc=share(1), auc_lo=np.nan, auc_hi=np.nan))
            rows.append(dict(dataset=name, length=L, n=len(y), method="share_with_a_year_nonmembers",
                             auc=share(0), auc_lo=np.nan, auc_hi=np.nan))
            if L in (64, "all"):
                feats[name] = {"raw": top_features(texts, y), "years_masked": top_features(masked, y)}
            print(f"{name:13s} len {L!s:>3}  n={len(y):>4}  " + "  ".join(
                f"{m}={roc_auc_score(y, s):.3f}" for m, s in scorers.items()))

    df = pd.DataFrame(rows)
    df.to_csv(out / "summary.csv", index=False)
    json.dump(feats, open(out / "top_features_len64.json", "w"), indent=1)
    print("\nTop TF-IDF features at length 64:")
    print(json.dumps(feats, indent=1))


if __name__ == "__main__":
    main()
