"""Analyse the human recency annotation and compare humans, LLM judges, Min-K% and the blind classifier
on the same 100 member/non-member pairs (WikiMIA-2026, 64 words, years masked).

Inputs
  data/annotation/pairs.jsonl                  answer key (scripts/make_annotation.py)
  data/annotation/responses/*.json             one results file per annotator, as downloaded from the page
  results/llm_judge/pairs_*.jsonl              judge probabilities, both A/B orders (notebook 04; optional)
  results/wikimia2026/scores/pythia-2.8b_len64.jsonl        Min-K% per passage (notebook 02)
  results/year_masked/wikimia-2026/scores/pythia-2.8b_len64.jsonl   the same with years masked (optional)
  results/blind_baseline/scores/WikiMIA-2026_len64.jsonl    blind TF-IDF, out of fold

A "correct" answer picks the 2026 passage (the non-member) as the more recent event. For any scorer, the share
of random member/non-member pairs it orders correctly is an unbiased estimate of its AUC, so pairwise accuracy
here is directly comparable to the AUCs elsewhere in the project.

Outputs (no initials, no names: annotators become A1, A2, ...; the author is A1)
  results/annotation/summary.json
  results/annotation/per_pair.csv
Usage:  python scripts/analyse_annotation.py [--author AR] [--responses DIR]
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import cohen_kappa_score

from make_annotation import mask_years

ROOT = Path(__file__).resolve().parent.parent
SEEN_BY_AUTHOR = {"p000", "p001", "p002"}       # their answers were visible to the author while building the task


def read(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def fleiss_kappa(table):
    """table: items x categories counts, every item rated by the same number of raters."""
    table = np.asarray(table, dtype=float)
    n = table.sum(1)[0]
    assert (table.sum(1) == n).all() and n > 1
    p_j = table.sum(0) / table.sum()
    P_i = ((table ** 2).sum(1) - n) / (n * (n - 1))
    P_e = (p_j ** 2).sum()
    return (P_i.mean() - P_e) / (1 - P_e)


def accuracy_row(name, verdicts, gold, ids, exclude=()):
    """verdicts: dict id -> 0/1 (index judged newer) or None (can't tell)."""
    use = [i for i in ids if i in verdicts and i not in exclude]
    decided = [i for i in use if verdicts[i] is not None]
    k = sum(verdicts[i] == gold[i] for i in decided)
    lo, hi = wilson(k, len(decided))
    return {"who": name, "answered": len(use), "cant_tell": len(use) - len(decided), "decided": len(decided),
            "correct": int(k), "accuracy_decided": k / len(decided) if decided else np.nan,
            "ci95": [lo, hi],
            "accuracy_cant_tell_as_half": (k + 0.5 * (len(use) - len(decided))) / len(use) if use else np.nan,
            "p_vs_chance": binomtest(int(k), len(decided), 0.5).pvalue if decided else np.nan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--author", default="AR,ASHIKUR", help="comma-separated initials the author used on the page")
    ap.add_argument("--responses", default=str(ROOT / "data/annotation/responses"))
    ap.add_argument("--round", type=int, default=1, choices=[1, 2],
                    help="1: 100 pairs, no time lock; 2: 40 of them, answer buttons locked 10 s per pair")
    ap.add_argument("--out", default=None, help="default results/annotation/round<N>")
    args = ap.parse_args()
    args.out = args.out or str(ROOT / f"results/annotation/round{args.round}")

    pairs = read(ROOT / "data/annotation/pairs.jsonl")
    ids = [p["id"] for p in pairs]
    if args.round == 2:
        ids = json.loads((ROOT / "annotation/round2_ids.json").read_text(encoding="utf-8"))
    gold = {p["id"]: 1 - p["member_index"] for p in pairs}          # index of the newer (non-member) passage

    # ---------------------------------------------------------------- annotators
    files = sorted(Path(args.responses).glob("*.json"))
    raw = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    raw = [r for r in raw if r.get("consent") and r.get("task") == f"recency-annotation-v{args.round}"]
    # A1 = the author, then the others by initials, so the same person keeps the same label in both rounds
    authors = {a.strip().upper() for a in args.author.split(",")}
    raw.sort(key=lambda r: (r.get("initials", "").upper() not in authors, r.get("initials", "").upper()))
    humans = {}
    timing = {}
    for k, r in enumerate(raw, 1):
        name = f"A{k}"
        humans[name] = {pid: a["choice"] for pid, a in r["answers"].items()}
        timing[name] = (float(np.median([a["ms"] for a in r["answers"].values()])) / 1000,
                        sum(a["ms"] for a in r["answers"].values()) / 60000)
    author = "A1" if raw and raw[0].get("initials", "").upper() in authors else None

    rows = []
    for h, v in humans.items():
        row = accuracy_row(h + (" (author)" if h == author else ""), v, gold, ids)
        row["median_seconds_per_pair"], row["total_minutes"] = timing[h]
        rows.append(row)
        if h == author and args.round == 1:                      # round 2 already leaves those pairs out
            rows.append(accuracy_row(h + " (author, excluding p000-p002)", v, gold, ids, SEEN_BY_AUTHOR))

    # majority vote over humans who decided (ties and all-"can't tell" -> None)
    majority = {}
    for pid in ids:
        votes = [v[pid] for v in humans.values() if v.get(pid) is not None]
        c = sum(votes)
        majority[pid] = None if not votes or 2 * c == len(votes) else int(2 * c > len(votes))
    if humans:
        rows.append(accuracy_row("human majority", majority, gold, ids))

    agreement = {}
    cat = lambda x: "?" if x is None else str(x)
    complete = [pid for pid in ids if all(pid in v for v in humans.values())]
    if len(humans) >= 2:
        table = [[sum(cat(v[pid]) == c for v in humans.values()) for c in ("0", "1", "?")] for pid in complete]
        agreement["fleiss_kappa_3cat"] = fleiss_kappa(table)
        both = [pid for pid in complete if all(v[pid] is not None for v in humans.values())]
        if both:
            agreement["fleiss_kappa_decided_only"] = fleiss_kappa(
                [[sum(v[pid] == c for v in humans.values()) for c in (0, 1)] for pid in both])
            agreement["n_items_all_decided"] = len(both)
        agreement["n_items"] = len(complete)
        for a, b in combinations(humans, 2):
            agreement[f"cohen_kappa_{a}_{b}"] = cohen_kappa_score([cat(humans[a][p]) for p in complete],
                                                                  [cat(humans[b][p]) for p in complete])

    # ---------------------------------------------------------------- LLM judges
    judges = {}
    for f in sorted((ROOT / "results/llm_judge").glob("pairs_*.jsonl")):
        J = [r for r in read(f) if r["id"] in set(ids)]
        name = J[0]["judge"].split("/")[-1]
        judges[name] = {r["id"]: int(r["p0_newer"] < 0.5) for r in J}     # index judged newer, both orders averaged
        rows.append(accuracy_row(f"judge {name}", judges[name], gold, ids))
        o1 = {r["id"]: int(r["ab"][1] > r["ab"][0]) for r in J}
        o2 = {r["id"]: int(r["ba"][0] > r["ba"][1]) for r in J}
        agreement[f"{name}_same_verdict_both_orders"] = float(np.mean([o1[i] == o2[i] for i in o1]))
        agreement[f"{name}_mean_P(A)"] = float(np.mean([r["ab"][0] / sum(r["ab"]) for r in J] +
                                                     [r["ba"][0] / sum(r["ba"]) for r in J]))
    for j, jv in judges.items():
        for h, hv in list(humans.items()) + ([("majority", majority)] if humans else []):
            common = [p for p in ids if hv.get(p) is not None]
            agreement[f"cohen_kappa_{j}_{h}"] = cohen_kappa_score([hv[p] for p in common], [jv[p] for p in common])
    for a, b in combinations(judges, 2):
        agreement[f"cohen_kappa_{a}_{b}"] = cohen_kappa_score([judges[a][p] for p in ids], [judges[b][p] for p in ids])

    # ---------------------------------------------------------------- Min-K% and the blind classifier
    rows64 = read(ROOT / "data/wikimia2026/WikiMIA2026_length64.jsonl")
    by_masked = {mask_years(r["input"]): r["input"] for r in rows64}
    scorers = {}
    sc = {r["input"]: -r["pred"]["Min_20.0% Prob"] for r in read(ROOT / "results/wikimia2026/scores/pythia-2.8b_len64.jsonl")}
    scorers["Min-20% (Pythia-2.8B)"] = lambda t: sc[by_masked[t]]
    mpath = ROOT / "results/year_masked/wikimia-2026/scores/pythia-2.8b_len64.jsonl"
    if mpath.exists():
        msc = {r["input"]: -r["pred"]["Min_20.0% Prob"] for r in read(mpath)}
        scorers["Min-20% (Pythia-2.8B, years masked)"] = lambda t: msc[t]
    blind = read(ROOT / "results/blind_baseline/scores/WikiMIA-2026_len64.jsonl")
    bsc = {mask_years(r["input"]): b["tfidf_lr"] for r, b in zip(rows64, blind)}
    bsc_m = {mask_years(r["input"]): b["tfidf_lr_noyr"] for r, b in zip(rows64, blind)}
    scorers["blind TF-IDF (raw text)"] = lambda t: bsc[t]
    scorers["blind TF-IDF (years masked)"] = lambda t: bsc_m[t]
    machine = {}
    for name, f in scorers.items():
        # every scorer gives a HIGHER score to the passage it thinks is the member, i.e. the OLDER one
        machine[name] = {p["id"]: int(f(p["passages"][0]) > f(p["passages"][1])) for p in pairs}
        rows.append(accuracy_row(name, machine[name], gold, ids))
    if humans:
        for name, mv in machine.items():
            common = [p for p in ids if majority[p] is not None]
            agreement[f"cohen_kappa_{name}_majority"] = cohen_kappa_score([majority[p] for p in common],
                                                                          [mv[p] for p in common])

    # ---------------------------------------------------------------- write
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per = pd.DataFrame({"id": ids, "newer_index": [gold[i] for i in ids]})
    for h, v in humans.items():
        per[h] = pd.array([v.get(i) for i in ids], dtype="Int64")        # empty = "can't tell"
    per["majority"] = pd.array([majority.get(i) for i in ids], dtype="Int64")
    for j, v in judges.items():
        per[f"judge_{j}"] = [v[i] for i in ids]
    for m, v in machine.items():
        per[m] = [v[i] for i in ids]
    per.to_csv(out / "per_pair.csv", index=False)
    summary = {"n_pairs": len(ids), "n_annotators": len(humans), "author": author,
               "accuracy": rows, "agreement": agreement}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")

    show = pd.DataFrame(rows)[["who", "decided", "cant_tell", "accuracy_decided", "accuracy_cant_tell_as_half", "p_vs_chance"]]
    print(show.round(3).to_string(index=False))
    print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in agreement.items()}, indent=1))


if __name__ == "__main__":
    main()
