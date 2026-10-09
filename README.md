# COMP8240 Project: Replicating *Detecting Pretraining Data from Large Language Models*

Replication study of **Min-K% Prob** (Shi et al., ICLR 2024) for COMP8240 Applications of Data Science,
Macquarie University, Session 2 2026.

- **Paper:** [Detecting Pretraining Data from Large Language Models](https://openreview.net/forum?id=zWqr3MQuNs), ICLR 2024
- **Original code:** [swj0419/detect-pretrain-code](https://github.com/swj0419/detect-pretrain-code) (Apache-2.0), pinned at commit `a5048f9`
- **Original data:** [swj0419/WikiMIA](https://huggingface.co/datasets/swj0419/WikiMIA) (MIT)

## Project structure

| Phase | Data | Question | Status |
|---|---|---|---|
| P1 Replication | WikiMIA | Do the paper's AUC / TPR@5%FPR numbers and its model-size trend reproduce? | Pythia 160M → 2.8B sweep: `notebooks/01_replication_sweep.ipynb` |
| P2 Existing unused data | arXivTection, BookTection, MIMIR | Does the claim transfer? | arXivTection adapted + audited; MIMIR now gated on Hugging Face (access requested) |
| P3 New data | **WikiMIA-2026** (built here); ND-2 shift-controlled split (planned) | Does the claim survive without a temporal confound? | WikiMIA-2026 built (`data/wikimia2026/`); blind-baseline audit done |

P3 exists because the headline result is contested: [Duan et al. (COLM 2024)](https://arxiv.org/abs/2402.07841)
find membership inference on LLMs barely beats chance once members and non-members share a distribution,
and [Das et al. (2024)](https://arxiv.org/abs/2406.16201) show blind baselines beating published attacks.

## Repository layout

```
notebooks/01_replication_sweep.ipynb    P1: original code on WikiMIA, Pythia 160M-2.8B, all 4 lengths (Colab T4)
notebooks/02_wikimia2026_attack.ipynb   P3: the same attack on WikiMIA-2026
notebooks/feasibility_check.ipynb       the proposal's first end-to-end run (Pythia-160M)
scripts/build_wikimia2026.py            builds WikiMIA-2026 from the MediaWiki API
scripts/blind_baseline.py               text-only membership classifiers (no model access)
scripts/make_figures.py                 figures + LaTeX numbers for the reports, from results/
scripts/make_notebooks.py               generates the notebooks (keeps them diffable)
data/wikimia2026/                       the new dataset, its build log and a README on re-acquiring it
results/                                outputs of every run (CSV summaries, per-passage scores, figures)
report/                                 proposal, update presentation (Beamer), final report
```

## Results so far

**Blind-baseline audit** (`results/blind_baseline/summary.csv`). A TF-IDF + logistic-regression classifier
that never queries any model separates WikiMIA members from non-members with mean AUC **0.96** across the
four lengths, versus 0.67 for Min-K% on Pythia-2.8B in the paper. Simply taking the latest year mentioned
in the passage gives 0.89; masking all years still leaves 0.80 (tense: "will be held" vs "was").
On arXivTection, grouped by paper, the blind classifier reaches 0.95 (0.997 if the cross-validation is not
grouped by paper, which leaks paper identity across folds). On WikiMIA-2026, built here with topic-matched
categories, it still reaches 0.95. The **control** is MIMIR's Wikipedia split (members = Pile train,
non-members = Pile test, no time gap): there the same classifier scores 0.48–0.55, i.e. chance. So the
classifier is not simply strong; the date-defined benchmarks leak time. (MIMIR is gated on Hugging Face and
is used locally only; place the two files in `data/external/mimir/` to reproduce.)

**Replication sweep** (`results/replication/`, Colab T4, transformers 5.18, fp16). Pythia-2.8B, mean AUC over
the four WikiMIA lengths, mine vs. the paper's Table 1:

| Method | Mine | Paper |
|---|---|---|
| PPL | 0.626 | 0.61 |
| Zlib | 0.642 | 0.65 |
| Lowercase | 0.603 | 0.59 |
| Smaller Ref (Pythia-70M) | 0.610 | 0.60 |
| **Min-20%** | **0.649** | **0.67** |

All five are within 0.02 of the paper. The model-size trend reproduces (Min-20%: 0.558, 0.591, 0.617, 0.618,
0.649 for 160M → 2.8B), as does the length trend (0.617 at 32 words → 0.701 at 256). k = 20% is the best k in
a 5–60% sweep. Min-K%'s margin over Zlib (0.007) is well inside the 95% bootstrap CIs. The Pythia-160M,
length-32 numbers match the August feasibility run to three decimals (Min-20% 0.532, PPL 0.553).

**Min-K% on WikiMIA-2026** (`results/wikimia2026/`, `notebooks/02_wikimia2026_attack.ipynb`). Pythia-2.8B,
mean over the four lengths: Min-20% **0.675**, PPL 0.661, Zlib 0.627, Lowercase 0.642, Smaller Ref 0.611.
Min-K% is again the best method and the size trend holds (0.561 → 0.641 → 0.675 for 160M → 1.4B → 2.8B):
the paper's claim transfers to fresh data. On the same data a text-only classifier reaches 0.95, so these
AUCs cannot by themselves be attributed to memorisation.

## Reproducibility notes on the original code

Each is a real trap for a replicator.

1. No `requirements.txt`: versions have to be pinned and recorded by hand (`results/*/environment.json`).
2. `run.py` imports `openai` unconditionally, and its GPT-3 path targets `text-davinci-003`, retired in
   January 2024. The proprietary-model rows of the paper's tables cannot be reproduced at all.
3. `run.py` stores the **negated** Min-K% mean and `eval.py` negates again inside `roc_curve`; a naive
   re-implementation silently inverts the ROC curve.
4. The repository ships a committed `out/` directory with the **authors' own LLaMA-13B results**; any
   script that globs for `auc.txt` can pick those up instead of your own run.
5. `eval.py` labels its third number "TPR@0.1%FPR" in `auc.txt`, but computes TPR at **5%** FPR.

Also: Table 1 of the paper does not state which length bucket it reports, so all four are reported here,
plus their mean.

## Running it

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; use .venv/bin/activate elsewhere
pip install -r requirements.txt
python scripts/build_wikimia2026.py --out data/wikimia2026     # ~20 min, polite single-threaded API use
python scripts/blind_baseline.py                                # seconds, CPU only
python scripts/make_figures.py
```

The model runs need a GPU: open the notebooks in Google Colab (T4) and *Run all*.

## Licence

Code in this repository: MIT. The original implementation is Apache-2.0 and is not vendored here.
WikiMIA and arXivTection are MIT-licensed and downloaded at run time. WikiMIA-2026 text is from English
Wikipedia (CC BY-SA 4.0).
