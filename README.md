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
| P2 Existing unused data | MIMIR (Wikipedia), arXivTection | Does the claim transfer? | MIMIR: Min-K% on Pythia 160M-2.8B (no time gap; the control). arXivTection: blind audit |
| P3 New data | **WikiMIA-2026**, **WikiMIA-2026-matched**, a 100-pair **human annotation set** (all built here) | Does the claim survive without a temporal confound? | Done: attack, blind audit, year masking, propensity matching, 3 human annotators (2 rounds), 2 LLM judges |

P3 exists because the headline result is contested: [Duan et al. (COLM 2024)](https://arxiv.org/abs/2402.07841)
find membership inference on LLMs barely beats chance once members and non-members share a distribution,
and [Das et al. (2024)](https://arxiv.org/abs/2406.16201) show blind baselines beating published attacks.

## Repository layout

```
notebooks/01_replication_sweep.ipynb    P1: original code on WikiMIA, Pythia 160M-2.8B, all 4 lengths (Colab T4)
notebooks/02_wikimia2026_attack.ipynb   P3: the same attack on WikiMIA-2026
notebooks/04_controls_and_judge.ipynb   controls (MIMIR, years masked) and the LLM-as-judge recency audit
scripts/make_annotation.py              builds the human annotation task (100 pairs, years masked) and its 40-pair round 2
annotation/annotate.html                round-1 offline annotation page given to annotators (no answer key)
annotation/annotate_round2.html         round-2 page: 40 pairs, answer buttons locked 10 s per pair
scripts/analyse_annotation.py           accuracy, Wilson CIs, Fleiss'/Cohen's kappa for humans, judges, Min-K%, blind classifier
scripts/shift_controlled.py             propensity matching / stratification on the blind score -> data/wikimia2026_matched/
scripts/make_report_assets.py           every number, table and figure of the final report, from results/
notebooks/feasibility_check.ipynb       the proposal's first end-to-end run (Pythia-160M)
scripts/build_wikimia2026.py            builds WikiMIA-2026 from the MediaWiki API
scripts/blind_baseline.py               text-only membership classifiers (no model access)
scripts/paired_bootstrap.py             paired bootstrap of the Min-K% minus Zlib AUC difference -> results/replication/paired_delta.json
scripts/make_figures.py                 figures + LaTeX numbers for the reports, from results/
scripts/make_notebooks.py               generates the notebooks (keeps them diffable)
data/wikimia2026/                       the new dataset, its build log and a README on re-acquiring it
results/                                outputs of every run (CSV summaries, per-passage scores, figures)
report/                                 proposal, update presentation (Beamer), final report
```

## Results

**Blind-baseline audit** (`results/blind_baseline/summary.csv`). A TF-IDF + logistic-regression classifier
that never queries any model separates WikiMIA members from non-members with mean AUC **0.96** across the
four lengths, versus 0.67 for Min-K% on Pythia-2.8B in the paper. Simply taking the latest year mentioned
in the passage gives 0.89; masking all years still leaves 0.79 (tense: "will be held" vs "was").
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

**MIMIR control** (`results/mimir/`, notebook 04). Members = Pile train, non-members = Pile test, same period.
Min-20% with Pythia-2.8B: **0.529** (per-length 95% CIs 0.50-0.56), PPL 0.525, Zlib 0.528; size trend
0.512 -> 0.522 -> 0.529 (160M -> 1.4B -> 2.8B), against +0.09 on WikiMIA. Near chance, as Duan et al. report.
Released files hold passage indices and hashes only (MIMIR is gated).

**Years masked** (`results/year_masked/`). Replacing every year with `YEAR` and re-running the attack:
Min-20% 0.649 -> 0.610 on WikiMIA and 0.675 -> 0.628 on WikiMIA-2026, i.e. 26-27% of its margin over chance.
Min-K% loses more than any other attack and is no longer the best method; the blind classifier on the same
masked text still reaches 0.79-0.80.

**Shift-controlled subset** (`results/shift_controlled/`, `data/wikimia2026_matched/`). Matching members to
non-members on the blind classifier's score brings that classifier to 0.50-0.62; Min-K% keeps 0.53-0.76
(weighted mean 0.64). Small subsets (26-194 passages), wide CIs.

**Human annotation and LLM judges** (`results/annotation/round1|round2/`, `results/llm_judge/`). Which of two
64-word passages (years masked) is about the newer event? Round 1 (100 pairs, no time limit): annotators
skimmed at 1.5-4 s per pair and were at chance (0.47-0.64, Fleiss' kappa 0.005). Round 2 (40 pairs, buttons
locked 10 s): 0.47-0.68, majority 0.62, Fleiss' kappa 0.10; Min-K% on the same pairs 0.68, blind classifier
(years masked) 0.80. The pairwise LLM judges (Qwen2.5-3B, Phi-3.5-mini) always pick passage B, so both orders
are averaged; pointwise (one masked passage at a time, no access to Pythia) they reach AUC 0.71/0.73 on
WikiMIA, 0.67/0.64 on WikiMIA-2026 and 0.49/0.51 on MIMIR. Annotators appear only as A1-A3 (A1 = author).

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
6. `eval.load_jsonl` silently **shuffles** the rows (`random.seed(0); random.shuffle`). AUC is unaffected, but
   per-passage outputs no longer line up with the input file by position, so any later join must use the
   text (or a hash of it), never the row index. Notebook 04 reads local data itself, in file order.

Also: Table 1 of the paper does not state which length bucket it reports, so all four are reported here,
plus their mean.

## Running it

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; use .venv/bin/activate elsewhere
pip install -r requirements.txt
python scripts/build_wikimia2026.py --out data/wikimia2026     # ~20 min, polite single-threaded API use
python scripts/blind_baseline.py                                # seconds, CPU only
python scripts/paired_bootstrap.py                               # seconds, CPU only
python scripts/shift_controlled.py
python scripts/analyse_annotation.py --round 1     # needs the annotators' files in data/annotation/responses/
python scripts/analyse_annotation.py --round 2
python scripts/make_report_assets.py               # then: cd report/final_report && latexmk -pdf main.tex
```

Or open the repository in **GitHub Codespaces**: `.devcontainer/devcontainer.json` installs the same
requirements, and every CPU script above runs there unchanged.

The model runs need a GPU: open the notebooks in Google Colab (T4) and *Run all*.

## Licence

Code in this repository: MIT. The original implementation is Apache-2.0 and is not vendored here.
WikiMIA and arXivTection are MIT-licensed and downloaded at run time. WikiMIA-2026 text is from English
Wikipedia (CC BY-SA 4.0).
