"""Generate the Colab notebooks in notebooks/ from plain Python strings.

Keeping the notebook source here (rather than hand-editing JSON) makes the
notebooks diffable under version control. Run:  python scripts/make_notebooks.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip("\n")}


def write(name, cells):
    nb = {"cells": cells,
          "metadata": {"accelerator": "GPU",
                       "colab": {"provenance": [], "gpuType": "T4"},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    path = ROOT / "notebooks" / name
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", path.relative_to(ROOT))


# --------------------------------------------------------------------------
# Shared cells: environment + driver around the ORIGINAL, unmodified code
# --------------------------------------------------------------------------
SETUP = code(r'''
# Clone the ORIGINAL implementation at the exact commit used throughout the project.
%cd -q /content
!git clone --quiet https://github.com/swj0419/detect-pretrain-code.git 2>/dev/null || echo 'already cloned'
!cd /content/detect-pretrain-code && git checkout --quiet a5048f93c4fa6491aab0638bb3e6fd5a662d7698 && git log -1 --format='original code @ %h (%ad)' --date=short
# run.py needs accelerate (device_map='auto') and imports openai unconditionally.
!pip install -q accelerate openai

import json, os, platform, sys
import torch, transformers, datasets, sklearn, numpy as np
ENV = {'python': platform.python_version(), 'torch': torch.__version__,
       'transformers': transformers.__version__, 'datasets': datasets.__version__,
       'sklearn': sklearn.__version__, 'numpy': np.__version__,
       'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}
print(json.dumps(ENV, indent=1))
assert torch.cuda.is_available(), 'No GPU: Runtime -> Change runtime type -> T4 GPU, then Run all again.'
''')

DRIVER = code(r'''
# Import the original modules and call THEIR functions. Nothing in run.py / eval.py is edited;
# we only drive them from Python so the per-passage scores can be saved (run.py discards them).
SRC = '/content/detect-pretrain-code/src'
os.chdir(SRC); sys.path.insert(0, SRC)
import run as R            # original run.py  (load_model, evaluate_data, ...)
import gc, re, time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

REF = 'EleutherAI/pythia-70m'   # the paper's reference model for Pythia-2.8B (Section 4.2)

def metric_name(key):
    """Map run.py's prediction keys to the paper's method names."""
    if key == 'ppl': return 'PPL'
    if key.startswith('ppl/Ref'): return 'Smaller Ref'
    if key.startswith('ppl/lowercase'): return 'Lowercase'
    if key.startswith('ppl/zlib'): return 'Zlib'
    m = re.match(r'Min_([0-9.]+)% Prob', key)          # note: run.py writes 'Min_30.000000000000004% Prob'
    return f'Min-{round(float(m.group(1)))}%' if m else key

BOOT_METRICS = {'PPL', 'Zlib', 'Lowercase', 'Smaller Ref', 'Min-20%'}

def summarise(outputs, model, dataset, length, n_boot=1000, seed=0):
    """AUC + TPR@5%FPR per method using the ORIGINAL eval.sweep, plus a bootstrap 95% CI on AUC."""
    labels = np.array([ex['label'] for ex in outputs], dtype=bool)
    rng = np.random.default_rng(seed)
    idx = [rng.integers(0, len(labels), len(labels)) for _ in range(n_boot)]
    rows = []
    for key in outputs[0]['pred']:
        scores = np.array([ex['pred'][key] for ex in outputs], dtype=float)
        if np.isnan(scores).any():
            continue
        fpr, tpr, auc, acc = R.sweep(scores, labels)          # original: roc_curve(labels, -score)
        tpr5 = tpr[np.where(fpr < .05)[0][-1]]
        lo = hi = np.nan
        name = metric_name(key)
        if name in BOOT_METRICS:
            boots = [roc_auc_score(labels[i], -scores[i]) for i in idx if 0 < labels[i].sum() < len(i)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append(dict(dataset=dataset, model=model, length=length, n=len(labels), method=name,
                         auc=auc, auc_lo=lo, auc_hi=hi, tpr_at_5fpr=tpr5, best_acc=acc))
    return rows

def save_jsonl(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for r in rows:
            f.write(json.dumps({'input': r['input'], 'label': int(r['label']), 'pred': r['pred']}) + '\n')

def run_model(target, jobs, out_root, dataset_name):
    """jobs: list of (length, list_of_examples). Loads the model once, scores every job."""
    rows = []
    todo = [(L, data) for L, data in jobs
            if not os.path.exists(f'{out_root}/scores/{target.split("/")[-1]}_len{L}.jsonl')]
    for L, data in jobs:                                  # resume: reuse finished runs
        p = f'{out_root}/scores/{target.split("/")[-1]}_len{L}.jsonl'
        if os.path.exists(p):
            done = [json.loads(l) for l in open(p)]
            rows += summarise(done, target, dataset_name, L)
    if not todo:
        return rows
    t0 = time.time()
    m1, m2, t1, t2 = R.load_model(target, REF)             # original loader
    print(f'{target}: loaded in {time.time()-t0:.0f}s, dtype={m1.dtype}')
    for L, data in todo:
        t0 = time.time()
        out = R.evaluate_data(data, m1, m2, t1, t2, 'input', target, REF)   # original scoring loop
        d = f'{out_root}/original_eval/{target}_{REF}/len{L}'
        os.makedirs(d, exist_ok=True)
        try:
            R.fig_fpr_tpr(out, d)                          # original auc.txt + auc.png
        except ValueError as e:                            # e.g. NaN from Min-5% on a very short passage
            print('  original eval.py failed:', e)
        plt.close('all')
        save_jsonl(out, f'{out_root}/scores/{target.split("/")[-1]}_len{L}.jsonl')
        rows += summarise(out, target, dataset_name, L)
        print(f'  length {L}: {len(data)} passages in {time.time()-t0:.0f}s')
    del m1, m2; gc.collect(); torch.cuda.empty_cache()
    return rows
print('driver ready')
''')

# --------------------------------------------------------------------------
# Notebook 1: replication sweep on the original WikiMIA
# --------------------------------------------------------------------------
write("01_replication_sweep.ipynb", [
    md(r'''
# P1 Replication sweep: Min-K% Prob on WikiMIA across the Pythia suite

**Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all**. About 30–40 minutes on a free T4.

What this does
* Runs the **original, unmodified** code (`swj0419/detect-pretrain-code` @ `a5048f9`) on the **original data**
  (`swj0419/WikiMIA`, all four lengths: 32 / 64 / 128 / 256 words).
* Five target sizes from one model family: Pythia **160M, 410M, 1B, 1.4B, 2.8B**, with Pythia-70M as the
  Smaller-Ref model, exactly the reference the paper used for Pythia-2.8B.
* Tests two claims from the paper: the Pythia-2.8B row of **Table 1** (Min-K% 0.67 vs PPL 0.61) and the
  **model-size trend** of Figure 2a (detection improves with size).
* Saves every passage's scores (the original `run.py` throws them away), so we can add bootstrap 95% CIs
  and plot score distributions.

If Colab disconnects, just **Run all** again: finished (model, length) runs are skipped.
At the end a zip of `results/` downloads automatically. Put it in the repo under `results/replication/`.
'''),
    SETUP,
    DRIVER,
    md("## Run the sweep"),
    code(r'''
MODELS = ['EleutherAI/pythia-160m', 'EleutherAI/pythia-410m', 'EleutherAI/pythia-1b',
          'EleutherAI/pythia-1.4b', 'EleutherAI/pythia-2.8b']
LENGTHS = [32, 64, 128, 256]
OUT = '/content/results/replication'

wikimia = {L: R.convert_huggingface_data_to_list_dic(
               R.load_dataset('swj0419/WikiMIA', split=f'WikiMIA_length{L}')) for L in LENGTHS}
print({L: len(v) for L, v in wikimia.items()})       # expect 776 / 542 / 250 / 82

all_rows = []
for target in MODELS:
    all_rows += run_model(target, [(L, wikimia[L]) for L in LENGTHS], OUT, 'WikiMIA')
    pd.DataFrame(all_rows).to_csv(f'{OUT}/summary.csv', index=False)   # checkpoint after each model
json.dump(ENV, open(f'{OUT}/environment.json', 'w'), indent=1)
print('done')
'''),
    md("## Compare with the paper (Table 1, Pythia-2.8B, original setting)"),
    code(r'''
df = pd.read_csv(f'{OUT}/summary.csv')
main = ['PPL', 'Zlib', 'Lowercase', 'Smaller Ref', 'Min-20%']
PAPER = {'PPL': 0.61, 'Zlib': 0.65, 'Lowercase': 0.59, 'Smaller Ref': 0.60, 'Min-20%': 0.67}
PAPER_TPR = {'PPL': 9.4, 'Zlib': 18.7, 'Lowercase': 10.8, 'Smaller Ref': 10.1, 'Min-20%': 13.7}

d28 = df[(df.model == 'EleutherAI/pythia-2.8b') & df.method.isin(main)]
cmp = d28.pivot_table(index='method', columns='length', values='auc').reindex(main)
cmp['mean (4 lengths)'] = cmp.mean(axis=1)
cmp['paper Table 1'] = pd.Series(PAPER)
cmp['TPR@5%FPR mean (%)'] = (d28.groupby('method').tpr_at_5fpr.mean() * 100).reindex(main)
cmp['paper Table 6 (%)'] = pd.Series(PAPER_TPR)
display(cmp.round(3))
cmp.round(4).to_csv(f'{OUT}/pythia28_vs_paper.csv')

print('\nMin-20% AUC by model size (rows) and length (columns):')
size = df[df.method == 'Min-20%'].pivot_table(index='model', columns='length', values='auc').reindex(MODELS)
size['mean'] = size.mean(axis=1)
display(size.round(3))
'''),
    md("## Figures for the presentation"),
    code(r'''
FIG = f'{OUT}/figures'; os.makedirs(FIG, exist_ok=True)
PARAMS = {'EleutherAI/pythia-160m': 0.16, 'EleutherAI/pythia-410m': 0.41, 'EleutherAI/pythia-1b': 1.0,
          'EleutherAI/pythia-1.4b': 1.4, 'EleutherAI/pythia-2.8b': 2.8}

# (1) AUC vs model size, averaged over the four lengths
fig, ax = plt.subplots(figsize=(5, 3.4))
for m, style in [('Min-20%', 'o-'), ('PPL', 's--'), ('Zlib', '^:')]:
    g = df[df.method == m].groupby('model').auc.mean().reindex(MODELS)
    ax.plot([PARAMS[x] for x in MODELS], g.values, style, label=m)
ax.axhline(0.5, color='grey', lw=0.8, ls='--'); ax.set_xscale('log')
ax.set_xticks([PARAMS[x] for x in MODELS]); ax.set_xticklabels(['160M', '410M', '1B', '1.4B', '2.8B'])
ax.set_xlabel('Pythia model size'); ax.set_ylabel('AUC (mean of 4 lengths)'); ax.legend()
ax.set_title('WikiMIA: detection vs model size'); fig.tight_layout()
fig.savefig(f'{FIG}/auc_vs_size.pdf'); fig.savefig(f'{FIG}/auc_vs_size.png', dpi=200); plt.show()

# (2) AUC vs passage length for Pythia-2.8B, with bootstrap 95% CIs
fig, ax = plt.subplots(figsize=(5, 3.4))
for m, style in [('Min-20%', 'o-'), ('PPL', 's--'), ('Zlib', '^:')]:
    g = d28[d28.method == m].sort_values('length')
    ax.errorbar(g.length, g.auc, yerr=[g.auc - g.auc_lo, g.auc_hi - g.auc], fmt=style, capsize=3, label=m)
ax.axhline(0.5, color='grey', lw=0.8, ls='--'); ax.set_xscale('log', base=2); ax.set_xticks(LENGTHS)
ax.set_xticklabels(LENGTHS); ax.set_xlabel('passage length (words)'); ax.set_ylabel('AUC')
ax.set_title('Pythia-2.8B on WikiMIA (95% bootstrap CI)'); ax.legend(); fig.tight_layout()
fig.savefig(f'{FIG}/auc_vs_length_28b.pdf'); fig.savefig(f'{FIG}/auc_vs_length_28b.png', dpi=200); plt.show()

# (3) What the detector actually sees: Min-20% score distributions, members vs non-members
rows = [json.loads(l) for l in open(f'{OUT}/scores/pythia-2.8b_len128.jsonl')]
key = [k for k in rows[0]['pred'] if k.startswith('Min_20')][0]
s = np.array([-r['pred'][key] for r in rows]); y = np.array([r['label'] for r in rows])
fig, ax = plt.subplots(figsize=(5, 3.4))
ax.hist(s[y == 1], bins=30, alpha=0.55, density=True, label='members (pre-2017)')
ax.hist(s[y == 0], bins=30, alpha=0.55, density=True, label='non-members (post-2023)')
ax.set_xlabel('Min-20% score (mean log-prob of 20% least likely tokens)'); ax.set_ylabel('density')
ax.set_title('Pythia-2.8B, WikiMIA length 128'); ax.legend(); fig.tight_layout()
fig.savefig(f'{FIG}/score_hist_28b_len128.pdf'); fig.savefig(f'{FIG}/score_hist_28b_len128.png', dpi=200); plt.show()
'''),
    md("## Download everything"),
    code(r'''
!cd /content && zip -qr replication_results.zip results/replication
from google.colab import files
files.download('/content/replication_results.zip')
'''),
])

# --------------------------------------------------------------------------
# Notebook 2: the same attack on the newly constructed WikiMIA-2026
# --------------------------------------------------------------------------
write("02_wikimia2026_attack.ipynb", [
    md(r'''
# P3 First results on new data: Min-K% Prob on WikiMIA-2026

**T4 GPU**, then **Run all** (about 10–15 minutes).

WikiMIA-2026 is built by `scripts/build_wikimia2026.py` in this repository, following the paper's own recipe
(Section 2.2) with a **1 January 2026** cut-off. It is written in the same JSONL schema the original code reads,
so the original `run.py` functions score it **unmodified**.
'''),
    SETUP,
    DRIVER,
    code(r'''
!git clone --quiet https://github.com/Ashik-here/COMP8240-MinK-Replication.git /content/mine 2>/dev/null || (cd /content/mine && git pull --quiet)
DATA = '/content/mine/data/wikimia2026'
LENGTHS = [32, 64, 128, 256]
new = {L: R.load_jsonl(f'{DATA}/WikiMIA2026_length{L}.jsonl') for L in LENGTHS}   # original loader
print({L: len(v) for L, v in new.items()})

MODELS = ['EleutherAI/pythia-160m', 'EleutherAI/pythia-1.4b', 'EleutherAI/pythia-2.8b']
OUT = '/content/results/wikimia2026'
rows = []
for target in MODELS:
    rows += run_model(target, [(L, new[L]) for L in LENGTHS], OUT, 'WikiMIA-2026')
    pd.DataFrame(rows).to_csv(f'{OUT}/summary.csv', index=False)
json.dump(ENV, open(f'{OUT}/environment.json', 'w'), indent=1)

df = pd.read_csv(f'{OUT}/summary.csv')
main = ['PPL', 'Zlib', 'Lowercase', 'Smaller Ref', 'Min-20%']
t = df[df.method.isin(main)].pivot_table(index=['model', 'method'], columns='length', values='auc')
t['mean'] = t.mean(axis=1); display(t.round(3))
'''),
    code(r'''
!cd /content && zip -qr wikimia2026_results.zip results/wikimia2026
from google.colab import files
files.download('/content/wikimia2026_results.zip')
'''),
])
