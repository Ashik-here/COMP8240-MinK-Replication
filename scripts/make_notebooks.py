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
# Notebook 3: a one-minute live demo for the update video
# --------------------------------------------------------------------------
write("03_live_demo.ipynb", [
    md(r'''
# Live demo: the original Min-K% code, start to finish, in about a minute

**T4 GPU**, then run the cells one by one while recording. Uses Pythia-160M (smallest, fastest) on
200 WikiMIA passages of 64 words; the full sweep is `01_replication_sweep.ipynb`.
'''),
    SETUP,
    DRIVER,
    md("## 0. Load the model and data (run this BEFORE you start recording)"),
    code(r'''
TARGET = 'EleutherAI/pythia-160m'
m1, m2, t1, t2 = R.load_model(TARGET, REF)                       # original loader
data = R.convert_huggingface_data_to_list_dic(R.load_dataset('swj0419/WikiMIA', split='WikiMIA_length64'))
print('ready:', TARGET, '+', REF, '|', len(data), 'WikiMIA passages of 64 words')
'''),
    md("## 1. Score passages with the original `run.py` (live, ~10 s)"),
    code(r'''
out = R.evaluate_data(data[:200], m1, m2, t1, t2, 'input', TARGET, REF)   # original scoring loop
'''),
    md("## 2. What Min-K% looks at: the least likely 20% of tokens in one passage"),
    code(r'''
def show(ex):
    _, logp, _ = R.calculatePerplexity(ex['input'], m1, t1, m1.device)     # original per-token log-probs
    toks = t1.convert_ids_to_tokens(t1.encode(ex['input']))[1:]
    k = int(len(logp) * 0.2)
    worst = sorted(zip(logp, toks))[:k]
    kind = 'MEMBER (pre-2017)' if ex['label'] == 1 else 'NON-MEMBER (post-2023)'
    print(f'{kind}: {ex["input"][:110]}...')
    print('  lowest-20% tokens:', ', '.join(f'{t.replace("Ġ", "")}({p:.1f})' for p, t in worst[:8]), '...')
    print(f'  Min-20% score = {np.mean([p for p, _ in worst]):.2f}   (higher = more likely a member)\n')

show(next(ex for ex in out if ex['label'] == 1))
show(next(ex for ex in out if ex['label'] == 0))
'''),
    md("## 3. The original evaluation: `eval.py` writes `auc.txt` and the ROC plot"),
    code(r'''
os.makedirs('/content/demo', exist_ok=True)
R.fig_fpr_tpr(out, '/content/demo')                              # original evaluation
print(open('/content/demo/auc.txt').read())
from IPython.display import Image, display
display(Image('/content/demo/auc.png'))
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

# --------------------------------------------------------------------------
# Notebook 4: the controls (MIMIR, years masked) and the LLM-as-judge recency audit
# --------------------------------------------------------------------------
write("04_controls_and_judge.ipynb", [
    md(r'''
# Final experiments: is Min-K% detecting membership, or detecting *time*?

**Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all**. About 1.5 hours.
You will be asked twice for something:
1. **Google Drive access**: allow it. Every result is written to `My Drive/COMP8240/` as soon as it is
   computed, so if Colab disconnects you just **Run all** again and finished parts are skipped.
2. **Upload MIMIR** (first run only): choose the two `.jsonl` files from `data/external/mimir/` on your
   laptop. MIMIR is gated on Hugging Face, so it is never put in the public repository.

| Part | Question | Data | Model |
|---|---|---|---|
| A | Does Min-K% still work when members and non-members come from the **same time**? | MIMIR Wikipedia (Pile train vs Pile test) | Pythia 160M / 1.4B / 2.8B |
| B | How much of the AUC comes from **year tokens**? | WikiMIA and WikiMIA-2026, every year → `YEAR` | Pythia-2.8B |
| C | Can a reader with **no access to Pythia** tell members from non-members? (LLM-as-judge) | the 100 human-annotated pairs + every benchmark, years masked | Qwen2.5-3B-Instruct, Phi-3.5-mini-instruct |

Parts A and B call the **original, unmodified** `run.py` exactly as notebooks 01–02 do. At the end a zip of
the new results downloads; unzip it in the repository folder (it contains `results/...`).
'''),
    SETUP,
    DRIVER,
    code(r'''
# ---- Results go to Google Drive, so a disconnect loses nothing (Run all again resumes) ----
USE_DRIVE = True
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE = '/content/drive/MyDrive/COMP8240'
else:
    BASE = '/content'
RES = f'{BASE}/results'
os.makedirs(RES, exist_ok=True)

# ---- This project's repository: WikiMIA-2026, the annotation pairs and the year-masking function ----
!git clone --quiet https://github.com/Ashik-here/COMP8240-MinK-Replication.git /content/mine 2>/dev/null || (cd /content/mine && git pull --quiet)
!cd /content/mine && git log -1 --format='project repo @ %h (%ad)' --date=short
sys.path.append('/content/mine/scripts')
from make_annotation import mask_years          # exactly the masking the human annotators saw
LENGTHS = [32, 64, 128, 256]
TARGET = 'EleutherAI/pythia-2.8b'
MAIN = ['PPL', 'Zlib', 'Lowercase', 'Smaller Ref', 'Min-20%']

# Gated data must never reach the public repository. With KEEP_TEXT = False the per-passage score
# files store the passage index and a short fingerprint (hash) of its text, not the text itself.
# (This replaces the driver's save_jsonl.)
import hashlib
fp = lambda text: hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]
KEEP_TEXT = True
def save_jsonl(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for i, r in enumerate(rows):
            rec = {'idx': i, 'h': fp(r['input']), 'label': int(r['label']), 'pred': r['pred']}
            if KEEP_TEXT:
                rec['input'] = r['input']
            f.write(json.dumps(rec) + '\n')

def boot_auc(y, s, n=1000, seed=0):
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            vals.append(roc_auc_score(y[i], s[i]))
    return np.percentile(vals, [2.5, 97.5])

# WikiMIA-2026 is read in FILE order. (The original eval.load_jsonl shuffles with random.seed(0), so the
# scores from notebook 02 are in that shuffled order: harmless for AUC, but it would break joins by index.)
wm26 = {L: [json.loads(l) for l in open(f'/content/mine/data/wikimia2026/WikiMIA2026_length{L}.jsonl',
                                        encoding='utf-8')] for L in LENGTHS}
wikimia = {L: R.convert_huggingface_data_to_list_dic(
               R.load_dataset('swj0419/WikiMIA', split=f'WikiMIA_length{L}')) for L in LENGTHS}
print('WikiMIA', {L: len(v) for L, v in wikimia.items()}, '| WikiMIA-2026', {L: len(v) for L, v in wm26.items()})
print('results ->', RES)
'''),
    md(r'''
## Part A. Min-K% on MIMIR: the control with no time gap

MIMIR (Duan et al., COLM 2024) takes members from the Pile's **training** split and non-members from its
**test** split: same source, same time period, so there is no date or topic shift to exploit. The blind
text classifier already scores chance here (0.48–0.55). If Min-K% also falls to about 0.5, its WikiMIA
AUC reflects the time gap rather than memorisation.
'''),
    code(r'''
MIMIR_DIR = f'{BASE}/mimir_private'             # your Drive only; never committed
MIMIR_FILES = {1: 'wikipedia_en_ngram_13_0.8_train_members.jsonl',     # Pile train -> members
               0: 'wikipedia_en_ngram_13_0.8_test_nonmembers.jsonl'}   # Pile test  -> non-members
os.makedirs(MIMIR_DIR, exist_ok=True)
if not all(os.path.exists(f'{MIMIR_DIR}/{f}') for f in MIMIR_FILES.values()):
    from google.colab import files
    print('Click "Choose files" and select BOTH of these (from data/external/mimir/ on your laptop):',
          *MIMIR_FILES.values(), sep='\n  ')
    for name, content in files.upload().items():
        clean = re.sub(r' \(\d+\)(?=\.jsonl$)', '', name)     # Colab renames repeats to "name (1).jsonl"
        open(f'{MIMIR_DIR}/{clean}', 'wb').write(content)
missing = [f for f in MIMIR_FILES.values() if not os.path.exists(f'{MIMIR_DIR}/{f}')]
assert not missing, f'still missing {missing}: run this cell again and choose both files'

# Same construction as scripts/blind_baseline.py (members first, each passage cut to its first L words),
# so passage indices line up with the blind-baseline results.
mimir_text = {lab: [json.loads(l) for l in open(f'{MIMIR_DIR}/{f}', encoding='utf-8') if l.strip()]
              for lab, f in MIMIR_FILES.items()}
MIMIR_LENGTHS = [32, 64, 128]                   # every MIMIR passage has at least 100 words
mimir = {L: [{'input': ' '.join(t.split()[:L]), 'label': lab} for lab in (1, 0) for t in mimir_text[lab]]
         for L in MIMIR_LENGTHS}
print({L: len(v) for L, v in mimir.items()}, '| members:', sum(x['label'] for x in mimir[64]))
'''),
    code(r'''
MIMIR_MODELS = ['EleutherAI/pythia-2.8b', 'EleutherAI/pythia-1.4b', 'EleutherAI/pythia-160m']   # biggest first
OUT_MIMIR = f'{RES}/mimir'
rows = []
KEEP_TEXT = False                               # gated data: indices only
try:
    for target in MIMIR_MODELS:
        rows += run_model(target, [(L, mimir[L]) for L in MIMIR_LENGTHS], OUT_MIMIR, 'MIMIR-Wikipedia')
        pd.DataFrame(rows).to_csv(f'{OUT_MIMIR}/summary.csv', index=False)
finally:
    KEEP_TEXT = True
json.dump(ENV, open(f'{OUT_MIMIR}/environment.json', 'w'), indent=1)
t = pd.DataFrame(rows)
t = t[t.method.isin(MAIN)].pivot_table(index=['model', 'method'], columns='length', values='auc')
t['mean'] = t.mean(axis=1); display(t.round(3))
'''),
    md(r'''
## Part B. Min-K% with every year masked

The same Pythia-2.8B attack on the same passages, with every year (and season such as `2016–17`) replaced
by `YEAR`, the masking the annotators and the blind classifier saw. Non-member passages mention recent
years that Pythia has rarely seen, which are exactly the low-probability tokens Min-K% averages over.
If the AUC drops a lot, the year tokens were doing much of the work.
'''),
    code(r'''
masked = {'WikiMIA': {L: [dict(ex, input=mask_years(ex['input'])) for ex in wikimia[L]] for L in LENGTHS},
          'WikiMIA-2026': {L: [dict(ex, input=mask_years(ex['input'])) for ex in wm26[L]] for L in LENGTHS}}
for name, src in [('WikiMIA', wikimia), ('WikiMIA-2026', wm26)]:
    n = sum(len(src[L]) for L in LENGTHS)
    k = sum(m['input'] != s['input'] for L in LENGTHS for m, s in zip(masked[name][L], src[L]))
    print(f'{name}: {k} of {n} passages had at least one year masked')

rows = []
for name, d in masked.items():
    rows += run_model(TARGET, [(L, d[L]) for L in LENGTHS], f'{RES}/year_masked/{name.lower()}',
                      f'{name} (years masked)')
    pd.DataFrame(rows).to_csv(f'{RES}/year_masked/summary.csv', index=False)
json.dump(ENV, open(f'{RES}/year_masked/environment.json', 'w'), indent=1)
'''),
    md("### Parts A + B together: Pythia-2.8B on every benchmark (AUC, mean over lengths)"),
    code(r'''
allres = pd.concat([pd.read_csv('/content/mine/results/replication/summary.csv'),
                    pd.read_csv('/content/mine/results/wikimia2026/summary.csv'),
                    pd.read_csv(f'{RES}/year_masked/summary.csv'),
                    pd.read_csv(f'{OUT_MIMIR}/summary.csv')])
ORDER = ['WikiMIA', 'WikiMIA (years masked)', 'WikiMIA-2026', 'WikiMIA-2026 (years masked)', 'MIMIR-Wikipedia']
cmp = (allres[(allres.model == TARGET) & allres.method.isin(MAIN)]
       .groupby(['dataset', 'method']).auc.mean().unstack().reindex(ORDER)[MAIN])
display(cmp.round(3))      # MIMIR: mean of lengths 32/64/128; the others: 32/64/128/256
'''),
    md(r'''
## Part C. LLM-as-judge recency audit

Two small open judges from different model families do the **human annotators' task** on the same 100 pairs
(years masked, same instructions). Each pair is asked twice with the passages swapped, to measure and cancel
**position bias**. The verdict is read from the **probability of the answer token** (`A` vs `B`), not from
generated text, so nothing needs parsing and every answer comes with a confidence.

Once the judges' agreement with the humans is known (Cohen's κ, computed locally), the same judges score
**every passage of every benchmark** one at a time. Their AUC is a "world-knowledge blind baseline": how well
members and non-members can be told apart by someone who has never seen Pythia's training data.
'''),
    code(r'''
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm

JUDGES = ['Qwen/Qwen2.5-3B-Instruct', 'microsoft/Phi-3.5-mini-instruct']   # open licences, fit a T4 in fp16
OUT_J = f'{RES}/llm_judge'

HINT = ('Clues people use: names of politicians or office holders, technology, how many times an event has '
        'already happened ("the 61st edition"), whether the text says something will happen, and so on.')
PAIR_PROMPT = ('Below are two passages, each from the start of a Wikipedia article. Every year has been replaced '
               'with the word YEAR. One passage describes an older event and the other describes a newer event.'
               '\n\n' + HINT + '\n\nPassage A: """{a}"""\n\nPassage B: """{b}"""\n\n'
               'Which passage describes the more recent event? Answer with a single letter: A or B.')
POINT_PROMPT = ('Below is a passage from the start of a Wikipedia article. Every year has been replaced with the '
                'word YEAR.\n\n' + HINT + '\n\nPassage: """{text}"""\n\n'
                'Is the event or topic of this article from before 2017, or from 2023 or later?\n'
                'A) before 2017\nB) 2023 or later\n\nAnswer with a single letter: A or B.')

def letter_ids(tok, letter):
    """Every token id that spells the bare answer letter, however the tokenizer writes it."""
    vocab = tok.get_vocab()
    ids = {vocab[s] for s in (letter, '▁' + letter, 'Ġ' + letter) if s in vocab}
    for s in (letter, ' ' + letter, '\n' + letter):
        t = tok.encode(s, add_special_tokens=False)
        if t and tok.decode(t[-1:]).strip() == letter:
            ids.add(t[-1])
    return sorted(ids)

def load_judge(name):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16).to('cuda').eval()
    return model, tok, letter_ids(tok, 'A'), letter_ids(tok, 'B')

@torch.no_grad()
def ask(judge, prompt):
    """Probability-of-token verdict: P('A') and P('B') as the first token of the answer."""
    model, tok, A, B = judge
    text = tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                                   add_generation_prompt=True)
    ids = tok(text, return_tensors='pt', add_special_tokens=False).input_ids.to('cuda')
    try:
        logits = model(ids, logits_to_keep=1).logits[0, -1]
    except TypeError:                           # older transformers without logits_to_keep
        logits = model(ids).logits[0, -1]
    p = torch.softmax(logits.float(), -1)
    pa, pb = p[A].sum().item(), p[B].sum().item()
    assert np.isfinite(pa + pb) and pa + pb > 0, 'non-finite answer probabilities (fp16 overflow?)'
    return pa, pb

pairs = [json.loads(l) for l in open('/content/mine/annotation/pairs_for_annotators.jsonl', encoding='utf-8')]
POINT_SETS = {'WikiMIA': masked['WikiMIA'], 'WikiMIA-2026': masked['WikiMIA-2026'],
              'MIMIR-Wikipedia': {L: [dict(ex, input=mask_years(ex['input'])) for ex in mimir[L]]
                                  for L in MIMIR_LENGTHS}}

def write_rows(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.writelines(json.dumps(r) + '\n' for r in rows)

def run_judge(name):
    tag = name.split('/')[-1]
    pair_file = f'{OUT_J}/pairs_{tag}.jsonl'
    jobs = [(ds, L) for ds, d in POINT_SETS.items() for L in d
            if not os.path.exists(f'{OUT_J}/pointwise/{tag}_{ds}_len{L}.jsonl')]
    if os.path.exists(pair_file) and not jobs:
        return print(f'{name}: already done')
    t0 = time.time(); judge = load_judge(name)
    print(f'{name}: loaded in {time.time()-t0:.0f}s; answer tokens A={judge[2]} B={judge[3]}')
    if not os.path.exists(pair_file):
        rows = []
        for p in tqdm(pairs, desc='100 pairs x 2 orders'):
            a, b = p['passages']
            ab = ask(judge, PAIR_PROMPT.format(a=a, b=b))          # passage 0 shown as A
            ba = ask(judge, PAIR_PROMPT.format(a=b, b=a))          # swapped: passage 0 shown as B
            rows.append({'id': p['id'], 'judge': name, 'ab': ab, 'ba': ba,
                         'p0_newer': 0.5 * (ab[0] / sum(ab) + ba[1] / sum(ba))})
        write_rows(rows, pair_file)
    for ds, L in jobs:                          # no passage text is stored, only indices and scores
        rows = []
        for i, ex in enumerate(tqdm(POINT_SETS[ds][L], desc=f'{ds} len {L}')):
            pa, pb = ask(judge, POINT_PROMPT.format(text=ex['input']))
            rows.append({'idx': i, 'h': fp(ex['input']), 'label': int(ex['label']),
                         'p_newer': pb / (pa + pb), 'mass': pa + pb})
        write_rows(rows, f'{OUT_J}/pointwise/{tag}_{ds}_len{L}.jsonl')
        y = np.array([r['label'] for r in rows]); s = 1 - np.array([r['p_newer'] for r in rows])
        print(f'  {ds} len {L}: AUC {roc_auc_score(y, s):.3f}')
    del judge

for name in JUDGES:                             # one judge failing must not stop the other
    try:
        run_judge(name)
    except Exception as e:
        print(f'!! {name} failed and was skipped: {type(e).__name__}: {e}')
    gc.collect(); torch.cuda.empty_cache()
'''),
    md("### Judge results"),
    code(r'''
# Ground truth for this printout only: map each masked WikiMIA-2026 passage back to its label.
truth = {mask_years(r['input']): r['label'] for r in wm26[64]}
newer = {p['id']: [truth[x] for x in p['passages']].index(0) for p in pairs}   # index of the 2026 passage

rows = []
for name in JUDGES:
    f = f'{OUT_J}/pairs_{name.split("/")[-1]}.jsonl'
    if not os.path.exists(f):
        continue
    J = [json.loads(l) for l in open(f)]
    gold = np.array([newer[r['id']] == 0 for r in J])            # True when passage 0 is the newer one
    v1 = np.array([r['ab'][0] > r['ab'][1] for r in J])          # "passage 0 is newer", shown as A
    v2 = np.array([r['ba'][1] > r['ba'][0] for r in J])          # "passage 0 is newer", shown as B
    deb = np.array([r['p0_newer'] > 0.5 for r in J])
    pA = [r['ab'][0] / sum(r['ab']) for r in J] + [r['ba'][0] / sum(r['ba']) for r in J]
    rows.append({'judge': name, 'accuracy (A/B order 1)': (v1 == gold).mean(),
                 'accuracy (order 2)': (v2 == gold).mean(), 'accuracy (both orders averaged)': (deb == gold).mean(),
                 'same verdict both orders': (v1 == v2).mean(), 'mean P(answer A)': np.mean(pA),
                 'P(A)+P(B)': np.mean([sum(r['ab']) for r in J] + [sum(r['ba']) for r in J])})
pair_summary = pd.DataFrame(rows); pair_summary.to_csv(f'{OUT_J}/summary_pairs.csv', index=False)
print('On the 100 human-annotated pairs (chance = 0.5; mean P(answer A) = 0.5 means no position bias):')
display(pair_summary.round(3))

rows = []
for name in JUDGES:
    for ds, d in POINT_SETS.items():
        for L in d:
            f = f'{OUT_J}/pointwise/{name.split("/")[-1]}_{ds}_len{L}.jsonl'
            if not os.path.exists(f):
                continue
            J = [json.loads(l) for l in open(f)]
            y = np.array([r['label'] for r in J]); s = 1 - np.array([r['p_newer'] for r in J])
            lo, hi = boot_auc(y, s)
            rows.append(dict(judge=name, dataset=ds, length=L, n=len(y), auc=roc_auc_score(y, s),
                             auc_lo=lo, auc_hi=hi, answer_mass=np.mean([r['mass'] for r in J])))
point_summary = pd.DataFrame(rows); point_summary.to_csv(f'{OUT_J}/summary_pointwise.csv', index=False)
json.dump(ENV, open(f'{OUT_J}/environment.json', 'w'), indent=1)
print('Pointwise AUC, years masked (compare with Min-K% above; MIMIR should be about 0.5):')
t = point_summary.pivot_table(index=['judge', 'dataset'], columns='length', values='auc')
t['mean'] = t.mean(axis=1); display(t.round(3))
'''),
    md("## Download the new results (also kept in `My Drive/COMP8240/results`)"),
    code(r'''
!rm -f /content/new_results.zip && cd {BASE} && zip -qr /content/new_results.zip results/mimir results/year_masked results/llm_judge
!unzip -l /content/new_results.zip | tail -1
from google.colab import files
files.download('/content/new_results.zip')
'''),
])
