"""Figures + LaTeX number macros for the update presentation, built only from files in results/ and data/.

Re-run after new results arrive; the slides pick everything up from report/update_presentation/.
Usage:  python scripts/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "report" / "update_presentation" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
MACROS = ROOT / "report" / "update_presentation" / "numbers.tex"

# Validated categorical palette (fixed order), recessive greys for text and reference lines.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
PAPER_28B = {"Min-20%": 0.67, "PPL": 0.61, "Zlib": 0.65, "Lowercase": 0.59, "Smaller Ref": 0.60}

plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "pdf.fonttype": 42})
macros = {}


def fmt(x, nd=3):
    return f"{x:.{nd}f}"


# ---------------------------------------------------------------- blind-baseline audit
blind = pd.read_csv(ROOT / "results/blind_baseline/summary.csv")
order = [d for d in ["WikiMIA", "WikiMIA-2026", "arXivTection", "MIMIR-Wikipedia"] if d in set(blind.dataset)]
methods = [("max_year", "Year rule (no learning)", BLUE),
           ("tfidf_lr", "TF-IDF + logistic regression", ORANGE),
           ("tfidf_lr_noyr", "Same, years masked", AQUA)]
mean = blind.groupby(["dataset", "method"]).auc.mean()

fig, ax = plt.subplots(figsize=(7.0, 3.6))
w = 0.26
for j, (m, label, colour) in enumerate(methods):
    xs = [i + (j - 1) * w for i in range(len(order))]
    vals = [mean[(d, m)] for d in order]
    ax.bar(xs, vals, w - 0.03, color=colour, label=label, zorder=3)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=9, color=INK)
ax.hlines(0.5, -0.5, len(order) - 0.5, color=INK2, lw=1, ls="--", zorder=2)
ax.hlines(0.67, -0.5, len(order) - 0.5, color=INK, lw=1, ls=":", zorder=2)
ax.set_xlim(-0.5, len(order) - 0.5 + 0.85)               # right margin holds the line labels
ax.text(len(order) - 0.45, 0.5, "chance", color=INK2, fontsize=8, va="center", ha="left")
ax.text(len(order) - 0.45, 0.67, "paper's Min-K%\n(Pythia-2.8B)", color=INK, fontsize=8,
        va="center", ha="left")
ax.set_xticks(range(len(order)))
ax.set_xticklabels([{"WikiMIA-2026": "WikiMIA-2026\n(mine)", "MIMIR-Wikipedia": "MIMIR\n(no time gap)"}.get(d, d)
                    for d in order])
ax.set_ylim(0.4, 1.05)
ax.set_ylabel("AUC with NO model access")
ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
ax.legend(loc="upper center", bbox_to_anchor=(0.45, 1.27), ncol=2, fontsize=9.5)
fig.tight_layout()
fig.savefig(FIG / "blind_audit.pdf")
fig.savefig(FIG / "blind_audit.png", dpi=200)
plt.close(fig)

for d in order:
    key = d.replace("-", "").replace("2026", "New").replace("Wikipedia", "")
    for m, _, _ in methods:
        macros[f"blind{key}{m.replace('_', '')}"] = fmt(mean[(d, m)], 2)
wk = blind[(blind.dataset == "WikiMIA") & (blind.method == "tfidf_lr")]
macros["blindWikiMIAtfidfMax"] = fmt(wk.auc.max(), 2)
ax_leak = blind[(blind.dataset == "arXivTection") & (blind.method == "tfidf_lr_ungrouped_LEAKY")]
if len(ax_leak):
    macros["blindarXivLeaky"] = fmt(ax_leak.auc.iloc[0], 3)

# ---------------------------------------------------------------- WikiMIA-2026 build stats
stats_path = ROOT / "data/wikimia2026/stats.json"
if stats_path.exists():
    st = json.load(open(stats_path))
    for L, b in st["buckets"].items():                    # LaTeX names cannot hold digits: 32->N, 64->NA, ...
        macros[f"newN{['', 'A', 'B', 'C'][[32, 64, 128, 256].index(int(L))]}"] = str(b["n"])
    macros["newCheckedNon"] = str(st["creation_checked"]["non-members"])
    macros["newRejectedNon"] = str(sum(st["rejected"]["non-members"].values()))
    macros["newCheckedMem"] = str(st["creation_checked"]["members"])
    macros["newApiCalls"] = str(st["api_calls"])
    macros["newBuilt"] = st["built"]

# ---------------------------------------------------------------- replication sweep (from Colab)
rep_path = ROOT / "results/replication/summary.csv"
if rep_path.exists():
    rep = pd.read_csv(rep_path)
    sizes = {"EleutherAI/pythia-160m": 0.16, "EleutherAI/pythia-410m": 0.41, "EleutherAI/pythia-1b": 1.0,
             "EleutherAI/pythia-1.4b": 1.4, "EleutherAI/pythia-2.8b": 2.8}
    models = [m for m in sizes if m in set(rep.model)]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for method, colour, marker in [("Min-20%", BLUE, "o"), ("PPL", ORANGE, "s"), ("Zlib", AQUA, "^")]:
        g = rep[rep.method == method].groupby("model").auc.mean().reindex(models)
        xs = [sizes[m] for m in models]
        ax.plot(xs, g.values, marker=marker, ms=6, lw=2, color=colour, label=f"{method} (mine)", zorder=3)
        ax.plot([2.8 * 1.16], [PAPER_28B[method]], marker=marker, ms=9, mfc="none", mew=1.6, color=colour,
                ls="none", zorder=4)                     # paper's value, drawn just right of my 2.8B point
    ax.plot([], [], marker="o", mfc="none", color=INK2, ls="none", label="paper, Pythia-2.8B (hollow)")
    ax.axhline(0.5, color=INK2, lw=1, ls="--", zorder=2)
    ax.set_xscale("log")
    ax.set_xticks([sizes[m] for m in models])
    ax.set_xticklabels([m.split("-")[-1].upper() for m in models])
    ax.minorticks_off()
    ax.set_xlim(0.12, 4.2)
    ax.set_xlabel("Pythia model size (parameters)")
    ax.set_ylabel("AUC on WikiMIA\n(mean of 4 lengths)")
    ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "auc_vs_size.pdf")
    fig.savefig(FIG / "auc_vs_size.png", dpi=200)
    plt.close(fig)

    d28 = rep[rep.model == "EleutherAI/pythia-2.8b"]
    names = {"Min-20%": "MinK", "PPL": "PPL", "Zlib": "Zlib", "Lowercase": "Lowercase", "Smaller Ref": "SmallerRef"}
    for method in PAPER_28B:
        g = d28[d28.method == method]
        if len(g):
            k = names[method]
            macros[f"rep{k}"] = fmt(g.auc.mean(), 2)
            macros[f"rep{k}Tpr"] = fmt(g.tpr_at_5fpr.mean() * 100, 1)
    mk = rep[rep.method == "Min-20%"].groupby("model").auc.mean()
    macros["repMinSmall"] = fmt(mk.get("EleutherAI/pythia-160m", float("nan")), 2)
    macros["repMinLarge"] = fmt(mk.get("EleutherAI/pythia-2.8b", float("nan")), 2)
    best_len = d28[d28.method == "Min-20%"].set_index("length").auc
    macros["repMinLenShort"] = fmt(best_len.get(32, float("nan")), 2)
    macros["repMinLenLong"] = fmt(best_len.get(256, float("nan")), 2)

# ---------------------------------------------------------------- Min-K% on WikiMIA-2026 (from Colab)
new_path = ROOT / "results/wikimia2026/summary.csv"
if new_path.exists():
    nw = pd.read_csv(new_path)
    g = nw.groupby(["model", "method"]).auc.mean()
    for model, tag in [("EleutherAI/pythia-160m", "Small"), ("EleutherAI/pythia-1.4b", "Mid"),
                       ("EleutherAI/pythia-2.8b", "Large")]:
        if (model, "Min-20%") in g:
            macros[f"newMinK{tag}"] = fmt(g[(model, "Min-20%")], 2)
    for method, k in [("PPL", "PPL"), ("Zlib", "Zlib")]:
        if ("EleutherAI/pythia-2.8b", method) in g:
            macros[f"new{k}Large"] = fmt(g[("EleutherAI/pythia-2.8b", method)], 2)

# ---------------------------------------------------------------- write macros
lines = ["% Generated by scripts/make_figures.py -- do not edit by hand."]
lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
MACROS.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {len(macros)} macros to {MACROS.relative_to(ROOT)}; figures in {FIG.relative_to(ROOT)}")
print("\n".join(lines[1:]))
