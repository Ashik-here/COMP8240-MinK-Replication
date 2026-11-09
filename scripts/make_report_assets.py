r"""Figures, tables and number macros for the final report, built only from results/ and data/.

Every number in report/final_report/main.tex that comes from an experiment is a macro defined here, so the
report cannot drift from the results. Results that do not exist yet (a run still in progress) become
\pending, so the report always compiles and a grep for "pending" lists what is missing.

Usage:  python scripts/make_report_assets.py
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report/final_report"
FIG, GEN = OUT / "figures", OUT / "generated"
FIG.mkdir(parents=True, exist_ok=True)
GEN.mkdir(parents=True, exist_ok=True)

# Validated categorical palette (light mode), fixed order; text stays in ink colours.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "pdf.fonttype": 42})
MAIN = ["PPL", "Zlib", "Lowercase", "Smaller Ref", "Min-20%"]
T28 = "EleutherAI/pythia-2.8b"
PARAMS = {"EleutherAI/pythia-160m": 0.16, "EleutherAI/pythia-410m": 0.41, "EleutherAI/pythia-1b": 1.0,
          "EleutherAI/pythia-1.4b": 1.4, "EleutherAI/pythia-2.8b": 2.8}
JUDGES = ["Qwen2.5-3B-Instruct", "Phi-3.5-mini-instruct"]
PENDING = r"\pending{}"
macros, missing = {}, []


def csv(path):
    p = ROOT / path
    return pd.read_csv(p) if p.exists() else None


def f3(x):
    return PENDING if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def f2(x):
    return PENDING if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"


def put(name, value):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    macros[name] = value
    if PENDING in str(value):
        missing.append(name)


def tex_escape(s):
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    s = "".join(rep.get(c, c) for c in s)
    return s.replace("–", "--").replace("—", "---").replace("’", "'").replace("‘", "`")


def mean_auc(df, model, method, dataset=None):
    if df is None:
        return None
    d = df[(df.model == model) & (df.method == method)]
    if dataset is not None:
        d = d[d.dataset == dataset]
    return d.auc.mean() if len(d) else None


# ---------------------------------------------------------------- replication (WikiMIA)
rep = csv("results/replication/summary.csv")
for m, name in zip(MAIN, ["PPL", "Zlib", "Lower", "Ref", "MinK"]):
    put(f"Rep{name}", f3(mean_auc(rep, T28, m)))
    put(f"New{name}", f3(mean_auc(csv("results/wikimia2026/summary.csv"), T28, m)))
put("RepMinKTPR", f"{rep[(rep.model == T28) & (rep.method == 'Min-20%')].tpr_at_5fpr.mean() * 100:.1f}")
put("RepZlibTPR", f"{rep[(rep.model == T28) & (rep.method == 'Zlib')].tpr_at_5fpr.mean() * 100:.1f}")
size = rep[rep.method == "Min-20%"].groupby("model").auc.mean()
put("RepSizeTrend", ", ".join(f"{size[m]:.3f}" for m in PARAMS))
lens = rep[(rep.model == T28) & (rep.method == "Min-20%")].set_index("length")
put("RepMinKShort", f3(lens.auc[32])); put("RepMinKLong", f3(lens.auc[256]))
put("RepCIShort", f"{lens.auc_lo[32]:.2f}--{lens.auc_hi[32]:.2f}")
put("RepCILong", f"{lens.auc_lo[256]:.2f}--{lens.auc_hi[256]:.2f}")
ks = rep[(rep.model == T28) & rep.method.str.startswith("Min-")].groupby("method").auc.mean()
put("RepBestK", ks.idxmax().replace("Min-", "").replace("%", r"\%"))
put("RepMarginZlib", f"{mean_auc(rep, T28, 'Min-20%') - mean_auc(rep, T28, 'Zlib'):.3f}")

new = csv("results/wikimia2026/summary.csv")
nsize = new[new.method == "Min-20%"].groupby("model").auc.mean()
put("NewSizeTrend", ", ".join(f"{nsize[m]:.3f}" for m in PARAMS if m in nsize))

tab = pd.read_csv(ROOT / "results/replication/pythia28_vs_paper.csv").set_index("method").loc[MAIN]
lines = [r"\begin{tabular}{@{}lccccc>{\bfseries}cc@{}}", r"\toprule",
         r" & \multicolumn{5}{c}{\textbf{AUC (mine)}} & & \textbf{TPR@5\%} \\",
         r"\cmidrule(lr){2-6}",
         r"\textbf{Method} & 32 & 64 & 128 & 256 & mean & \textbf{paper} & mine / paper \\", r"\midrule"]
for m, r in tab.iterrows():
    nm = r"\textbf{Min-K\% (k=20)}" if m == "Min-20%" else m
    cells = [f"{r[c]:.3f}" for c in ["32", "64", "128", "256", "mean (4 lengths)"]]
    if m == "Min-20%":
        cells = [rf"\textbf{{{c}}}" for c in cells]
    lines.append(f"{nm} & " + " & ".join(cells) + f" & {r['paper Table 1']:.2f} & "
                 f"{r['TPR@5%FPR mean (%)']:.1f} / {r['paper Table 6 (%)']:.1f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(GEN / "tab_replication.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

# ---------------------------------------------------------------- blind baseline
blind = csv("results/blind_baseline/summary.csv")
bm = blind.groupby(["dataset", "method"]).auc.mean()
for ds, tag in [("WikiMIA", "Wiki"), ("WikiMIA-2026", "New"), ("arXivTection", "Arx"), ("MIMIR-Wikipedia", "Mimir")]:
    put(f"Blind{tag}Raw", f3(bm.get((ds, "tfidf_lr"))))
    put(f"Blind{tag}Noyr", f3(bm.get((ds, "tfidf_lr_noyr"))))
    put(f"Blind{tag}Year", f3(bm.get((ds, "max_year"))))
put("BlindArxLeaky", f3(bm.get(("arXivTection", "tfidf_lr_ungrouped_LEAKY"))))
put("BlindMimirRange", f"{blind[(blind.dataset == 'MIMIR-Wikipedia') & blind.method.str.startswith('tfidf')].auc.min():.2f}--"
                       f"{blind[(blind.dataset == 'MIMIR-Wikipedia') & blind.method.str.startswith('tfidf')].auc.max():.2f}")

# ---------------------------------------------------------------- MIMIR and year-masked Min-K% (notebook 04)
mimir = csv("results/mimir/summary.csv")
masked = csv("results/year_masked/summary.csv")
for m, name in zip(MAIN, ["PPL", "Zlib", "Lower", "Ref", "MinK"]):
    put(f"Mimir{name}", f3(mean_auc(mimir, T28, m)))
    put(f"MaskWiki{name}", f3(mean_auc(masked, T28, m, "WikiMIA (years masked)")))
    put(f"MaskNew{name}", f3(mean_auc(masked, T28, m, "WikiMIA-2026 (years masked)")))
if mimir is not None:
    ms = mimir[mimir.method == "Min-20%"].groupby("model").auc.mean()
    put("MimirSizeTrend", ", ".join(f"{ms[m]:.3f}" for m in PARAMS if m in ms))
    d = mimir[(mimir.model == T28) & (mimir.method == "Min-20%")]
    put("MimirCI", f"{d.auc_lo.min():.2f}--{d.auc_hi.max():.2f}")
else:
    put("MimirSizeTrend", PENDING); put("MimirCI", PENDING)

# ---------------------------------------------------------------- LLM judge (notebook 04)
jp = csv("results/llm_judge/summary_pointwise.csv")
jpairs = csv("results/llm_judge/summary_pairs.csv")
for j, tag in zip(JUDGES, ["Qwen", "Phi"]):
    for ds, dtag in [("WikiMIA", "Wiki"), ("WikiMIA-2026", "New"), ("MIMIR-Wikipedia", "Mimir")]:
        v = None
        if jp is not None:
            d = jp[(jp.judge.str.endswith(j)) & (jp.dataset == ds)]
            v = d.auc.mean() if len(d) else None
        put(f"Judge{tag}{dtag}", f3(v))
    v = None
    if jpairs is not None:
        d = jpairs[jpairs.judge.str.endswith(j)]
        if len(d):
            v = d.iloc[0]
            put(f"Judge{tag}PairAcc", f2(v["accuracy (both orders averaged)"]))
            put(f"Judge{tag}OrderAcc", f"{v['accuracy (A/B order 1)']:.2f}/{v['accuracy (order 2)']:.2f}")
            put(f"Judge{tag}Consistent", f2(v["same verdict both orders"]))
            put(f"Judge{tag}PA", f2(v["mean P(answer A)"]))
            put(f"Judge{tag}Mass", f2(v["P(A)+P(B)"]))
    if v is None:
        for k in ["PairAcc", "OrderAcc", "Consistent", "PA", "Mass"]:
            put(f"Judge{tag}{k}", PENDING)

# ---------------------------------------------------------------- shift-controlled (ND-2)
sc = csv("results/shift_controlled/summary.csv")
rows = [r"\begin{tabular}{@{}llrcccc@{}}", r"\toprule",
        r"& & & \multicolumn{2}{c}{\textbf{matched subset}} & \multicolumn{2}{c}{\textbf{Min-K\% AUC}} \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"\textbf{Data} & $L$ & $n$ & blind & Min-K\% [95\% CI] & full set & stratified \\", r"\midrule"]
for ds in ["WikiMIA", "WikiMIA-2026"]:
    for L in [32, 64, 128, 256]:
        d = sc[(sc.dataset == ds) & (sc.length == L)]
        mt = d[(d.control == "matched") & (d.confound == "tfidf_lr")].set_index("method")
        st = d[(d.control == "stratified") & (d.confound == "tfidf_lr")].set_index("method")
        full = d[(d.control == "none")].set_index("method")
        k = mt.loc["Min-20%"]
        rows.append(f"{ds if L == 32 else ''} & {L} & {int(k.n)} & {mt.loc['blind (itself)'].auc:.2f} & "
                    f"{k.auc:.2f} [{k.auc_lo:.2f}, {k.auc_hi:.2f}] & {full.loc['Min-20%'].auc:.2f} & "
                    f"{st.loc['Min-20%'].auc:.2f} \\\\")
    if ds == "WikiMIA":
        rows.append(r"\addlinespace")
rows += [r"\bottomrule", r"\end{tabular}"]
(GEN / "tab_shift.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
mt = sc[(sc.control == "matched") & (sc.confound == "tfidf_lr")]
put("ShiftMinKRange", f"{mt[mt.method == 'Min-20%'].auc.min():.2f}--{mt[mt.method == 'Min-20%'].auc.max():.2f}")
put("ShiftBlindRange", f"{mt[mt.method == 'blind (itself)'].auc.min():.2f}--{mt[mt.method == 'blind (itself)'].auc.max():.2f}")
w = mt[mt.method == "Min-20%"]
put("ShiftMinKPooled", f"{np.average(w.auc, weights=w.n):.2f}")
put("ShiftNMin", str(int(w.n.min()))); put("ShiftNMax", str(int(w.n.max())))

# ---------------------------------------------------------------- results on new data: one table
def cell(df, ds, method, model=T28, judge=None):
    if df is None:
        return PENDING
    d = df[df.dataset == ds]
    if judge is not None:
        d = d[d.judge.str.endswith(judge)]
    else:
        if "model" in d:
            d = d[d.model == model]
        d = d[d.method == method]
    return f"{d.auc.mean():.3f}" if len(d) else "--"

cols = [("WikiMIA", rep, blind, "WikiMIA"), ("WikiMIA-2026", new, blind, "WikiMIA-2026"),
        ("MIMIR", mimir, blind, "MIMIR-Wikipedia")]
rows = [r"\begin{tabular}{@{}lccc|cc@{}}", r"\toprule",
        r"& \multicolumn{3}{c|}{\textbf{as released}} & \multicolumn{2}{c}{\textbf{years masked}} \\",
        r"\textbf{Method} & WikiMIA & WikiMIA-26 & MIMIR & WikiMIA & WikiMIA-26 \\", r"\midrule",
        r"\multicolumn{6}{@{}l}{\emph{Membership attacks, Pythia-2.8B (original code)}} \\"]
for m in MAIN:
    nm = r"Min-K\% (k=20)" if m == "Min-20%" else m
    c = [cell(df, dsname, m) for _, df, _, dsname in cols]
    c += [cell(masked, "WikiMIA (years masked)", m), cell(masked, "WikiMIA-2026 (years masked)", m)]
    rows.append(f"{nm} & " + " & ".join(c) + r" \\")
rows += [r"\addlinespace", r"\multicolumn{6}{@{}l}{\emph{Blind baselines: text only, no model access}} \\"]
for meth, nm in [("max_year", "Latest year in text"), ("tfidf_lr", "TF-IDF + LR"), ("tfidf_lr_noyr", "TF-IDF + LR, years masked")]:
    c = [f"{bm[(ds, meth)]:.3f}" for ds in ["WikiMIA", "WikiMIA-2026", "MIMIR-Wikipedia"]]
    c += ["--", "--"]
    rows.append(f"{nm} & " + " & ".join(c) + r" \\")
rows += [r"\addlinespace", r"\multicolumn{6}{@{}l}{\emph{LLM judges, years masked, one passage at a time}} \\"]
for j in JUDGES:
    c = ["--", "--", "--"] + [cell(jp, "WikiMIA", None, judge=j), cell(jp, "WikiMIA-2026", None, judge=j)]
    # judges only ever see masked text; MIMIR goes in the masked block too
    c[2] = cell(jp, "MIMIR-Wikipedia", None, judge=j)
    rows.append(f"{j} & " + " & ".join(c) + r" \\")
rows += [r"\bottomrule", r"\end{tabular}"]
(GEN / "tab_results.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

# ---------------------------------------------------------------- human annotation
ann_path = ROOT / "results/annotation/summary.json"
if ann_path.exists():
    ann = json.loads(ann_path.read_text(encoding="utf-8"))
    acc = {r["who"]: r for r in ann["accuracy"]}
    ag = ann["agreement"]
    rows = [r"\begin{tabular}{@{}lrrc@{}}", r"\toprule",
            r"\textbf{Who} & \textbf{decided} & \textbf{can't tell} & \textbf{accuracy [95\% CI]} \\", r"\midrule"]
    for who, r in acc.items():
        nm = tex_escape(who).replace("Min-20%", r"Min-K\%")
        rows.append(f"{nm} & {r['decided']} & {r['cant_tell']} & {r['accuracy_decided']:.2f} "
                    f"[{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}] \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_annotation.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    put("NAnnotators", str(ann["n_annotators"]))
    put("FleissK", f2(ag.get("fleiss_kappa_3cat", np.nan)))
    put("FleissKDecided", f2(ag.get("fleiss_kappa_decided_only", np.nan)))
    ck = [v for k, v in ag.items() if re.fullmatch(r"cohen_kappa_A\d_A\d", k)]
    put("CohenHumanRange", f"{min(ck):.2f}--{max(ck):.2f}" if ck else PENDING)
    maj = acc.get("human majority")
    put("MajorityAcc", f2(maj["accuracy_decided"]) if maj else PENDING)
    hum = [r for w, r in acc.items() if re.match(r"A\d( \(author\))?$", w)]
    put("HumanAccRange", f"{min(r['accuracy_decided'] for r in hum):.2f}--{max(r['accuracy_decided'] for r in hum):.2f}")
    for j, tag in zip(JUDGES, ["Qwen", "Phi"]):
        put(f"Kappa{tag}Majority", f2(ag.get(f"cohen_kappa_{j}_majority", np.nan)))
    put("KappaJudges", f2(ag.get(f"cohen_kappa_{JUDGES[0]}_{JUDGES[1]}", np.nan)))
    put("PairMinK", f2(acc["Min-20% (Pythia-2.8B)"]["accuracy_decided"]))
    put("PairBlind", f2(acc["blind TF-IDF (raw text)"]["accuracy_decided"]))
    put("PairBlindNoyr", f2(acc["blind TF-IDF (years masked)"]["accuracy_decided"]))
    put("KappaMinKMajority", f2(ag.get("cohen_kappa_Min-20% (Pythia-2.8B)_majority", np.nan)))
else:
    (GEN / "tab_annotation.tex").write_text(r"\textcolor{red}{[annotation table pending]}" + "\n", encoding="utf-8")
    for k in ["NAnnotators", "FleissK", "FleissKDecided", "CohenHumanRange", "MajorityAcc", "HumanAccRange",
              "KappaQwenMajority", "KappaPhiMajority", "KappaJudges", "PairMinK", "PairBlind", "PairBlindNoyr",
              "KappaMinKMajority"]:
        put(k, PENDING)

# ---------------------------------------------------------------- data snippets figure
PERSON_EVENT = re.compile(r"^(Killing|Death|Murder|Shooting|Assassination|Disappearance|Kidnapping|Stabbing) of", re.I)


def first(path, label):
    """First passage of a class, skipping pages about the death or harm of a named person (see Ethics)."""
    for l in (ROOT / path).read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        if int(r["label"]) == label and not PERSON_EVENT.search(r.get("title", "")):
            return r

snip = {"wm_m": first("data/external/wikimia/WikiMIA_length32.jsonl", 1)["input"],
        "wm_n": first("data/external/wikimia/WikiMIA_length32.jsonl", 0)["input"],
        "n26_m": first("data/wikimia2026/WikiMIA2026_length32.jsonl", 1)["input"],
        "n26_n": first("data/wikimia2026/WikiMIA2026_length32.jsonl", 0)["input"]}
for k, v in snip.items():
    put("Snip" + {"wm_m": "WikiMember", "wm_n": "WikiNon", "n26_m": "NewMember", "n26_n": "NewNon"}[k], tex_escape(v))
row26 = first("data/wikimia2026/WikiMIA2026_length32.jsonl", 0)
put("SnipNewJSON", tex_escape(json.dumps({k: (row26[k][:38] + "...") if k == "input" else row26[k]
                                          for k in ["input", "label", "title", "created", "pageid", "revid"]},
                                         ensure_ascii=False)))

# ---------------------------------------------------------------- figure: Min-K% vs model size on 3 benchmarks
fig, ax = plt.subplots(figsize=(3.15, 2.15))
series = [("WikiMIA", rep, BLUE, "o"), ("WikiMIA-2026", new, ORANGE, "s"), ("MIMIR (no time gap)", mimir, AQUA, "^")]
for name, df, colour, mk in series:
    if df is None:
        continue
    g = df[df.method == "Min-20%"].groupby("model").auc.mean()
    xs = [PARAMS[m] for m in PARAMS if m in g]
    ys = [g[m] for m in PARAMS if m in g]
    ax.plot(xs, ys, marker=mk, color=colour, lw=1.6, ms=4.5, label=name)
    ax.annotate(f"{ys[-1]:.2f}", (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=7.5, color=INK)
ax.axhline(0.5, color=INK2, lw=0.8, ls="--")
ax.text(0.15, 0.505, "chance", fontsize=7, color=INK2, va="bottom")
ax.set_xscale("log"); ax.set_xticks(list(PARAMS.values())); ax.set_xticklabels(["160M", "410M", "1B", "1.4B", "2.8B"])
ax.minorticks_off(); ax.set_xlim(0.13, 3.9); ax.set_ylim(0.45, 0.72)
ax.set_xlabel("Pythia model size"); ax.set_ylabel("Min-K% AUC (mean over lengths)")
ax.legend(loc="upper left", fontsize=7.5, handlelength=1.6)
fig.tight_layout(pad=0.2); fig.savefig(FIG / "size_trend.pdf"); plt.close(fig)

# ---------------------------------------------------------------- figure: what separates members, per benchmark
groups = [("WikiMIA", "WikiMIA"), ("WikiMIA-2026\n(new)", "WikiMIA-2026"), ("arXivTection", "arXivTection"),
          ("MIMIR\n(no time gap)", "MIMIR-Wikipedia")]
judge_auc = {}
if jp is not None:
    d = jp[jp.judge.str.endswith(JUDGES[0])]
    judge_auc = d.groupby("dataset").auc.mean().to_dict()
bars = [("Min-K% on Pythia-2.8B", BLUE, lambda ds: {"WikiMIA": mean_auc(rep, T28, "Min-20%"),
                                                    "WikiMIA-2026": mean_auc(new, T28, "Min-20%"),
                                                    "MIMIR-Wikipedia": mean_auc(mimir, T28, "Min-20%")}.get(ds)),
        ("Blind TF-IDF (text only)", ORANGE, lambda ds: bm.get((ds, "tfidf_lr"))),
        ("Blind TF-IDF, years masked", AQUA, lambda ds: bm.get((ds, "tfidf_lr_noyr"))),
        (f"LLM judge ({JUDGES[0].split('-Instruct')[0]}), years masked", YELLOW, lambda ds: judge_auc.get(ds))]
fig, ax = plt.subplots(figsize=(6.3, 2.25))
wdt = 0.2
for j, (label, colour, get) in enumerate(bars):
    for i, (_, ds) in enumerate(groups):
        v = get(ds)
        x = i + (j - 1.5) * wdt
        if v is None or (isinstance(v, float) and np.isnan(v)):
            ax.text(x, 0.415, "n/a", ha="center", fontsize=6, color=INK2)
            continue
        ax.bar(x, v - 0.4, wdt - 0.03, bottom=0.4, color=colour, label=label if i == 0 else None, zorder=3)
        ax.text(x, v + 0.008, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK)
ax.axhline(0.5, color=INK2, lw=0.8, ls="--", zorder=2)
ax.set_xticks(range(len(groups))); ax.set_xticklabels([g for g, _ in groups])
ax.set_ylim(0.4, 1.04); ax.set_ylabel("AUC (mean over lengths)")
handles, labels = ax.get_legend_handles_labels()
for label, colour, _ in bars:                     # keep the legend complete even if a series is all n/a
    if label not in labels:
        handles.append(plt.Rectangle((0, 0), 1, 1, color=colour)); labels.append(label)
ax.legend(handles, labels, loc="upper center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, 1.13), handlelength=1.2)
fig.tight_layout(pad=0.2); fig.savefig(FIG / "benchmarks.pdf"); plt.close(fig)

# ---------------------------------------------------------------- write macros
body = ["% Generated by scripts/make_report_assets.py from results/ -- do not edit by hand."]
for k, v in macros.items():
    body.append(f"\\newcommand{{\\{k}}}{{{v}}}")
(GEN / "numbers.tex").write_text("\n".join(body) + "\n", encoding="utf-8")
print(f"{len(macros)} macros, {len(missing)} pending:", ", ".join(missing) if missing else "none")
