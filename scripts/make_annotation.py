"""Build the human recency-annotation task: 100 member/non-member pairs from WikiMIA-2026, years masked.

Writes
  data/annotation/pairs.jsonl           the pairs WITH the answer key (which passage is the member)
  annotation/pairs_for_annotators.jsonl the same pairs WITHOUT the key (input for the LLM judge, notebook 04)
  annotation/annotate.html              a self-contained, offline page for annotators, WITHOUT the key

Each pair is one member (page created before 2017) and one non-member (created in 2026), both 64 words.
Annotators are asked which passage describes the more recent event, so a correct answer = the non-member.
The same pairs are later given to the LLM judge, so human-human and human-judge agreement can be measured.

Usage:  python scripts/make_annotation.py
"""
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
N_PAIRS = 100
SEED = 2026

SEASON = re.compile(r"\b(?:1[5-9]|20)\d\d\s*[–—/-]\s*(?:\d{4}|\d{2})\b")   # 2025-26, 2011-2012
YEAR = re.compile(r"\b(?:1[5-9]\d\d|20\d\d)\b")


def mask_years(text):
    return YEAR.sub("YEAR", SEASON.sub("YEAR", text))


def main():
    rows = [json.loads(l) for l in (ROOT / "data/wikimia2026/WikiMIA2026_length64.jsonl").read_text(encoding="utf-8").splitlines()]
    rng = random.Random(SEED)
    members = rng.sample([r for r in rows if r["label"] == 1], N_PAIRS)
    nonmembers = rng.sample([r for r in rows if r["label"] == 0], N_PAIRS)

    pairs, page_items = [], []
    for i, (m, n) in enumerate(zip(members, nonmembers)):
        member_first = rng.random() < 0.5                     # file order; the page shuffles again per annotator
        first, second = (m, n) if member_first else (n, m)
        pid = f"p{i:03d}"
        pairs.append({"id": pid, "member_index": 0 if member_first else 1,
                      "passages": [mask_years(first["input"]), mask_years(second["input"])],
                      "titles": [first["title"], second["title"]],
                      "pageids": [first["pageid"], second["pageid"]],
                      "created": [first["created"], second["created"]]})
        page_items.append({"id": pid, "p": [mask_years(first["input"]), mask_years(second["input"])]})

    out = ROOT / "data/annotation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "pairs.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs), encoding="utf-8")
    (ROOT / "annotation/pairs_for_annotators.jsonl").write_text(
        "".join(json.dumps({"id": it["id"], "passages": it["p"]}, ensure_ascii=False) + "\n" for it in page_items),
        encoding="utf-8")

    template = (ROOT / "annotation/annotate_template.html").read_text(encoding="utf-8")
    html = template.replace("/*__ITEMS__*/[]", json.dumps(page_items, ensure_ascii=False))
    (ROOT / "annotation/annotate.html").write_text(html, encoding="utf-8")
    print(f"{len(pairs)} pairs -> data/annotation/pairs.jsonl and annotation/annotate.html")
    print("years left after masking:", sum(bool(YEAR.search(t)) for p in pairs for t in p["passages"]))


if __name__ == "__main__":
    main()
