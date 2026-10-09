"""Build WikiMIA-2026: the WikiMIA recipe (Shi et al., ICLR 2024, Sec. 2.2) re-run with a 2026 cut-off.

Recipe, as in the paper:
  1. non-members: Wikipedia event articles whose page was CREATED on/after the cut-off (1 Jan 2026);
  2. members:     event articles created before 2017 (the paper's Step 2);
  3. drop pages without meaningful text ("List of ...", "Timeline of ...", year overviews);
  4. truncate to 32 / 64 / 128 / 256 words and balance members and non-members in each bucket.

One deliberate change: both sides are drawn from the SAME event-category types (disasters, elections,
sports, ...), so members and non-members differ in date but not in topic mix. Membership is still
defined by date, exactly as in WikiMIA.

Output is the JSONL schema the original code reads ({"input", "label"}, label 1 = member), so the
original run.py scores it unmodified. Every passage keeps its title, page id, creation timestamp and the
revision id the text came from, so the exact text can be re-fetched later.

Usage:  python scripts/build_wikimia2026.py --out data/wikimia2026
"""
import argparse
import datetime as dt
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import requests

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "COMP8240-MinK-Replication/0.1 "
                         "(https://github.com/Ashik-here/COMP8240-MinK-Replication; student research project)"}

# Event-type subcategories of "Category:<year>", identical on both sides.
EVENT_TOPICS = ["disasters", "crimes", "controversies", "competitions", "conferences", "festivals",
                "awards", "in politics", "in science", "in sports", "in technology", "in transport",
                "in military history", "in law", "in the environment", "in health"]

SKIP_TITLE = re.compile(r"^(List of|Lists of|Timeline of|Index of|Outline of|Glossary of)\b|"
                        r"\(disambiguation\)$|^\d{4}(\s|–|-)?(\d{4})? in |^\d{4}$", re.I)
LENGTHS = [32, 64, 128, 256]


class Wiki:
    """Tiny, polite MediaWiki client: one request at a time, retries, maxlag."""

    def __init__(self, delay=0.05):
        self.s = requests.Session()
        self.s.headers.update(HEADERS)
        self.delay = delay
        self.calls = 0

    def get(self, **params):
        params.update(format="json", formatversion=2, maxlag=5)
        for attempt in range(6):
            try:
                r = self.s.get(API, params=params, timeout=30)
                self.calls += 1
                if r.status_code == 200 and "error" not in r.json():
                    time.sleep(self.delay)
                    return r.json()
                wait = int(r.headers.get("Retry-After", 2 ** attempt))
            except (requests.RequestException, ValueError):
                wait = 2 ** attempt
            time.sleep(wait)
        raise RuntimeError(f"MediaWiki API kept failing for {params}")

    def category_pages(self, root, max_pages, max_depth=2):
        """Article titles under a category, breadth-first through subcategories."""
        seen_cats, queue, pages = {root}, [(root, 0)], []
        while queue and len(pages) < max_pages:
            cat, depth = queue.pop(0)
            cont = {}
            while True:
                d = self.get(action="query", list="categorymembers", cmtitle=cat, cmtype="page|subcat",
                             cmlimit=500, **cont)
                for m in d["query"]["categorymembers"]:
                    if m["ns"] == 0:
                        pages.append(m["title"])
                    elif m["ns"] == 14 and depth < max_depth and m["title"] not in seen_cats:
                        seen_cats.add(m["title"])
                        queue.append((m["title"], depth + 1))
                if "continue" not in d or len(pages) >= max_pages:
                    break
                cont = d["continue"]
        return pages[:max_pages]

    def created(self, title):
        """Timestamp of the page's first revision (= creation), following redirects."""
        d = self.get(action="query", prop="revisions", titles=title, rvlimit=1, rvdir="newer",
                     rvprop="timestamp", redirects=1)
        page = d["query"]["pages"][0]
        if page.get("missing") or "revisions" not in page:
            return None, None
        return page["title"], page["revisions"][0]["timestamp"]

    def leads(self, titles):
        """Plain-text lead section + page id + current revision id, 20 titles per call."""
        out = {}
        for i in range(0, len(titles), 20):
            d = self.get(action="query", prop="extracts|info", titles="|".join(titles[i:i + 20]),
                         exintro=1, explaintext=1, exlimit=20, redirects=1)
            for p in d["query"]["pages"]:
                if "extract" in p:
                    out[p["title"]] = {"pageid": p["pageid"], "revid": p.get("lastrevid"),
                                       "text": p["extract"]}
        return out


def clean(text):
    """Regex clean-up of the API's plain-text extract (the Week 8 'messy data' steps)."""
    text = re.sub(r"\[(\d+|citation needed|note \d+)\]", "", text)   # leftover reference markers
    text = re.sub(r"\(\s*[;,]?\s*\)", "", text)                       # empty brackets from stripped templates
    text = re.sub(r"\s+", " ", text)                                  # newlines / runs of spaces
    return re.sub(r"\s+([,.;:])", r"\1", text).strip()


def collect(wiki, years, keep, max_pages, target, rng, log):
    """Candidate titles from the event categories of `years`, filtered by creation date."""
    titles = []
    for y in years:
        for topic in EVENT_TOPICS:
            got = wiki.category_pages(f"Category:{y} {topic}", max_pages)
            titles += got
            log(f"  Category:{y} {topic}: {len(got)} pages")
    titles = [t for t in dict.fromkeys(titles) if not SKIP_TITLE.search(t)]
    rng.shuffle(titles)
    log(f"  {len(titles)} unique candidate titles after title filters; checking creation dates ...")

    kept, checked, rejected = [], 0, Counter()
    for t in titles:
        if len(kept) >= target:
            break
        title, ts = wiki.created(t)                  # follows redirects, so filter the RESOLVED title
        checked += 1
        if ts is None:
            rejected["missing"] += 1
        elif SKIP_TITLE.search(title) or title in {k for k, _ in kept}:
            rejected["redirect to a list page or a duplicate"] += 1
        elif keep(ts):
            kept.append((title, ts))
        else:
            rejected["created on the wrong side of the date"] += 1
        if checked % 100 == 0:
            log(f"    checked {checked}, kept {len(kept)}")
    return kept, checked, rejected


def dedupe(records, log):
    """Drop repeated pages (two titles redirecting to one page) and list pages reached via a redirect."""
    seen, out = set(), []
    for r in records:
        if r["pageid"] in seen or SKIP_TITLE.search(r["title"]):
            log(f"  dropped: {r['title']} (label {r['label']})")
            continue
        seen.add(r["pageid"])
        out.append(r)
    return out


def bucket(records, out, rng, stats, log):
    """Length buckets, balanced per bucket exactly as WikiMIA is (equal members and non-members)."""
    stats["buckets"] = {}
    for L in LENGTHS:
        pool = {lab: [r for r in records if r["label"] == lab and r["n_words"] >= L] for lab in (0, 1)}
        n = min(len(pool[0]), len(pool[1]))
        chosen = rng.sample(pool[0], n) + rng.sample(pool[1], n)
        rng.shuffle(chosen)
        with open(out / f"WikiMIA2026_length{L}.jsonl", "w", encoding="utf-8") as f:
            for r in chosen:
                f.write(json.dumps({"input": " ".join(r["text"].split()[:L]), "label": r["label"],
                                    "title": r["title"], "created": r["created"],
                                    "pageid": r["pageid"], "revid": r["revid"]}, ensure_ascii=False) + "\n")
        stats["buckets"][L] = {"n": 2 * n, "members": n, "non_members": n,
                               "available": {"non-members": len(pool[0]), "members": len(pool[1])}}
        log(f"length {L:>3}: {2 * n} passages ({n} + {n})")


def rebucket(cand_path, out, seed):
    """Re-make the length buckets from a saved candidates.jsonl, without any API calls."""
    log = print
    records = dedupe([json.loads(l) for l in open(cand_path, encoding="utf-8")], log)
    stats = json.load(open(out / "stats.json"))
    stats["rebucketed_from"] = str(cand_path).replace("\\", "/")
    stats["kept_with_text"] = dict(Counter("member" if r["label"] else "non-member" for r in records))
    bucket(records, out, random.Random(seed), stats, log)
    json.dump(stats, open(out / "stats.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="data/wikimia2026")
    ap.add_argument("--cutoff", default="2026-01-01", help="non-members: page created on/after this date")
    ap.add_argument("--member-before", default="2017-01-01", help="members: page created before this date")
    ap.add_argument("--member-years", default="2012-2015", help="event years the members are drawn from")
    ap.add_argument("--max-pages", type=int, default=120, help="cap on candidate pages per category")
    ap.add_argument("--target", type=int, default=600, help="pages to keep per side before length bucketing")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--from-candidates", help="rebuild the buckets from a saved raw/<date>/candidates.jsonl")
    args = ap.parse_args()

    out = Path(args.out)
    if args.from_candidates:
        return rebucket(Path(args.from_candidates), out, args.seed)
    today = dt.date.today().isoformat()
    raw = out / "raw" / today
    raw.mkdir(parents=True, exist_ok=True)
    logf = open(raw / "build.log", "w", encoding="utf-8")

    def log(msg):
        print(msg)
        logf.write(msg + "\n")
        logf.flush()

    rng = random.Random(args.seed)
    wiki = Wiki()
    cutoff_year = int(args.cutoff[:4])
    y0, y1 = map(int, args.member_years.split("-"))

    log(f"WikiMIA-2026 build, {dt.datetime.now().isoformat(timespec='seconds')}")
    log(f"non-members: event pages of {cutoff_year}, created >= {args.cutoff}")
    nonmem, nm_checked, nm_rej = collect(wiki, [cutoff_year], lambda ts: ts >= args.cutoff,
                                         args.max_pages * 3, args.target, rng, log)
    log(f"members: event pages of {y0}-{y1}, created < {args.member_before}")
    mem, m_checked, m_rej = collect(wiki, range(y0, y1 + 1), lambda ts: ts < args.member_before,
                                    args.max_pages, args.target, rng, log)

    log("fetching lead sections ...")
    records = []
    for label, side in [(0, nonmem), (1, mem)]:
        leads = wiki.leads([t for t, _ in side])
        for title, ts in side:
            if title in leads:
                text = clean(leads[title]["text"])
                records.append({"title": title, "label": label, "created": ts,
                                "pageid": leads[title]["pageid"], "revid": leads[title]["revid"],
                                "n_words": len(text.split()), "text": text})
    records = dedupe(records, log)
    with open(raw / "candidates.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Length buckets, balanced per bucket exactly as WikiMIA is (equal members and non-members).
    stats = {"built": today, "cutoff": args.cutoff, "member_before": args.member_before,
             "member_years": args.member_years, "seed": args.seed, "api_calls": wiki.calls,
             "creation_checked": {"non-members": nm_checked, "members": m_checked},
             "rejected": {"non-members": dict(nm_rej), "members": dict(m_rej)},
             "kept_with_text": dict(Counter("member" if r["label"] else "non-member" for r in records)),
             "buckets": {}}
    bucket(records, out, rng, stats, log)

    json.dump(stats, open(out / "stats.json", "w"), indent=1)
    log(json.dumps(stats, indent=1))
    log(f"done: {wiki.calls} API calls")


if __name__ == "__main__":
    sys.exit(main())
