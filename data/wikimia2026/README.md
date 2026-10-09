# WikiMIA-2026

A re-run of the WikiMIA construction recipe (Shi et al., ICLR 2024, Section 2.2) with a **1 January 2026**
cut-off, built for this replication project.

| | Members (label 1) | Non-members (label 0) |
|---|---|---|
| Source | Wikipedia event articles from 2012–2015 event categories | Wikipedia event articles from 2026 event categories |
| Membership rule | page **created before 1 Jan 2017** (paper's Step 2) | page **created on/after 1 Jan 2026** |
| Why the label holds | old pages are in the Wikipedia dumps used to pretrain Pythia / OPT / LLaMA | created after every target model's training cut-off |

Creation date = timestamp of the page's first revision (MediaWiki API, `rvdir=newer`).

## Files

| File | Contents |
|---|---|
| `WikiMIA2026_length{32,64,128,256}.jsonl` | balanced passages, first *L* words of the article's lead section |
| `stats.json` | counts, rejections, API calls, parameters of the build |
| `raw/<yyyy-mm-dd>/candidates.jsonl` | every kept page before length bucketing, with full lead text |
| `raw/<yyyy-mm-dd>/build.log` | the build's log (categories crawled, pages checked/kept) |

Each JSONL line: `input` (the passage), `label`, `title`, `created` (first-revision timestamp),
`pageid`, `revid` (the revision the text was taken from). `input` + `label` is the schema the original
`detect-pretrain-code` reads, so `run.py --data <file>.jsonl` works unmodified.

## How to re-acquire exactly

```bash
python scripts/build_wikimia2026.py --out data/wikimia2026 --max-pages 120 --target 600 --seed 0
```

Wikipedia changes daily, so a re-run on a later date gives different pages. To recover the **exact** text
used here, fetch each passage's stored `revid`:
`https://en.wikipedia.org/w/index.php?oldid=<revid>`.

## Differences from the original WikiMIA (deliberate, documented)

1. **Matched topics.** Both sides are drawn from the same 16 event-category types (`<year> disasters`,
   `<year> in sports`, `<year> in politics`, ...). WikiMIA does not describe how its member events were
   chosen beyond "pre-2017 event pages".
2. **Text = the lead section** (`prop=extracts&exintro`), cleaned with regular expressions (reference markers,
   empty brackets, whitespace).
3. **Redirects.** Creation dates are read after following redirects, and the title filters are applied
   again to the resolved title; pages reached twice through different redirects are kept once. (The
   first build applied the filters before resolution; the buckets were re-made from the saved
   `candidates.jsonl` with `--from-candidates`, which is recorded in `stats.json`.)
4. Member text is the **current** revision, as in WikiMIA. It may have been edited since the model saw it;
   this is a known weakness of date-defined membership, and one reason for the shift-controlled ND-2 split.

## Known label noise

A page *created* after the cut-off can still carry *old* text when it was split off or copied from an older
article (e.g. "Legault ministry", created January 2026 but describing 2018–2026). Such non-members may
in fact overlap with training data. Detecting splits (edit summaries such as "split from ...") is future work.

## Known confound (measured, not fixed)

Non-member passages mention "2026" and use future/present tense ("will be held"); member passages use the
past tense. `scripts/blind_baseline.py` measures how far a text-only classifier can exploit this: see
`results/blind_baseline/`.

## Licence

Text is from English Wikipedia, CC BY-SA 4.0; attribution is each article's page history (page id and
revision id are kept for every passage). The build script is MIT.
