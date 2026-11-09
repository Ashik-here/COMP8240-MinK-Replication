# WikiMIA-2026-matched: a shift-controlled subset (ND-2)

Derived from WikiMIA-2026 by `scripts/shift_controlled.py`; same JSONL schema as WikiMIA and WikiMIA-2026.

**Why.** On WikiMIA-2026 a blind TF-IDF classifier that never sees the language model separates members
(pages created before 2017) from non-members (created in 2026) with AUC 0.95, so an attack's AUC on the full
set partly measures the time shift. This subset keeps only member/non-member pairs the blind classifier
cannot tell apart.

**How.** Each passage gets the blind classifier's out-of-fold probability of being a member
(`results/blind_baseline/scores/WikiMIA-2026_len{L}.jsonl`). Members are matched 1:1 to non-members by nearest
neighbour on the logit of that probability (caliper 0.2 SD, without replacement, seed 0). Only matched
passages are kept, so the classes stay balanced.

| length | passages | blind AUC on the subset |
|---|---|---|
| 32 | 194 | 0.54 |
| 64 | 126 | 0.55 |
| 128 | 70 | 0.53 |
| 256 | 26 | 0.50 |

**Limits.** The subset is small, and it only controls for the cues a bag-of-words model can learn. Novelty a
TF-IDF model cannot represent, such as a name never seen before, is still present. MIMIR (same source, same
time period) is the stronger control. The pages and licence are those of WikiMIA-2026 (Wikipedia, CC BY-SA 4.0).
