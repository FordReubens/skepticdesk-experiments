# SkepticDesk Experiments

Reproducible, code-backed analyses behind [SkepticDesk](https://skepticdesk.com)
investigations. Every claim in a SkepticDesk thread that rests on a measurement
has a script here you can re-run yourself.

Each investigation gets its own folder. First up:

## `voynich/` — the Voynich Manuscript

A quantitative investigation into what the Voynich text *is*. ~25 experiments,
pure-Python stdlib (no dependencies), each scoring the manuscript against real
controls (Latin prose, recipe Latin, Bengalese finch song).

**Run it:**
```bash
cd voynich
python fetch_data.py          # downloads the corpora from their original sources
python voynich_parse_stats.py # reproduce the baseline anomalies (h2 ≈ 2.15, etc.)
python voynich_recursive.py   # the recursive self-citation model
python voynich_reversible.py  # the reversible-cipher battery (roundtrip-verified)
python voynich_bacon_methods.py # Roger Bacon's concealment methods applied to Latin
# ...and the rest
```

**What it found, in one line:** Voynichese is not random gibberish and not
ordinary encrypted prose — it behaves like a *structured, self-citing signal*
("weird words breeding weird words"), reproducible by a single recursive
copy-and-mutate rule, and the features that define it are mutually exclusive
under any reversible cipher. Whether it carries a hidden payload is the last open
question, and it is no longer answerable from the statistics alone.

The full multi-party discussion (human + GPT-5.5 + Claude + a local DeepSeek
model) lives on the SkepticDesk thread; this repo is the math.

**Data is fetched, not redistributed** — see [`voynich/SOURCES.md`](voynich/SOURCES.md)
for every corpus and its attribution.

Code: MIT (see [LICENSE](LICENSE)). The data corpora keep their own terms.
