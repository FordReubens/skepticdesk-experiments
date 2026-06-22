#!/usr/bin/env python3
"""M6: collapse near-neighbour word variants to stems, then re-test.

Voynich words drift by one glyph (qokedy / qokeey / qokeedy). If that surface
churn is inflection/spelling around an underlying stem, normalising it may expose
a cleaner grammatical layer. We build a frequency-anchored stemmer: process word
types most-frequent-first; each word within Levenshtein 1 of an existing, more
frequent 'hub' is merged into it (no transitive chaining). Then we re-measure:
  - vocabulary reduction
  - adjacent same-STEM rate (the one-edit drift should fold into repetition)
  - the function-word test on stems, within a section / Currier A
to see whether the high-frequency layer flattens toward a real-language profile.
"""
import math, random
from collections import Counter, defaultdict
from voynich_lib import parse, lev1

random.seed(7)
rows = parse()
toks = [r[6] for r in rows]
counts = Counter(toks)

# ---- frequency-anchored stemming (merge Lev<=1 into the more frequent hub) ----
reps_by_len = defaultdict(list)   # length -> [hub words, freq-desc]
stem = {}
for w in sorted(counts, key=lambda x: -counts[x]):
    hub = None
    for L in (len(w), len(w)-1, len(w)+1):
        for r in reps_by_len.get(L, ()):
            if lev1(w, r): hub = r; break
        if hub: break
    if hub:
        stem[w] = hub
    else:
        stem[w] = w; reps_by_len[len(w)].append(w)

stems = [stem[t] for t in toks]
print(f"vocabulary: {len(counts)} word-types -> {len(set(stems))} stems "
      f"({(1-len(set(stems))/len(counts))*100:.0f}% reduction)\n")

# adjacent same-word vs same-stem (within line)
def adj_same(seq):
    same = pairs = 0; prev = pk = None
    for (folio, line, *_ ), s in zip(rows, seq):
        k = (folio, line)
        if prev is not None and k == pk:
            pairs += 1; same += (s == prev)
        prev, pk = s, k
    return same / pairs * 100
print(f"adjacent same-WORD: {adj_same(toks):.2f}%   adjacent same-STEM: {adj_same(stems):.2f}%"
      "   (one-edit drift folds into repetition)\n")

B, MINC, M = 24, 6, 120
def profile(label, seq):
    N = len(seq)
    if N < 1500: print(f"{label:<30} {N} toks — skipped"); return
    binof = [i*B//N for i in range(N)]
    c = Counter(seq); elig = [w for w, n in c.items() if n >= MINC]; nw = {w: c[w] for w in elig}
    def kls(s):
        h = defaultdict(lambda: [0]*B)
        for i, w in enumerate(s):
            if w in nw: h[w][binof[i]] += 1
        o = {}
        for w, hh in h.items():
            n = nw[w]; kl = 0.0
            for x in hh:
                if x: p = x/n; kl += p*math.log2(p*B)
            o[w] = kl
        return o
    real = kls(seq); s = seq[:]; acc = defaultdict(float); acc2 = defaultdict(float)
    for _ in range(M):
        random.shuffle(s)
        for w, v in kls(s).items(): acc[w] += v; acc2[w] += v*v
    z = {}
    for w in elig:
        mu = acc[w]/M; sd = math.sqrt(max(acc2[w]/M-mu*mu, 1e-9)); z[w] = (real[w]-mu)/sd
    rk = sorted(elig, key=lambda w: -c[w])
    mz = sum(z[w] for w in rk[:10])/min(10, len(rk)); fw = sum(1 for w in rk[:20] if z[w] < 5)
    print(f"{label:<30}{N:>7}{len(elig):>6}{mz:>9.1f}{fw:>7}/20   " + " ".join(rk[:5]))

secH = [i for i, r in enumerate(rows) if r[5] == "H"]
curA = [i for i, r in enumerate(rows) if r[3] == "A"]
print(f"{'function-word test':<30}{'toks':>7}{'elig':>6}{'top10z':>9}{'func20':>10}   top")
print("-"*86)
profile("Section H  WORDS", [toks[i] for i in secH])
profile("Section H  STEMS", [stems[i] for i in secH])
profile("Currier A  WORDS", [toks[i] for i in curA])
profile("Currier A  STEMS", [stems[i] for i in curA])
print("\nLatin reference: top10 z ~1.7-2.2, func-like ~15-17/20.")
