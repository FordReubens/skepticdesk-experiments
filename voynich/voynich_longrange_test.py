#!/usr/bin/env python3
"""Milestone 2: controlled long-range structure test for Voynichese.

Metric: word CLUSTERING = frequency-weighted KL(occurrence-distribution || uniform)
across B equal position-bins. High = words cluster in regions (topical organization).

We compare the REAL corpus against progressively stronger nulls and two generators:
  - full shuffle        : destroys all order (keeps word frequencies)
  - within-section shuffle: keeps each position's SECTION, destroys finer order
  - within-folio shuffle  : keeps each position's FOLIO, destroys finer order
  - global word-Markov    : order-2 Markov over whole text (local structure only)
  - per-section Markov    : order-1 Markov trained/generated PER SECTION
                            (reproduces section vocabulary by construction)

Reading: if real clustering >> within-section null, there is topical structure FINER
than sections. If real ~ per-section generator, "topical structure" reduces to
"sections use different words" (a sectioned generator suffices). z-scores vs the
Monte-Carlo null distributions quantify each.
"""
import math, random, statistics, sys
from collections import Counter, defaultdict
from voynich_lib import parse

random.seed(7)
B, MINC, M = 40, 10, 150     # bins, min count to be "eligible", Monte-Carlo shuffles
rows = parse()
toks = [r[6] for r in rows]
N = len(toks)
sec = [r[5] for r in rows]
fol = [r[0] for r in rows]
binof = [i * B // N for i in range(N)]   # fixed position-bin per index

counts = Counter(toks)
eligible = {w for w, c in counts.items() if c >= MINC}
nw = {w: counts[w] for w in eligible}
totw = sum(nw.values())
print(f"corpus: {N} tokens, {len(counts)} types; eligible (>={MINC}): {len(eligible)} types "
      f"covering {totw} tokens ({totw/N*100:.0f}%)\n")

def clustering(seq):
    """freq-weighted mean KL(occurrence dist || uniform) over eligible words."""
    hist = defaultdict(lambda: [0]*B)
    for i, w in enumerate(seq):
        if w in eligible: hist[w][binof[i]] += 1
    acc = 0.0
    for w, h in hist.items():
        n = nw[w]; kl = 0.0
        for b in h:
            if b:
                p = b / n
                kl += p * math.log2(p * B)      # vs uniform 1/B
        acc += n * kl
    return acc / totw

C_real = clustering(toks)

def shuffled(kind):
    s = toks[:]
    if kind == "full":
        random.shuffle(s)
    else:
        key = sec if kind == "section" else fol
        groups = defaultdict(list)
        for i, k in enumerate(key): groups[k].append(i)
        for idxs in groups.values():
            vals = [s[i] for i in idxs]; random.shuffle(vals)
            for i, v in zip(idxs, vals): s[i] = v
    return s

def null_dist(kind):
    vals = [clustering(shuffled(kind)) for _ in range(M)]
    return statistics.mean(vals), statistics.pstdev(vals)

def markov_global(order=2):
    nxt = defaultdict(list)
    for i in range(len(toks)-order):
        nxt[tuple(toks[i:i+order])].append(toks[i+order])
    starts = [tuple(toks[i:i+order]) for i in range(len(toks)-order)]
    out = list(random.choice(starts))
    while len(out) < N:
        k = tuple(out[-order:]); nx = nxt.get(k)
        out.append(random.choice(nx) if nx else random.choice(toks))
    return out[:N]

def markov_per_section():
    # contiguous section blocks in reading order; order-1 Markov within each block
    out = [None]*N
    blocks = []   # (indices_in_order) grouped by contiguous section runs
    start = 0
    for i in range(1, N+1):
        if i == N or sec[i] != sec[start]:
            blocks.append(list(range(start, i))); start = i
    for idxs in blocks:
        bt = [toks[i] for i in idxs]
        nxt = defaultdict(list)
        for a, b in zip(bt, bt[1:]): nxt[a].append(b)
        gen = [random.choice(bt)]
        while len(gen) < len(bt):
            nx = nxt.get(gen[-1]); gen.append(random.choice(nx) if nx else random.choice(bt))
        for i, v in zip(idxs, gen): out[i] = v
    return out

def z(x, mu, sd): return (x - mu) / sd if sd else float("nan")

print(f"REAL clustering C = {C_real:.4f}\n")
print(f"{'comparison':<26}{'C':>9}{'z vs real':>12}")
for kind in ("full", "section", "folio"):
    mu, sd = null_dist(kind)
    print(f"{'null: '+kind+' shuffle':<26}{mu:>9.4f}{z(C_real, mu, sd):>12.1f}")
# generators: average a few samples
for name, fn in (("gen: global word-Markov", markov_global), ("gen: per-section Markov", markov_per_section)):
    samp = [clustering(fn()) for _ in range(8)]
    mu, sd = statistics.mean(samp), statistics.pstdev(samp)
    print(f"{name:<26}{mu:>9.4f}{z(C_real, mu, sd):>12.1f}")
