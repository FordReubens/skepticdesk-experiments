#!/usr/bin/env python3
"""M5b: rerun the function-word/burstiness test WITHIN single strata, to kill the
Currier A/B and section-vocabulary caveat. If Voynich still shows no evenly-spread
high-frequency (function-word) layer even within one Currier language or one
section, the A/B split was not what produced the M5 result."""
import math, random
from collections import Counter, defaultdict
from voynich_lib import parse

random.seed(7)
rows = parse()

def funcword_profile(label, toks, B, MINC, M=120):
    N = len(toks)
    if N < 2000:
        print(f"\n=== {label}: only {N} tokens — skipped ==="); return
    binof = [i * B // N for i in range(N)]
    counts = Counter(toks)
    elig = [w for w, c in counts.items() if c >= MINC]
    nw = {w: counts[w] for w in elig}
    def kls(seq):
        hist = defaultdict(lambda: [0]*B)
        for i, w in enumerate(seq):
            if w in nw: hist[w][binof[i]] += 1
        out = {}
        for w, h in hist.items():
            n = nw[w]; kl = 0.0
            for x in h:
                if x: p = x/n; kl += p*math.log2(p*B)
            out[w] = kl
        return out
    real = kls(toks); s = toks[:]; acc = defaultdict(float); acc2 = defaultdict(float)
    for _ in range(M):
        random.shuffle(s)
        for w, v in kls(s).items(): acc[w] += v; acc2[w] += v*v
    z = {}
    for w in elig:
        mu = acc[w]/M; sd = math.sqrt(max(acc2[w]/M - mu*mu, 1e-9)); z[w] = (real[w]-mu)/sd
    ranked = sorted(elig, key=lambda w: -counts[w])
    top10 = ranked[:10]
    mz10 = sum(z[w] for w in top10)/len(top10)
    fw = sum(1 for w in ranked[:20] if z[w] < 5)
    print(f"\n=== {label}: {N} tokens, {len(elig)} eligible (>= {MINC}) ===")
    print(f"  top-10 mean z = {mz10:.1f}   function-like top-20 (z<5) = {fw}/20")
    print("  top words: " + "  ".join(f"{w}(z{z[w]:.0f})" for w in top10))

A = [r[6] for r in rows if r[3] == "A"]
Bt = [r[6] for r in rows if r[3] == "B"]
funcword_profile("Currier A only", A, 40, 10)
funcword_profile("Currier B only", Bt, 40, 10)
for sec in ("H", "B", "S", "P"):   # Herbal, Biological, Stars/recipes, Pharma
    toks = [r[6] for r in rows if r[5] == sec]
    funcword_profile(f"Section {sec} only", toks, 24, 6)
print("\nReminder — Latin (whole): top-10 mean z = 1.7, function-like 19/20.")
print("Voynich (whole): top-10 mean z = 26.2, function-like 1/20.")
