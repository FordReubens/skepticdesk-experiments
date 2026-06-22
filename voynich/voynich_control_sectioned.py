#!/usr/bin/env python3
"""M8a: calibrate the cross-section result against a REAL sectioned text.

Run the identical cross-section-uniformity test on Apicius sliced into its 5
(topical) books, vs Voynich sliced into its 8 sections. The decisive contrast:
  - %% of frequent tokens that are cross-section uniform, and
  - how many of the TOP-10 tokens are uniform (the natural-language function-word
    signature: in real prose the commonest words ARE the topic-independent glue).
If real Latin's top words are uniform but Voynich's are not, the difference is the
point. If both look the same, Voynich's shared layer is unremarkable.
"""
import re, math, random, pathlib
from collections import Counter, defaultdict
from voynich_lib import parse

random.seed(7)
M, T = 80, 15
ROOT = pathlib.Path(__file__).resolve().parent

def measure(label, toks, secof, nsec):
    N = len(toks)
    secsize = [secof.count(i) for i in range(nsec)]
    secfrac = [s/N for s in secsize]
    c = Counter(toks); elig = [w for w, n in c.items() if n >= T]; nw = {w: c[w] for w in elig}
    def klmap(order):
        h = defaultdict(lambda: [0]*nsec)
        for i, w in enumerate(toks):
            if w in nw: h[w][order[i]] += 1
        o = {}
        for w, hh in h.items():
            n = nw[w]; kl = 0.0
            for si, x in enumerate(hh):
                if x: p = x/n; kl += p*math.log2(p/secfrac[si])
            o[w] = kl
        return o
    real = klmap(secof)
    sh = secof[:]; acc = defaultdict(float); acc2 = defaultdict(float)
    for _ in range(M):
        random.shuffle(sh)
        for w, v in klmap(sh).items(): acc[w] += v; acc2[w] += v*v
    z = {}
    for w in elig:
        mu = acc[w]/M; sd = math.sqrt(max(acc2[w]/M-mu*mu, 1e-9)); z[w] = (real[w]-mu)/sd
    rk = sorted(elig, key=lambda w: -c[w])
    uni = [w for w in elig if z[w] < 2]
    top10_uni = sum(1 for w in rk[:10] if z[w] < 2)
    pct = len(uni)/len(elig)*100 if elig else 0
    print(f"{label:<26}{nsec:>5}{N:>8}{len(elig):>7}{len(uni):>8}{pct:>7.0f}%{top10_uni:>9}/10   "
          + " ".join(f"{w}(z{z[w]:.0f})" for w in rk[:6]))

# Apicius by book
ap = (ROOT/"data/apicius_books.txt").read_text(encoding="utf-8", errors="ignore")
ap_toks, ap_sec, bi = [], [], -1
for line in ap.splitlines():
    if line.startswith("### BOOK"): bi += 1; continue
    for w in re.findall(r"[a-z]+", line.lower()): ap_toks.append(w); ap_sec.append(bi)

# Voynich by section
rows = parse()
secs = sorted({r[5] for r in rows}); sidx = {s: i for i, s in enumerate(secs)}
v_toks = [r[6] for r in rows]; v_sec = [sidx[r[5]] for r in rows]

print(f"{'corpus':<26}{'nsec':>5}{'toks':>8}{'elig':>7}{'unif':>8}{'unif%':>8}{'top10unif':>12}   top words (z)")
print("-"*108)
measure("Apicius (real, by book)", ap_toks, ap_sec, bi+1)
measure("Voynich (by section)", v_toks, v_sec, len(secs))
print("\ntop10unif = how many of the 10 commonest tokens are topic-independent (function-word signature).")
