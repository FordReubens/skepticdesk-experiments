#!/usr/bin/env python3
"""M7 (climax test): is there a TOPIC-INDEPENDENT layer?

Every test so far was within a single section. A true grammatical / function-word
layer is topic-independent: it recurs evenly across ALL sections (a herbal's "et",
a recipe's "et", an astronomy page's "et"). We test whether ANY normalization
exposes frequent tokens/cores that are spread evenly across the 8 Voynich sections.

Normalizations: raw | Lev-1 stem | prefix-stripped | suffix-stripped | core-only
(affixes derived empirically from the data, Stolfi prefix-core-suffix style).

Cross-section metric: for each token, KL of its section distribution from the
section-SIZE distribution (proportional = topic-independent). z-score vs shuffles.
A 'function-like-across-sections' token = frequent (>=30) and z < 2 (no more
section-concentrated than chance). Count them per normalization.

If a normalization yields a real set of cross-section-uniform tokens -> grammar /
cipher layer gains. If every normalization stays section-bound -> the structure is
the generator's grammar, not the language's.
"""
import math, random
from collections import Counter, defaultdict
from voynich_lib import parse, lev1

random.seed(7)
M = 80
rows = parse()
toks = [r[6] for r in rows]
secs = sorted({r[5] for r in rows})
sidx = {s: i for i, s in enumerate(secs)}
secof = [sidx[r[5]] for r in rows]
N = len(toks)
secsize = [secof.count(i) for i in range(len(secs))]
secfrac = [s / N for s in secsize]
print(f"{N} tokens; sections {secs} sizes {secsize}\n")

# ---- affix inventory (empirical, token-weighted) ----
def top_affixes(end, lengths, k):
    chosen = set()
    for L in lengths:
        c = Counter(t[-L:] if end else t[:L] for t in toks if len(t) > L)
        for a, _ in c.most_common(k): chosen.add(a)
    return chosen
SUF = top_affixes(True, (4, 3, 2, 1), 4)
PRE = top_affixes(False, (3, 2, 1), 4)
print("suffixes:", sorted(SUF, key=len, reverse=True))
print("prefixes:", sorted(PRE, key=len, reverse=True), "\n")

def strip_suf(w):
    for a in sorted(SUF, key=len, reverse=True):
        if w.endswith(a) and len(w) - len(a) >= 2: return w[:-len(a)]
    return w
def strip_pre(w):
    for a in sorted(PRE, key=len, reverse=True):
        if w.startswith(a) and len(w) - len(a) >= 2: return w[len(a):]
    return w

# Lev-1 frequency-anchored stems
counts = Counter(toks); reps = defaultdict(list); stem = {}
for w in sorted(counts, key=lambda x: -counts[x]):
    hub = None
    for L in (len(w), len(w)-1, len(w)+1):
        for r in reps.get(L, ()):
            if lev1(w, r): hub = r; break
        if hub: break
    stem[w] = hub or w
    if not hub: reps[len(w)].append(w)

NORM = {
    "raw":      lambda w: w,
    "lev-stem": lambda w: stem[w],
    "prefix-strip": strip_pre,
    "suffix-strip": strip_suf,
    "core-only": lambda w: strip_suf(strip_pre(w)),
}

def crosssec_uniform(seq):
    c = Counter(seq)
    elig = [w for w, n in c.items() if n >= 30]
    nw = {w: c[w] for w in elig}
    def klmap(s):
        h = defaultdict(lambda: [0]*len(secs))
        for i, w in enumerate(s):
            if w in nw: h[w][secof[i]] += 1
        o = {}
        for w, hh in h.items():
            n = nw[w]; kl = 0.0
            for si, x in enumerate(hh):
                if x: p = x/n; kl += p*math.log2(p/secfrac[si])
            o[w] = kl
        return o
    real = klmap(seq); s = seq[:]; acc = defaultdict(float); acc2 = defaultdict(float)
    for _ in range(M):
        random.shuffle(s)
        for w, v in klmap(s).items(): acc[w] += v; acc2[w] += v*v
    z = {}
    for w in elig:
        mu = acc[w]/M; sd = math.sqrt(max(acc2[w]/M-mu*mu, 1e-9)); z[w] = (real[w]-mu)/sd
    uni = sorted([w for w in elig if z[w] < 2], key=lambda w: -c[w])
    return len(elig), uni, c, z, real

print(f"{'normalization':<15}{'vocab':>7}{'freq>=30':>9}{'cross-sec uniform':>18}   examples (count)")
print("-"*92)
for name, fn in NORM.items():
    seq = [fn(t) for t in toks]
    nelig, uni, c, z, real = crosssec_uniform(seq)
    ex = "  ".join(f"{w}({c[w]})" for w in uni[:5]) if uni else "(none)"
    print(f"{name:<15}{len(set(seq)):>7}{nelig:>9}{len(uni):>18}   {ex}")
print("\nmany cross-section-uniform frequent tokens => topic-independent (grammar/cipher) layer.")
print("near-zero across every normalization => structure is section-bound (generator-grammar).")
