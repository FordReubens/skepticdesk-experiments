#!/usr/bin/env python3
"""M8b: the two-layer generator baseline.

GPT-5.5's stronger null: not a dumb per-section generator, but one with a GLOBAL
core inventory + SECTION-SPECIFIC affixes. We decompose Voynich words into
prefix-core-suffix, pool the cores globally, keep prefix/suffix distributions per
section, then generate: core ~ global, prefix & suffix ~ section. Run the same
cross-section test on the output.

If this reproduces Voynich's signature (uniform layer present but mid-frequency,
top words section-bound), then that signature is achievable by a rule-mediated
generator and does not by itself imply meaning. If it can't, the meaning/cipher
side gains weight.
"""
import math, random, pathlib
from collections import Counter, defaultdict
from voynich_lib import parse

random.seed(7)
M, T = 80, 15
rows = parse()
toks = [r[6] for r in rows]
secs = sorted({r[5] for r in rows}); sidx = {s: i for i, s in enumerate(secs)}
secof = [sidx[r[5]] for r in rows]
nsec = len(secs)

# affix inventory (same construction as M7)
def top_affixes(end, lengths, k):
    chosen = set()
    for L in lengths:
        c = Counter(t[-L:] if end else t[:L] for t in toks if len(t) > L)
        for a, _ in c.most_common(k): chosen.add(a)
    return chosen
SUF = sorted(top_affixes(True, (4, 3, 2, 1), 4), key=len, reverse=True)
PRE = sorted(top_affixes(False, (3, 2, 1), 4), key=len, reverse=True)

def decompose(w):
    pre = ""
    for a in PRE:
        if w.startswith(a) and len(w)-len(a) >= 2: pre = a; w = w[len(a):]; break
    suf = ""
    for a in SUF:
        if w.endswith(a) and len(w)-len(a) >= 1: suf = a; w = w[:-len(a)]; break
    return pre, w, suf   # prefix, core, suffix

cores = []                                   # global core pool
pre_by_sec = [Counter() for _ in range(nsec)]
suf_by_sec = [Counter() for _ in range(nsec)]
for i, w in enumerate(toks):
    p, c, s = decompose(w)
    cores.append(c); pre_by_sec[secof[i]][p] += 1; suf_by_sec[secof[i]][s] += 1
core_pool = cores[:]                          # sample uniformly-by-frequency

def wpick(counter):
    tot = sum(counter.values()); r = random.uniform(0, tot); a = 0
    for k, v in counter.items():
        a += v
        if r <= a: return k
    return ""

gen = []
for i in range(len(toks)):
    si = secof[i]
    p = wpick(pre_by_sec[si]); c = random.choice(core_pool); s = wpick(suf_by_sec[si])
    gen.append(p + c + s)

def measure(label, seq, order):
    N = len(seq); secsize = [order.count(i) for i in range(nsec)]; secfrac = [x/N for x in secsize]
    c = Counter(seq); elig = [w for w, n in c.items() if n >= T]; nw = {w: c[w] for w in elig}
    def klmap(o2):
        h = defaultdict(lambda: [0]*nsec)
        for i, w in enumerate(seq):
            if w in nw: h[w][o2[i]] += 1
        out = {}
        for w, hh in h.items():
            n = nw[w]; kl = 0.0
            for si, x in enumerate(hh):
                if x: pp = x/n; kl += pp*math.log2(pp/secfrac[si])
            out[w] = kl
        return out
    real = klmap(order); sh = order[:]; acc = defaultdict(float); acc2 = defaultdict(float)
    for _ in range(M):
        random.shuffle(sh)
        for w, v in klmap(sh).items(): acc[w] += v; acc2[w] += v*v
    z = {}
    for w in elig:
        mu = acc[w]/M; sd = math.sqrt(max(acc2[w]/M-mu*mu, 1e-9)); z[w] = (real[w]-mu)/sd
    rk = sorted(elig, key=lambda w: -c[w]); uni = [w for w in elig if z[w] < 2]
    t10 = sum(1 for w in rk[:10] if z[w] < 2)
    print(f"{label:<30}{len(elig):>6}{len(uni)/len(elig)*100:>7.0f}%{t10:>9}/10   "
          + " ".join(f"{w}(z{z[w]:.0f})" for w in rk[:6]))

print(f"affixes: PRE={PRE}\n         SUF={SUF}\n")
print(f"{'corpus':<30}{'elig':>6}{'unif%':>8}{'top10unif':>12}   top words")
print("-"*100)
measure("Voynich REAL (by section)", toks, secof)
measure("Two-layer GENERATOR", gen, secof)
print("\nIf the generator matches Voynich (uniform% similar, top10unif ~0), the signature is")
print("reproducible by a rule-mediated machine -> not by itself evidence of meaning.")
