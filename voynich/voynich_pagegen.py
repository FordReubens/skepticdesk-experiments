#!/usr/bin/env python3
"""M13 (Ford's generator test): can a per-page-seeded procedural generator
reproduce Voynich's statistics?

Generator: slot-grammar words (prefix + core + suffix from inventories learned
from Voynich) + a PER-PAGE seed that reweights which cores/affixes are common on
that page (the "salty RNG per page" -> page-local vocabulary) + a repeat / one-
glyph-drift mechanism. Then score generated text vs real Voynich on every
signature we've measured. Matches => procedural generation is SUFFICIENT to
explain that signature (does not prove it, but removes it as evidence of meaning).
"""
import math, random
from collections import Counter, defaultdict
from voynich_lib import parse, lev1

rows = parse()
toks = [r[6] for r in rows]
fol = [r[0] for r in rows]
GLYPHS = sorted({ch for w in toks for ch in w})

def top_affixes(end, lengths, k):
    ch = set()
    for L in lengths:
        c = Counter(t[-L:] if end else t[:L] for t in toks if len(t) > L)
        for a, _ in c.most_common(k): ch.add(a)
    return ch
SUF = sorted(top_affixes(True, (4, 3, 2, 1), 4), key=len, reverse=True)
PRE = sorted(top_affixes(False, (3, 2, 1), 4), key=len, reverse=True)

def decomp(w):
    p = ""
    for a in PRE:
        if w.startswith(a) and len(w)-len(a) >= 2: p = a; w = w[len(a):]; break
    s = ""
    for a in SUF:
        if w.endswith(a) and len(w)-len(a) >= 1: s = a; w = w[:-len(a)]; break
    return p, w, s

cores = Counter(); pres = Counter(); sufs = Counter()
for w in toks:
    p, c, s = decomp(w); cores[c] += 1; pres[p] += 1; sufs[s] += 1
CI = [c for c, _ in cores.most_common(350)]; CW = [cores[c] for c in CI]   # cap -> more reuse
PI = list(pres); PW = [pres[p] for p in PI]
SI = list(sufs); SW = [sufs[s] for s in SI]
GC = Counter(ch for w in toks for ch in w); GLI = list(GC); GLW = [GC[g] for g in GLI]

order = []; cnt = Counter()
for f in fol:
    if f not in cnt: order.append(f)
    cnt[f] += 1

def wpick(items, weights, rng):
    r = rng.random()*sum(weights); a = 0.0
    for it, wt in zip(items, weights):
        a += wt
        if r <= a: return it
    return items[-1]

def mutate(w, rng):
    if not w: return "o"
    g = wpick(GLI, GLW, rng)                     # mutate toward common glyphs (lower entropy)
    i = rng.randrange(len(w)); op = rng.random()
    if op < 0.34 and len(w) > 1: return w[:i]+w[i+1:]
    if op < 0.67: return w[:i]+g+w[i:]
    return w[:i]+g+w[i+1:]

def generate():
    out = []; of = []
    for pi, f in enumerate(order):
        rng = random.Random(1000+pi)                 # salty per-page seed (page number)
        cw = [wt*rng.gammavariate(0.6, 1) for wt in CW]   # page favors some cores (gentler)
        pw = [wt*rng.gammavariate(0.8, 1) for wt in PW]
        sw = [wt*rng.gammavariate(0.8, 1) for wt in SW]
        prev = None
        for _ in range(cnt[f]):
            x = rng.random()
            if prev and x < 0.008: w = prev
            elif prev and x < 0.035: w = mutate(prev, rng)
            else:
                w = wpick(PI, pw, rng) + wpick(CI, cw, rng) + wpick(SI, sw, rng)
                if not w: w = wpick(CI, cw, rng) or "o"
            out.append(w); of.append(f); prev = w
    return out, of

def scoreboard(name, tk, fl):
    N = len(tk); types = len(set(tk)); mwl = sum(len(w) for w in tk)/N
    pairs = same = one = 0; prev = pf = None
    for w, f in zip(tk, fl):
        if prev is not None and f == pf:
            pairs += 1
            if w == prev: same += 1
            elif lev1(w, prev): one += 1
        prev, pf = w, f
    uni = Counter(); big = Counter()
    for w in tk:
        for ch in w: uni[ch] += 1
        for a, b in zip(w, w[1:]): big[(a, b)] += 1
    tu = sum(uni.values()); tb = sum(big.values())
    h1 = -sum(c/tu*math.log2(c/tu) for c in uni.values())
    h2 = (-sum(c/tb*math.log2(c/tb) for c in big.values())) - h1
    pagesz = Counter(fl); Nt = len(fl); pf2 = {f: pagesz[f]/Nt for f in pagesz}
    wp = defaultdict(Counter); c = Counter(tk)
    for w, f in zip(tk, fl): wp[w][f] += 1
    num = den = 0.0
    for w in c:
        if c[w] >= 10:
            n = c[w]; kl = sum((cf/n)*math.log2((cf/n)/pf2[f]) for f, cf in wp[w].items())
            num += n*kl; den += n
    print(f"{name:<16}{N:>7}{types:>7}{mwl:>6.2f}{same/pairs*100:>7.2f}%{one/pairs*100:>7.2f}%{h2/h1:>7.2f}{num/den:>8.2f}")

gtk, gfl = generate()
print(f"{'system':<16}{'toks':>7}{'types':>7}{'wlen':>6}{'repeat':>8}{'drift':>7}{'h2/h1':>7}{'page-loc':>8}")
print("-"*72)
scoreboard("Real Voynich", toks, fol)
scoreboard("Page-gen", gtk, gfl)
print("\nrepeat=adjacent same word; drift=adjacent one-glyph variant; h2/h1=glyph redundancy;")
print("page-loc=how page-concentrated frequent words are (the per-page-seed target).")
