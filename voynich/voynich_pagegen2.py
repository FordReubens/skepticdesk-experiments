#!/usr/bin/env python3
"""M13b: refined page-seeded generator.

The crude slot generator left two gaps: glyph-entropy too high (free prefix+core+
suffix junctions aren't realistic) and vocabulary not reused tightly enough. Fix:
  - words come from an ORDER-2 GLYPH Markov model trained on real Voynich words,
    so letter-adjacency is realistic (targets h2/h1).
  - each page draws a page-specific lexicon (page-number seed) PLUS a shared global
    lexicon, mixed -> page-local vocabulary with a cross-page common layer.
  - repeat + one-glyph drift as before.
Score against real Voynich on the full scoreboard.
"""
import math, random
from collections import Counter, defaultdict
from voynich_lib import parse, lev1

rows = parse()
toks = [r[6] for r in rows]
fol = [r[0] for r in rows]
GC = Counter(ch for w in toks for ch in w); GLI = list(GC); GLW = [GC[g] for g in GLI]

# order-2 glyph Markov over real words
gm = defaultdict(Counter)
for w in toks:
    s = "^^" + w + "$"
    for i in range(len(s)-2): gm[(s[i], s[i+1])][s[i+2]] += 1

def wpick(items, weights, rng):
    r = rng.random()*sum(weights); a = 0.0
    for it, wt in zip(items, weights):
        a += wt
        if r <= a: return it
    return items[-1]

def gen_word(rng):
    s = ["^", "^"]
    for _ in range(12):
        nx = gm.get((s[-2], s[-1]))
        if not nx: break
        items = list(nx); ws = [nx[k] for k in items]
        c = wpick(items, ws, rng)
        if c == "$": break
        s.append(c)
    w = "".join(s[2:])
    return w or "o"

def lexicon(rng, n):
    words = [gen_word(rng) for _ in range(n)]
    wts = [1.0/(j+1)**0.5 for j in range(n)]        # flatter Zipf -> more types, less repeat
    return words, wts

def mutate(w, rng):
    g = wpick(GLI, GLW, rng); i = rng.randrange(len(w)); op = rng.random()
    if op < 0.34 and len(w) > 1: return w[:i]+w[i+1:]
    if op < 0.67: return w[:i]+g+w[i:]
    return w[:i]+g+w[i+1:]

order = []; cnt = Counter()
for f in fol:
    if f not in cnt: order.append(f)
    cnt[f] += 1

gl_rng = random.Random(1)
GLEX, GLEXW = lexicon(gl_rng, 300)                 # shared global lexicon

def generate():
    out = []; of = []
    for pi, f in enumerate(order):
        rng = random.Random(1000+pi)               # page-number seed
        L = max(20, round(cnt[f]*0.6))
        plex, plexw = lexicon(rng, L)
        prev = None
        for _ in range(cnt[f]):
            x = rng.random()
            if prev and x < 0.006: w = prev
            elif prev and x < 0.033: w = mutate(prev, rng)
            elif x < 0.30: w = wpick(GLEX, GLEXW, rng)    # shared layer
            else: w = wpick(plex, plexw, rng)             # page-local layer
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
scoreboard("Page-gen v3", gtk, gfl)
