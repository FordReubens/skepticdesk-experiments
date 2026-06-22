#!/usr/bin/env python3
"""M10: class-level syntax. Does Voynich predict the NEXT word-CLASS from the
current class, the way grammar does (DET->NOUN->VERB), even if the raw tokens
vary by section? Raw-token bigrams understate syntax if the "the" of Voynich is
a CLASS of tokens, not one token (GPT-5.5's point).

Method (identical for every corpus, no semantics, no dependencies):
  1. induce word classes distributionally: cluster the top-N tokens by their
     left+right neighbour context (k-means, cosine), rare tokens -> OTHER.
  2. measure predictive GAIN at the class level: H(class) - H(next|current),
     plus the proportional reduction. (GPT-5.5's normalization fix.)
  3. compare Voynich vs Apicius vs Caesar, each against a random-class null
     (same class sizes, coherence destroyed).

If Voynich's class-level gain climbs to Latin-like levels -> class syntax exists.
If it stays near its random-class null and well below Latin -> the word-machine
replaces grammar rather than encoding it.
"""
import re, math, random, pathlib
from collections import Counter, defaultdict
from voynich_lib import parse

random.seed(7)
ROOT = pathlib.Path(__file__).resolve().parent
K, N, M, ITERS = 12, 300, 40, 25

def H(counter):
    t = sum(counter.values())
    return -sum(c/t*math.log2(c/t) for c in counter.values()) if t else 0.0

def gain(seq):
    Hu = H(Counter(seq)); Hc = H(Counter(zip(seq, seq[1:]))) - Hu
    g = Hu - Hc
    return Hu, g, (g/Hu*100 if Hu else 0)

def context_vectors(toks, top, ctx):
    idx = {w: i for i, w in enumerate(ctx)}
    L = len(ctx)
    vecs = {w: [0.0]*(2*L) for w in top}
    tset = set(top)
    for i, w in enumerate(toks):
        if w in tset:
            if i > 0 and toks[i-1] in idx: vecs[w][idx[toks[i-1]]] += 1
            if i+1 < len(toks) and toks[i+1] in idx: vecs[w][L+idx[toks[i+1]]] += 1
    for w, v in vecs.items():
        n = math.sqrt(sum(x*x for x in v)) or 1.0
        vecs[w] = [x/n for x in v]
    return vecs

def kmeans(items, vecs, k):
    pts = [vecs[w] for w in items]; dim = len(pts[0])
    cent = [pts[i][:] for i in random.sample(range(len(pts)), k)]
    assign = [0]*len(pts)
    for _ in range(ITERS):
        for i, p in enumerate(pts):
            assign[i] = max(range(k), key=lambda c: sum(p[d]*cent[c][d] for d in range(dim)))
        new = [[0.0]*dim for _ in range(k)]; cnt = [0]*k
        for i, c in enumerate(assign):
            cnt[c] += 1
            for d in range(dim): new[c][d] += pts[i][d]
        for c in range(k):
            if cnt[c]:
                nrm = math.sqrt(sum(x*x for x in new[c])) or 1.0
                cent[c] = [x/nrm for x in new[c]]
            else:
                cent[c] = pts[random.randrange(len(pts))][:]
    return {items[i]: assign[i] for i in range(len(items))}

def analyze(name, toks):
    c = Counter(toks)
    top = [w for w, _ in c.most_common(N)]
    ctx = [w for w, _ in c.most_common(M)]
    vecs = context_vectors(toks, top, ctx)
    cls = kmeans(top, vecs, K)                      # token -> class 0..K-1; rare -> K (OTHER)
    seq = [cls.get(w, K) for w in toks]
    Hu, g, gp = gain(seq)
    # random-class null: permute the class labels among the clustered types
    perm_g = []
    for _ in range(5):
        labels = list(cls.values()); random.shuffle(labels)
        rmap = {w: labels[i] for i, w in enumerate(cls)}
        _, gg, _ = gain([rmap.get(w, K) for w in toks]); perm_g.append(gg)
    nullg = sum(perm_g)/len(perm_g)
    tHu, tg, tgp = gain(toks)                        # token-level for reference
    print(f"{name:<12}{tg:>9.2f}{tgp:>7.0f}%   {g:>10.2f}{gp:>7.0f}%   {nullg:>10.2f}")

def latin(path, strip_gut=False):
    t = (ROOT/path).read_text(encoding="utf-8", errors="ignore")
    if strip_gut:
        a = t.find("*** START"); b = t.find("*** END")
        if a >= 0: t = t[t.find("\n", a)+1:(b if b > 0 else len(t))]
    return re.findall(r"[a-z]+", t.lower())

print(f"K={K} classes, top-{N} clustered, {M} context words\n")
print(f"{'corpus':<12}{'tok-gain':>9}{'tok%':>8}   {'CLASS-gain':>10}{'cls%':>7}   {'rand-class':>11}")
print("-"*64)
analyze("Voynich", [r[6] for r in parse()])
analyze("Apicius", latin("data/apicius_books.txt"))
analyze("Caesar", latin("data/latin_dbg.txt", True))
print("\nCLASS-gain >> rand-class and ~ Latin => class-level syntax present.")
print("CLASS-gain near rand-class / << Latin => weak syntax even at the class level.")
