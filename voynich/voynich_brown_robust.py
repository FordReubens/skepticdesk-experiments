#!/usr/bin/env python3
"""M10c: robustness of the class-level syntax signal (GPT-5.5's checks 1-3).

A. SHAPE vs ROLE (the decisive one). Compare the distributional (Brown) class
   signal to a SHAPE-based class signal (cluster tokens purely by spelling). If
   Brown ~ shape, the "syntax" is morphology in disguise. If Brown >> shape,
   there is above-word structure beyond spelling. Run on Voynich AND Caesar.
B. SEED stability: Brown signal over several inits (mean +/- sd).
C. K/N sweep: does the Voynich signal persist across class counts / cutoffs?

Signal here = MI(real, classes) - MI(shuffle, same classes): how much class
predictability is destroyed by scrambling order (same null procedure for both
Brown and shape classes, so they are directly comparable).
"""
import re, math, random, pathlib
from collections import Counter, defaultdict
from voynich_lib import parse

ROOT = pathlib.Path(__file__).resolve().parent

def mi(seq):
    uni = Counter(seq); big = Counter(zip(seq, seq[1:])); tot = sum(big.values())
    if not tot: return 0.0
    Hu = -sum(c/sum(uni.values())*math.log2(c/sum(uni.values())) for c in uni.values())
    Hb = -sum(c/tot*math.log2(c/tot) for c in big.values())
    return 2*Hu - Hb   # = I(C1;C2) when marginals ~ equal

def reduce_seq(toks, N):
    c = Counter(toks); top = [w for w, _ in c.most_common(N)]
    tid = {w: i for i, w in enumerate(top)}
    return [tid.get(w, N) for w in toks], N, top   # OTHER = N

def class_seq(red, OTHER, clsmap):
    return [OTHER if t == OTHER else clsmap[t] for t in red]

def exchange(red, OTHER, K, seed, passes=8):
    rnd = random.Random(seed)
    left = defaultdict(Counter); right = defaultdict(Counter)
    for i in range(1, len(red)):
        a, b = red[i-1], red[i]
        if b != OTHER: left[b][a] += 1
        if a != OTHER: right[a][b] += 1
    movable = sorted(t for t in set(red) if t != OTHER)
    cls = {t: rnd.randrange(K) for t in movable}
    def clsK(t): return K if t == OTHER else cls[t]
    Kt = K+1; CB = [[0]*Kt for _ in range(Kt)]
    for i in range(1, len(red)): CB[clsK(red[i-1])][clsK(red[i])] += 1
    def cbmi():
        tot = sum(sum(r) for r in CB)
        if not tot: return 0.0
        row = [sum(r) for r in CB]; col = [sum(CB[x][y] for x in range(Kt)) for y in range(Kt)]
        m = 0.0
        for x in range(Kt):
            if not row[x]: continue
            for y in range(Kt):
                v = CB[x][y]
                if v: m += v/tot*math.log2(v*tot/(row[x]*col[y]))
        return m
    def apply(t, a, b):
        lc = left[t]; rc = right[t]; sc = lc.get(t, 0)
        for p, cnt in lc.items():
            if p == t: continue
            x = clsK(p); CB[x][a] -= cnt; CB[x][b] += cnt
        for q, cnt in rc.items():
            if q == t: continue
            y = clsK(q); CB[a][y] -= cnt; CB[b][y] += cnt
        if sc: CB[a][a] -= sc; CB[b][b] += sc
        cls[t] = b
    for _ in range(passes):
        moved = 0
        for t in movable:
            a = cls[t]; best, bm = a, None
            for b in range(K):
                apply(t, a, b); m = cbmi(); apply(t, b, a)
                if bm is None or m > bm: bm, best = m, b
            if best != a: apply(t, a, best); moved += 1
        if not moved: break
    return dict(cls)

def shape_classes(top, K, seed):
    alpha = sorted({ch for w in top for ch in w})
    ai = {c: i for i, c in enumerate(alpha)}; A = len(alpha)
    def feat(w):
        v = [0.0]*(A+2+2*A)
        for ch in w: v[ai[ch]] += 1
        v[A] = len(w)/10.0
        v[A+1] = 1.0
        v[A+2+ai[w[0]]] += 1; v[A+2+A+ai[w[-1]]] += 1
        n = math.sqrt(sum(x*x for x in v)) or 1; return [x/n for x in v]
    pts = [feat(w) for w in top]; dim = len(pts[0]); rnd = random.Random(seed)
    cent = [pts[i][:] for i in rnd.sample(range(len(pts)), K)]; a = [0]*len(pts)
    for _ in range(20):
        for i, p in enumerate(pts): a[i] = max(range(K), key=lambda c: sum(p[d]*cent[c][d] for d in range(dim)))
        nw = [[0.0]*dim for _ in range(K)]; cn = [0]*K
        for i, c in enumerate(a):
            cn[c] += 1
            for d in range(dim): nw[c][d] += pts[i][d]
        for c in range(K):
            if cn[c]: nr = math.sqrt(sum(x*x for x in nw[c])) or 1; cent[c] = [x/nr for x in nw[c]]
            else: cent[c] = pts[rnd.randrange(len(pts))][:]
    return {i: a[i] for i in range(len(top))}   # token-id -> shape class

def signal(toks, N, K, method, seed=1):
    red, OTHER, top = reduce_seq(toks, N)
    cls = exchange(red, OTHER, K, seed) if method == "brown" else shape_classes(top, K, seed)
    real = mi(class_seq(red, OTHER, cls))
    rnd = random.Random(seed); sh = red[:]; rnd.shuffle(sh)
    return real - mi(class_seq(sh, OTHER, cls))

def latin(p, g=False):
    t = (ROOT/p).read_text(encoding="utf-8", errors="ignore")
    if g:
        a = t.find("*** START"); b = t.find("*** END")
        if a >= 0: t = t[t.find("\n", a)+1:(b if b > 0 else len(t))]
    return re.findall(r"[a-z]+", t.lower())

V = [r[6] for r in parse()]; C = latin("data/latin_dbg.txt", True)
print("A. SHAPE vs ROLE  (signal = real - shuffled, K=12 N=200)")
print(f"   {'corpus':<10}{'BROWN(role)':>13}{'SHAPE':>9}{'ratio':>8}")
for nm, tk in (("Voynich", V), ("Caesar", C)):
    b = signal(tk, 200, 12, "brown"); s = signal(tk, 200, 12, "shape")
    print(f"   {nm:<10}{b:>13.3f}{s:>9.3f}{b/s if s else 0:>8.1f}x")

print("\nB. SEED stability (Voynich Brown signal, K=12 N=200)")
vals = [signal(V, 200, 12, "brown", seed=s) for s in range(1, 7)]
m = sum(vals)/len(vals); sd = (sum((x-m)**2 for x in vals)/len(vals))**0.5
print(f"   mean {m:.3f}  sd {sd:.3f}  range [{min(vals):.3f}, {max(vals):.3f}]")

print("\nC. K/N sweep (Voynich Brown signal)")
print(f"   {'':<6}" + "".join(f"N={n:<6}" for n in (150, 300)))
for K in (8, 12, 16):
    print(f"   K={K:<4}" + "".join(f"{signal(V,n,K,'brown'):<8.3f}" for n in (150, 300)))
