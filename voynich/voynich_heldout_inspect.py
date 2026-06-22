#!/usr/bin/env python3
"""M11: held-out validation + class inspection (GPT-5.5's final checks).

HELD-OUT: Brown clustering optimizes on the sequence it scores. Cleaner test:
train classes on HALF the folios, freeze them, measure the class-transition
signal on the OTHER half. If the class grammar generalizes out-of-sample it is a
real property, not fitted noise. Caesar (split in half) is the positive control.

INSPECT: for each induced Voynich class, show members and quantify whether it
groups by shape, by section, by line position, or (the syntax-like case)
visually different words sharing contexts.
"""
import re, math, random, pathlib
from collections import Counter, defaultdict
from voynich_lib import parse

ROOT = pathlib.Path(__file__).resolve().parent
K, N, PASSES = 12, 200, 8

def mi(seq):
    uni = Counter(seq); big = Counter(zip(seq, seq[1:]))
    tu = sum(uni.values()); tb = sum(big.values())
    if not tb: return 0.0
    Hu = -sum(c/tu*math.log2(c/tu) for c in uni.values())
    Hb = -sum(c/tb*math.log2(c/tb) for c in big.values())
    return 2*Hu - Hb

def train_classes(toks, seed=1):
    c = Counter(toks); top = [w for w, _ in c.most_common(N)]
    tid = {w: i for i, w in enumerate(top)}; OTHER = N
    red = [tid.get(w, OTHER) for w in toks]
    left = defaultdict(Counter); right = defaultdict(Counter)
    for i in range(1, len(red)):
        a, b = red[i-1], red[i]
        if b != OTHER: left[b][a] += 1
        if a != OTHER: right[a][b] += 1
    rnd = random.Random(seed); mov = sorted(t for t in set(red) if t != OTHER)
    cls = {t: rnd.randrange(K) for t in mov}
    def cK(t): return K if t == OTHER else cls[t]
    Kt = K+1; CB = [[0]*Kt for _ in range(Kt)]
    for i in range(1, len(red)): CB[cK(red[i-1])][cK(red[i])] += 1
    def cbmi():
        tot = sum(sum(r) for r in CB)
        if not tot: return 0.0
        row = [sum(r) for r in CB]; col = [sum(CB[x][y] for x in range(Kt)) for y in range(Kt)]
        return sum(CB[x][y]/tot*math.log2(CB[x][y]*tot/(row[x]*col[y]))
                   for x in range(Kt) for y in range(Kt) if CB[x][y] and row[x] and col[y])
    def ap(t, a, b):
        lc = left[t]; rc = right[t]; sc = lc.get(t, 0)
        for p, n in lc.items():
            if p != t: x = cK(p); CB[x][a] -= n; CB[x][b] += n
        for q, n in rc.items():
            if q != t: y = cK(q); CB[a][y] -= n; CB[b][y] += n
        if sc: CB[a][a] -= sc; CB[b][b] += sc
        cls[t] = b
    for _ in range(PASSES):
        mv = 0
        for t in mov:
            a = cls[t]; best, bm = a, None
            for b in range(K):
                ap(t, a, b); m = cbmi(); ap(t, b, a)
                if bm is None or m > bm: bm, best = m, b
            if best != a: ap(t, a, best); mv += 1
        if not mv: break
    return {top[t]: cls[t] for t in mov}   # token string -> class

def signal(toks, clsmap, seed=1):
    OT = K
    seq = [clsmap.get(w, OT) for w in toks]
    real = mi(seq)
    rnd = random.Random(seed); sh = seq[:]; rnd.shuffle(sh)
    return real - mi(sh), sum(1 for w in toks if w in clsmap)/len(toks)

# ---------- HELD-OUT ----------
rows = parse()
folios = sorted({r[0] for r in rows})
setA = set(folios[::2]);
toksA = [r[6] for r in rows if r[0] in setA]
toksB = [r[6] for r in rows if r[0] not in setA]
clsA = train_classes(toksA)
in_s, _ = signal(toksA, clsA)            # train A, test A (in-sample)
out_s, cov = signal(toksB, clsA)         # train A, test B (held-out)

ca = (ROOT/"data/latin_dbg.txt").read_text(encoding="utf-8", errors="ignore")
a = ca.find("*** START"); b = ca.find("*** END")
ca = re.findall(r"[a-z]+", ca[ca.find(chr(10), a)+1:(b if b>0 else len(ca))].lower())
ca1, ca2 = ca[:len(ca)//2], ca[len(ca)//2:]
clsC = train_classes(ca1); cin, _ = signal(ca1, clsC); cout, ccov = signal(ca2, clsC)

print("HELD-OUT validation (signal = real - shuffled, K=12 N=200)")
print(f"   {'corpus':<10}{'in-sample':>11}{'held-out':>11}{'retained':>10}{'coverage':>10}")
print(f"   {'Voynich':<10}{in_s:>11.3f}{out_s:>11.3f}{out_s/in_s*100:>9.0f}%{cov*100:>9.0f}%")
print(f"   {'Caesar':<10}{cin:>11.3f}{cout:>11.3f}{cout/cin*100:>9.0f}%{ccov*100:>9.0f}%")

# ---------- INSPECT ----------
clsFull = train_classes([r[6] for r in rows])
freq = Counter(r[6] for r in rows)
sec_of = defaultdict(Counter);
for r in rows: sec_of[r[6]][r[5]] += 1
members = defaultdict(list)
for w, k in clsFull.items(): members[k].append(w)
print("\nCLASS INSPECTION (Voynich, K=12; is each class shape / section / role based?)")
print(f"   {'cls':<4}{'n':>4}{'suffix-homog':>13}{'sect-conc':>10}   examples")
for k in sorted(members, key=lambda k: -len(members[k])):
    ms = sorted(members[k], key=lambda w: -freq[w])
    suf = Counter(w[-2:] for w in ms if len(w) >= 2)
    shom = suf.most_common(1)[0][1]/len(ms)*100
    secc = Counter()
    for w in ms:
        secc[sec_of[w].most_common(1)[0][0]] += 1
    sconc = secc.most_common(1)[0][1]/len(ms)*100
    print(f"   {k:<4}{len(ms):>4}{shom:>11.0f}%{sconc:>9.0f}%   {' '.join(ms[:6])}")
print("\nlow suffix-homog + low sect-conc + coherent context = syntax-like (role) class.")
