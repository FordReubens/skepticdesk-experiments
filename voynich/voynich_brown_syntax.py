#!/usr/bin/env python3
"""M10b: rigorous class-level syntax via EXCHANGE (Brown) clustering.

The k-means proxy failed to validate (Caesar didn't separate). Exchange clustering
directly maximizes the adjacent-class mutual information I(C1;C2) -- exactly the
class-bigram predictability we want -- so it is the right tool.

For each corpus we cluster the top-N tokens into K classes by exchange, and report
the optimized class MI on (i) the REAL sequence and (ii) a SHUFFLED sequence (same
unigrams, order destroyed = the overfitting floor). The SIGNAL = MI_real - MI_shuf
is the validated amount of class-level sequential structure. Caesar (real syntax)
must show a clear positive signal for the test to be trusted; then we read Voynich.
"""
import re, math, random, pathlib
from collections import Counter, defaultdict
from voynich_lib import parse

random.seed(7)
ROOT = pathlib.Path(__file__).resolve().parent
N, K, PASSES = 200, 12, 8

def reduce_seq(toks):
    c = Counter(toks); top = [w for w, _ in c.most_common(N)]
    tid = {w: i for i, w in enumerate(top)}
    OTHER = N
    return [tid.get(w, OTHER) for w in toks], OTHER

def left_right(red, OTHER):
    left = defaultdict(Counter); right = defaultdict(Counter)
    for i in range(1, len(red)):
        a, b = red[i-1], red[i]
        if b != OTHER: left[b][a] += 1
        if a != OTHER: right[a][b] += 1
    return left, right

def mi_from_cb(CB, Kt):
    tot = sum(sum(r) for r in CB)
    if tot == 0: return 0.0
    row = [sum(r) for r in CB]; col = [sum(CB[x][y] for x in range(Kt)) for y in range(Kt)]
    mi = 0.0
    for x in range(Kt):
        if not row[x]: continue
        for y in range(Kt):
            v = CB[x][y]
            if v: mi += v/tot*math.log2(v*tot/(row[x]*col[y]))
    return mi

def exchange(red, OTHER):
    left, right = left_right(red, OTHER)
    movable = [t for t in set(red) if t != OTHER]
    cls = {t: i % K for i, t in enumerate(sorted(movable))}
    cls_of = lambda t: K-1 if t == OTHER else cls[t]   # put OTHER in last class slot? keep separate
    Kt = K + 1                                          # classes 0..K-1 plus OTHER=K
    def clsK(t): return K if t == OTHER else cls[t]
    CB = [[0]*Kt for _ in range(Kt)]
    for i in range(1, len(red)):
        CB[clsK(red[i-1])][clsK(red[i])] += 1
    def apply(t, a, b):
        lc = left[t]; rc = right[t]; self_c = lc.get(t, 0)
        for p, cnt in lc.items():
            if p == t: continue
            x = clsK(p); CB[x][a] -= cnt; CB[x][b] += cnt
        for q, cnt in rc.items():
            if q == t: continue
            y = clsK(q); CB[a][y] -= cnt; CB[b][y] += cnt
        if self_c: CB[a][a] -= self_c; CB[b][b] += self_c
        cls[t] = b
    for _ in range(PASSES):
        moved = 0
        for t in movable:
            a = cls[t]; best, bestmi = a, None
            for b in range(K):
                apply(t, a, b); m = mi_from_cb(CB, Kt); apply(t, b, a)  # try then revert
                if bestmi is None or m > bestmi: bestmi, best = m, b
            if best != a: apply(t, a, best); moved += 1
        if moved == 0: break
    return mi_from_cb(CB, Kt)

def run(name, toks):
    red, OTHER = reduce_seq(toks)
    real = exchange(red, OTHER)
    sh = red[:]; random.shuffle(sh)
    shuf = exchange(sh, OTHER)
    print(f"{name:<10}{real:>10.3f}{shuf:>10.3f}{real-shuf:>10.3f}")

def latin(p, g=False):
    t = (ROOT/p).read_text(encoding="utf-8", errors="ignore")
    if g:
        a = t.find("*** START"); b = t.find("*** END")
        if a >= 0: t = t[t.find("\n", a)+1:(b if b > 0 else len(t))]
    return re.findall(r"[a-z]+", t.lower())

print(f"exchange clustering: N={N} types, K={K} classes, {PASSES} passes")
print(f"class MI (bits) = adjacent-class predictability\n")
print(f"{'corpus':<10}{'MI real':>10}{'MI shuf':>10}{'SIGNAL':>10}")
print("-"*40)
run("Caesar", latin("data/latin_dbg.txt", True))   # positive control
run("Apicius", latin("data/apicius_books.txt"))
run("Voynich", [r[6] for r in parse()])
print("\nSIGNAL = MI_real - MI_shuffled = validated class-level sequential structure.")
print("Caesar must show a clear positive SIGNAL; then compare Voynich to it.")
