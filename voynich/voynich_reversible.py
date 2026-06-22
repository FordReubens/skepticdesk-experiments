#!/usr/bin/env python3
"""M18: the reversible test (decryption door).

A reversible cipher must PRESERVE the plaintext's information (else you can't
invert). So the question is whether reversibility and Voynich's statistics can
coexist. We build genuinely invertible ciphers of real Latin, VERIFY the roundtrip
(decode(encode(x))==x), and measure whether their output can match Voynich on:
  - h2/h1 glyph redundancy
  - one-glyph drift (the self-citation signature)
  - Brown class sequencing
  - compressibility (lzma) = how redundant / how much true information per token
Compared to Voynich itself and to the (non-reversible) self-citation generator.

Crux: self-citation gets Voynich's low entropy by COPYING (redundancy). A reversible
cipher cannot be that redundant -- copies carry no new plaintext. So the prediction
is a tension: reversible -> information-rich (incompressible, high entropy); Voynich
-> redundant (compressible, low entropy). You can't have both.
"""
import lzma, re, math, random
from collections import Counter, defaultdict
from voynich_lib import parse, lev1

rows = parse()
VOY = [r[6] for r in rows]
GLY = sorted({c for w in VOY for c in w})

def latin(path, g=False):
    t = (pathlib_read(path));
    if g:
        a = t.find("*** START"); b = t.find("*** END")
        if a >= 0: t = t[t.find("\n", a)+1:(b if b > 0 else len(t))]
    return re.findall(r"[a-z]+", t.lower())
import pathlib
def pathlib_read(p): return (pathlib.Path(__file__).resolve().parent/p).read_text(encoding="utf-8", errors="ignore")

APIC = latin("data/apicius_books.txt")
LAT = sorted({c for w in APIC for c in w})

# --- reversible cipher 1: simple 1->1 glyph substitution ---
rng = random.Random(0); gl = GLY[:]; rng.shuffle(gl)
SUB = {c: gl[i] for i, c in enumerate(LAT)}; INV = {v: k for k, v in SUB.items()}
def enc1(words): return ["".join(SUB[c] for c in w) for w in words]
def dec1(toks): return ["".join(INV[c] for c in t) for t in toks]

# --- reversible cipher 2: verbose 1->2-glyph (fixed-length, uniquely decodable) ---
pairs = [a+b for a in GLY for b in GLY]; rng.shuffle(pairs)
SUB2 = {c: pairs[i] for i, c in enumerate(LAT)}; INV2 = {v: k for k, v in SUB2.items()}
def enc2(words): return ["".join(SUB2[c] for c in w) for w in words]
def dec2(toks): return ["".join(INV2[t[i:i+2]] for i in range(0, len(t), 2)) for t in toks]

CIP1 = enc1(APIC); CIP2 = enc2(APIC)
print("roundtrip exact (must be True):", dec1(CIP1) == APIC, "/", dec2(CIP2) == APIC, "\n")

def Hf(c):
    t = sum(c.values()); return -sum(v/t*math.log2(v/t) for v in c.values()) if t else 0.0
def exch_signal(tk, N=150, KK=10, passes=6, seed=1):
    rnd = random.Random(seed); c = Counter(tk); top = [w for w, _ in c.most_common(N)]; tid = {w: i for i, w in enumerate(top)}
    OT = N; red = [tid.get(w, OT) for w in tk]
    def mi(seq): return 2*Hf(Counter(seq)) - Hf(Counter(zip(seq, seq[1:])))
    left = defaultdict(Counter); right = defaultdict(Counter)
    for i in range(1, len(red)):
        a, b = red[i-1], red[i]
        if b != OT: left[b][a] += 1
        if a != OT: right[a][b] += 1
    mov = sorted(t for t in set(red) if t != OT); cls = {t: rnd.randrange(KK) for t in mov}
    def cK(t): return KK if t == OT else cls[t]
    Kt = KK+1; CB = [[0]*Kt for _ in range(Kt)]
    for i in range(1, len(red)): CB[cK(red[i-1])][cK(red[i])] += 1
    def cbmi():
        tot = sum(sum(r) for r in CB)
        if not tot: return 0.0
        row = [sum(r) for r in CB]; col = [sum(CB[x][y] for x in range(Kt)) for y in range(Kt)]
        return sum(CB[x][y]/tot*math.log2(CB[x][y]*tot/(row[x]*col[y])) for x in range(Kt) for y in range(Kt) if CB[x][y] and row[x] and col[y])
    def ap(t, a, b):
        for p, n in left[t].items():
            if p != t: x = cK(p); CB[x][a] -= n; CB[x][b] += n
        for q, n in right[t].items():
            if q != t: y = cK(q); CB[a][y] -= n; CB[b][y] += n
        sc = left[t].get(t, 0)
        if sc: CB[a][a] -= sc; CB[b][b] += sc
        cls[t] = b
    for _ in range(passes):
        for t in mov:
            a = cls[t]; best, bm = a, None
            for b in range(KK):
                ap(t, a, b); m = cbmi(); ap(t, b, a)
                if bm is None or m > bm: bm, best = m, b
            if best != a: ap(t, a, best)
    real = mi([cK(x) for x in red]); sh = red[:]; rnd.shuffle(sh)
    return real - mi([cK(x) for x in sh])

def stats(name, tk, reversible):
    N = len(tk); mwl = sum(len(w) for w in tk)/N
    pairs_ = same = one = 0; prev = None
    for w in tk:
        if prev is not None:
            pairs_ += 1
            if w == prev: same += 1
            elif lev1(w, prev): one += 1
        prev = w
    uni = Counter(); big = Counter()
    for w in tk:
        for ch in w: uni[ch] += 1
        for a, b in zip(w, w[1:]): big[(a, b)] += 1
    h1 = Hf(uni); h2 = Hf(big) - h1
    blob = " ".join(tk).encode()
    ratio = len(lzma.compress(blob, preset=6)) / len(blob)
    print(f"{name:<22}{'yes' if reversible else 'no':>5}{mwl:>6.2f}{one/pairs_*100:>7.1f}%{h2/h1:>7.2f}{exch_signal(tk):>8.3f}{ratio:>9.3f}")

print(f"{'system':<22}{'rev?':>5}{'wlen':>6}{'drift':>7}{'h2/h1':>7}{'brownS':>8}{'lzma':>9}")
print("-"*66)
stats("Voynich", VOY, None)
stats("Latin (Apicius)", APIC, None)
stats("Subst cipher (1->1)", CIP1, True)
stats("Verbose cipher (1->2)", CIP2, True)
print("\nlzma = compressed/original (LOWER = more redundant = less true information per token).")
print("Reversible ciphers must carry the plaintext's information -> expect them to resist")
print("compression (high lzma) and show Latin-like entropy, unlike redundant Voynich.")
