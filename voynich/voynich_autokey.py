#!/usr/bin/env python3
"""M19: the strongest reversible model -- a stateful AUTOKEY cipher.

c_0 = (p_0 + key) % N ;  c_i = (p_i + c_{i-1}) % N   (continuous over the letter
stream, word boundaries preserved). Decrypt: p_i = (c_i - c_{i-1}) % N. Reversible,
stateful, compact key (one number). Each Latin letter's glyph depends on the running
ciphertext, so the SAME plaintext word produces ever-changing cipher words --
exactly the "verbose / near-variant" behaviour GPT-5.5 wanted, but invertible.

Strict rules: roundtrip verified first; compact key; no post-hoc Voynich tuning;
score the hard features. Plus the information-theoretic check: a bijective cipher
preserves adjacent mutual information, so a substitution's Brown signal EQUALS the
plaintext's -- a reversible map cannot push grammar BELOW the plaintext without
losing information. Does autokey escape that?
"""
import lzma, re, math, random, pathlib
from collections import Counter, defaultdict
from voynich_lib import parse, lev1

rows = parse(); VOY = [r[6] for r in rows]
GLY = sorted({c for w in VOY for c in w}); N = len(GLY)

def latin(path, g=False):
    t = (pathlib.Path(__file__).resolve().parent/path).read_text(encoding="utf-8", errors="ignore")
    if g:
        a = t.find("*** START"); b = t.find("*** END")
        if a >= 0: t = t[t.find("\n", a)+1:(b if b > 0 else len(t))]
    return re.findall(r"[a-z]+", t.lower())
APIC = latin("data/apicius_books.txt")
LAT = sorted({c for w in APIC for c in w})
L2I = {c: i for i, c in enumerate(LAT)}; I2L = {i: c for c, i in L2I.items()}   # <= N letters

# --- reversible substitution (MI-invariance demo) ---
rng = random.Random(0); gl = GLY[:]; rng.shuffle(gl)
SUB = {c: gl[i] for i, c in enumerate(LAT)}; INV = {v: k for k, v in SUB.items()}
CIPSUB = ["".join(SUB[c] for c in w) for w in APIC]
assert ["".join(INV[ch] for ch in t) for t in CIPSUB] == APIC

# --- autokey cipher (continuous over letter stream, word boundaries kept) ---
KEY = 7
def autokey_encode(words):
    out = []; prev = KEY
    for w in words:
        cw = []
        for c in w:
            ci = (L2I[c] + prev) % N; cw.append(GLY[ci]); prev = ci
        out.append("".join(cw))
    return out
def autokey_decode(toks):
    out = []; prev = KEY
    for t in toks:
        pw = []
        for ch in t:
            ci = GLY.index(ch); pw.append(I2L[(ci - prev) % N]); prev = ci
        out.append("".join(pw))
    return out
CIPAK = autokey_encode(APIC)
print("roundtrip exact (must be True):", autokey_decode(CIPAK) == APIC, "\n")

def Hf(c):
    t = sum(c.values()); return -sum(v/t*math.log2(v/t) for v in c.values()) if t else 0.0
def exch_signal(tk, Nn=150, KK=10, passes=6, seed=1):
    rnd = random.Random(seed); c = Counter(tk); top = [w for w, _ in c.most_common(Nn)]; tid = {w: i for i, w in enumerate(top)}
    OT = Nn; red = [tid.get(w, OT) for w in tk]
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

def stats(name, tk, rev):
    Ntok = len(tk); types = len(set(tk)); mwl = sum(len(w) for w in tk)/Ntok
    pr = same = one = 0; prev = None
    for w in tk:
        if prev is not None:
            pr += 1
            if w == prev: same += 1
            elif lev1(w, prev): one += 1
        prev = w
    uni = Counter(); big = Counter()
    for w in tk:
        for ch in w: uni[ch] += 1
        for a, b in zip(w, w[1:]): big[(a, b)] += 1
    h1 = Hf(uni); h2 = Hf(big) - h1
    blob = " ".join(tk).encode(); ratio = len(lzma.compress(blob, preset=6))/len(blob)
    print(f"{name:<22}{rev:>5}{types:>7}{mwl:>6.2f}{one/pr*100:>7.1f}%{h2/h1:>7.2f}{exch_signal(tk):>8.3f}{ratio:>8.3f}")

print(f"{'system':<22}{'rev?':>5}{'types':>7}{'wlen':>6}{'drift':>7}{'h2/h1':>7}{'brownS':>8}{'lzma':>8}")
print("-"*72)
stats("Voynich", VOY, "no")
stats("Latin (Apicius)", APIC, "--")
stats("Subst cipher", CIPSUB, "yes")
stats("Autokey cipher", CIPAK, "yes")
print("\nMI-invariance: a bijective substitution leaves adjacent mutual information unchanged")
print("-> its Brown signal EQUALS Latin's. A reversible map cannot push grammar below the")
print("plaintext's without destroying information. Voynich's brownS (0.087) is far below Latin's.")
