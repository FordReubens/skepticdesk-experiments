#!/usr/bin/env python3
"""M20: page-keyed (book) cipher -- Ford's "each page is its own cipher".

The global reversible ciphers had no page structure, so they couldn't make
page-local vocabulary or line effects. Here each PAGE enciphers with its own key
(derived from the page number) -> the same plaintext word ciphers differently per
page = page-local vocabulary, and it's still reversible (decoder knows page # -> key).
We lay real Latin into Voynich's exact page/line geometry, encipher per page with
two bases (autokey, substitution), verify roundtrip, and score the FULL board
including page-locality and line effects.
"""
import lzma, re, math, random, pathlib
from collections import Counter, defaultdict
from voynich_lib import parse, lev1

rows = parse(); VOY = [r[6] for r in rows]
GLY = sorted({c for w in VOY for c in w}); N = len(GLY)

def latin(p, g=False):
    t = (pathlib.Path(__file__).resolve().parent/p).read_text(encoding="utf-8", errors="ignore")
    if g:
        a = t.find("*** START"); b = t.find("*** END")
        if a >= 0: t = t[t.find("\n", a)+1:(b if b > 0 else len(t))]
    return re.findall(r"[a-z]+", t.lower())
PLAIN = latin("data/latin_dbg.txt", True) + latin("data/apicius_books.txt")
_freq = Counter(c for w in PLAIN for c in w)
KEEP = set(c for c, _ in _freq.most_common(N))            # plaintext alphabet must fit in N glyphs
PLAIN = [w for w in PLAIN if all(c in KEEP for c in w)]   # keep reversibility exact
LAT = sorted(KEEP); L2I = {c: i for i, c in enumerate(LAT)}; I2L = {i: c for c, i in L2I.items()}

# Voynich layout: list of (folio, line, count), folio order
line_items = []; cur = None; cnt = 0
for folio, line, pos, *_ in rows:
    k = (folio, line)
    if k != cur:
        if cur is not None: line_items.append((cur[0], cur[1], cnt))
        cur = k; cnt = 0
    cnt += 1
line_items.append((cur[0], cur[1], cnt))
TOTAL = sum(c for *_ , c in line_items)
PW = [PLAIN[i % len(PLAIN)] for i in range(TOTAL)]      # cycle Latin to fill the geometry
folorder = {}
for f, _, _ in line_items: folorder.setdefault(f, len(folorder))

def lay(enc):
    """enc(folio_index, word) -> cipher token. Returns items + plaintext-by-page for roundtrip."""
    items = []; ppage = defaultdict(list); idx = 0
    for f, line, c in line_items:
        g = folorder[f]
        for _ in range(c):
            pw = PW[idx]; idx += 1
            items.append((f, line, _, enc(g, pw))); ppage[g].append(pw)
    return items

# --- base 1: page-keyed autokey (re-seed per page with page key) ---
def mk_autokey():
    state = {}
    def enc(g, w):
        prev = state.get(g, (7 + 31*g) % N)
        cw = []
        for ch in w:
            ci = (L2I[ch] + prev) % N; cw.append(GLY[ci]); prev = ci
        state[g] = prev; return "".join(cw)
    return enc
def dec_autokey(items):
    state = {}; out = []
    for f, line, pos, t in items:
        g = folorder[f]; prev = state.get(g, (7 + 31*g) % N); pw = []
        for ch in t:
            ci = GLY.index(ch); pw.append(I2L[(ci - prev) % N]); prev = ci
        state[g] = prev; out.append("".join(pw))
    return out

# --- base 2: page-keyed substitution (rotate the cipher alphabet by page) ---
_perm = list(range(N)); random.Random(0).shuffle(_perm)
L2G = {c: _perm[i] for i, c in enumerate(LAT)}            # letter -> base glyph index
G2L = {v: k for k, v in L2G.items()}
GIDX = {g: i for i, g in enumerate(GLY)}
def mk_subst():
    def enc(g, w):
        sh = g % N
        return "".join(GLY[(L2G[ch] + sh) % N] for ch in w)
    return enc
def dec_subst(items):
    out = []
    for f, line, pos, t in items:
        sh = folorder[f] % N
        out.append("".join(G2L[(GIDX[ch] - sh) % N] for ch in t))
    return out

AK = lay(mk_autokey()); SB = lay(mk_subst())
print("roundtrip autokey:", dec_autokey(AK) == PW, "| roundtrip subst:", dec_subst(SB) == PW, "\n")

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
def scoreboard(name, items):
    tk = [x[3] for x in items]; Nt = len(tk); types = len(set(tk)); mwl = sum(len(w) for w in tk)/Nt
    pr = same = one = 0; prev = pk = None
    for f, l, p, w in items:
        if prev is not None and (f, l) == pk:
            pr += 1
            if w == prev: same += 1
            elif lev1(w, prev): one += 1
        prev, pk = w, (f, l)
    uni = Counter(); big = Counter()
    for w in tk:
        for ch in w: uni[ch] += 1
        for a, b in zip(w, w[1:]): big[(a, b)] += 1
    h1 = Hf(uni); h2 = Hf(big) - h1
    fl = [x[0] for x in items]; pagesz = Counter(fl); Nn = len(fl); pf = {f: pagesz[f]/Nn for f in pagesz}
    wp = defaultdict(Counter); c = Counter(tk)
    for w, f in zip(tk, fl): wp[w][f] += 1
    num = den = 0.0
    for w in c:
        if c[w] >= 10:
            n = c[w]; kl = sum((cf/n)*math.log2((cf/n)/pf[f]) for f, cf in wp[w].items()); num += n*kl; den += n
    ploc = num/den if den else 0
    ini = Counter(); inter = Counter()
    for f, l, p, w in items: (ini if p == 0 else inter)[w] += 1
    elig = {w for w in c if c[w] >= 10}
    ti = sum(ini[w] for w in elig) or 1; tr = sum(inter[w] for w in elig) or 1
    le = sum((ini[w]/ti)*math.log2((ini[w]/ti)/((inter[w]/tr) or 1e-9)) for w in elig if ini[w])
    blob = " ".join(tk).encode(); ratio = len(lzma.compress(blob, preset=6))/len(blob)
    print(f"{name:<20}{types:>7}{mwl:>6.2f}{one/pr*100:>7.1f}%{h2/h1:>7.2f}{ploc:>8.2f}{le:>8.2f}{exch_signal(tk):>8.3f}{ratio:>8.3f}")

print(f"{'system':<20}{'types':>7}{'wlen':>6}{'drift':>7}{'h2/h1':>7}{'pageloc':>8}{'lineeff':>8}{'brownS':>8}{'lzma':>8}")
print("-"*80)
scoreboard("Voynich", [(r[0], r[1], r[2], r[6]) for r in rows])
scoreboard("Page-autokey", AK)
scoreboard("Page-subst", SB)
print("\nBoth reversible (roundtrip True). Does per-page keying add page-locality/line-effects?")
