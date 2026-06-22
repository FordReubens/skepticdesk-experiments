#!/usr/bin/env python3
"""M17: recursive self-citation generator (Ford's "recursion" fix; Timm & Schinner family).

Each new word is, most of the time, a COPY of a recently generated word -- with an
occasional one-glyph mutation. Words beget words. A few fresh glyph-Markov words seed
new material. One recursive mechanism, not four fighting knobs:
  - mutation spawns new types        -> vocabulary diversity grows organically
  - copy from a RECENT window        -> moderate page-locality (not siloed)
  - copy+tweak                        -> one-glyph drift is the mechanism itself
  - copying local structure           -> class sequencing for free
"""
import math, random
from collections import Counter, defaultdict
from voynich_lib import parse, lev1

rows = parse()
toks = [r[6] for r in rows]
GW = Counter(c for w in toks for c in w); GLI = list(GW); GLWT = [GW[g] for g in GLI]

gm = defaultdict(Counter)
for w in toks:
    s = "^^" + w + "$"
    for i in range(len(s)-2): gm[(s[i], s[i+1])][s[i+2]] += 1

line_items = []; cur = None; cnt = 0
for folio, line, pos, L, Hh, I, tok in rows:
    k = (folio, line)
    if k != cur:
        if cur is not None: line_items.append((cur, cnt))
        cur = k; cnt = 0
    cnt += 1
line_items.append((cur, cnt))

def wpick(items, wts, rng):
    r = rng.random()*sum(wts); a = 0.0
    for it, wt in zip(items, wts):
        a += wt
        if r <= a: return it
    return items[-1]

def gen_word(rng):
    s = ["^", "^"]
    for _ in range(12):
        nx = gm.get((s[-2], s[-1]))
        if not nx: break
        c = wpick(list(nx), [nx[k] for k in nx], rng)
        if c == "$": break
        s.append(c)
    return "".join(s[2:]) or "o"

def ctx_glyph(w, i, rng):                                  # a glyph plausible at position i
    s = "^^" + w; nx = gm.get((s[i], s[i+1]))
    if nx:
        items = [c for c in nx if c != "$"]
        if items: return wpick(items, [nx[c] for c in items], rng)
    return wpick(GLI, GLWT, rng)

def mutate(w, rng):                                       # substitution only -> length never creeps
    i = rng.randrange(len(w)); return w[:i] + ctx_glyph(w, i, rng) + w[i+1:]

# --- knobs ---
P_ADJ = 0.03      # copy the immediately previous word + mutate -> adjacent one-glyph drift
P_COPY = 0.82     # else copy from history (lower -> more fresh seeds, anchors length & h2/h1)
P_MUT = 0.45      # given a copy, chance of mutation
P_GLOBAL = 0.45   # of copies, fraction from ALL history (vs local window) -> lowers page-locality
WINDOW = 500      # local recency window

def generate():
    out = []; hist = []; folorder = {}
    for (folio, line), count in line_items:
        g = folorder.setdefault(folio, len(folorder)); rng = random.Random(1000+g)
        for w in range(count):
            r = rng.random()
            if hist and r < P_ADJ:
                tok = mutate(hist[-1], rng)                          # adjacent drift
            elif hist and r < P_COPY:
                if rng.random() < P_GLOBAL:
                    src = hist[rng.randrange(len(hist))]             # global -> cross-page recurrence
                else:
                    lo = max(0, len(hist)-WINDOW)
                    src = hist[rng.randint(lo, len(hist)-1)]         # local -> line/class structure
                tok = mutate(src, rng) if rng.random() < P_MUT else src
            else:
                tok = gen_word(rng)                                  # seed fresh
            out.append((folio, line, w, tok)); hist.append(tok)
    return out

# --- scoreboard ---
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
def scoreboard(name, items):
    tk = [x[3] for x in items]; N = len(tk); types = len(set(tk)); mwl = sum(len(w) for w in tk)/N
    pairs = same = one = 0; prev = pk = None
    for f, l, p, w in items:
        if prev is not None and (f, l) == pk:
            pairs += 1
            if w == prev: same += 1
            elif lev1(w, prev): one += 1
        prev, pk = w, (f, l)
    uni = Counter(); big = Counter()
    for w in tk:
        for ch in w: uni[ch] += 1
        for a, b in zip(w, w[1:]): big[(a, b)] += 1
    h1 = Hf(uni); h2 = Hf(big) - h1
    fl = [x[0] for x in items]; pagesz = Counter(fl); Nt = len(fl); pf = {f: pagesz[f]/Nt for f in pagesz}
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
    print(f"{name:<16}{N:>7}{types:>7}{mwl:>6.2f}{same/pairs*100:>7.1f}%{one/pairs*100:>7.1f}%{h2/h1:>7.2f}{ploc:>8.2f}{le:>8.2f}{exch_signal(tk):>8.3f}")

print(f"P_ADJ={P_ADJ} P_COPY={P_COPY} P_MUT={P_MUT} P_GLOBAL={P_GLOBAL} WINDOW={WINDOW}\n")
print(f"{'system':<16}{'toks':>7}{'types':>7}{'wlen':>6}{'rep':>8}{'drift':>7}{'h2/h1':>7}{'pageloc':>8}{'lineeff':>8}{'brownS':>8}")
print("-"*94)
scoreboard("Real Voynich", [(r[0], r[1], r[2], r[6]) for r in rows])
scoreboard("Recursive-gen", generate())
