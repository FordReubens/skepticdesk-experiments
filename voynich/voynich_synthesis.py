#!/usr/bin/env python3
"""M16: synthesis generator -- try to hit the ENTIRE adversarial board at once.

Combines the pieces that each worked:
  - glyph order-2 Markov word-builder  -> realistic word length + glyph entropy (h2/h1)
  - page-number-seeded lexicon          -> page-local vocabulary
  - moderate class coupling (tunable)   -> Brown class sequencing at the REAL level (not 4x)
  - line-initial class bias             -> line-edge effect
This is the sufficiency arm: tuning to match is the explicit goal (can a procedural
machine match every signature simultaneously?).
"""
import math, random
from collections import Counter, defaultdict
from voynich_lib import parse, lev1

rows = parse()
toks = [r[6] for r in rows]
GLY = sorted({c for w in toks for c in w}); GW = Counter(c for w in toks for c in w)
GLI = list(GW); GLWT = [GW[g] for g in GLI]
K = 12

gm = defaultdict(Counter)
for w in toks:
    s = "^^" + w + "$"
    for i in range(len(s)-2): gm[(s[i], s[i+1])][s[i+2]] += 1

line_items = []; cur = None; cnt = 0
for folio, line, pos, L, Hh, I, tok in rows:
    k = (folio, line)
    if k != cur:
        if cur is not None: line_items.append((cur, cnt));
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

def mutate(w, rng):
    g = wpick(GLI, GLWT, rng); i = rng.randrange(len(w))
    return (w[:i] + g + w[i:]) if rng.random() < .5 else (w[:i] + g + w[i+1:])

def build_T(s, seed=3):
    rng = random.Random(seed); T = []
    for c in range(K):
        row = [1.0]*K
        for tgt in rng.sample(range(K), 2): row[tgt] += s*K     # prefer ~2 next-classes
        tot = sum(row); T.append([x/tot for x in row])
    return T

# tunable knobs
S = 0.05; BETA = 0.0; REP = 0.005; DRIFT = 0.016; GAMMA = 6.0; MASTER = 9000
T = build_T(S)

# one shared master vocabulary; pages reweight it (words recur across pages)
mr = random.Random(7)
MWORDS = [gen_word(mr) for _ in range(MASTER)]
MCLS = [mr.randrange(K) for _ in range(MASTER)]
MBASE = [1.0]*MASTER                                   # flat base -> pages can use the whole vocab
CLS_IDX = defaultdict(list)
for i in range(MASTER): CLS_IDX[MCLS[i]].append(i)

def generate():
    out = []; folorder = {}
    for (folio, line), count in line_items:
        g = folorder.setdefault(folio, len(folorder))
        rng = random.Random(1000+g)
        pw = [MBASE[i]*rng.gammavariate(GAMMA, 1) for i in range(MASTER)]   # page-favored weights
        clist = {c: (CLS_IDX[c], [pw[i] for i in CLS_IDX[c]]) for c in range(K)}
        state = 0; prev = None
        for w in range(count):
            if w == 0:
                state = 0 if rng.random() < BETA else wpick(list(range(K)), T[state], rng)
            else:
                state = wpick(list(range(K)), T[state], rng)
            x = rng.random()
            if prev and x < REP: tok = prev
            elif prev and x < REP+DRIFT: tok = mutate(prev, rng)
            else:
                idxs, wts = clist[state] if clist[state][0] else clist[(state+1) % K]
                tok = MWORDS[wpick(idxs, wts, rng)]
            out.append((folio, line, w, tok)); prev = tok
    return out

# ---- scoreboard ----
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

print(f"S={S} BETA={BETA} GAMMA={GAMMA} MASTER={MASTER}\n")
print(f"{'system':<16}{'toks':>7}{'types':>7}{'wlen':>6}{'rep':>8}{'drift':>7}{'h2/h1':>7}{'pageloc':>8}{'lineeff':>8}{'brownS':>8}")
print("-"*94)
scoreboard("Real Voynich", [(r[0], r[1], r[2], r[6]) for r in rows])
scoreboard("Synthesis-gen", generate())
