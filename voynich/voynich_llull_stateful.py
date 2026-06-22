#!/usr/bin/env python3
"""M15: STATEFUL paper machine. The memoryless wheel failed the Brown class-
sequencing test (0.019 vs real 0.087) because each word ignored the last. Here the
wheels CARRY (positions persist) and the previous word feeds the next word's
affixes -- a period-plausible 'advance from where you left off, and let the last
reading steer the next' rule. Does that mechanical memory create above-word class
sequencing without being trained on Voynich's transitions?
"""
import math, random
from collections import Counter, defaultdict
from voynich_lib import parse, lev1

rows = parse()
toks = [r[6] for r in rows]
GLYPHS = sorted({c for w in toks for c in w})
PRE_SET = ["qo","qok","ch","sh","cth","o","y","d","s","q","l","k",""]
def decomp(w):
    p = ""
    for a in sorted(PRE_SET, key=len, reverse=True):
        if a and w.startswith(a) and len(w)-len(a) >= 2: p = a; w = w[len(a):]; break
    s = ""
    for a in ["aiin","eedy","edy","dy","iin","ain","ol","or","ar","al","ey","y","n",""]:
        if a and w.endswith(a) and len(w)-len(a) >= 1: s = a; w = w[:-len(a)]; break
    return p, w, s
cores = Counter(); pres = Counter(); sufs = Counter()
for w in toks:
    p, c, s = decomp(w); cores[c] += 1; pres[p] += 1; sufs[s] += 1
PRE = [p for p, _ in pres.most_common(12)]
CORE = [c for c, _ in cores.most_common(120)]
SUF = [s for s, _ in sufs.most_common(16)]

line_items = []; cur = None; cnt = 0
for folio, line, pos, L, Hh, I, tok in rows:
    k = (folio, line)
    if k != cur:
        if cur is not None: line_items.append((cur, cnt))
        cur = k; cnt = 0
    cnt += 1
line_items.append((cur, cnt))

def mutate(w, seed):
    if not w: return "o"
    i = seed % len(w); g = GLYPHS[(seed // max(len(w),1)) % len(GLYPHS)]
    return w[:i] + g + w[i:]

def llull_stateful():
    out = []; folorder = {}; prev = None; pci = 0; oc = 0
    for (folio, line), count in line_items:
        g = folorder.setdefault(folio, len(folorder))
        oc = (oc + g*13) % len(CORE)                       # page jump (carry persists)
        for w in range(count):
            if prev and (g*line + w) % 40 == 0:
                tok = prev
            elif prev and (g + line + w) % 22 == 0:
                tok = mutate(prev, g*7 + line*3 + w)
            else:
                oc = (oc + 1 + (pci % 3)) % len(CORE)       # core advances; step depends on prev core
                op = (pci*2 + line) % len(PRE)              # prefix steered by previous word
                os = (pci*3 + oc) % len(SUF)               # suffix steered by prev + current core
                tok = (PRE[op] + CORE[oc] + SUF[os]) or "o"
                pci = oc                                    # state = last core index
            out.append((folio, line, w, tok)); prev = tok
    return out

real_items = [(r[0], r[1], r[2], r[6]) for r in rows]
gen_items = llull_stateful()

def Hf(c):
    t = sum(c.values()); return -sum(v/t*math.log2(v/t) for v in c.values()) if t else 0.0

def exch_signal(tk, N=150, K=10, passes=6, seed=1):
    rnd = random.Random(seed)
    c = Counter(tk); top = [w for w, _ in c.most_common(N)]; tid = {w: i for i, w in enumerate(top)}
    OT = N; red = [tid.get(w, OT) for w in tk]
    def mi(seq): return 2*Hf(Counter(seq)) - Hf(Counter(zip(seq, seq[1:])))
    left = defaultdict(Counter); right = defaultdict(Counter)
    for i in range(1, len(red)):
        a, b = red[i-1], red[i]
        if b != OT: left[b][a] += 1
        if a != OT: right[a][b] += 1
    mov = sorted(t for t in set(red) if t != OT); cls = {t: rnd.randrange(K) for t in mov}
    def cK(t): return K if t == OT else cls[t]
    Kt = K+1; CB = [[0]*Kt for _ in range(Kt)]
    for i in range(1, len(red)): CB[cK(red[i-1])][cK(red[i])] += 1
    def cbmi():
        tot = sum(sum(r) for r in CB)
        if not tot: return 0.0
        row = [sum(r) for r in CB]; col = [sum(CB[x][y] for x in range(Kt)) for y in range(Kt)]
        return sum(CB[x][y]/tot*math.log2(CB[x][y]*tot/(row[x]*col[y]))
                   for x in range(Kt) for y in range(Kt) if CB[x][y] and row[x] and col[y])
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
            for b in range(K):
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

print(f"{'system':<16}{'toks':>7}{'types':>7}{'wlen':>6}{'rep':>8}{'drift':>7}{'h2/h1':>7}{'pageloc':>8}{'lineeff':>8}{'brownS':>8}")
print("-"*94)
scoreboard("Real Voynich", real_items)
scoreboard("Llull-stateful", gen_items)
print("\n(memoryless Llull last round: brownS 0.019, lineeff 0.93, pageloc 4.20)")
