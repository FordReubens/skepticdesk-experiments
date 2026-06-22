#!/usr/bin/env python3
"""M14: period-plausible Llull-wheel simulator + adversarial scoreboard + info budget.

A Llull paper machine = concentric wheels (prefix / core / suffix). Word at
(page g, line l, word w) is read by rotating each wheel to an offset that is a
fixed linear function of g, l, w (the "gear ratios") -- NO per-word randomness,
NO tuning against the scoreboard. Two mechanical imperfections: the wheel
occasionally sticks (repeat) or slips (one-glyph change), at fixed rates.

We then ask GPT-5.5's adversarial question: does this low-DOF device reproduce
features it was never shown -- page-local vocabulary, line-initial effects, and
held-out Brown class sequencing -- not just word length / entropy?

Finally: the information-budget check for the decryption question.
"""
import math
from collections import Counter, defaultdict
from voynich_lib import parse, lev1

rows = parse()
toks = [r[6] for r in rows]
GLYPHS = sorted({c for w in toks for c in w})

# --- wheel inventories from the manuscript (alphabet is observable; this is not tuning) ---
PRE_SET = sorted({p for p in ["qo","qok","ch","sh","cth","o","y","d","s","q","l","k",""]})
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
CORE = [c for c, _ in cores.most_common(120)]      # large wheel -> page samples a window
SUF = [s for s, _ in sufs.most_common(16)]

# --- real layout: words per (folio,line), in reading order ---
line_items = []
seen = {}
cur = None; cnt = 0
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

def llull():
    out = []; folorder = {}; prev = None
    for (folio, line), count in line_items:
        g = folorder.setdefault(folio, len(folorder))
        for w in range(count):
            if prev and (g*line + w) % 40 == 0:           # wheel sticks -> repeat
                tok = prev
            elif prev and (g + line + w) % 22 == 0:        # wheel slips -> one-glyph change
                tok = mutate(prev, g*7 + line*3 + w)
            else:
                p = PRE[(3*g + line + w) % len(PRE)]
                c = CORE[(7*g + 2*line + w) % len(CORE)]
                s = SUF[(5*g + 3*line + 2*w) % len(SUF)]
                tok = (p + c + s) or "o"
            out.append((folio, line, w, tok)); prev = tok
    return out

# real items in the same (folio,line,pos,tok) shape
real_items = [(r[0], r[1], r[2], r[6]) for r in rows]
gen_items = llull()

def H(c):
    t = sum(c.values()); return -sum(v/t*math.log2(v/t) for v in c.values()) if t else 0.0

def exch_signal(tk, N=150, K=10, passes=6, seed=1):
    import random as _r
    rnd = _r.Random(seed)
    c = Counter(tk); top = [w for w, _ in c.most_common(N)]; tid = {w: i for i, w in enumerate(top)}
    OT = N; red = [tid.get(w, OT) for w in tk]
    def mi(seq):
        uni = Counter(seq); big = Counter(zip(seq, seq[1:]));
        return 2*H(uni) - H(big)
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
        tot = sum(sum(r) for r in CB);
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
    real = mi([cK(x) for x in red])
    sh = red[:]; rnd.shuffle(sh)
    return real - mi([cK(x) for x in sh])

def scoreboard(name, items):
    tk = [x[3] for x in items]; N = len(tk); types = len(set(tk))
    mwl = sum(len(w) for w in tk)/N
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
    h1 = H(uni); h2 = H(big) - h1
    # page-loc
    fl = [x[0] for x in items]; pagesz = Counter(fl); Nt = len(fl); pf = {f: pagesz[f]/Nt for f in pagesz}
    wp = defaultdict(Counter); c = Counter(tk)
    for w, f in zip(tk, fl): wp[w][f] += 1
    num = den = 0.0
    for w in c:
        if c[w] >= 10:
            n = c[w]; kl = sum((cf/n)*math.log2((cf/n)/pf[f]) for f, cf in wp[w].items()); num += n*kl; den += n
    ploc = num/den if den else 0
    # line effect: KL(line-initial token dist || interior token dist) over eligible
    ini = Counter(); inter = Counter()
    for f, l, p, w in items:
        (ini if p == 0 else inter)[w] += 1
    elig = {w for w in c if c[w] >= 10}
    ti = sum(ini[w] for w in elig) or 1; tr = sum(inter[w] for w in elig) or 1
    le = sum((ini[w]/ti)*math.log2((ini[w]/ti)/((inter[w]/tr) or 1e-9)) for w in elig if ini[w])
    bs = exch_signal(tk)
    print(f"{name:<14}{N:>7}{types:>7}{mwl:>6.2f}{same/pairs*100:>7.1f}%{one/pairs*100:>7.1f}%{h2/h1:>7.2f}{ploc:>8.2f}{le:>8.2f}{bs:>8.3f}")

print(f"wheels: {len(PRE)} prefixes x {len(CORE)} cores x {len(SUF)} suffixes  (gear-driven, untuned)\n")
print(f"{'system':<14}{'toks':>7}{'types':>7}{'wlen':>6}{'rep':>8}{'drift':>7}{'h2/h1':>7}{'pageloc':>8}{'lineeff':>8}{'brownS':>8}")
print("-"*92)
scoreboard("Real Voynich", real_items)
scoreboard("Llull-wheel", gen_items)

# --- information budget for the decryption question ---
print("\n--- information budget (is there room to encode a real book?) ---")
glyphs = sum(len(w) for w in toks);
ug = Counter(c for w in toks for c in w); bg = Counter()
for w in toks:
    for a, b in zip(w, w[1:]): bg[(a, b)] += 1
h1g = H(ug); h2g = H(bg) - h1g
for label, rate in (("h1 (no context)", h1g), ("h2 (1-glyph context)", h2g), ("conservative 1.0 b/gl", 1.0)):
    bits = glyphs * rate
    pchars = bits / 1.1            # natural-language ~1.1 bits/char (Shannon)
    print(f"  at {label:<22} -> {bits/8/1024:6.0f} KB info -> ~{pchars/5.5:6.0f} plaintext words "
          f"(~{pchars/5.5/300:.0f} book-pages)")
print("  (a real ~230pp book is ~60-90k words; compare above.)")
