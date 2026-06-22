#!/usr/bin/env python3
"""Milestone 3: long-range CHARACTER mutual information vs distance.

A second, independent long-range signal (Montemurro-Pury style). For glyphs
separated by distance d:  I(d) = sum p(x,y;d) log2[ p(x,y;d) / (p(x)p(y)) ].

Natural language shows long-range correlations that persist (slow decay) well
above a local-Markov baseline. A purely local generator decays fast to the
shuffle floor. We compare:
  - REAL              (whole glyph stream, reading order, words joined by space)
  - full shuffle      (correlation floor / finite-size bias)
  - char-Markov(3)    (matched local structure; excess of REAL over this = long-range)
And the CONTROLLED version, counting only pairs inside the SAME folio, so a
real-vs-Markov excess can't be just page-to-page vocabulary drift.
"""
import math, random
from collections import Counter, defaultdict
from voynich_lib import parse

random.seed(7)
rows = parse()
# glyph stream with folio id per glyph; words joined by space
chars, fols = [], []
last_f = None
for folio, line, pos, L, Hh, I, tok in rows:
    if chars: chars.append(" "); fols.append(folio)   # space between tokens
    for c in tok: chars.append(c); fols.append(folio)
N = len(chars)
DS = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

def mi(seq, d, same_folio=False, fol=None):
    px = Counter(seq); tot = len(seq)
    joint = Counter()
    n = 0
    for i in range(len(seq) - d):
        if same_folio and fol[i] != fol[i+d]: continue
        joint[(seq[i], seq[i+d])] += 1; n += 1
    if n == 0: return 0.0
    I = 0.0
    for (x, y), c in joint.items():
        pxy = c / n
        I += pxy * math.log2(pxy / ((px[x]/tot) * (px[y]/tot)))
    return I

def char_markov(order=3):
    nxt = defaultdict(list)
    for i in range(N - order):
        nxt[tuple(chars[i:i+order])].append(chars[i+order])
    out = list(chars[:order])
    while len(out) < N:
        k = tuple(out[-order:]); nx = nxt.get(k)
        out.append(random.choice(nx) if nx else random.choice(chars))
    return out[:N]

sh = chars[:]; random.shuffle(sh)
mk = char_markov(3)

print(f"glyph stream: {N} chars, alphabet {len(set(chars))}\n")
print(f"{'d':>5}{'REAL':>9}{'shuffle':>9}{'Markov3':>9}{'REAL|folio':>12}{'Mk3|folio':>11}")
for d in DS:
    r  = mi(chars, d)
    s  = mi(sh, d)
    m  = mi(mk, d)
    rf = mi(chars, d, True, fols)
    mf = mi(mk, d, True, fols)
    print(f"{d:>5}{r:>9.4f}{s:>9.4f}{m:>9.4f}{rf:>12.4f}{mf:>11.4f}")
print("\nbits of mutual information. REAL>>Markov at large d => long-range structure beyond local rules.")
print("REAL|folio vs Mk3|folio isolates within-page long-range signal (controls page vocabulary drift).")
