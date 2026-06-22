#!/usr/bin/env python3
"""Milestone 4: does the long-range MI tail survive a STRONGER generator?

The order-3 Markov baseline was weak. Here we test two stronger nulls, both of
which reproduce Voynich structure WITHOUT any semantic content, and re-measure
long-range character mutual information I(d):

  SLOT+LINE generator  -- a slot-grammar word model: P(char | position-in-word,
                          position-in-line, prev-char), word lengths sampled per
                          line-position, generated line-by-line matching each real
                          line's word count. Captures morphology + line/LAAFU
                          structure, but generates NOVEL words (no vocabulary memory).
  FOLIO word-shuffle    -- keeps each folio's exact word multiset but shuffles word
                          ORDER within the folio. Preserves page vocabulary, destroys
                          sequence. (Decomposition: is the within-page MI tail just
                          word-reuse, or real order structure?)

If SLOT+LINE reproduces REAL's I(d) tail -> the long-range signal is positional/
morphological rule structure. If FOLIO-shuffle reproduces REAL|folio's tail ->
the within-page tail is just vocabulary reuse, not order. If REAL beats both ->
structure beyond morphology, line layout, and page vocabulary.
"""
import math, random
from collections import Counter, defaultdict
from voynich_lib import parse

random.seed(7)
rows = parse()

# group into lines: [(folio, [words]), ...] in reading order
lines = []
cur_key, cur_words = None, []
for folio, line, pos, L, Hh, I, tok in rows:
    if (folio, line) != cur_key:
        if cur_words: lines.append((cur_key[0], cur_words))
        cur_key, cur_words = (folio, line), []
    cur_words.append(tok)
if cur_words: lines.append((cur_key[0], cur_words))

def stream_from_lines(lns):
    chars, fols = [], []
    for folio, words in lns:
        for w in words:
            if chars: chars.append(" "); fols.append(folio)
            for c in w: chars.append(c); fols.append(folio)
    return chars, fols

real_chars, real_fols = stream_from_lines(lines)
N = len(real_chars)
DS = [1, 5, 8, 13, 21, 34, 55, 89, 144]

def mi(seq, d, same=False, fol=None):
    px = Counter(seq); tot = len(seq); joint = Counter(); n = 0
    for i in range(len(seq) - d):
        if same and fol[i] != fol[i+d]: continue
        joint[(seq[i], seq[i+d])] += 1; n += 1
    if not n: return 0.0
    return sum((c/n) * math.log2((c/n) / ((px[x]/tot)*(px[y]/tot))) for (x, y), c in joint.items())

# ---- train SLOT+LINE model ----
def wbucket(j, L): return "F" if j == 0 else ("L" if j == L-1 else "M")
m3 = defaultdict(Counter); m2 = defaultdict(Counter); m1 = defaultdict(Counter); g0 = Counter()
lens = defaultdict(Counter)
for folio, words in lines:
    for wi, w in enumerate(words):
        lp = "F" if wi == 0 else ("L" if wi == len(words)-1 else "M")
        lens[lp][len(w)] += 1
        prev = "^"
        for j, c in enumerate(w):
            wp = wbucket(j, len(w))
            m3[(lp, wp, prev)][c] += 1; m2[(wp, prev)][c] += 1; m1[prev][c] += 1; g0[c] += 1
            prev = c

def pick(counter):
    tot = sum(counter.values()); r = random.uniform(0, tot); acc = 0
    for k, v in counter.items():
        acc += v
        if r <= acc: return k
    return next(iter(counter))

def gen_slotline():
    out = []
    for folio, words in lines:
        new = []
        for wi, w in enumerate(words):
            lp = "F" if wi == 0 else ("L" if wi == len(words)-1 else "M")
            L = pick(lens[lp]) or 1
            chars = []; prev = "^"
            for j in range(L):
                wp = wbucket(j, L)
                ctx = m3.get((lp, wp, prev)) or m2.get((wp, prev)) or m1.get(prev) or g0
                c = pick(ctx); chars.append(c); prev = c
            new.append("".join(chars))
        out.append((folio, new))
    return out

def gen_folioshuffle():
    byf = defaultdict(list)
    for folio, words in lines: byf[folio] += words
    for f in byf: random.shuffle(byf[f])
    out = []; idx = defaultdict(int)
    for folio, words in lines:
        k = len(words); new = byf[folio][idx[folio]:idx[folio]+k]; idx[folio] += k
        out.append((folio, new))
    return out

sl_c, sl_f = stream_from_lines(gen_slotline())
fs_c, fs_f = stream_from_lines(gen_folioshuffle())

print(f"streams ~{N} chars\n")
print(f"{'d':>5}{'REAL':>9}{'SlotLine':>10}{'FolioShuf':>11}{'  |':>4}{'REAL|f':>9}{'SlotLn|f':>10}{'FolShuf|f':>11}")
for d in DS:
    print(f"{d:>5}{mi(real_chars,d):>9.4f}{mi(sl_c,d):>10.4f}{mi(fs_c,d):>11.4f}{'  |':>4}"
          f"{mi(real_chars,d,True,real_fols):>9.4f}{mi(sl_c,d,True,sl_f):>10.4f}{mi(fs_c,d,True,fs_f):>11.4f}")
print("\nbits MI. SlotLine = morphology+line rules, novel words. FolioShuf = page vocab, no order.")
