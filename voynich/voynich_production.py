#!/usr/bin/env python3
"""M22: production signatures (GPT-5.5's battlefield), from transcription metadata.
A) rare-token ancestry: are rare words born by one-glyph mutation from a RECENT word?
B) do scribal hands ($H) preserve the same recurrence rule?
"""
import random
from collections import Counter
from voynich_lib import parse, lev1

rows = parse()
toks = [r[6] for r in rows]
hands = [r[4] for r in rows]
counts = Counter(toks); N = len(toks); W = 100

def ancestry(seq, only_rare):
    hit = tot = 0
    for i in range(W, len(seq)):
        w = seq[i]
        if only_rare and counts[w] > 2: continue
        tot += 1
        if any(lev1(w, x) for x in seq[i-W:i]): hit += 1
    return (hit/tot*100 if tot else 0), tot

sh = toks[:]; random.Random(1).shuffle(sh)
print("A) RARE-TOKEN ANCESTRY: % with a 1-glyph 'parent' in the previous", W, "words")
for label, only_rare in (("all tokens", False), ("rare types (count<=2)", True)):
    r, rt = ancestry(toks, only_rare); s, _ = ancestry(sh, only_rare)
    print(f"   {label:<22} real {r:5.1f}%   shuffled {s:5.1f}%   lift x{r/s:.1f}" if s else f"   {label}: real {r:.1f}%")

print("\nB) PER-HAND RECURRENCE (does each scribe copy-and-mutate the same way?)")
print(f"   {'hand':<6}{'tokens':>8}{'repeat':>8}{'drift':>8}{'rare-ancestry':>15}")
for h in sorted(set(hands)):
    ht = [toks[i] for i in range(N) if hands[i] == h]
    if len(ht) < 1000: continue
    pr = same = one = 0
    for a, b in zip(ht, ht[1:]):
        pr += 1; same += (a == b); one += (lev1(a, b))
    # rare ancestry within this hand's stream
    hit = tot = 0
    for i in range(W, len(ht)):
        if counts[ht[i]] > 2: continue
        tot += 1
        if any(lev1(ht[i], x) for x in ht[i-W:i]): hit += 1
    ra = hit/tot*100 if tot else 0
    print(f"   {h:<6}{len(ht):>8}{same/pr*100:>7.1f}%{one/pr*100:>7.1f}%{ra:>14.1f}%")
