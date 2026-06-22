#!/usr/bin/env python3
"""M12 (Ford's birdsong angle): how does animal song fit the entropy profile?

Compare the atomic-symbol statistics of Bengalese finch song (syllable sequences)
against Voynich (glyph sequences) and Latin (letter sequences). All treated the
same way: a sequence of atomic symbols within its natural unit (song bout / line /
text). Because alphabet sizes differ, the comparable measures are the NORMALIZED
ones: h2/h1 (fraction of per-symbol uncertainty left after the previous symbol)
and adjacent-repeat rate.
"""
import re, math, pathlib
from collections import Counter
from voynich_lib import parse

ROOT = pathlib.Path(__file__).resolve().parent

def stats(name, seqs):
    uni = Counter(); big = Counter(); rep = 0; pairs = 0
    for s in seqs:
        for x in s: uni[x] += 1
        for a, b in zip(s, s[1:]):
            big[(a, b)] += 1; pairs += 1; rep += (a == b)
    tu = sum(uni.values()); tb = sum(big.values())
    h1 = -sum(c/tu*math.log2(c/tu) for c in uni.values())
    hb = -sum(c/tb*math.log2(c/tb) for c in big.values())
    h2 = hb - h1
    A = len(uni)
    print(f"{name:<22}{A:>6}{tu:>9}{h1:>8.2f}{h2:>8.2f}{h2/h1:>8.2f}{rep/pairs*100:>9.1f}%")

# --- Bengalese finch: syllable-label sequences per song bout ---
xml = (ROOT/"data/birdsong_Bird0.xml").read_text(encoding="utf-8", errors="ignore")
bird = []
for seq in re.findall(r"<Sequence>(.*?)</Sequence>", xml, re.S):
    labels = re.findall(r"<Label>([^<]*)</Label>", seq)
    if len(labels) >= 2: bird.append(labels)

# --- Voynich: glyph sequence per line (words' glyphs concatenated, no spaces) ---
rows = parse()
vlines = {}
for folio, ln, pos, L, Hh, I, tok in rows:
    vlines.setdefault((folio, ln), []).append(tok)
voy = ["".join(ws) for ws in vlines.values()]            # each line -> glyph string
voy = [list(s) for s in voy if len(s) >= 2]

def latin_letters(path, g=False):
    t = (ROOT/path).read_text(encoding="utf-8", errors="ignore")
    if g:
        a = t.find("*** START"); b = t.find("*** END")
        if a >= 0: t = t[t.find("\n", a)+1:(b if b > 0 else len(t))]
    return [list(re.sub(r"[^a-z]", "", t.lower()))]        # one long letter stream

print(f"{'system':<22}{'alpha':>6}{'symbols':>9}{'h1':>8}{'h2':>8}{'h2/h1':>8}{'rep':>10}")
print("-"*72)
stats("Bengalese finch song", bird)
stats("Voynich (glyphs)", voy)
stats("Latin Caesar (letters)", latin_letters("data/latin_dbg.txt", True))
stats("Latin Apicius (letters)", latin_letters("data/apicius_books.txt"))
print("\nh2/h1 = fraction of per-symbol uncertainty remaining after the previous symbol")
print("(lower = more rule-bound/predictable). rep = adjacent identical-symbol rate.")
print(f"\n[birdsong: {len(bird)} bouts]")
