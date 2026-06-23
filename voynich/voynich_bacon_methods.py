#!/usr/bin/env python3
"""M-Bacon: apply Roger Bacon's concealment methods to real Latin and ask whether ANY
of them moves Latin toward the Voynich's statistical signature.

Bacon's Epistola de secretis operibus lists ~7 ways to hide knowledge. Most are not
quantifiable on a glyph stream (figurative language, geometric symbols). Three ARE
mechanical transforms we can apply and measure:
  (2) omit the vowels        -> abjad / consonantal skeleton
  (3/4) exotic/invented alphabet -> 1:1 substitution (relabel glyphs)
  (5) shorthand/abbreviation -> drop interior vowels + collapse doubled letters
  (combo) vowel-drop then substitution

We transform Latin (Apicius, a real herbal/recipe text) and compare every output to
the Voynich on four axes:
  wlen  mean token length
  TTR   type/token ratio (vocabulary richness)
  Hc    conditional char entropy  H(c2|c1) = H(bigram) - H(unigram)   [grammar/rigidity]
  bpc   lzma bits-per-char        [redundancy: low = repetitive/compressible]
Voynich is SHORT, LOW-Hc (rigid), LOW-bpc (redundant). The question: can a reversible
surface cipher manufacture that profile, or does it need a generative/copying process?

Run `python fetch_data.py` first to download the corpora into ./data/.
"""
import lzma, math, random, re, pathlib
from collections import Counter
from voynich_lib import parse

ROOT = pathlib.Path(__file__).resolve().parent
def latin(p):
    return re.findall(r"[a-z]+", (ROOT/p).read_text(encoding="utf-8", errors="ignore").lower())

VOY = [r[6] for r in parse()]
APIC = latin("data/apicius_books.txt")

VOW = set("aeiou")
def drop_vowels(words):
    out = []
    for w in words:
        s = "".join(c for c in w if c not in VOW)
        if s: out.append(s)
    return out
def abbreviate(words):                       # keep 1st letter, drop interior vowels, collapse doubles
    out = []
    for w in words:
        if not w: continue
        s = w[0] + "".join(c for c in w[1:] if c not in VOW)
        s = re.sub(r"(.)\1+", r"\1", s)
        if s: out.append(s)
    return out
def substitute(words, seed=0):               # reversible 1:1 relabel (exotic/invented alphabet)
    alpha = sorted({c for w in words for c in w}); shuf = alpha[:]
    random.Random(seed).shuffle(shuf); m = dict(zip(alpha, shuf))
    return ["".join(m[c] for c in w) for w in words]

def metrics(words):
    n = len(words); chars = "".join(words)
    wlen = sum(len(w) for w in words)/n
    ttr = len(set(words))/n
    uni = Counter(chars); bi = Counter(zip(chars, chars[1:]))
    def H(c):
        t = sum(c.values()); return -sum(v/t*math.log2(v/t) for v in c.values()) if t else 0.0
    Hc = H(bi) - H(uni)                       # conditional entropy of next char given prev
    raw = chars.encode()
    bpc = 8*len(lzma.compress(raw, preset=6))/len(raw)
    return dict(wlen=wlen, TTR=ttr, Hc=Hc, bpc=bpc)

streams = {
    "VOYNICH (target)":        VOY,
    "Latin (raw)":             APIC,
    "Latin + vowel-drop":      drop_vowels(APIC),
    "Latin + substitution":    substitute(APIC),
    "Latin + abbreviation":    abbreviate(APIC),
    "Latin + vdrop+subst":     substitute(drop_vowels(APIC)),
}
M = {k: metrics(v) for k, v in streams.items()}
keys = ["wlen", "TTR", "Hc", "bpc"]
print(f"{'stream':<24}" + "".join(f"{k:>9}" for k in keys))
for name, m in M.items():
    print(f"{name:<24}" + "".join(f"{m[k]:>9.3f}" for k in keys))

# distance to Voynich, normalized per-metric by Voynich value, averaged
V = M["VOYNICH (target)"]
print("\ndistance to Voynich (mean |Δ|/V over the 4 metrics; lower = more Voynich-like):")
rank = []
for name, m in M.items():
    if name.startswith("VOYNICH"): continue
    d = sum(abs(m[k]-V[k])/abs(V[k]) for k in keys)/len(keys)
    rank.append((d, name))
for d, name in sorted(rank):
    print(f"  {name:<24}{d:>7.3f}")
