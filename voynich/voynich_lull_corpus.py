#!/usr/bin/env python3
"""M-Lull-corpus: does genuine pseudo-Lullian ALCHEMICAL Latin sit any closer to the
Voynich than ordinary Latin? Closes the Franciscan/Lullist authorship thread empirically.

If the Voynich were Lullist-tradition text (or a cipher of it), we'd hope its word
structure resembled real Lullist alchemical Latin more than classical or culinary
Latin. We compare three real Latin corpora to the Voynich on the same axes as
voynich_bacon_methods.py:
  wlen  mean token length
  TTR   type/token ratio
  Hc    conditional char entropy H(c2|c1)   [grammar/rigidity -- the Voynich tell]
  bpc   lzma bits/char                       [redundancy]

Corpora (run `python fetch_data.py` first):
  classical  = Caesar, De Bello Gallico (Gutenberg 218)            latin_dbg.txt
  culinary   = Apicius, De re coquinaria (recipe/herbal register)  apicius_books.txt
  alchemical = pseudo-Lullian *Testamentum* (Cologne 1566 OCR)     pseudolull_testamentum.txt

NOTE: the alchemical text is OCR of 16th-c. print -- long-s reads as 'f', ligatures,
running heads. OCR noise tends to RAISE entropy, so if even this corpus lands far from
the Voynich, the conclusion (Latin morphology != Voynich) only gets safer.
"""
import lzma, math, re, pathlib
from collections import Counter
from voynich_lib import parse

ROOT = pathlib.Path(__file__).resolve().parent
# normalise common early-modern / medieval glyphs before extracting words
TRANS = str.maketrans({"ſ": "s", "ꝛ": "r", "ꝰ": "s", "æ": "a", "œ": "o",
                       "õ": "o", "ũ": "u", "ã": "a", "ẽ": "e", "ĩ": "i",
                       "ꝑ": "p", "ꝓ": "p", "ꝯ": "c", "—": " "})
# OCR / scan boilerplate to drop (English + library furniture)
STOP = set("the and of to in for by with internet archive google digitized copyright "
           "proquest llc library images courtesy wellcome trust london european books "
           "page early reproduced significat".split())
def load(p):
    t = (ROOT/"data"/p).read_text(encoding="utf-8", errors="ignore").translate(TRANS).lower()
    return [w for w in re.findall(r"[a-z]+", t) if len(w) >= 2 and w not in STOP]

VOY = [r[6] for r in parse()]
corpora = {
    "VOYNICH (target)":             VOY,
    "Latin: classical (Caesar)":    load("latin_dbg.txt"),
    "Latin: culinary (Apicius)":    load("apicius_books.txt"),
    "Latin: alchemical (ps-Llull)": load("pseudolull_testamentum.txt"),
}

def metrics(words):
    n = len(words); chars = "".join(words)
    wlen = sum(len(w) for w in words)/n
    ttr = len(set(words))/n
    uni = Counter(chars); bi = Counter(zip(chars, chars[1:]))
    def H(c):
        t = sum(c.values()); return -sum(v/t*math.log2(v/t) for v in c.values()) if t else 0.0
    Hc = H(bi) - H(uni)
    raw = chars.encode()
    bpc = 8*len(lzma.compress(raw, preset=6))/len(raw)
    return dict(n=n, wlen=wlen, TTR=ttr, Hc=Hc, bpc=bpc)

M = {k: metrics(v) for k, v in corpora.items()}
keys = ["n", "wlen", "TTR", "Hc", "bpc"]
print(f"{'corpus':<30}" + "".join(f"{k:>9}" for k in keys))
for name, m in M.items():
    print(f"{name:<30}" + f"{m['n']:>9d}" + "".join(f"{m[k]:>9.3f}" for k in keys[1:]))

V = M["VOYNICH (target)"]
mk = ["wlen", "TTR", "Hc", "bpc"]
print("\ndistance to Voynich (mean |Δ|/V over wlen,TTR,Hc,bpc; lower = closer):")
for dist, nm in sorted((sum(abs(M[n][k]-V[k])/abs(V[k]) for k in mk)/len(mk), n)
                       for n in M if not n.startswith("VOYNICH")):
    print(f"  {nm:<30}{dist:>7.3f}")
print("\nHc gap to Voynich (the rigidity tell; Voynich Hc = %.3f):" % V["Hc"])
for name in corpora:
    if not name.startswith("VOYNICH"):
        print(f"  {name:<30}{M[name]['Hc']-V['Hc']:>+7.3f}")
