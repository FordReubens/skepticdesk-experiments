#!/usr/bin/env python3
"""Milestone 1 of the Voynich controlled-structure experiment.

Parse the ZL EVA IVTFF transliteration into tokens tagged with their controls
(folio, line, in-line position, Currier language $L, scribal hand $H, section $I),
then REPRODUCE the known statistical anomalies as a pipeline sanity check:
  - adjacent-word repetition rate (known: high vs natural language)
  - one-edit-neighbour rate (the "drift by one glyph" signature)
  - character conditional entropy h1/h2 (known: h2 ~2 bits, unusually low)
If these match the literature, the parser is trustworthy for the controlled tests.
"""
import re, math, pathlib
from collections import Counter

SRC = pathlib.Path(__file__).resolve().parent / "data/ZL3b-n.txt"
TAG = re.compile(r"<[^>]*>")            # inline tags: <%>, <$>, <!...>, <->, <~>
ALT = re.compile(r"\[([^:\]]*)[:][^\]]*\]")  # [a:b] uncertain reading -> first option
HEAD = re.compile(r"^<(f[^.>]+)>\s+<!\s*(.*?)>")    # page header w/ metadata
LOCUS = re.compile(r"^<(f[^.>]+)\.(\d+)[^>]*>\s*(.*)$")  # locus line: folio.line + text

def meta(s):
    return dict(re.findall(r"\$([A-Z])=([^\s>]+)", s))

def clean_tokens(text):
    text = TAG.sub("", text)
    text = ALT.sub(r"\1", text)
    text = text.replace("!", "").replace("%", "")
    toks = re.split(r"[.,]", text)
    out = []
    for t in toks:
        t = t.strip()
        if t and re.fullmatch(r"[a-z]+", t):   # drop illegible/uncertain (*, ?, etc.)
            out.append(t)
    return out

def lev1(a, b):
    """True iff Levenshtein(a,b) == 1 (one substitution/insertion/deletion)."""
    if a == b: return False
    la, lb = len(a), len(b)
    if abs(la - lb) > 1: return False
    if la == lb:
        return sum(x != y for x, y in zip(a, b)) == 1
    if la > lb: a, b, la, lb = b, a, lb, la   # ensure a shorter
    i = j = 0; diff = 0
    while i < la and j < lb:
        if a[i] == b[j]: i += 1; j += 1
        else:
            diff += 1; j += 1
            if diff > 1: return False
    return True

rows = []   # (folio, line, pos, L, H, I, token)
cur = {"f": None, "L": "?", "H": "?", "I": "?"}
for raw in SRC.read_text(encoding="utf-8", errors="ignore").splitlines():
    if raw.startswith("#") or not raw.strip():
        continue
    m = HEAD.match(raw)
    if m:
        md = meta(m.group(2))
        cur = {"f": m.group(1), "L": md.get("L", "?"), "H": md.get("H", "?"), "I": md.get("I", "?")}
        continue
    m = LOCUS.match(raw)
    if m:
        folio, line, text = m.group(1), int(m.group(2)), m.group(3)
        for pos, tok in enumerate(clean_tokens(text)):
            rows.append((folio, line, pos, cur["L"], cur["H"], cur["I"], tok))

toks = [r[6] for r in rows]
N = len(toks)
print(f"parsed: {N} tokens, {len(set(toks))} types, {len(set(r[0] for r in rows))} folios")
print(f"controls present -> Currier $L: {sorted(set(r[3] for r in rows))} | "
      f"hands $H: {sorted(set(r[4] for r in rows))} | sections $I: {sorted(set(r[5] for r in rows))}")

# --- adjacent repetition & one-edit-neighbour (within line, in reading order) ---
pairs = same = oneoff = 0
prev = None; prev_key = None
for folio, line, pos, L, H, I, tok in rows:
    key = (folio, line)
    if prev is not None and key == prev_key:
        pairs += 1
        if tok == prev: same += 1
        elif lev1(tok, prev): oneoff += 1
    prev, prev_key = tok, key
print(f"\nadjacent-word repetition: {same/pairs*100:.2f}%  (natural language typically <0.5%)")
print(f"one-edit-neighbour rate : {oneoff/pairs*100:.2f}%  (the 'mutate by one glyph' drift)")
print(f"either (repeat or 1-edit): {(same+oneoff)/pairs*100:.2f}%  of adjacent pairs")

# --- character conditional entropy over the glyph stream (letters + word-space) ---
stream = " ".join(toks)
uni = Counter(stream); tot = len(stream)
h1 = -sum(c/tot * math.log2(c/tot) for c in uni.values())
bg = Counter(zip(stream, stream[1:])); tb = sum(bg.values())
h_joint = -sum(c/tb * math.log2(c/tb) for c in bg.values())
h2 = h_joint - h1   # H(next char | current char)
mw = sum(len(t) for t in toks) / N
print(f"\nmean word length        : {mw:.2f} glyphs")
print(f"char entropy h1         : {h1:.2f} bits/char")
print(f"char conditional h2     : {h2:.2f} bits/char  (Voynich known ~2.0; English letters ~3.0-3.6)")
