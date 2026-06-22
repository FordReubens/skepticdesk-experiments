#!/usr/bin/env python3
"""Shared Voynich parsing: ZL EVA IVTFF -> tokens tagged with controls.
Each row: (folio, line, pos_in_line, currier_L, hand_H, section_I, token)."""
import re, pathlib
SRC = pathlib.Path(__file__).resolve().parent / "data/ZL3b-n.txt"
_TAG = re.compile(r"<[^>]*>")
_ALT = re.compile(r"\[([^:\]]*)[:][^\]]*\]")
_HEAD = re.compile(r"^<(f[^.>]+)>\s+<!\s*(.*?)>")
_LOCUS = re.compile(r"^<(f[^.>]+)\.(\d+)[^>]*>\s*(.*)$")

def _meta(s): return dict(re.findall(r"\$([A-Z])=([^\s>]+)", s))

def clean_tokens(text):
    text = _TAG.sub("", text); text = _ALT.sub(r"\1", text)
    text = text.replace("!", "").replace("%", "")
    out = []
    for t in re.split(r"[.,]", text):
        t = t.strip()
        if t and re.fullmatch(r"[a-z]+", t): out.append(t)
    return out

def lev1(a, b):
    if a == b: return False
    la, lb = len(a), len(b)
    if abs(la - lb) > 1: return False
    if la == lb: return sum(x != y for x, y in zip(a, b)) == 1
    if la > lb: a, b, la, lb = b, a, lb, la
    i = j = diff = 0
    while i < la and j < lb:
        if a[i] == b[j]: i += 1; j += 1
        else:
            diff += 1; j += 1
            if diff > 1: return False
    return True

def parse(path=SRC):
    rows = []; cur = {"f": None, "L": "?", "H": "?", "I": "?"}
    for raw in pathlib.Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        if raw.startswith("#") or not raw.strip(): continue
        m = _HEAD.match(raw)
        if m:
            md = _meta(m.group(2))
            cur = {"f": m.group(1), "L": md.get("L", "?"), "H": md.get("H", "?"), "I": md.get("I", "?")}
            continue
        m = _LOCUS.match(raw)
        if m:
            folio, line, text = m.group(1), int(m.group(2)), m.group(3)
            for pos, tok in enumerate(clean_tokens(text)):
                rows.append((folio, line, pos, cur["L"], cur["H"], cur["I"], tok))
    return rows
