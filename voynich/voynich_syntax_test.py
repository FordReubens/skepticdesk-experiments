#!/usr/bin/env python3
"""M9: is the shared layer SYNTACTIC or merely STRUCTURAL?

Three probes, no semantics needed:

P1  Line-position slots. How predictable is each line slot (initial / 2nd /
    interior / penult / final) vs a within-line shuffle null? Strong line-edge
    structure is real -- but note: in ordinary language line position is
    grammatically irrelevant, so line-driven structure leans STRUCTURAL/layout,
    not syntactic.

P2  Section-specific role-players (GPT-5.5's #5). Is the line-initial (and final)
    slot owned by the SAME token across sections (universal, like Latin 'et'), or
    by DIFFERENT tokens playing the same positional role (grammar with a
    section-specific lexicon)?

P3  Transition predictability. Word-level conditional entropy H(next|current),
    Voynich vs Latin. Lower = more frame-like / formulaic transitions.
"""
import re, math, pathlib, random
from collections import Counter, defaultdict
from voynich_lib import parse

random.seed(7)
ROOT = pathlib.Path(__file__).resolve().parent
rows = parse()

# group into lines
lines = []   # (section, [words])
key, buf = None, []
for folio, ln, pos, L, Hh, I, tok in rows:
    if (folio, ln) != key:
        if buf: lines.append((sec0, buf))
        key, buf, sec0 = (folio, ln), [], I
    buf.append(tok)
if buf: lines.append((sec0, buf))

def H(counter):
    tot = sum(counter.values())
    return -sum(c/tot*math.log2(c/tot) for c in counter.values()) if tot else 0.0

# ---- P1: slot entropy vs within-line shuffle ----
def slot_counters(lns):
    s = {"initial": Counter(), "2nd": Counter(), "interior": Counter(), "penult": Counter(), "final": Counter()}
    for _, w in lns:
        if len(w) < 4: continue
        s["initial"][w[0]] += 1; s["2nd"][w[1]] += 1; s["penult"][w[-2]] += 1; s["final"][w[-1]] += 1
        for t in w[2:-2]: s["interior"][t] += 1
    return s
real = slot_counters(lines)
# null: shuffle words within each line
shuf_lines = [(sec, random.sample(w, len(w))) for sec, w in lines]
null = slot_counters(shuf_lines)
print("P1  line-position slot predictability (entropy, bits; lower = more fixed)")
print(f"    {'slot':<10}{'real H':>8}{'shuffled H':>12}{'top token (share)':>22}")
for k in ("initial", "2nd", "interior", "penult", "final"):
    top, n = real[k].most_common(1)[0]; share = n/sum(real[k].values())*100
    print(f"    {k:<10}{H(real[k]):>8.2f}{H(null[k]):>12.2f}      {top+' ('+format(share,'.0f')+'%)':>18}")

# ---- P2: section-specific role-players at line edges ----
secs = sorted({s for s, _ in lines})
print("\nP2  who owns the line-INITIAL slot, per section (universal token or section-specific?)")
print(f"    {'section':<9}{'top initial (share)':<24}{'top final (share)'}")
for sec in secs:
    ini = Counter(); fin = Counter()
    for s, w in lines:
        if s == sec and len(w) >= 2: ini[w[0]] += 1; fin[w[-1]] += 1
    if sum(ini.values()) < 30: continue
    ti, ni = ini.most_common(1)[0]; tf, nf = fin.most_common(1)[0]
    print(f"    {sec:<9}{ti+' ('+format(ni/sum(ini.values())*100,'.0f')+'%)':<24}{tf+' ('+format(nf/sum(fin.values())*100,'.0f')+'%)'}")

# ---- P3: transition predictability vs Latin ----
def cond_entropy(toks):
    uni = Counter(toks); big = Counter(zip(toks, toks[1:]))
    Hjoint = H(big); Huni = H(uni)
    return Huni, Hjoint - Huni   # H(word), H(next|current)
vH, vC = cond_entropy([r[6] for r in rows])
ap = re.findall(r"[a-z]+", (ROOT/"data/apicius_books.txt").read_text(encoding="utf-8", errors="ignore").lower())
ca = (ROOT/"data/latin_dbg.txt").read_text(encoding="utf-8", errors="ignore")
a = ca.find("*** START"); b = ca.find("*** END"); ca = re.findall(r"[a-z]+", ca[ca.find(chr(10), a)+1:(b if b>0 else len(ca))].lower())
aH, aC = cond_entropy(ap); cH, cC = cond_entropy(ca)
print("\nP3  word transition predictability (bits)")
print(f"    {'corpus':<14}{'H(word)':>9}{'H(next|cur)':>13}")
print(f"    {'Voynich':<14}{vH:>9.2f}{vC:>13.2f}")
print(f"    {'Apicius':<14}{aH:>9.2f}{aC:>13.2f}")
print(f"    {'Caesar':<14}{cH:>9.2f}{cC:>13.2f}")
print("\nlower H(next|current) = more predictable next word = more formulaic/frame-like.")
