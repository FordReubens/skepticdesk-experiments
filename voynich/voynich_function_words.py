#!/usr/bin/env python3
"""Milestone 5: the function-word / grammar signature, Voynich vs real Latin.

In a real language a tiny set of ultra-frequent words (the, of, et, in, est...) are
grammatical glue: very frequent yet spread EVENLY across the text (low burstiness),
unlike content words which clump where their topic is discussed. A page-by-page
generator has no such function/content split.

Per word we compute burstiness = KL(occurrence distribution || uniform) over B bins,
then a z-score vs frequency-matched random placement (whole-corpus shuffles). A
function word = high frequency but z ~ 0 (uniform). Content word = high z (clumped).
We compare the frequency -> mean-z curve, and the top words, in both corpora.
"""
import re, math, random, pathlib
from collections import Counter, defaultdict
from voynich_lib import parse

random.seed(7)
B, MINC, M = 40, 10, 120
ROOT = pathlib.Path(__file__).resolve().parent

def latin_tokens():
    t = (ROOT / "data/latin_dbg.txt").read_text(encoding="utf-8", errors="ignore")
    a = t.find("*** START"); b = t.find("*** END")
    if a >= 0: t = t[t.find("\n", a)+1 : (b if b > 0 else len(t))]
    return re.findall(r"[a-z]+", t.lower())

def voynich_tokens():
    return [r[6] for r in parse()]

def analyze(name, toks):
    N = len(toks)
    binof = [i * B // N for i in range(N)]
    counts = Counter(toks)
    elig = [w for w, c in counts.items() if c >= MINC]
    nw = {w: counts[w] for w in elig}

    def kls(seq):
        hist = defaultdict(lambda: [0]*B)
        for i, w in enumerate(seq):
            if w in nw: hist[w][binof[i]] += 1
        out = {}
        for w, h in hist.items():
            n = nw[w]; kl = 0.0
            for x in h:
                if x: p = x/n; kl += p*math.log2(p*B)
            out[w] = kl
        return out

    real = kls(toks)
    s = toks[:]; acc = defaultdict(float); acc2 = defaultdict(float)
    for _ in range(M):
        random.shuffle(s)
        for w, v in kls(s).items(): acc[w] += v; acc2[w] += v*v
    z = {}
    for w in elig:
        mu = acc[w]/M; sd = math.sqrt(max(acc2[w]/M - mu*mu, 1e-9))
        z[w] = (real[w] - mu)/sd

    print(f"\n=== {name}: {N} tokens, {len(counts)} types, {len(elig)} eligible (>= {MINC}) ===")
    top = sorted(elig, key=lambda w: -counts[w])[:12]
    print(f"  {'top word':<12}{'count':>7}{'burst KL':>10}{'z':>8}   (low z = function-like / uniform)")
    for w in top:
        print(f"  {w:<12}{counts[w]:>7}{real[w]:>10.3f}{z[w]:>8.1f}")
    # mean z by frequency-rank group
    ranked = sorted(elig, key=lambda w: -counts[w])
    groups = [("top 10", ranked[:10]), ("11-50", ranked[10:50]),
              ("51-150", ranked[50:150]), ("151+", ranked[150:])]
    print(f"  freq-rank group   mean z   (natural language: top group DIPS low = function words)")
    for label, ws in groups:
        if ws:
            mz = sum(z[w] for w in ws)/len(ws)
            print(f"    {label:<14}{mz:>8.1f}")
    # how many of the top-20 are 'uniform' (function-like, z<5)?
    top20 = ranked[:20]
    fw = sum(1 for w in top20 if z[w] < 5)
    print(f"  function-like among top-20 (z<5): {fw}/20")

analyze("LATIN (De Bello Gallico)", latin_tokens())
analyze("VOYNICH (ZL EVA)", voynich_tokens())
