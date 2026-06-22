#!/usr/bin/env python3
"""M5c: genre-matched control. Does a formulaic, entry-by-entry Latin RECIPE text
(Apicius) show the function-word signature (uniform high-frequency words)? Compare
recipe Latin vs narrative Latin (Caesar) vs Voynich within-section, identical params.

If even recipe Latin keeps low top-rank burstiness, the function-word signature is
genre-robust and Voynich's within-section numbers are the comparison that matters.
"""
import re, math, random, pathlib
from collections import Counter, defaultdict
from voynich_lib import parse

random.seed(7)
B, MINC, M = 24, 6, 120
ROOT = pathlib.Path(__file__).resolve().parent

def profile(label, toks):
    N = len(toks)
    if N < 1500:
        print(f"{label:<26} only {N} tokens — skipped"); return
    binof = [i*B//N for i in range(N)]
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
    real = kls(toks); s = toks[:]; acc = defaultdict(float); acc2 = defaultdict(float)
    for _ in range(M):
        random.shuffle(s)
        for w, v in kls(s).items(): acc[w] += v; acc2[w] += v*v
    z = {}
    for w in elig:
        mu = acc[w]/M; sd = math.sqrt(max(acc2[w]/M - mu*mu, 1e-9)); z[w] = (real[w]-mu)/sd
    ranked = sorted(elig, key=lambda w: -counts[w])
    mz = sum(z[w] for w in ranked[:10])/min(10, len(ranked))
    fw = sum(1 for w in ranked[:20] if z[w] < 5)
    print(f"{label:<26}{N:>7}{len(elig):>7}{mz:>10.1f}{fw:>9}/20   " + " ".join(ranked[:5]))

def caesar():
    t = (ROOT/"data/latin_dbg.txt").read_text(encoding="utf-8", errors="ignore")
    a = t.find("*** START"); b = t.find("*** END")
    if a >= 0: t = t[t.find("\n", a)+1:(b if b > 0 else len(t))]
    return re.findall(r"[a-z]+", t.lower())

apicius = re.findall(r"[a-z]+", (ROOT/"data/latin_apicius.txt").read_text(encoding="utf-8", errors="ignore").lower())
rows = parse()
print(f"{'corpus':<26}{'toks':>7}{'elig':>7}{'top10 z':>10}{'func20':>9}     top words")
print("-"*92)
profile("Latin RECIPE (Apicius)", apicius)
profile("Latin NARRATIVE (Caesar)", caesar())
profile("Voynich Section H herbal", [r[6] for r in rows if r[5] == "H"])
profile("Voynich Section P pharma", [r[6] for r in rows if r[5] == "P"])
profile("Voynich Section B biol.", [r[6] for r in rows if r[5] == "B"])
profile("Voynich Currier A", [r[6] for r in rows if r[3] == "A"])
profile("Voynich Currier B", [r[6] for r in rows if r[3] == "B"])
profile("Voynich WHOLE (pooled)", [r[6] for r in rows])
print("\nlow top10 z + high func20 = function-word layer present (language-like).")
