#!/usr/bin/env python3
"""M21: capacity of the reversible constrained-generation channel (GPT-5.5's last loophole).

A reversible steganographic transducer embeds payload into the *choices* of a
Voynich-like generator. The MAX payload it can carry = the cover text's own
information content (you can't reversibly embed more bits than the cover holds).
The practical estimate of that information content is its compressed size.

So: compress the Voynich token stream (lzma, bz2), convert to a max reversible
payload in plaintext-words, and compare to a real book. References: real Latin
(should be similar), a max-entropy random stream (incompressible = max capacity),
and a near-constant stream (min capacity). This tells us whether the loophole is
book-scale (door reopens, but needs a coding-theory mechanism) or tiny (door shuts).
"""
import lzma, bz2, re, math, random, pathlib
from collections import Counter
from voynich_lib import parse

VOY = [r[6] for r in parse()]
def latin(p, g=False):
    t = (pathlib.Path(__file__).resolve().parent/p).read_text(encoding="utf-8", errors="ignore")
    if g:
        a = t.find("*** START"); b = t.find("*** END")
        if a >= 0: t = t[t.find("\n", a)+1:(b if b > 0 else len(t))]
    return re.findall(r"[a-z]+", t.lower())
APIC = latin("data/apicius_books.txt"); CAES = latin("data/latin_dbg.txt", True)

GLY = sorted({c for w in VOY for c in w}); rng = random.Random(1)
RAND = ["".join(rng.choice(GLY) for _ in range(rng.randint(3, 7))) for _ in range(len(VOY))]   # max entropy
CONST = (["daiin"] * len(VOY))                                                                  # min entropy

BITS_PER_WORD = 6.0   # ~1.1 bits/char (Shannon) x ~5.5 chars/word -> info in a plaintext word
BOOK = 70000          # a ~230-page book, words

def cap(name, toks):
    blob = " ".join(toks).encode()
    lz = len(lzma.compress(blob, preset=9)); bz = len(bz2.compress(blob, 9))
    best = min(lz, bz)
    bits = best * 8
    payload_words = bits / BITS_PER_WORD
    print(f"{name:<20}{len(toks):>7}{len(blob)/1024:>8.0f}{lz/1024:>8.0f}{best/len(blob):>8.3f}"
          f"{bits/1024/8:>9.0f}{payload_words:>11.0f}{payload_words/BOOK:>8.2f}")

print(f"{'corpus':<20}{'toks':>7}{'rawKB':>8}{'lzmaKB':>8}{'ratio':>8}{'infoKB':>9}{'payloadW':>11}{'books':>8}")
print("-"*80)
cap("Voynich", VOY)
cap("Latin Apicius", APIC)
cap("Latin Caesar", CAES)
cap("Random (max cap)", RAND)
cap("Constant (min cap)", CONST)
print(f"\npayloadW = max reversible payload in plaintext words (info / {BITS_PER_WORD} bits-per-word)")
print(f"books = payloadW / {BOOK} (a ~230-page book). >~1 => book-scale capacity; <<1 => only a short message.")
