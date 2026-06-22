#!/usr/bin/env python3
"""Download the corpora the Voynich experiments need, into ./data/.
We fetch from each source rather than redistributing it (see SOURCES.md)."""
import re, time, pathlib, urllib.request as u
DATA = pathlib.Path(__file__).resolve().parent / "data"; DATA.mkdir(exist_ok=True)
UA = {"User-Agent": "skepticdesk-experiments (research; https://skepticdesk.com)"}

def get(url):
    return u.urlopen(u.Request(url, headers=UA), timeout=60).read()

def save(name, data):
    (DATA / name).write_bytes(data); print(f"  saved {name} ({len(data)} bytes)")

print("1/4 Voynich ZL EVA transliteration (voynich.nu)")
save("ZL3b-n.txt", get("http://www.voynich.nu/data/ZL3b-n.txt"))

print("2/4 Latin control: Caesar, De Bello Gallico (Project Gutenberg #218)")
save("latin_dbg.txt", get("https://www.gutenberg.org/cache/epub/218/pg218.txt"))

print("3/4 Latin recipe control: Apicius, De re coquinaria (The Latin Library)")
parts = []
for i in range(1, 6):
    html = get(f"https://www.thelatinlibrary.com/apicius/apicius{i}.shtml").decode("latin-1", "ignore")
    body = re.sub(r"(?is)<head.*?</head>", " ", html)
    body = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", body)
    txt = re.sub(r"(?s)<[^>]+>", " ", body)
    txt = re.sub(r"(?i)the latin library|the classics page|apicius|liber [ivxlcdm]+", " ", txt)
    parts.append(f"### BOOK {i}\n" + txt); time.sleep(1)
save("apicius_books.txt", "\n".join(parts).encode("utf-8"))

print("4/4 Bengalese finch song (Koumura & Okanoya 2016, via NickleDave repo)")
save("birdsong_Bird0.xml", get("https://raw.githubusercontent.com/NickleDave/birdsong-recognition-dataset/main/tests/test_data/Bird0/Annotation.xml"))

print("done -> ./data/")
