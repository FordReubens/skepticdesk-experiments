# Data sources

The experiments fetch these corpora from their original hosts (`fetch_data.py`).
We do not redistribute them; each keeps its own terms and attribution.

- **Voynich Manuscript text** — ZL (Zandbergen–Landini) EVA transliteration,
  IVTFF 2.0, René Zandbergen, http://www.voynich.nu/data/ZL3b-n.txt
  (transcription index: http://www.voynich.nu/transcr.html). Page images:
  Beinecke Rare Book & Manuscript Library, Yale (MS 408), public domain.
- **Latin narrative control** — Julius Caesar, *De Bello Gallico*, Project
  Gutenberg eBook #218, https://www.gutenberg.org/ebooks/218 (public domain;
  Project Gutenberg License).
- **Latin recipe control** — Apicius, *De re coquinaria* (Books I–V), The Latin
  Library, https://www.thelatinlibrary.com/apicius/
- **Bengalese finch song** — Koumura, T. & Okanoya, K. (2016), *Automatic
  Recognition of Element Classes and Boundaries in the Birdsong with Variable
  Sequences*, PLoS ONE; annotation via
  https://github.com/NickleDave/birdsong-recognition-dataset (orig. figshare
  dataset 3470165).

Method references cited in the analyses: Brown et al. (1992) class-based n-gram
models; Timm & Schinner (2019) self-citation; Montemurro & Zanette (2013);
Rugg (2004); Suzuki, Buck & Tyack (2006); Kershenbaum et al. (2016).

## Pseudo-Lullian alchemical Latin (control)
- *Raymundi Lulli Testamentum*, Cologne 1566 — Internet Archive item `ARes25626`, OCR full text (`ARes25626_djvu.txt`). Used as a genuine Lullist-tradition alchemical Latin corpus to test whether the Voynich resembles alchemical Latin more than ordinary Latin (it does not).
