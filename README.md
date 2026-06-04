# IR-Project — Information Retrieval Engine

A search engine built **from scratch** in Python: text is preprocessed, indexed with three different index structures, and queried through a small Flask web interface. No search library does the heavy lifting — the indexes, ranking inputs, and text normalization are all implemented directly.

## What it does

- **Preprocessing pipeline** — tokenization, stop-word removal, NLTK lemmatization with POS tagging, and Porter stemming (`Lemmatizer.py`, `Stopwords.py`).
- **Inverted index** — term → document postings list (`InverseIndex.py`).
- **Positional index** — term → `(doc_id, offset)` pairs, enabling phrase/proximity queries (`PositinalIndex.py`).
- **Biword (extended binary) index** — adjacent-term pairs for two-word phrase retrieval (`ExtendedBinaryRetrieval.py`).
- **Vector/TF matrix** — term–document matrix built with NumPy for similarity-based ranking (`matrix.py`).
- **Web interface** — query the indexes through a Flask app (`app.py`, `templates/`).

## Project layout

| File | Role |
| --- | --- |
| `app.py` | Flask app — query entry point and result rendering |
| `InverseIndex.py` | Inverted index |
| `PositinalIndex.py` | Positional index (phrase/proximity) |
| `ExtendedBinaryRetrieval.py` | Biword index |
| `Lemmatizer.py` | NLTK lemmatization + POS tagging |
| `Stopwords.py` | Stop-word list |
| `matrix.py` | Term–document TF matrix (NumPy) |
| `Dataset/` | Corpus of text documents to index |

## Run it

```bash
pip install flask nltk numpy
python app.py
```

The first run downloads the required NLTK resources (`wordnet`, `averaged_perceptron_tagger`) automatically. Then open the local Flask URL and search the indexed corpus.

## Tech

Python · Flask · NLTK · NumPy

---

*Course project demonstrating core information-retrieval internals — inverted/positional/biword indexing, linguistic preprocessing, and vector-space ranking.*
