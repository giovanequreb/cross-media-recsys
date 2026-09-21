# Cross-Media Recommender

Recommend **music** based on your taste in **movies and TV series**.

Most recommenders stay inside one domain: they suggest songs because you liked other songs. This project tries the opposite: if you love a slow, melancholic sci-fi film, which tracks would fit that same mood? It is inspired by the Podiums app, where taste is captured through pairwise comparisons instead of star ratings.

> **Status: work in progress.** The environment is set up and verified with a smoke test (`python src/smoke_test.py`); the dataset and the recommender are not built yet. See the [roadmap](#roadmap) for what exists and what is coming.

## How it will work

The project is built in three levels, each one a working step on its own.

### Level 1 — Prototype without training (current)

Every movie, series and track gets a short English text description (generated with an LLM). Descriptions are turned into vectors with a local embedding model, so items from different media end up in the **same vector space**. Recommendations are the tracks whose vectors are closest to the movies you like, ranked by **cosine similarity**.

```
movie/series description ─┐
                          ├─► sentence embedding (384-d) ─► cosine similarity ─► top-N tracks
track description ────────┘
```

- Embedding model: [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) via `sentence-transformers`, running locally (free, no API key).
- No training involved: this is the baseline that later levels have to beat.

### Level 2 — Trained model

Train on cross-domain datasets (Douban, Amazon Reviews for movies and CDs) so the model learns real movie ↔ music taste correlations instead of relying only on text similarity.

### Level 3 — Pairwise comparisons and web app

Collect preferences through pairwise comparisons ("which do you prefer?"), rank items with Elo / Bradley-Terry, and wrap everything in a full web app.

## Getting started

Requires Python 3.12 (developed on macOS).

```bash
# 1. Clone the repository
git clone <repo-url>
cd cross-media-recsys

# 2. Create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

The first time the embedding model is used, `sentence-transformers` downloads it (about 90 MB) and caches it locally.

## Project structure

```
cross-media-recsys/
├── data/            # movies, series, tracks
├── src/             # project code
├── notebooks/       # exploratory experiments
├── requirements.txt # pinned dependencies
├── README.md
├── CLAUDE.md        # working notes shared between Claude Code and Claude chat
└── .gitignore
```

## Tech stack

| Purpose | Tool |
| --- | --- |
| Language | Python 3.12 |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) on PyTorch |
| Numerics and data | `numpy`, `pandas`, `scikit-learn` |

## Roadmap

- [x] Project setup: virtual environment, dependencies, folder structure
- [x] Smoke test: embed 3 sentences and check the output shape is `(3, 384)`
- [ ] Level 1: minimal dataset (~20 movies/series, ~50 tracks with basic metadata)
- [ ] Level 1: LLM-generated descriptions, embeddings, cosine-similarity recommendations
- [ ] Level 2: train on Douban / Amazon Reviews (movies + CDs)
- [ ] Level 3: pairwise comparisons (Elo / Bradley-Terry) and web app

## About

Built by Adam Kouribiy as a portfolio project on applied AI. The code is written step by step, with each step documented and committed separately.
