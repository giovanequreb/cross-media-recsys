# Cross-Media Recommender

Recommend **music** based on your taste in **movies and TV series**.

Most recommenders stay inside one domain: they suggest songs because you liked other songs. This project tries the opposite: if you love a slow, melancholic sci-fi film, which tracks would fit that same mood? It is inspired by the Podiums app, where taste is captured through pairwise comparisons instead of star ratings.

> **Status: work in progress.** The environment is set up and the minimal dataset (20 movies/series, 50 tracks) is ready; descriptions, embeddings and the recommender are not built yet. See the [roadmap](#roadmap) for what exists and what is coming.

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

# 4. Check that the embedding model works (expected: Shape: (3, 384))
python src/smoke_test.py
```

The first time the embedding model is used, `sentence-transformers` downloads it (about 90 MB) and caches it locally.

The CSV files in `data/` are already committed, so nothing else is needed to run the project. To regenerate them, see [Data](#data).

## Data

Level 1 uses a small, hand-picked dataset (committed in `data/`):

| File | Content | Source |
| --- | --- | --- |
| `data/titles.csv` | 20 movies and TV series: TMDB id, title, year, type, genres, English overview | [TMDB API](https://developer.themoviedb.org/) |
| `data/tracks.csv` | 50 tracks: id, title, artist, album, popularity and audio features (`danceability`, `energy`, `valence`, `acousticness`, `instrumentalness`, `tempo`) | [`maharshipandya/spotify-tracks-dataset`](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) on Hugging Face (BSD license) |

The 20 titles were chosen to cover very different moods (dark, dreamy, joyful, epic), and the 50 tracks to span ambient, classical, synthwave, jazz, indie and rock. The `genre_label` column in `tracks.csv` comes from the source dataset and is **noisy** (for example, Hans Zimmer's "Time" is labelled `german`), so it is kept for reference only.

To regenerate the files:

```bash
# Tracks: downloads the raw dataset to data/raw/ (git-ignored) and extracts the subset
python src/build_tracks.py

# Titles: needs a free TMDB API key in .env (copy .env.example to .env)
python src/fetch_titles.py
```

Spotify's own API is not used: audio features have been unavailable to new apps since November 2024, and since February 2026 developer apps also require a Premium account.

### Credits

This product uses the TMDB API but is not endorsed or certified by TMDB. TMDB data is free for non-commercial use only.

## Project structure

```
cross-media-recsys/
├── data/
│   ├── titles.csv       # movies and series (from TMDB)
│   ├── tracks.csv       # curated tracks with audio features
│   └── raw/             # raw downloads (git-ignored)
├── src/
│   ├── smoke_test.py    # checks the embedding model output shape
│   ├── fetch_titles.py  # builds data/titles.csv from the TMDB API
│   └── build_tracks.py  # builds data/tracks.csv from the Hugging Face dataset
├── notebooks/           # exploratory experiments
├── requirements.txt     # pinned dependencies
├── .env.example         # template for the TMDB API key (.env is git-ignored)
├── README.md
├── CLAUDE.md            # working notes shared between Claude Code and Claude chat
└── .gitignore
```

## Tech stack

| Purpose | Tool |
| --- | --- |
| Language | Python 3.12 |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) on PyTorch |
| Numerics and data | `numpy`, `pandas`, `scikit-learn` |
| Data access | `httpx` (TMDB API), `huggingface_hub`, `python-dotenv` |

## Roadmap

- [x] Project setup: virtual environment, dependencies, folder structure
- [x] Smoke test: embed 3 sentences and check the output shape is `(3, 384)`
- [x] Level 1: minimal dataset (20 movies/series, 50 tracks with basic metadata)
- [ ] Level 1: LLM-generated descriptions, embeddings, cosine-similarity recommendations
- [ ] Level 2: train on Douban / Amazon Reviews (movies + CDs)
- [ ] Level 3: pairwise comparisons (Elo / Bradley-Terry) and web app

## About

Built by Adam Kouribiy as a portfolio project on applied AI. The code is written step by step, with each step documented and committed separately.
