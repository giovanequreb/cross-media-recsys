# Cross-Media Recommender

Recommend **music** based on your taste in **movies and TV series**.

Most recommenders stay inside one domain: they suggest songs because you liked other songs. This project tries the opposite: if you love a slow, melancholic sci-fi film, which tracks would fit that same mood? It is inspired by the Podiums app, where taste is captured through pairwise comparisons instead of star ratings.

> **Status: work in progress.** The environment is set up and Level 1 works end to end: 20 movies/series and 50 tracks with mood descriptions, embeddings, and a command-line recommender. Levels 2 and 3 are not built yet. See the [roadmap](#roadmap) for what exists and what is coming.

## How it works

The project is built in three levels, each one a working step on its own.

### Level 1 — Prototype without training (current)

Every movie, series and track gets a short English text description (generated with an LLM) in three parts: the emotions it evokes, a light summary of what it is about, and its pop-culture references (era, scene, aesthetic, where you have heard it). The item's own title and artist are never mentioned. Descriptions are turned into vectors with a local embedding model, so items from different media end up in the **same vector space**. Recommendations are the tracks whose vectors are closest to the movies you like, ranked by **cosine similarity**.

```
movie/series description ─┐
                          ├─► sentence embedding (384-d) ─► cosine similarity ─► top-N tracks
track description ────────┘
```

- Embedding model: [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) via `sentence-transformers`, running locally (free, no API key).
- No training involved: this is the baseline that later levels have to beat.

### Level 2 — Trained model

Train on cross-domain datasets (Douban, Amazon Reviews for movies and CDs) so the model learns real movie ↔ music taste correlations instead of relying only on text similarity.

**Feasibility check (done).** A cross-domain model needs users who are active in both domains. `python src/check_overlap.py` streams the rating files of [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (17.2M ratings in `Movies_and_TV`, 4.8M in `CDs_and_Vinyl`) and counts them:

| Ratings in **both** domains | Users | Their ratings |
| --- | --- | --- |
| at least 1 | 713,375 | 6.8M |
| at least 3 | 123,527 | 3.4M |
| at least 5 | 54,260 | 2.3M |
| at least 10 | 17,292 | 1.4M |
| at least 20 | 5,153 | 0.75M |

So there is enough overlap to train on: about 54,000 users with five or more ratings on each side. The dataset has no explicit licence (the authors ask to cite [Hou et al., 2024](https://arxiv.org/abs/2403.03952)), so it is used here for research only and no raw data is committed.

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

The data files in `data/` are already committed, so nothing else is needed to run the project. To regenerate them, see [Data](#data).

## Usage

```bash
# Recommend tracks from one or more titles you like (any title in data/titles.csv)
python src/recommend.py "Blade Runner 2049" "Drive" --top 5
```

```
Because you like: Mad Max: Fury Road
 1. Firestarter — The Prodigy  (0.584)
 2. Killing In The Name — Rage Against The Machine  (0.534)
 3. Master Of Puppets — Metallica  (0.516)
 4. HUMBLE. — Kendrick Lamar  (0.484)
 5. Feeling Good — Nina Simone  (0.408)
```

The liked titles are averaged into one "taste" vector and the 50 tracks are ranked by cosine similarity to it. The number in brackets is the similarity score.

### Sanity check

Five tracks in the dataset come from the soundtrack of a title in the dataset. Descriptions never mention the item's own title or artist, so a good recommender should still rank each of them high for its own title:

```bash
python src/evaluate.py
```

| Title | Soundtrack track | Rank (of 50) |
| --- | --- | --- |
| Amélie | Comptine d'un autre été, l'après-midi | 1 |
| Drive | Nightcall | 1 |
| Drive | A Real Hero | 7 |
| Spirited Away | One Summer Day | 1 |
| Pulp Fiction | Son Of A Preacher Man | 44 |

Mean reciprocal rank: **0.63**, against 0.09 for random guessing. Two caveats: five pairs are far too few for a real evaluation, and the descriptions were written by an LLM that knows these works and deliberately include shared cultural references ("synthwave", "Japanese animation soundtrack"), so part of the match comes from how they were written. The honest test is whether real people like the recommendations, which is what Level 3 is for.

## Data

Level 1 uses a small, hand-picked dataset (committed in `data/`):

| File | Content | Source |
| --- | --- | --- |
| `data/titles.csv` | 20 movies and TV series: TMDB id, title, year, type, genres, English overview | [TMDB API](https://developer.themoviedb.org/) |
| `data/tracks.csv` | 50 tracks: id, title, artist, album, popularity and audio features (`danceability`, `energy`, `valence`, `acousticness`, `instrumentalness`, `tempo`) | [`maharshipandya/spotify-tracks-dataset`](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) on Hugging Face (BSD license) |
| `data/descriptions.csv` | 70 short English descriptions (emotions, light plot or theme, pop-culture references), one per title and track: item type (`title`/`track`), item id, name, description | Written with an LLM (Claude) and reviewed by hand |
| `data/embeddings.npz` | One 384-d unit vector per description, with its item type and id | Built by `src/embed.py` |

The 20 titles were chosen to cover very different moods (dark, dreamy, joyful, epic), and the 50 tracks to span ambient, classical, synthwave, jazz, indie and rock. The `genre_label` column in `tracks.csv` comes from the source dataset and is **noisy** (for example, Hans Zimmer's "Time" is labelled `german`), so it is kept for reference only.

To regenerate the files:

```bash
# Tracks: downloads the raw dataset to data/raw/ (git-ignored) and extracts the subset
python src/build_tracks.py

# Titles: needs a free TMDB API key in .env (copy .env.example to .env)
python src/fetch_titles.py

# Embeddings: re-run after editing data/descriptions.csv
python src/embed.py
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
│   ├── descriptions.csv # mood descriptions for titles and tracks
│   ├── embeddings.npz   # one vector per description
│   └── raw/             # raw downloads (git-ignored)
├── src/
│   ├── smoke_test.py    # checks the embedding model output shape
│   ├── fetch_titles.py  # builds data/titles.csv from the TMDB API
│   ├── build_tracks.py  # builds data/tracks.csv from the Hugging Face dataset
│   ├── embed.py         # builds data/embeddings.npz from the descriptions
│   ├── recommend.py     # recommends tracks from the titles you like
│   ├── evaluate.py      # soundtrack sanity check
│   └── check_overlap.py # Level 2 feasibility: users who rated both movies and music
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
- [x] Level 1: LLM-generated mood descriptions for every title and track
- [x] Level 1: embeddings, cosine-similarity recommendations, soundtrack sanity check
- [x] Level 2: feasibility check on Amazon Reviews 2023 (movie/music user overlap)
- [ ] Level 2: train on Amazon Reviews (movies + CDs)
- [ ] Level 3: pairwise comparisons (Elo / Bradley-Terry) and web app

## About

Built by Adam Kouribiy as a portfolio project on applied AI. The code is written step by step, with each step documented and committed separately.
