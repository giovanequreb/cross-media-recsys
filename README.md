# Cross-Media Recommender

Recommend **music** based on your taste in **movies and TV series**.

Most recommenders stay inside one domain: they suggest songs because you liked other songs. This project tries the opposite: if you love a slow, melancholic sci-fi film, which tracks would fit that same mood? It is inspired by the Podiums app, where taste is captured through pairwise comparisons instead of star ratings.

> **Status: work in progress.** The environment is set up and Level 1 works end to end: 111 movies/series and 200 tracks with four-facet descriptions, embeddings, a small weighted similarity model, a command-line recommender and a static web app. Levels 2 and 3 are not built yet. See the [roadmap](#roadmap) for what exists and what is coming.

## How it works

The project is built in three levels, each one a working step on its own.

### Level 1 — Prototype without training (current)

Every movie, series and track gets a short English description (generated with an LLM) split into four **facets**:

| Facet | What it says | Example (a neon-noir film) |
| --- | --- | --- |
| `emotions` | three adjectives for how it feels | "Cool, romantic and dangerous." |
| `plot` | a light summary: what happens, or what the song sounds like and is about | "A silent getaway driver falls for his neighbor and is pulled into a heist gone wrong." |
| `setting` | where and when it takes place; for a track, the place and time the music evokes | "Los Angeles at night, present day with an 80s glow: empty freeways, motels, neon streets." |
| `references` | pop-culture touchstones: era, scene, aesthetic, where you have heard it | "80s-inspired neon noir, synthwave, night drives through Los Angeles..." |

The item's own title and artist are never mentioned. Each facet is turned into a vector with a local embedding model, so movies and tracks end up in the **same vector space**, facet by facet.

```
            emotions ───► embedding ─► cosine ─┐ × 0.35
title/track plot ───────► embedding ─► cosine ─┤ × 0.05
            setting ────► embedding ─► cosine ─┼ × 0.25 ─► weighted sum ─► hub correction ─► top-N tracks
            references ─► embedding ─► cosine ─┘ × 0.35
```

The model ([`src/model.py`](src/model.py), parameters in [`data/model.json`](data/model.json)) scores a (title, track) pair as:

```
score = Σ over facets of  weight[facet] × cosine(title[facet], track[facet])
        − hub_correction × (average score of that track over all titles)
```

- **Facet weights.** Recommendations should follow emotions, pop-culture references and setting much more than plot, so the weights are emotions 0.35, plot 0.05, setting 0.25, references 0.35. They are a design choice, not learned (see [Soundtrack check](#soundtrack-check) for why).
- **Hub correction.** Some tracks are a little similar to everything and would show up for every title. Subtracting each track's average score keeps only what is specific to *this* title. Without it the top-5 lists of the 111 titles use 157 different tracks and one track appears in 22 of them; with it they use 171 different tracks and the worst case is 12.
- Embedding model: [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) via `sentence-transformers`, running locally (free, no API key).
- No training on user data yet: this is the baseline that later levels have to beat.

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

The first time the embedding model is used, `sentence-transformers` downloads it (about 130 MB) and caches it locally.

The data files in `data/` are already committed, so nothing else is needed to run the project. To regenerate them, see [Data](#data).

## Usage

```bash
# Recommend tracks from one or more titles you like (any title in data/titles.csv)
python src/recommend.py "Blade Runner 2049" "Drive" --top 5
```

```
Because you like: Mad Max: Fury Road
 1. Master Of Puppets — Metallica  (+0.175)
 2. Don't Stop Me Now - Remastered 2011 — Queen  (+0.133)
 3. Killing In The Name — Rage Against The Machine  (+0.123)
 4. Danger Zone - From "Top Gun" Original Soundtrack — Kenny Loggins  (+0.112)
 5. Eye of the Tiger — Survivor  (+0.098)
```

With several liked titles, each track gets the average of its scores. The number in brackets is the model score: positive means "more fitting for this title than for the average title".

### Web app

A small static web app, with no backend, runs the same model in the browser:

```bash
python -m http.server 5173 --directory app
# then open http://localhost:5173
```

It also works by opening `app/index.html` directly. Pick the titles you like, move the four facet sliders to see how the weights change the results, and rate each track on a four-step scale (`++` fits, `+` fits a little, `−` not far off, `−−` doesn't fit) or mark it `?` if you don't know it. **Rating mode** shuffles the list, hides ranks and scores, and mixes in three lower-ranked wildcards, so the ratings also cover tracks the model would not have shown. Ratings stay in your browser and can be downloaded as JSON: they are the feedback the model needs to learn its weights instead of having them set by hand.

`app/data.js` is generated; after changing descriptions, embeddings or `data/model.json`, rebuild it with `python src/export_app.py`.

### Soundtrack check

[`data/soundtrack_pairs.csv`](data/soundtrack_pairs.csv) lists 83 (title, track) pairs where the track is famously used in the title, for example *Trainspotting* and "Lust For Life". Since descriptions never name the item itself, a good model should still rank each track high for its own title:

```bash
python src/evaluate.py
```

| Setting | MRR | Hit@1 | Hit@5 | Hit@10 | Mean rank (of 200) |
| --- | --- | --- | --- | --- | --- |
| Random guess | 0.03 | | | | 100 |
| Only `plot` | 0.06 | 0.01 | 0.10 | 0.14 | 70.8 |
| Only `emotions` | 0.25 | 0.18 | 0.31 | 0.35 | 52.7 |
| Only `setting` | 0.33 | 0.25 | 0.41 | 0.43 | 35.3 |
| Only `references` | 0.70 | 0.65 | 0.76 | 0.86 | 7.3 |
| Equal weights | 0.53 | 0.45 | 0.60 | 0.67 | 11.3 |
| Model weights, no hub correction | 0.57 | 0.48 | 0.63 | 0.72 | 10.2 |
| **Model** (`data/model.json`) | **0.61** | **0.53** | **0.67** | **0.71** | **9.9** |

MRR is the mean reciprocal rank (1.0 = always first); Hit@5 is how often the track is in the top 5. The script also runs a grid search over the weights with 5-fold cross-validation by title.

What this check does and does not show:

- **Plot is nearly useless** for matching music (0.07, about random), which supports giving it a small weight.
- **Setting carries real signal** (0.33 on its own, second only to references).
- **The hub correction helps** (0.57 → 0.61) on top of making results more varied.
- **The embedding model matters.** Eight local models were compared on this check with the same descriptions and weights:

  | Model | Size | MRR | Hit@1 | Hit@10 |
  | --- | --- | --- | --- | --- |
  | `all-MiniLM-L6-v2` (the first choice) | 90 MB | 0.52 | 0.42 | 0.73 |
  | `all-MiniLM-L12-v2` | 130 MB | 0.53 | 0.45 | 0.66 |
  | `all-mpnet-base-v2` | 420 MB | 0.52 | 0.45 | 0.66 |
  | `thenlper/gte-base` | 220 MB | 0.59 | 0.52 | 0.71 |
  | `mixedbread-ai/mxbai-embed-large-v1` | 1.3 GB | 0.59 | 0.51 | 0.78 |
  | `BAAI/bge-base-en-v1.5` | 440 MB | 0.60 | 0.54 | 0.75 |
  | **`BAAI/bge-small-en-v1.5`** (used) | 130 MB | **0.61** | 0.53 | 0.71 |
  | `BAAI/bge-large-en-v1.5` | 1.3 GB | 0.63 | 0.57 | 0.80 |

  `bge-small` gives almost all of the gain of the largest model at a tenth of the size, with the same 384 dimensions as before.
- **The check is biased towards `references`.** A track's references often describe the scene it is famous for ("boxing training montage"), so the grid search picks 70% references (MRR 0.74, 0.72 on held-out titles). That finds a title's famous songs, but it is not the same as matching someone's taste, so the weights are not taken from it. Learning them properly needs real feedback, which is what Level 3 is for.
- The descriptions were written by an LLM that knows these works, so part of the match comes from how they were written.

## Data

Level 1 uses a small, hand-picked dataset (committed in `data/`):

| File | Content | Source |
| --- | --- | --- |
| `data/titles.csv` | 111 movies and TV series: TMDB id, title, year, type, genres, English overview | [TMDB API](https://developer.themoviedb.org/) |
| `data/tracks.csv` | 200 tracks: id, title, artist, album, popularity and audio features (`danceability`, `energy`, `valence`, `acousticness`, `instrumentalness`, `tempo`) | [`maharshipandya/spotify-tracks-dataset`](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) on Hugging Face (BSD license) |
| `data/descriptions.csv` | 311 descriptions, one per title and track, in four facets: `emotions`, `plot`, `setting`, `references` (plus item type, id and name) | Written with an LLM (Claude) |
| `data/soundtrack_pairs.csv` | 83 (title, track) pairs where the track is famously used in the title | Hand-picked |
| `data/embeddings.npz` | 384-d unit vectors, shape (311 items, 4 facets, 384) | Built by `src/embed.py` |
| `data/model.json` | Model parameters: facet weights and hub correction | Hand-set |

The first 20 titles were chosen to cover very different moods (dark, dreamy, joyful, epic) and the first 50 tracks to span ambient, classical, synthwave, jazz, indie and rock; 37 more titles and 43 more tracks were then added as famous title/song pairs for the soundtrack check, and finally 107 more tracks (classical, classic rock, new wave, alternative, hip-hop, pop, electronic, soul, folk) so that more titles have a fitting track, and 54 more popular titles. The `genre_label` column in `tracks.csv` comes from the source dataset and is **noisy** (for example, Hans Zimmer's "Time" is labelled `german`), so it is kept for reference only.

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
│   ├── descriptions.csv # four-facet descriptions for titles and tracks
│   ├── soundtrack_pairs.csv # known title/track pairs used by the check
│   ├── embeddings.npz   # one vector per description facet
│   ├── model.json       # model parameters (facet weights, hub correction)
│   └── raw/             # raw downloads (git-ignored)
├── app/
│   ├── index.html       # static web app: pick titles, tune weights, rate tracks
│   └── data.js          # generated by src/export_app.py
├── src/
│   ├── smoke_test.py    # checks the embedding model output shape
│   ├── fetch_titles.py  # builds data/titles.csv from the TMDB API
│   ├── build_tracks.py  # builds data/tracks.csv from the Hugging Face dataset
│   ├── embed.py         # builds data/embeddings.npz from the descriptions
│   ├── model.py         # the scoring model: weighted facets + hub correction
│   ├── recommend.py     # recommends tracks from the titles you like
│   ├── evaluate.py      # soundtrack check, ablation and weight grid search
│   ├── export_app.py    # builds app/data.js for the web app
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
| Embeddings | `sentence-transformers` (`BAAI/bge-small-en-v1.5`) on PyTorch |
| Numerics and data | `numpy`, `pandas`, `scikit-learn` |
| Data access | `httpx` (TMDB API), `huggingface_hub`, `python-dotenv` |

## Roadmap

- [x] Project setup: virtual environment, dependencies, folder structure
- [x] Smoke test: embed 3 sentences and check the output shape is `(3, 384)`
- [x] Level 1: minimal dataset (20 movies/series, 50 tracks with basic metadata)
- [x] Level 1: LLM-generated mood descriptions for every title and track
- [x] Level 1: embeddings, cosine-similarity recommendations, soundtrack sanity check
- [x] Level 1: four-facet descriptions (emotions, plot, setting, references), weighted model with hub correction, 57-pair soundtrack check
- [x] Level 1: static web app with live facet weights and feedback export
- [x] Level 1: 111 titles and 200 tracks, embedding model chosen by comparison (`bge-small-en-v1.5`)
- [x] Level 2: feasibility check on Amazon Reviews 2023 (movie/music user overlap)
- [ ] Level 2: train on Amazon Reviews (movies + CDs)
- [ ] Level 3: pairwise comparisons (Elo / Bradley-Terry) and web app

## About

Built by Adam Kouribiy as a portfolio project on applied AI. The code is written step by step, with each step documented and committed separately.
