# Cross-Media Recommender

Recommend **music** based on your taste in **movies and TV series**.

Most recommenders stay inside one domain: they suggest songs because you liked other songs. This project tries the opposite: if you love a slow, melancholic sci-fi film, which tracks would fit that same mood? It is inspired by the Podiums app, where taste is captured through pairwise comparisons instead of star ratings.

> **Status: work in progress.** The environment is set up and Level 1 works end to end: 57 movies/series and 93 tracks with four-facet descriptions, embeddings, a small weighted similarity model, a command-line recommender and a static web app. Levels 2 and 3 are not built yet. See the [roadmap](#roadmap) for what exists and what is coming.

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
- **Hub correction.** Some tracks are a little similar to everything and would show up for every title. Subtracting each track's average score keeps only what is specific to *this* title. Without it one track appeared in the top 5 of 17 titles out of 57; with it the worst case is 9.
- Embedding model: [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) via `sentence-transformers`, running locally (free, no API key).
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

The first time the embedding model is used, `sentence-transformers` downloads it (about 90 MB) and caches it locally.

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

It also works by opening `app/index.html` directly. Pick the titles you like, move the four facet sliders to see how the weights change the results, and rate the tracks with 👍/👎. Ratings stay in your browser and can be downloaded as JSON: they are the feedback the model needs to learn its weights instead of having them set by hand.

`app/data.js` is generated; after changing descriptions, embeddings or `data/model.json`, rebuild it with `python src/export_app.py`.

### Soundtrack check

[`data/soundtrack_pairs.csv`](data/soundtrack_pairs.csv) lists 57 (title, track) pairs where the track is famously used in the title, for example *Trainspotting* and "Lust For Life". Since descriptions never name the item itself, a good model should still rank each track high for its own title:

```bash
python src/evaluate.py
```

| Setting | MRR | Hit@1 | Hit@5 | Hit@10 | Mean rank (of 93) |
| --- | --- | --- | --- | --- | --- |
| Random guess | 0.06 | | | | 47 |
| Only `plot` | 0.07 | 0.00 | 0.09 | 0.23 | 34.5 |
| Only `emotions` | 0.31 | 0.23 | 0.37 | 0.46 | 24.5 |
| Only `setting` | 0.49 | 0.42 | 0.51 | 0.60 | 17.1 |
| Only `references` | 0.74 | 0.67 | 0.84 | 0.91 | 4.3 |
| Equal weights | 0.66 | 0.60 | 0.72 | 0.81 | 7.6 |
| Model weights, no hub correction | 0.66 | 0.60 | 0.74 | 0.81 | 6.6 |
| **Model** (`data/model.json`) | **0.67** | **0.58** | **0.81** | **0.86** | **5.6** |

MRR is the mean reciprocal rank (1.0 = always first); Hit@5 is how often the track is in the top 5. The script also runs a grid search over the weights with 5-fold cross-validation by title.

What this check does and does not show:

- **Plot is nearly useless** for matching music (0.07, about random), which supports giving it a small weight.
- **Setting carries real signal** (0.49 on its own) and adding it as a fourth facet moved the model from 0.65 to 0.67 MRR and from 0.75 to 0.81 Hit@5.
- **The hub correction helps a little** on this check (mean rank 6.6 → 5.6) on top of making results more varied.
- **The check is biased towards `references`.** A track's references often describe the scene it is famous for ("boxing training montage"), so the grid search picks 90% references and 10% setting (MRR 0.80, 0.79 on held-out titles). That finds a title's famous songs, but it is not the same as matching someone's taste, so the weights are not taken from it. Learning them properly needs real feedback, which is what Level 3 is for.
- The descriptions were written by an LLM that knows these works, so part of the match comes from how they were written.

## Data

Level 1 uses a small, hand-picked dataset (committed in `data/`):

| File | Content | Source |
| --- | --- | --- |
| `data/titles.csv` | 57 movies and TV series: TMDB id, title, year, type, genres, English overview | [TMDB API](https://developer.themoviedb.org/) |
| `data/tracks.csv` | 93 tracks: id, title, artist, album, popularity and audio features (`danceability`, `energy`, `valence`, `acousticness`, `instrumentalness`, `tempo`) | [`maharshipandya/spotify-tracks-dataset`](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) on Hugging Face (BSD license) |
| `data/descriptions.csv` | 150 descriptions, one per title and track, in four facets: `emotions`, `plot`, `setting`, `references` (plus item type, id and name) | Written with an LLM (Claude) |
| `data/soundtrack_pairs.csv` | 57 (title, track) pairs where the track is famously used in the title | Hand-picked |
| `data/embeddings.npz` | 384-d unit vectors, shape (150 items, 4 facets, 384) | Built by `src/embed.py` |
| `data/model.json` | Model parameters: facet weights and hub correction | Hand-set |

The first 20 titles were chosen to cover very different moods (dark, dreamy, joyful, epic) and the first 50 tracks to span ambient, classical, synthwave, jazz, indie and rock; 37 more titles and 43 more tracks were then added as famous title/song pairs for the soundtrack check. The `genre_label` column in `tracks.csv` comes from the source dataset and is **noisy** (for example, Hans Zimmer's "Time" is labelled `german`), so it is kept for reference only.

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
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) on PyTorch |
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
- [x] Level 2: feasibility check on Amazon Reviews 2023 (movie/music user overlap)
- [ ] Level 2: train on Amazon Reviews (movies + CDs)
- [ ] Level 3: pairwise comparisons (Elo / Bradley-Terry) and web app

## About

Built by Adam Kouribiy as a portfolio project on applied AI. The code is written step by step, with each step documented and committed separately.
