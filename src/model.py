"""The recommender model: weighted facet similarity with a hub correction.

Every title and track has one vector per description facet (emotions, plot,
setting, sound, references). The score of a (title, track) pair is

    score = sum over facets of  weight[facet] * cosine(title[facet], track[facet])
            - hub_correction * (average score of that track over all titles)

One more facet, "tone", is not text: it compares the energy and valence of a
title (hand-set, data/title_tone.csv) with the audio features of a track.

The second term removes "hub" tracks: a track that is fairly similar to every
title would otherwise show up in everyone's recommendations.
The parameters live in data/model.json.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TITLES_PATH = PROJECT_ROOT / "data" / "titles.csv"
TRACKS_PATH = PROJECT_ROOT / "data" / "tracks.csv"
TITLE_TONE_PATH = PROJECT_ROOT / "data" / "title_tone.csv"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings.npz"
TONE_FEATURES = ["energy", "valence"]
MODEL_PATH = PROJECT_ROOT / "data" / "model.json"


def load_model() -> tuple[dict[str, float], float]:
    """Return (weights per facet, hub correction strength) from data/model.json."""
    model = json.loads(MODEL_PATH.read_text())
    return model["weights"], model["hub_correction"]


def load_facet_scores() -> tuple[list[str], list[str], list[str], np.ndarray]:
    """Return (facets, title ids, track ids, cosine similarities per facet).

    The similarities have shape (facets, titles, tracks).
    """
    if not EMBEDDINGS_PATH.exists():
        raise SystemExit("Embeddings not found: run `python src/embed.py` first.")
    data = np.load(EMBEDDINGS_PATH)
    is_title = data["item_type"] == "title"
    vectors = data["vectors"].astype(np.float32)
    title_vectors = vectors[is_title]  # (titles, facets, dimensions)
    track_vectors = vectors[~is_title]  # (tracks, facets, dimensions)

    # Vectors are unit-length, so the dot product is the cosine similarity.
    # For each facet f: similarities[f] = title_vectors[:, f] @ track_vectors[:, f].T
    similarities = np.einsum("tfd,kfd->ftk", title_vectors, track_vectors)
    title_ids = data["item_id"][is_title].tolist()
    track_ids = data["item_id"][~is_title].tolist()

    tone = tone_similarity(title_ids, track_ids)
    return (
        data["facets"].tolist() + ["tone"],
        title_ids,
        track_ids,
        np.concatenate([similarities, tone[np.newaxis]]),
    )


def tone_similarity(title_ids: list[str], track_ids: list[str]) -> np.ndarray:
    """Return a (titles, tracks) matrix: 1 = same energy and valence, 0 = opposite."""
    titles = pd.read_csv(TITLE_TONE_PATH, dtype={"tmdb_id": str}).set_index("tmdb_id")
    tracks = pd.read_csv(TRACKS_PATH).set_index("track_id")
    title_tone = titles.loc[title_ids, TONE_FEATURES].to_numpy(dtype=float)
    track_tone = tracks.loc[track_ids, TONE_FEATURES].to_numpy(dtype=float)

    # Distance between every title and every track in the (energy, valence) square.
    distances = np.linalg.norm(title_tone[:, np.newaxis] - track_tone[np.newaxis], axis=2)
    return 1 - distances / np.sqrt(len(TONE_FEATURES))


def score_matrix(
    facets: list[str], similarities: np.ndarray, weights: dict[str, float], hub_correction: float
) -> np.ndarray:
    """Combine the facet similarities into one (titles, tracks) score matrix."""
    weight_vector = np.array([weights[facet] for facet in facets])
    scores = np.tensordot(weight_vector, similarities, axes=1)
    return scores - hub_correction * scores.mean(axis=0, keepdims=True)
