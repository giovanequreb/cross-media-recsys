"""The recommender model: weighted facet similarity with a hub correction.

Every title and track has one vector per description facet (emotions, plot,
setting, references). The score of a (title, track) pair is

    score = sum over facets of  weight[facet] * cosine(title[facet], track[facet])
            - hub_correction * (average score of that track over all titles)

The second term removes "hub" tracks: a track that is fairly similar to every
title would otherwise show up in everyone's recommendations.
The parameters live in data/model.json.
"""

import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TITLES_PATH = PROJECT_ROOT / "data" / "titles.csv"
TRACKS_PATH = PROJECT_ROOT / "data" / "tracks.csv"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings.npz"
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
    title_vectors = data["vectors"][is_title]  # (titles, facets, dimensions)
    track_vectors = data["vectors"][~is_title]  # (tracks, facets, dimensions)

    # Vectors are unit-length, so the dot product is the cosine similarity.
    # For each facet f: similarities[f] = title_vectors[:, f] @ track_vectors[:, f].T
    similarities = np.einsum("tfd,kfd->ftk", title_vectors, track_vectors)
    return (
        data["facets"].tolist(),
        data["item_id"][is_title].tolist(),
        data["item_id"][~is_title].tolist(),
        similarities,
    )


def score_matrix(
    facets: list[str], similarities: np.ndarray, weights: dict[str, float], hub_correction: float
) -> np.ndarray:
    """Combine the facet similarities into one (titles, tracks) score matrix."""
    weight_vector = np.array([weights[facet] for facet in facets])
    scores = np.tensordot(weight_vector, similarities, axes=1)
    return scores - hub_correction * scores.mean(axis=0, keepdims=True)
