"""The recommender model: weighted facet similarity with a hub correction.

Every title and track has one vector per description facet (emotions, plot,
setting, sound, references). The score of a (title, track) pair is

    score = sum over facets of  weight[facet] * cosine(title[facet], track[facet])
            - hub_correction * (average score of that track over all titles)

One more facet, "tone", is not text: it compares the energy and valence of a
title (hand-set, data/title_tone.csv) with the audio features of a track.

The second term removes "hub" tracks: a track that is fairly similar to every
title would otherwise show up in everyone's recommendations.

That formula, with hand-set weights, is the baseline. On top of it there is a
trained model (see src/train.py): a small neural network with two "towers", one
for titles and one for tracks. Each tower turns an item's description vectors
into 64 numbers, and the network is trained so that a title ends up close to the
tracks hand-picked for it in data/curated.csv. The final score adds the
network's score and the baseline score, each put on the same scale first.

The baseline parameters live in data/model.json, the network in data/towers.npz.
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
CURATED_PATH = PROJECT_ROOT / "data" / "curated.csv"
TOWERS_PATH = PROJECT_ROOT / "data" / "towers.npz"
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


def load_vectors() -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (text facets, title vectors, track vectors); vectors are (items, facets, dimensions)."""
    data = np.load(EMBEDDINGS_PATH)
    is_title = data["item_type"] == "title"
    vectors = data["vectors"].astype(np.float32)
    return data["facets"].tolist(), vectors[is_title], vectors[~is_title]


def load_curated(title_ids: list[str], track_ids: list[str]) -> np.ndarray:
    """Return a (titles, tracks) table of True where the track was hand-picked for the title."""
    curated = pd.read_csv(CURATED_PATH, dtype=str)
    picked = np.zeros((len(title_ids), len(track_ids)), dtype=bool)
    rows = [title_ids.index(tmdb_id) for tmdb_id in curated["tmdb_id"]]
    columns = [track_ids.index(track_id) for track_id in curated["track_id"]]
    picked[rows, columns] = True
    return picked


def standardise(scores: np.ndarray) -> np.ndarray:
    """Put a score matrix on a common scale: mean 0, standard deviation 1."""
    return (scores - scores.mean()) / scores.std()


def tower_scores(title_vectors: np.ndarray, track_vectors: np.ndarray, towers: dict[str, np.ndarray]) -> np.ndarray:
    """Run both towers and return the (titles, tracks) similarity of their outputs.

    Each tower is one linear layer: output = input @ weight.T + bias, then scaled
    to unit length, so the score is a cosine similarity in the learned space.
    """
    titles = title_vectors.reshape(len(title_vectors), -1) @ towers["title_weight"].T + towers["title_bias"]
    tracks = track_vectors.reshape(len(track_vectors), -1) @ towers["track_weight"].T + towers["track_bias"]
    titles /= np.linalg.norm(titles, axis=1, keepdims=True)
    tracks /= np.linalg.norm(tracks, axis=1, keepdims=True)
    return titles @ tracks.T


def trained_score_matrix() -> tuple[list[str], list[str], np.ndarray]:
    """Return (title ids, track ids, scores): the network's score plus the baseline score."""
    if not TOWERS_PATH.exists():
        raise SystemExit("Trained model not found: run `python src/train.py` first.")
    all_facets, title_ids, track_ids, similarities = load_facet_scores()
    weights, hub_correction = load_model()
    _, title_vectors, track_vectors = load_vectors()
    towers = {name: values.astype(np.float32) for name, values in np.load(TOWERS_PATH).items()}

    learned = tower_scores(title_vectors, track_vectors, towers)
    baseline = score_matrix(all_facets, similarities, weights, hub_correction)
    return title_ids, track_ids, standardise(learned) + standardise(baseline)
