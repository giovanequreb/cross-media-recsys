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
trained model (see src/train.py): a logistic regression that learns from
data/curated.csv, a list of tracks hand-picked for each title. Its inputs are

- every facet of the title against every facet of the track (5 x 5 similarities),
- the tone similarity and the baseline score,
- "neighbours": how often the track was picked for titles similar to this one.

The parameters of both live in data/model.json.
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


def title_similarity(title_vectors: np.ndarray, facets: list[str], weights: dict[str, float]) -> np.ndarray:
    """Return a (titles, titles) matrix: how alike two titles are, using the facet weights."""
    facet_weights = np.array([weights[facet] for facet in facets])
    per_facet = np.einsum("afd,bfd->fab", title_vectors, title_vectors)
    return np.tensordot(facet_weights / facet_weights.sum(), per_facet, axes=1)


def neighbour_scores(
    between_titles: np.ndarray, picked: np.ndarray, known: np.ndarray, sharpness: float
) -> np.ndarray:
    """For every title, the share of similar titles for which each track was picked.

    Only the titles in `known` lend their picks, and a title never uses its own:
    that is what lets the score be tested on titles the model has not seen.
    """
    closeness = np.exp(sharpness * between_titles[:, known])
    closeness[known, np.arange(len(known))] = 0  # leave each title's own picks out
    return (closeness @ picked[known].astype(float)) / closeness.sum(axis=1, keepdims=True)


def feature_names(facets: list[str]) -> list[str]:
    """Names of the trained model's inputs, in order."""
    return [f"{a}~{b}" for a in facets for b in facets] + ["tone", "baseline", "neighbours"]


def build_features(known: np.ndarray, sharpness: float) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """Return (title ids, track ids, features, picked).

    features has shape (titles, tracks, inputs); `known` are the row numbers of
    the titles whose hand-picked tracks may be used by the neighbours input.
    """
    all_facets, title_ids, track_ids, similarities = load_facet_scores()
    weights, hub_correction = load_model()
    facets, title_vectors, track_vectors = load_vectors()
    picked = load_curated(title_ids, track_ids)

    # cross[a, b] = facet a of every title against facet b of every track
    cross = np.einsum("tad,kbd->abtk", title_vectors, track_vectors)
    baseline = score_matrix(all_facets, similarities, weights, hub_correction)
    between_titles = title_similarity(title_vectors, facets, weights)
    neighbours = neighbour_scores(between_titles, picked, known, sharpness)

    columns = [cross[a, b] for a in range(len(facets)) for b in range(len(facets))]
    columns += [similarities[all_facets.index("tone")], baseline, neighbours]
    return title_ids, track_ids, np.stack(columns, axis=-1), picked


def trained_score_matrix() -> tuple[list[str], list[str], np.ndarray]:
    """Return (title ids, track ids, scores) from the trained model in data/model.json.

    Tracks hand-picked for a title get `curated_boost` added, so for titles in the
    catalogue the curated list itself counts, not only what was learned from it.
    """
    trained = json.loads(MODEL_PATH.read_text())["trained"]
    title_ids, track_ids, features, picked = build_features(
        known=np.arange(len(pd.read_csv(TITLES_PATH))), sharpness=trained["neighbour_sharpness"]
    )
    standardised = (features - np.array(trained["mean"])) / np.array(trained["scale"])
    scores = standardised @ np.array(trained["coefficients"]) + trained["intercept"]
    return title_ids, track_ids, scores + trained["curated_boost"] * picked
