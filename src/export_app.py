"""Export the data the web app needs to app/data.js.

The app (app/index.html) is a static page with no backend: it loads the titles,
the tracks and the per-facet similarities from this file and computes the
recommendations in the browser, so the facet weights can be changed live.
"""

import base64
import json

import numpy as np

import pandas as pd

from model import (
    EMBEDDINGS_PATH, PROJECT_ROOT, TITLES_PATH, TRACKS_PATH,
    load_facet_scores, load_model, score_matrix, trained_score_matrix,
)

DESCRIPTIONS_PATH = PROJECT_ROOT / "data" / "descriptions.csv"
OUTPUT_PATH = PROJECT_ROOT / "app" / "data.js"


def to_bytes(matrix: np.ndarray) -> tuple[float, float, str]:
    """Store a matrix as one byte per value: return (minimum, maximum, base64 text)."""
    low, high = float(matrix.min()), float(matrix.max())
    scaled = np.round((matrix - low) / (high - low) * 255).astype(np.uint8)
    return round(low, 4), round(high, 4), base64.b64encode(scaled.tobytes()).decode()


def track_similarity(weights: dict[str, float]) -> np.ndarray:
    """Return a (tracks, tracks) matrix: how alike two tracks are, using the model's facet weights."""
    data = np.load(EMBEDDINGS_PATH)
    vectors = data["vectors"][data["item_type"] == "track"].astype(np.float32)
    facet_weights = np.array([weights[facet] for facet in data["facets"]])
    # For each facet f: vectors[:, f] @ vectors[:, f].T, then the weighted average over facets.
    per_facet = np.einsum("afd,bfd->fab", vectors, vectors)
    return np.tensordot(facet_weights / facet_weights.sum(), per_facet, axes=1)


def main() -> None:
    facets, title_ids, track_ids, similarities = load_facet_scores()
    weights, hub_correction = load_model()
    # The app only shows these parts of a track's description.
    shown_facets = ["emotions", "sound", "references"]

    # One byte per similarity (0-255 between each facet's minimum and maximum),
    # base64-encoded: about six times smaller than writing the numbers as text.
    low, high, encoded = zip(*(to_bytes(facet) for facet in similarities))

    # Track-to-track similarity, used by the app to avoid near-duplicates and to
    # learn from ratings ("more like the tracks you liked").
    between_tracks = track_similarity(weights)
    track_low, track_high, track_bytes = to_bytes(between_tracks)

    # Scores of the trained model, rescaled to the same spread as the hand-weighted
    # scores so the app's other settings behave the same with either.
    _, _, trained = trained_score_matrix()
    spread = score_matrix(facets, similarities, weights, hub_correction).std()
    trained_low, trained_high, trained_bytes = to_bytes((trained - trained.mean()) / trained.std() * spread)

    descriptions = pd.read_csv(DESCRIPTIONS_PATH, dtype=str).set_index("item_id")
    titles = pd.read_csv(TITLES_PATH, dtype=str).fillna("").set_index("tmdb_id").loc[title_ids]
    tracks = pd.read_csv(TRACKS_PATH, dtype=str).set_index("track_id").loc[track_ids]

    data = {
        "facets": facets,
        "weights": weights,
        "hubCorrection": hub_correction,
        "titles": [
            {"id": item_id, "title": row["title"], "year": row["year"], "type": row["type"],
             "director": row["director"]}
            for item_id, row in titles.iterrows()
        ],
        "tracks": [
            {"id": item_id, "title": row["title"], "artist": row["artist"],
             **descriptions.loc[item_id, shown_facets].to_dict()}
            for item_id, row in tracks.iterrows()
        ],
        # similarities[facet] is a base64 string of titles x tracks bytes, row by row
        "similarities": {"low": low, "high": high, "bytes": encoded},
        "trained": {"low": trained_low, "high": trained_high, "bytes": trained_bytes},
        # trackSimilarity is one base64 string of tracks x tracks bytes
        "trackSimilarity": {"low": track_low, "high": track_high, "mean": round(float(between_tracks.mean()), 4),
                            # above this value two tracks count as near-duplicates (top 5% of pairs)
                            "near": round(float(np.percentile(between_tracks, 95)), 4),
                            "bytes": track_bytes},
    }

    # A .js file (not .json) so the page also works when opened straight from disk.
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text("window.APP_DATA = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    print(f"Saved {len(title_ids)} titles and {len(track_ids)} tracks to {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
