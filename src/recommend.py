"""Recommend tracks from the movies and series you like.

Usage:
    python src/recommend.py "Blade Runner 2049" "Drive"
    python src/recommend.py "Amélie" --top 10

The liked titles are averaged into a single "taste" vector, and the tracks are
ranked by cosine similarity to it. Run src/embed.py first to build the vectors.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TITLES_PATH = PROJECT_ROOT / "data" / "titles.csv"
TRACKS_PATH = PROJECT_ROOT / "data" / "tracks.csv"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings.npz"


def load_vectors(item_type: str) -> dict[str, np.ndarray]:
    """Return {item_id: vector} for one item type ("title" or "track")."""
    if not EMBEDDINGS_PATH.exists():
        raise SystemExit("Embeddings not found: run `python src/embed.py` first.")
    data = np.load(EMBEDDINGS_PATH)
    mask = data["item_type"] == item_type
    return dict(zip(data["item_id"][mask], data["vectors"][mask]))


def find_title(titles: pd.DataFrame, query: str) -> pd.Series:
    """Return the title row matching the query (case-insensitive, exact match first)."""
    names = titles["title"].str.casefold()
    matches = titles[names == query.casefold()]
    if matches.empty:
        matches = titles[names.str.contains(query.casefold(), regex=False)]
    if len(matches) != 1:
        available = ", ".join(titles["title"])
        problem = "not found" if matches.empty else "ambiguous"
        raise SystemExit(f"Title {query!r} is {problem}. Available titles: {available}")
    return matches.iloc[0]


def recommend(liked_ids: list[str], top: int) -> pd.DataFrame:
    """Rank the tracks by cosine similarity to the mean vector of the liked titles."""
    title_vectors = load_vectors("title")
    track_vectors = load_vectors("track")

    taste = np.mean([title_vectors[item_id] for item_id in liked_ids], axis=0)
    taste /= np.linalg.norm(taste)

    track_ids = list(track_vectors)
    # Vectors are unit-length, so the dot product is the cosine similarity.
    scores = np.stack([track_vectors[track_id] for track_id in track_ids]) @ taste

    tracks = pd.read_csv(TRACKS_PATH).set_index("track_id").loc[track_ids]
    tracks["score"] = scores
    return tracks.sort_values("score", ascending=False).head(top)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend tracks from movies and series you like.")
    parser.add_argument("titles", nargs="+", help="one or more titles from data/titles.csv")
    parser.add_argument("--top", type=int, default=5, help="how many tracks to show (default: 5)")
    args = parser.parse_args()

    titles = pd.read_csv(TITLES_PATH, dtype={"tmdb_id": str})
    liked = [find_title(titles, query) for query in args.titles]

    print("Because you like: " + ", ".join(row["title"] for row in liked))
    recommendations = recommend([row["tmdb_id"] for row in liked], args.top)
    for rank, (_, track) in enumerate(recommendations.iterrows(), start=1):
        print(f"{rank:>2}. {track['title']} — {track['artist']}  ({track['score']:.3f})")


if __name__ == "__main__":
    main()
