"""Recommend tracks from the movies and series you like.

Usage:
    python src/recommend.py "Blade Runner 2049" "Drive"
    python src/recommend.py "Amélie" --top 10

Each liked title scores every track (see src/model.py); the scores are averaged
over the liked titles and the best tracks are shown. Run src/embed.py first.
"""

import argparse

import pandas as pd

from model import TITLES_PATH, TRACKS_PATH, load_facet_scores, load_model, score_matrix


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
    """Rank the tracks by their average model score over the liked titles."""
    facets, title_ids, track_ids, similarities = load_facet_scores()
    weights, hub_correction = load_model()
    scores = score_matrix(facets, similarities, weights, hub_correction)

    liked_rows = [title_ids.index(item_id) for item_id in liked_ids]
    taste_scores = scores[liked_rows].mean(axis=0)

    tracks = pd.read_csv(TRACKS_PATH).set_index("track_id").loc[track_ids]
    tracks["score"] = taste_scores
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
        print(f"{rank:>2}. {track['title']} — {track['artist']}  ({track['score']:+.3f})")


if __name__ == "__main__":
    main()
