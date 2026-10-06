"""Recommend tracks from the movies and series you like.

Usage:
    python src/recommend.py "Blade Runner 2049" "Drive"
    python src/recommend.py "Amélie" --top 10

Each liked title scores every track with the trained model (see src/model.py and
src/train.py); the scores are averaged over the liked titles and the best tracks
are shown. Run src/embed.py and src/train.py first.
"""

import argparse

import pandas as pd

from model import TITLES_PATH, TRACKS_PATH, trained_score_matrix


def find_title(titles: pd.DataFrame, query: str) -> pd.Series:
    """Return the title row matching the query (case-insensitive, exact match first).

    Some names exist twice (a film and its remake): add the year to pick one, e.g. "Aladdin (2019)".
    """
    year = None
    if query.endswith(")") and query[-6:-5] == "(" and query[-5:-1].isdigit():
        query, year = query[:-6].strip(), query[-5:-1]
    names = titles["title"].str.casefold()
    matches = titles[names == query.casefold()]
    if matches.empty:
        matches = titles[names.str.contains(query.casefold(), regex=False)]
    if year is not None:
        matches = matches[matches["year"].astype(str) == year]
    if len(matches) != 1:
        if matches.empty:
            raise SystemExit(f"Title {query!r} is not in data/titles.csv.")
        options = ", ".join(f"{row.title} ({row.year})" for row in matches.head(10).itertuples())
        raise SystemExit(f"Title {query!r} is ambiguous: add the year, for example one of: {options}")
    return matches.iloc[0]


def recommend(liked_ids: list[str], top: int) -> pd.DataFrame:
    """Rank the tracks by their average model score over the liked titles."""
    title_ids, track_ids, scores = trained_score_matrix()

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
        print(f"{rank:>2}. {track['title']} — {track['artist']}  ({track['score']:+.2f})")


if __name__ == "__main__":
    main()
