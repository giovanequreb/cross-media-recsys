"""Add the most-voted movies and TV series on TMDB to data/titles.csv.

TMDB has about a million titles, but the model needs a description for each one, so
the catalogue grows in steps: this script appends the titles with the most votes on
TMDB (the ones nearly everybody knows) that are not in data/titles.csv yet. The
descriptions are written afterwards (see data/descriptions.csv).

    python src/fetch_popular.py --movies 520 --tv 240

Needs a TMDB API key in .env. This product uses the TMDB API but is not endorsed or
certified by TMDB.
"""

import argparse
import os

import httpx
import pandas as pd
from dotenv import load_dotenv

from fetch_titles import BASE_URL, FIELDS, OUTPUT_PATH, get_json

PAGE_SIZE = 20  # TMDB returns 20 results per page
# Genres left out: documentaries for movies; talk shows, news and soap operas for TV
# (there is no story to describe, or no film music to match).
EXCLUDED_GENRES = {"movie": "99", "tv": "10767,10763,10766"}


def candidates(client: httpx.Client, api_key: str, media_type: str, known: set[tuple[str, str]], wanted: int) -> list[int]:
    """Return the ids of the most-voted titles of one type that are not known yet.

    known holds (type, TMDB id) pairs: TMDB numbers movies and TV series separately,
    so the same number can be a movie and a different series.
    """
    found: list[int] = []
    page = 1
    while len(found) < wanted and page <= 500:  # TMDB serves at most 500 pages
        data = get_json(client, f"/discover/{media_type}", {
            "api_key": api_key, "language": "en-US", "sort_by": "vote_count.desc", "include_adult": "false",
            "without_genres": EXCLUDED_GENRES[media_type], "page": page,
        })
        for item in data["results"]:
            # Titles without a synopsis or a date cannot be described properly.
            if (media_type, str(item["id"])) in known or not item.get("overview") or not item.get(FIELDS[media_type]["date"]):
                continue
            found.append(item["id"])
        page += 1
    return found[:wanted]


def load_row(client: httpx.Client, api_key: str, media_type: str, tmdb_id: int) -> dict:
    """Load the details of one title and return a row with the same columns as data/titles.csv."""
    fields = FIELDS[media_type]
    details = get_json(client, f"/{media_type}/{tmdb_id}", {"api_key": api_key, "language": "en-US", "append_to_response": "credits"})
    if media_type == "movie":
        directors = [person["name"] for person in details["credits"]["crew"] if person["job"] == "Director"]
    else:
        directors = [person["name"] for person in details["created_by"]]
    return {
        "tmdb_id": tmdb_id,
        "title": details[fields["title"]],
        "year": int(details[fields["date"]][:4]),
        "type": media_type,
        "director": ", ".join(directors),
        "genres": "|".join(genre["name"] for genre in details["genres"]),
        "overview": details["overview"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Append the most-voted TMDB titles to data/titles.csv")
    parser.add_argument("--movies", type=int, default=520, help="how many new movies to add")
    parser.add_argument("--tv", type=int, default=240, help="how many new TV series to add")
    arguments = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("TMDB_API_KEY")
    if not api_key:
        raise SystemExit("TMDB_API_KEY is missing: add it to the .env file in the project root.")

    existing = pd.read_csv(OUTPUT_PATH, dtype=str)
    known = set(zip(existing["type"], existing["tmdb_id"]))
    used_ids = set(existing["tmdb_id"])  # the id column must stay unique across movies and series
    rows = []
    with httpx.Client(timeout=15) as client:
        for media_type, wanted in (("movie", arguments.movies), ("tv", arguments.tv)):
            ids = candidates(client, api_key, media_type, known, wanted)
            print(f"{media_type}: {len(ids)} new titles")
            for tmdb_id in ids:
                row = load_row(client, api_key, media_type, tmdb_id)
                if str(row["tmdb_id"]) in used_ids:  # same number as a title of the other type: prefix it, e.g. "tv1402"
                    row["tmdb_id"] = f"{media_type}{row['tmdb_id']}"
                used_ids.add(str(row["tmdb_id"]))
                rows.append(row)

    combined = pd.concat([existing, pd.DataFrame(rows).astype(str)], ignore_index=True)
    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"Added {len(rows)} titles: data/titles.csv now has {len(combined)}")


if __name__ == "__main__":
    main()
