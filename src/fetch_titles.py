"""Fetch metadata for the seed movies and TV series from TMDB into data/titles.csv.

Needs a TMDB API key in the TMDB_API_KEY environment variable (see .env).
This product uses the TMDB API but is not endorsed or certified by TMDB.
"""

import os
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv

BASE_URL = "https://api.themoviedb.org/3"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "titles.csv"

# (title as searched on TMDB, year, media type: "movie" or "tv")
SEED_TITLES = [
    ("Blade Runner 2049", 2017, "movie"),
    ("Interstellar", 2014, "movie"),
    ("Drive", 2011, "movie"),
    ("Lost in Translation", 2003, "movie"),
    ("Whiplash", 2014, "movie"),
    ("La La Land", 2016, "movie"),
    ("Pulp Fiction", 1994, "movie"),
    ("Spirited Away", 2001, "movie"),
    ("Parasite", 2019, "movie"),
    ("Mad Max: Fury Road", 2015, "movie"),
    ("The Grand Budapest Hotel", 2014, "movie"),
    ("Eternal Sunshine of the Spotless Mind", 2004, "movie"),
    ("Amélie", 2001, "movie"),
    ("La grande bellezza", 2013, "movie"),
    ("Breaking Bad", 2008, "tv"),
    ("Chernobyl", 2019, "tv"),
    ("Fleabag", 2016, "tv"),
    ("Arcane", 2021, "tv"),
    ("Severance", 2022, "tv"),
    ("Succession", 2018, "tv"),
]

# TMDB names the same fields differently for movies and TV series.
FIELDS = {
    "movie": {"title": "title", "date": "release_date", "year_filter": "primary_release_year"},
    "tv": {"title": "name", "date": "first_air_date", "year_filter": "first_air_date_year"},
}


def get_json(client: httpx.Client, path: str, params: dict) -> dict:
    """GET a TMDB endpoint and return the JSON body, without ever printing the API key."""
    try:
        response = client.get(f"{BASE_URL}{path}", params=params)
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        # The default message contains the full URL, which includes the API key.
        raise SystemExit(f"TMDB returned HTTP {error.response.status_code} for {path}") from None
    except httpx.RequestError as error:
        raise SystemExit(f"Network error while calling {path}: {type(error).__name__}") from None
    return response.json()


def fetch_title(client: httpx.Client, api_key: str, query: str, year: int, media_type: str) -> dict:
    """Search TMDB for one title, then load its details and return one CSV row."""
    fields = FIELDS[media_type]
    auth = {"api_key": api_key, "language": "en-US"}

    search = get_json(
        client,
        f"/search/{media_type}",
        {**auth, "query": query, fields["year_filter"]: year},
    )
    if not search["results"]:
        raise SystemExit(f"No TMDB result for {query!r} ({year}, {media_type})")

    tmdb_id = search["results"][0]["id"]
    details = get_json(client, f"/{media_type}/{tmdb_id}", auth)

    return {
        "tmdb_id": tmdb_id,
        "title": details[fields["title"]],
        "year": int(details[fields["date"]][:4]),
        "type": media_type,
        "genres": "|".join(genre["name"] for genre in details["genres"]),
        "overview": details["overview"],
    }


def main() -> None:
    load_dotenv()
    api_key = os.environ.get("TMDB_API_KEY")
    if not api_key:
        raise SystemExit("TMDB_API_KEY is missing: add it to the .env file in the project root.")

    with httpx.Client(timeout=15) as client:
        rows = [fetch_title(client, api_key, *seed) for seed in SEED_TITLES]

    titles = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    titles.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(titles)} titles to {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
