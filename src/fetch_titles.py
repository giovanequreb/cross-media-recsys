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
    # Added to widen the soundtrack check: each has a well-known song in data/tracks.csv
    ("Inception", 2010, "movie"),
    ("Kill Bill: Vol. 1", 2003, "movie"),
    ("The Virgin Suicides", 2000, "movie"),
    ("Arrival", 2016, "movie"),
    ("Shaun of the Dead", 2004, "movie"),
    ("Good Will Hunting", 1997, "movie"),
    ("Stranger Things", 2016, "tv"),
    ("The Intouchables", 2011, "movie"),
    ("Mamma Mia!", 2008, "movie"),
    ("Guardians of the Galaxy", 2014, "movie"),
    ("Trainspotting", 1996, "movie"),
    ("Fight Club", 1999, "movie"),
    ("The Graduate", 1967, "movie"),
    ("Reservoir Dogs", 1992, "movie"),
    ("Top Gun", 1986, "movie"),
    ("Rocky III", 1982, "movie"),
    ("Ghostbusters", 1984, "movie"),
    ("Saturday Night Fever", 1977, "movie"),
    ("Euphoria", 2019, "tv"),
    ("Call Me by Your Name", 2017, "movie"),
    ("A Star Is Born", 2018, "movie"),
    ("8 Mile", 2002, "movie"),
    ("Black Panther", 2018, "movie"),
    ("Frozen", 2013, "movie"),
    ("Shrek", 2001, "movie"),
    ("(500) Days of Summer", 2009, "movie"),
    ("The Breakfast Club", 1985, "movie"),
    ("Requiem for a Dream", 2000, "movie"),
    ("Skyfall", 2012, "movie"),
    ("Apocalypse Now", 1979, "movie"),
    ("Forrest Gump", 1994, "movie"),
    ("Almost Famous", 2000, "movie"),
    ("Easy Rider", 1969, "movie"),
    ("The Lion King", 1994, "movie"),
    ("Ghost", 1990, "movie"),
    ("Pretty Woman", 1990, "movie"),
    ("Game of Thrones", 2011, "tv"),
    # Wider choice of popular titles
    ("The Godfather", 1972, "movie"),
    ("The Matrix", 1999, "movie"),
    ("The Dark Knight", 2008, "movie"),
    ("Joker", 2019, "movie"),
    ("Dune", 2021, "movie"),
    ("Oppenheimer", 2023, "movie"),
    ("Her", 2013, "movie"),
    ("Moonlight", 2016, "movie"),
    ("Everything Everywhere All at Once", 2022, "movie"),
    ("The Social Network", 2010, "movie"),
    ("Gladiator", 2000, "movie"),
    ("Titanic", 1997, "movie"),
    ("The Lord of the Rings: The Fellowship of the Ring", 2001, "movie"),
    ("Back to the Future", 1985, "movie"),
    ("The Shawshank Redemption", 1994, "movie"),
    ("Se7en", 1995, "movie"),
    ("The Shining", 1980, "movie"),
    ("Taxi Driver", 1976, "movie"),
    ("GoodFellas", 1990, "movie"),
    ("A Clockwork Orange", 1971, "movie"),
    ("2001: A Space Odyssey", 1968, "movie"),
    ("Your Name.", 2016, "movie"),
    ("Oldboy", 2003, "movie"),
    ("City of God", 2002, "movie"),
    ("Cinema Paradiso", 1988, "movie"),
    ("The Truman Show", 1998, "movie"),
    ("Donnie Darko", 2001, "movie"),
    ("Baby Driver", 2017, "movie"),
    ("The Wolf of Wall Street", 2013, "movie"),
    ("Django Unchained", 2012, "movie"),
    ("Get Out", 2017, "movie"),
    ("Black Swan", 2010, "movie"),
    ("Thor: Ragnarok", 2017, "movie"),
    ("Iron Man 2", 2010, "movie"),
    ("Wayne's World", 1992, "movie"),
    ("Romeo + Juliet", 1996, "movie"),
    ("Slumdog Millionaire", 2008, "movie"),
    ("Ocean's Eleven", 2001, "movie"),
    ("Logan", 2017, "movie"),
    ("Dangerous Minds", 1995, "movie"),
    ("Mommy", 2014, "movie"),
    ("Good Morning, Vietnam", 1987, "movie"),
    ("Notting Hill", 1999, "movie"),
    ("Peaky Blinders", 2013, "tv"),
    ("The Sopranos", 1999, "tv"),
    ("Twin Peaks", 1990, "tv"),
    ("True Detective", 2014, "tv"),
    ("Mr. Robot", 2015, "tv"),
    ("Black Mirror", 2011, "tv"),
    ("Squid Game", 2021, "tv"),
    ("Dark", 2017, "tv"),
    ("Westworld", 2016, "tv"),
    ("Better Call Saul", 2015, "tv"),
    ("The Bear", 2022, "tv"),
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
