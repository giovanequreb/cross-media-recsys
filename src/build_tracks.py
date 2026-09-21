"""Build data/tracks.csv: 50 hand-picked tracks with audio features.

Source: the Hugging Face dataset maharshipandya/spotify-tracks-dataset (BSD license).
The raw CSV is downloaded to data/raw/ (ignored by git); only the curated
subset is committed.

The dataset's own genre labels are noisy (e.g. Hans Zimmer's "Time" is tagged
"german"), so they are kept as `genre_label` for reference only.
"""

from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "maharshipandya/spotify-tracks-dataset"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = PROJECT_ROOT / "data" / "tracks.csv"

# (substring of the track name, substring of the artists), grouped by mood so the
# subset covers the whole range: dark to joyful, acoustic to electronic.
CURATED_TRACKS = [
    # Ambient and cinematic
    ("Time", "Hans Zimmer"),
    ("Weightless", "Marconi Union"),
    ("On the Nature of Daylight", "Max Richter"),
    ("An Ending", "Brian Eno"),
    ("Nuvole Bianche", "Einaudi"),
    ("Avril 14th", "Aphex Twin"),
    # Classical and nostalgic
    ("Clair de Lune", "Debussy"),
    ("Gymnopédie No. 1", "Satie"),
    ("Comptine d'un autre été", "Tiersen"),
    ("One Summer Day", "Joe Hisaishi"),
    # Synthwave and electronic
    ("Nightcall", "Kavinsky"),
    ("A Real Hero", "College"),
    ("Blinding Lights", "The Weeknd"),
    ("Midnight City", "M83"),
    ("Little Dark Age", "MGMT"),
    ("Get Lucky", "Daft Punk"),
    ("Firestarter", "The Prodigy"),
    # Trip-hop, dream pop and indie
    ("Teardrop", "Massive Attack"),
    ("Glory Box", "Portishead"),
    ("Playground Love", "Air"),
    ("Space Song", "Beach House"),
    ("Apocalypse", "Cigarettes After Sex"),
    ("Fade Into You", "Mazzy Star"),
    ("Intro", "The xx"),
    ("Nights", "Frank Ocean"),
    # Melancholic singer-songwriters and alternative
    ("Between The Bars", "Elliott Smith"),
    ("Hallelujah", "Jeff Buckley"),
    ("Skinny Love", "Bon Iver"),
    ("Pink Moon", "Nick Drake"),
    ("Karma Police", "Radiohead"),
    # Jazz, soul and retro
    ("Take Five", "Dave Brubeck"),
    ("So What", "Miles Davis"),
    ("Feeling Good", "Nina Simone"),
    ("My Funny Valentine", "Chet Baker"),
    ("Son Of A Preacher Man", "Dusty Springfield"),
    ("Bang Bang", "Nancy Sinatra"),
    # Upbeat and joyful
    ("September", "Earth, Wind"),
    ("Uptown Funk", "Mark Ronson"),
    ("Dancing Queen", "ABBA"),
    ("Don't Stop Me Now", "Queen"),
    ("Here Comes The Sun", "The Beatles"),
    ("Alors on danse", "Stromae"),
    # Rock, metal and hip-hop
    ("Killing In The Name", "Rage Against"),
    ("Smells Like Teen Spirit", "Nirvana"),
    ("Master Of Puppets", "Metallica"),
    ("HUMBLE.", "Kendrick Lamar"),
    ("Beggin'", "Måneskin"),
    ("Comfortably Numb", "Pink Floyd"),
    # French and 80s pop
    ("La vie en rose", "Piaf"),
    ("Sweet Dreams", "Eurythmics"),
]

COLUMNS = {
    "track_id": "track_id",
    "track_name": "title",
    "artists": "artist",
    "album_name": "album",
    "track_genre": "genre_label",
    "popularity": "popularity",
    "danceability": "danceability",
    "energy": "energy",
    "valence": "valence",
    "acousticness": "acousticness",
    "instrumentalness": "instrumentalness",
    "tempo": "tempo",
}


def load_raw_dataset() -> pd.DataFrame:
    """Download the raw CSV once, then read it (the first column is just a row index)."""
    path = hf_hub_download(REPO_ID, "dataset.csv", repo_type="dataset", local_dir=RAW_DIR)
    return pd.read_csv(path, index_col=0)


def find_track(dataset: pd.DataFrame, name: str, artist: str) -> pd.Series:
    """Return the most popular row whose title and artists contain the given substrings."""
    matches = dataset[
        dataset["track_name"].str.contains(name, case=False, regex=False, na=False)
        & dataset["artists"].str.contains(artist, case=False, regex=False, na=False)
    ]
    if matches.empty:
        raise SystemExit(f"Track not found in the dataset: {name!r} by {artist!r}")
    return matches.sort_values("popularity", ascending=False).iloc[0]


def main() -> None:
    dataset = load_raw_dataset()
    rows = [find_track(dataset, name, artist) for name, artist in CURATED_TRACKS]

    tracks = pd.DataFrame(rows)[list(COLUMNS)].rename(columns=COLUMNS)
    tracks["artist"] = tracks["artist"].str.replace(";", ", ", regex=False)

    duplicated = tracks["track_id"].duplicated()
    if duplicated.any():
        raise SystemExit(f"Duplicate tracks picked: {tracks.loc[duplicated, 'title'].tolist()}")

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    tracks.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(tracks)} tracks to {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
