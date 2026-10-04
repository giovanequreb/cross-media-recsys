"""Build data/tracks.csv: hand-picked tracks with audio features.

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
    # Songs famously used in a title from data/titles.csv (see data/soundtrack_pairs.csv)
    ("Cornfield Chase", "Hans Zimmer"),
    ("You Never Can Tell", "Chuck Berry"),
    ("Just Like Honey", "Jesus and Mary Chain"),
    ("Baby Blue", "Badfinger"),
    ("Enemy", "Imagine Dragons"),
    ("Tick Of The Clock", "Chromatics"),
    ("Battle Without Honor", "HOTEI"),
    ("Running Up That Hill", "Kate Bush"),
    ("Should I Stay or Should I Go", "The Clash"),
    ("Una Mattina", "Einaudi"),
    ("Fly", "Einaudi"),
    ("Hooked On A Feeling", "Blue Swede"),
    ("Lust For Life", "Iggy Pop"),
    ("Perfect Day", "Lou Reed"),
    ("Where Is My Mind", "Pixies"),
    ("Mrs. Robinson", "Simon & Garfunkel"),
    ("The Sound of Silence", "Simon & Garfunkel"),
    ("Stuck In The Middle With You", "Stealers Wheel"),
    ("Danger Zone", "Kenny Loggins"),
    ("Take My Breath Away", "Berlin"),
    ("Eye of the Tiger", "Survivor"),
    ("Ghostbusters", "Ray Parker"),
    ("Stayin' Alive", "Bee Gees"),
    ("Still Don't Know My Name", "Labrinth"),
    ("Mystery of Love", "Sufjan Stevens"),
    ("Shallow", "Lady Gaga"),
    ("Lose Yourself", "Eminem"),
    ("All The Stars", "Kendrick Lamar"),
    ("Let It Go", "Idina Menzel"),
    ("All Star", "Smash Mouth"),
    ("There Is a Light That Never Goes Out", "The Smiths"),
    ("Don't You (Forget About Me)", "Simple Minds"),
    ("Lux Aeterna", "Clint Mansell"),
    ("Skyfall", "Adele"),
    ("The End", "The Doors"),
    ("Fortunate Son", "Creedence"),
    ("Tiny Dancer", "Elton John"),
    ("Born To Be Wild", "Steppenwolf"),
    ("Circle of Life", "Carmen Twillie"),
    ("Unchained Melody", "Righteous Brothers"),
    ("Oh, Pretty Woman", "Roy Orbison"),
    ("Main Title", "Ramin Djawadi"),
    ("Light of the Seven", "Ramin Djawadi"),
    # Wider catalogue, so more titles have a fitting track
    # Classical
    ("The Four Seasons - Winter", "Vivaldi"),
    ("Moonlight Sonata", "Beethoven"),
    ("Cello Suite No. 1", "Bach"),
    ("Nocturne No. 2", "Chopin"),
    ("Lacrimosa", "Mozart"),
    ("Ride of the Valkyries", "Wagner"),
    ("Swan Lake", "Tchaikovsky"),
    ("Bagatelle No. 25", "Beethoven"),
    # Modern classical
    ("Experience", "Einaudi"),
    ("River Flows In You", "Yiruma"),
    ("Saturn", "Sleeping At Last"),
    ("Says", "Nils Frahm"),
    ("Near Light", "Ólafur Arnalds"),
    # Classic rock
    ("Stairway to Heaven", "Led Zeppelin"),
    ("Immigrant Song", "Led Zeppelin"),
    ("Wish You Were Here", "Pink Floyd"),
    ("Thunderstruck", "AC/DC"),
    ("Highway to Hell", "AC/DC"),
    ("Welcome To The Jungle", "Guns N' Roses"),
    ("Bohemian Rhapsody", "Queen"),
    ("Paint It, Black", "The Rolling Stones"),
    ("Gimme Shelter", "The Rolling Stones"),
    ("Yesterday", "The Beatles"),
    ("Come Together", "The Beatles"),
    ("Riders on the Storm", "The Doors"),
    ("All Along the Watchtower", "Jimi Hendrix"),
    ("Hotel California", "Eagles"),
    ("Sultans Of Swing", "Dire Straits"),
    # Post-punk, new wave and 80s pop
    ("Love Will Tear Us Apart", "Joy Division"),
    ("Just Like Heaven", "The Cure"),
    ("Boys Don't Cry", "The Cure"),
    ("Blue Monday", "New Order"),
    ("Psycho Killer", "Talking Heads"),
    ("Tainted Love", "Soft Cell"),
    ("Everybody Wants To Rule The World", "Tears For Fears"),
    ("Take on Me", "a-ha"),
    ("Africa", "TOTO"),
    ("Don't Stop Believin'", "Journey"),
    ("Every Breath You Take", "The Police"),
    ("Time After Time", "Cyndi Lauper"),
    ("Under Pressure", "Queen"),
    ("Thriller", "Michael Jackson"),
    ("Purple Rain", "Prince"),
    # 90s and 2000s alternative
    ("Creep", "Radiohead"),
    ("Exit Music", "Radiohead"),
    ("No Surprises", "Radiohead"),
    ("Do I Wanna Know?", "Arctic Monkeys"),
    ("Reptilia", "The Strokes"),
    ("Feel Good Inc.", "Gorillaz"),
    ("Clint Eastwood", "Gorillaz"),
    ("In the End", "Linkin Park"),
    ("Chop Suey!", "System Of A Down"),
    ("Du hast", "Rammstein"),
    ("Hurt", "Johnny Cash"),
    ("Seven Nation Army", "The White Stripes"),
    ("Zombie", "The Cranberries"),
    ("Wonderwall", "Oasis"),
    ("Bitter Sweet Symphony", "The Verve"),
    ("Losing My Religion", "R.E.M."),
    ("Black Hole Sun", "Soundgarden"),
    ("Mr. Brightside", "The Killers"),
    # Hip-hop
    ("Still D.R.E.", "Dr. Dre"),
    ("California Love", "2Pac"),
    ("Juicy", "The Notorious B.I.G."),
    ("SICKO MODE", "Travis Scott"),
    ("Gangsta's Paradise", "Coolio"),
    ("C.R.E.A.M.", "Wu-Tang Clan"),
    ("N.Y. State of Mind", "Nas"),
    # Pop
    ("bad guy", "Billie Eilish"),
    ("bury a friend", "Billie Eilish"),
    ("lovely", "Billie Eilish"),
    ("Someone Like You", "Adele"),
    ("Rolling in the Deep", "Adele"),
    ("Starboy", "The Weeknd"),
    ("Take Me To Church", "Hozier"),
    ("Paper Planes", "M.I.A."),
    ("Believer", "Imagine Dragons"),
    # Electronic
    ("Hey Boy Hey Girl", "The Chemical Brothers"),
    ("Roygbiv", "Boards of Canada"),
    ("Archangel", "Burial"),
    ("The Model", "Kraftwerk"),
    ("Angel", "Massive Attack"),
    ("Sandstorm", "Darude"),
    ("Wake Me Up", "Avicii"),
    ("Resonance", "Home"),
    ("Outro", "M83"),
    # Soul, jazz and standards
    ("Ain't No Sunshine", "Bill Withers"),
    ("Dock of the Bay", "Otis Redding"),
    ("Back To Black", "Amy Winehouse"),
    ("At Last", "Etta James"),
    ("What A Wonderful World", "Louis Armstrong"),
    ("Blue in Green", "Miles Davis"),
    ("The Girl From Ipanema", "Stan Getz"),
    ("Imagine", "John Lennon"),
    ("Three Little Birds", "Bob Marley"),
    # Folk, country and indie
    ("Blowin' in the Wind", "Bob Dylan"),
    ("Like a Rolling Stone", "Bob Dylan"),
    ("Hallelujah", "Leonard Cohen"),
    ("Holocene", "Bon Iver"),
    ("Jolene", "Dolly Parton"),
    ("Take Me Home, Country Roads", "John Denver"),
    ("Sweet Home Alabama", "Lynyrd Skynyrd"),
    ("Kids", "MGMT"),
    ("Pumped Up Kicks", "Foster The People"),
    ("Sweater Weather", "The Neighbourhood"),
    ("Riptide", "Vance Joy"),
    ("Ho Hey", "The Lumineers"),
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
