"""Append more tracks to data/tracks.csv from the list of Spotify ids in data/added_track_ids.txt.

build_tracks.py picks the first tracks by name; this script adds a second group by id.
The ids were chosen among the most popular tracks of the Hugging Face dataset (the
raw CSV in data/raw/, see build_tracks.py), keeping well-known songs from many eras
and genres and at most two per artist. Tracks already in data/tracks.csv are skipped.

    python src/add_tracks.py

Descriptions (data/descriptions.csv) are written afterwards, then run
src/embed.py, src/train.py and src/export_app.py.
"""

import pandas as pd

from build_tracks import COLUMNS, OUTPUT_PATH, PROJECT_ROOT, load_raw_dataset

IDS_PATH = PROJECT_ROOT / "data" / "added_track_ids.txt"


def main() -> None:
    wanted = [line.strip() for line in IDS_PATH.read_text().splitlines() if line.strip()]
    if len(wanted) != len(set(wanted)):
        raise SystemExit("data/added_track_ids.txt has repeated ids")

    existing = pd.read_csv(OUTPUT_PATH, dtype={"track_id": str})
    new_ids = [track_id for track_id in wanted if track_id not in set(existing["track_id"])]

    dataset = load_raw_dataset().drop_duplicates("track_id").set_index("track_id", drop=False)
    missing = [track_id for track_id in new_ids if track_id not in dataset.index]
    if missing:
        raise SystemExit(f"Not in the dataset: {missing[:5]}")

    added = dataset.loc[new_ids, list(COLUMNS)].rename(columns=COLUMNS)
    added["artist"] = added["artist"].str.replace(";", ", ", regex=False)
    combined = pd.concat([existing, added], ignore_index=True)
    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"Added {len(added)} tracks: data/tracks.csv now has {len(combined)}")


if __name__ == "__main__":
    main()
