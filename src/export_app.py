"""Export the data the web app needs to app/data.js.

The app (app/index.html) is a static page with no backend: it loads the titles,
the tracks and the per-facet similarities from this file and computes the
recommendations in the browser, so the facet weights can be changed live.
"""

import json

import pandas as pd

from model import PROJECT_ROOT, TITLES_PATH, TRACKS_PATH, load_facet_scores, load_model

DESCRIPTIONS_PATH = PROJECT_ROOT / "data" / "descriptions.csv"
OUTPUT_PATH = PROJECT_ROOT / "app" / "data.js"


def main() -> None:
    facets, title_ids, track_ids, similarities = load_facet_scores()
    weights, hub_correction = load_model()

    descriptions = pd.read_csv(DESCRIPTIONS_PATH, dtype=str).set_index("item_id")
    titles = pd.read_csv(TITLES_PATH, dtype=str).set_index("tmdb_id").loc[title_ids]
    tracks = pd.read_csv(TRACKS_PATH, dtype=str).set_index("track_id").loc[track_ids]

    data = {
        "facets": facets,
        "weights": weights,
        "hubCorrection": hub_correction,
        "titles": [
            {"id": item_id, "title": row["title"], "year": row["year"], "type": row["type"],
             **descriptions.loc[item_id, facets].to_dict()}
            for item_id, row in titles.iterrows()
        ],
        "tracks": [
            {"id": item_id, "title": row["title"], "artist": row["artist"],
             **descriptions.loc[item_id, facets].to_dict()}
            for item_id, row in tracks.iterrows()
        ],
        # similarities[facet][title][track], rounded to keep the file small
        "similarities": similarities.astype(float).round(3).tolist(),
    }

    # A .js file (not .json) so the page also works when opened straight from disk.
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text("window.APP_DATA = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    print(f"Saved {len(title_ids)} titles and {len(track_ids)} tracks to {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
