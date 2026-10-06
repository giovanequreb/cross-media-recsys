"""Download the votes collected by the web app from Supabase into data/votes.csv.

Needs SUPABASE_URL and SUPABASE_SECRET_KEY in the environment (see .env.example).
The secret key is the only way to READ the votes: the key in app/config.js can only add them.
Never commit the secret key. data/votes.csv is git-ignored.

After downloading, `python src/train.py` uses the votes (see src/model.py, load_votes).
"""

import os
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "votes.csv"
PAGE_SIZE = 1000  # Supabase returns at most 1000 rows per request, so we ask page by page
COLUMNS = ["id", "device_id", "title_key", "track_id", "vote", "rank", "wildcard", "personalised", "model", "created_at"]


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")  # like reading process.env after dotenv.config() in Node
    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SECRET_KEY")
    if not url or not key:
        raise SystemExit("Set SUPABASE_URL and SUPABASE_SECRET_KEY (see .env.example).")

    rows: list[dict] = []
    # The new-style keys go in the "apikey" header only, not in "Authorization: Bearer".
    with httpx.Client(headers={"apikey": key}, timeout=30) as client:
        while True:
            response = client.get(
                f"{url.rstrip('/')}/rest/v1/votes",
                params={"select": ",".join(COLUMNS), "order": "id.asc", "limit": PAGE_SIZE, "offset": len(rows)},
            )
            # Do not print the URL or headers on errors: they contain the project address and the key.
            if response.status_code != 200:
                raise SystemExit(f"Supabase answered {response.status_code}: {response.text[:200]}")
            page = response.json()
            rows.extend(page)
            if len(page) < PAGE_SIZE:
                break

    votes = pd.DataFrame(rows, columns=COLUMNS)
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    votes.to_csv(OUTPUT_PATH, index=False)
    devices = votes["device_id"].nunique()
    print(f"Saved {len(votes)} vote events from {devices} device(s) to {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
