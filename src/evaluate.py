"""Sanity check: do titles rank their own soundtrack tracks near the top?

A few tracks in the dataset come from the soundtrack of a title in the dataset.
The descriptions never mention names, so if the recommender still ranks those
tracks high for their title, the mood descriptions carry real signal.
This is a small sanity check (5 pairs), not a proper evaluation.
"""

import numpy as np
import pandas as pd

from recommend import TITLES_PATH, TRACKS_PATH, load_vectors

# (tmdb_id of the title, track_id of a track from its soundtrack)
SOUNDTRACK_PAIRS = [
    ("194", "14rZjW3RioG7WesZhYESso"),  # Amélie / Comptine d'un autre été
    ("64690", "0U0ldCRmgCqhVvD6ksG63j"),  # Drive / Nightcall
    ("64690", "6ei4QrpcciclGH593uHKo8"),  # Drive / A Real Hero
    ("129", "3gFQOMoUwlR6aUZj81gCzu"),  # Spirited Away / One Summer Day
    ("680", "6ek9SiEj5a65WIs2EV7qiM"),  # Pulp Fiction / Son Of A Preacher Man
]


def main() -> None:
    title_vectors = load_vectors("title")
    track_vectors = load_vectors("track")
    track_ids = list(track_vectors)
    track_matrix = np.stack([track_vectors[track_id] for track_id in track_ids])

    title_names = pd.read_csv(TITLES_PATH, dtype={"tmdb_id": str}).set_index("tmdb_id")["title"]
    track_names = pd.read_csv(TRACKS_PATH).set_index("track_id")["title"]

    ranks = []
    for tmdb_id, track_id in SOUNDTRACK_PAIRS:
        scores = track_matrix @ title_vectors[tmdb_id]
        # Rank 1 = most similar track; count how many tracks score strictly higher.
        rank = int((scores > scores[track_ids.index(track_id)]).sum()) + 1
        ranks.append(rank)
        print(f"{title_names[tmdb_id]:<16} {track_names[track_id]:<40} rank {rank:>2} of {len(track_ids)}")

    mean_reciprocal_rank = np.mean([1 / rank for rank in ranks])
    random_mrr = np.mean([1 / rank for rank in range(1, len(track_ids) + 1)])
    print(f"\nMean rank: {np.mean(ranks):.1f} (random guess: {(len(track_ids) + 1) / 2:.1f})")
    print(f"Mean reciprocal rank: {mean_reciprocal_rank:.2f} (random guess: {random_mrr:.2f})")


if __name__ == "__main__":
    main()
