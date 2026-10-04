"""Feasibility check for Level 2: how many users rated both movies and music?

Uses the rating-only files of Amazon Reviews 2023 (McAuley Lab, UCSD):
Movies_and_TV and CDs_and_Vinyl. A cross-domain model needs users who are
active in both domains, so this counts them at a few activity thresholds.

The files are large (about 90 MB and 350 MB compressed), so they are streamed
and decompressed on the fly: nothing is written to disk.
"""

import zlib
from collections import Counter
from collections.abc import Iterator

import httpx

BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/0core/rating_only"
MUSIC_CATEGORY = "CDs_and_Vinyl"
MOVIES_CATEGORY = "Movies_and_TV"
THRESHOLDS = [1, 3, 5, 10, 20]


def stream_user_ids(category: str) -> Iterator[str]:
    """Yield the user_id of every rating in a category, one row at a time."""
    decompressor = zlib.decompressobj(wbits=zlib.MAX_WBITS | 16)  # 16 = gzip format
    leftover = b""
    is_header = True
    with httpx.stream("GET", f"{BASE_URL}/{category}.csv.gz", timeout=60, follow_redirects=True) as response:
        response.raise_for_status()
        for chunk in response.iter_raw():
            lines = (leftover + decompressor.decompress(chunk)).split(b"\n")
            leftover = lines.pop()  # the last piece may be an incomplete row
            for line in lines:
                if is_header:
                    is_header = False
                    continue
                # Row format: user_id,parent_asin,rating,timestamp
                yield line.split(b",", 1)[0].decode()
    if leftover:
        yield leftover.split(b",", 1)[0].decode()


def main() -> None:
    print(f"Streaming {MUSIC_CATEGORY}...")
    music_counts = Counter(stream_user_ids(MUSIC_CATEGORY))
    print(f"  {sum(music_counts.values()):,} ratings from {len(music_counts):,} users")

    # Only users already seen in music can be in the overlap, so the larger
    # movies file is counted for those users only (keeps memory small).
    print(f"Streaming {MOVIES_CATEGORY}...")
    movie_counts: Counter[str] = Counter()
    total_movie_ratings = 0
    for user_id in stream_user_ids(MOVIES_CATEGORY):
        total_movie_ratings += 1
        if user_id in music_counts:
            movie_counts[user_id] += 1
    print(f"  {total_movie_ratings:,} ratings")

    print("\nUsers with at least N ratings in BOTH domains:")
    for threshold in THRESHOLDS:
        users = [
            user_id
            for user_id, movie_count in movie_counts.items()
            if movie_count >= threshold and music_counts[user_id] >= threshold
        ]
        ratings = sum(movie_counts[user_id] + music_counts[user_id] for user_id in users)
        print(f"  N >= {threshold:>2}: {len(users):>9,} users, {ratings:>11,} ratings")


if __name__ == "__main__":
    main()
