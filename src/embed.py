"""Build data/embeddings.npz: one vector per mood description.

Reads data/descriptions.csv, encodes the `description` column with a local
sentence-transformers model and saves the vectors next to their item type and id.
Only the description text is embedded: titles and artist names are left out on
purpose, so similarity cannot "cheat" by matching names.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DESCRIPTIONS_PATH = PROJECT_ROOT / "data" / "descriptions.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "embeddings.npz"


def main() -> None:
    descriptions = pd.read_csv(DESCRIPTIONS_PATH, dtype={"item_id": str})

    model = SentenceTransformer(MODEL_NAME)
    # Unit-length vectors: cosine similarity becomes a plain dot product.
    vectors = model.encode(descriptions["description"].tolist(), normalize_embeddings=True)

    np.savez(
        OUTPUT_PATH,
        item_type=descriptions["item_type"].to_numpy(dtype=str),
        item_id=descriptions["item_id"].to_numpy(dtype=str),
        vectors=vectors.astype(np.float32),
    )
    print(f"Saved {vectors.shape[0]} vectors of size {vectors.shape[1]} to {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
