"""Build data/embeddings.npz: one vector per description facet.

Each row of data/descriptions.csv describes an item in three facets (emotions,
plot, references). Every facet is embedded separately with a local
sentence-transformers model, so the recommender can weigh them differently.
The item's own title and artist are never part of the text.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
FACETS = ["emotions", "plot", "references"]
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DESCRIPTIONS_PATH = PROJECT_ROOT / "data" / "descriptions.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "embeddings.npz"


def main() -> None:
    descriptions = pd.read_csv(DESCRIPTIONS_PATH, dtype={"item_id": str})

    model = SentenceTransformer(MODEL_NAME)
    # Unit-length vectors: cosine similarity becomes a plain dot product.
    per_facet = [
        model.encode(descriptions[facet].tolist(), normalize_embeddings=True) for facet in FACETS
    ]
    # Shape (items, facets, dimensions): vectors[i, j] is facet j of item i.
    vectors = np.stack(per_facet, axis=1).astype(np.float32)

    np.savez(
        OUTPUT_PATH,
        item_type=descriptions["item_type"].to_numpy(dtype=str),
        item_id=descriptions["item_id"].to_numpy(dtype=str),
        facets=np.array(FACETS),
        vectors=vectors,
    )
    print(f"Saved vectors of shape {vectors.shape} to {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
