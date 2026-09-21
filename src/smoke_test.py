"""Smoke test: check that the embedding model loads and produces (3, 384) vectors."""

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
EXPECTED_SHAPE = (3, 384)


def main() -> None:
    model = SentenceTransformer(MODEL_NAME)

    sentences = [
        "A slow, melancholic sci-fi film about memory and loss.",
        "An ambient electronic track with soft synths and a dreamy mood.",
        "A loud, fast-paced action movie full of car chases.",
    ]

    embeddings = model.encode(sentences)

    print(f"Shape: {embeddings.shape}")
    assert embeddings.shape == EXPECTED_SHAPE, (
        f"Expected {EXPECTED_SHAPE}, got {embeddings.shape}"
    )
    print("OK")


if __name__ == "__main__":
    main()
