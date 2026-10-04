"""Soundtrack check: do titles rank the songs from their own soundtrack near the top?

data/soundtrack_pairs.csv lists (title, track) pairs where the track is famously
used in the title. Descriptions never mention the item's own title or artist, so
ranking these tracks high means the descriptions carry real signal.

The script compares several settings of the model (an ablation) and then runs a
grid search over the facet weights with cross-validation, to show which weights
this check would choose on its own.

Caveat: the `references` facet of a track often describes the scene it is famous
for, so this check favours `references` by construction. It measures "can the
model find a title's famous songs", not "will this person like the track".
"""

from itertools import product

import numpy as np
import pandas as pd

from model import PROJECT_ROOT, load_facet_scores, load_model, score_matrix

PAIRS_PATH = PROJECT_ROOT / "data" / "soundtrack_pairs.csv"
HUB_VALUES = [0.0, 0.5, 1.0]
FOLDS = 5


def pair_ranks(scores: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    """Rank of each paired track for its title (1 = best).

    Other soundtrack tracks of the same title are left out of the ranking, so a
    title with three famous songs is not penalised for ranking them 1, 2 and 3.
    """
    relevant: dict[int, set[int]] = {}
    for title, track in pairs:
        relevant.setdefault(title, set()).add(track)

    ranks = []
    for title, track in pairs:
        competitors = np.ones(scores.shape[1], dtype=bool)
        competitors[list(relevant[title] - {track})] = False
        ranks.append(int((scores[title, competitors] > scores[title, track]).sum()) + 1)
    return np.array(ranks)


def mrr(scores: np.ndarray, pairs: list[tuple[int, int]]) -> float:
    """Mean reciprocal rank: 1.0 if every paired track is ranked first."""
    return float(np.mean(1 / pair_ranks(scores, pairs)))


def weight_grid(facets: list[str], steps: int = 5) -> list[dict[str, float]]:
    """All weight combinations in steps of 1/steps that sum to 1."""
    return [
        dict(zip(facets, (part / steps for part in parts)))
        for parts in product(range(steps + 1), repeat=len(facets))
        if sum(parts) == steps
    ]


def main() -> None:
    facets, title_ids, track_ids, similarities = load_facet_scores()
    weights, hub_correction = load_model()

    pairs_table = pd.read_csv(PAIRS_PATH, dtype=str)
    pairs = [
        (title_ids.index(tmdb_id), track_ids.index(track_id))
        for tmdb_id, track_id in zip(pairs_table["tmdb_id"], pairs_table["track_id"])
    ]
    print(f"{len(pairs)} soundtrack pairs, {len(title_ids)} titles, {len(track_ids)} tracks\n")

    # Ablation: the same check under different settings of the model.
    equal = {facet: 1 / len(facets) for facet in facets}
    settings = [(f"only {facet}", {f: float(f == facet) for f in facets}, 0.0) for facet in facets]
    settings += [
        ("equal weights", equal, 0.0),
        ("model weights, no hub correction", weights, 0.0),
        ("model (data/model.json)", weights, hub_correction),
    ]
    print(f"{'setting':<34}{'MRR':>6}{'hit@1':>7}{'hit@5':>7}{'hit@10':>8}{'mean rank':>11}")
    for name, setting_weights, setting_hub in settings:
        ranks = pair_ranks(score_matrix(facets, similarities, setting_weights, setting_hub), pairs)
        print(
            f"{name:<34}{np.mean(1 / ranks):>6.2f}{np.mean(ranks <= 1):>7.2f}"
            f"{np.mean(ranks <= 5):>7.2f}{np.mean(ranks <= 10):>8.2f}{ranks.mean():>11.1f}"
        )
    random_mrr = np.mean([1 / rank for rank in range(1, len(track_ids) + 1)])
    print(f"{'random guess':<34}{random_mrr:>6.2f}")

    # Grid search with cross-validation: pick the best weights on 4/5 of the
    # titles, measure them on the remaining 1/5, and repeat for every fold.
    candidates = [(w, hub) for w in weight_grid(facets) for hub in HUB_VALUES]
    candidate_scores = [score_matrix(facets, similarities, w, hub) for w, hub in candidates]

    paired_titles = sorted({title for title, _ in pairs})
    np.random.default_rng(0).shuffle(paired_titles)
    held_out_ranks = []
    for fold in range(FOLDS):
        test_titles = set(paired_titles[fold::FOLDS])
        train = [pair for pair in pairs if pair[0] not in test_titles]
        test = [pair for pair in pairs if pair[0] in test_titles]
        best = max(range(len(candidates)), key=lambda i: mrr(candidate_scores[i], train))
        held_out_ranks.extend(pair_ranks(candidate_scores[best], test))

    best = max(range(len(candidates)), key=lambda i: mrr(candidate_scores[i], pairs))
    best_weights, best_hub = candidates[best]
    print(f"\nGrid search ({len(candidates)} settings, {FOLDS}-fold cross-validation by title)")
    print(f"  best on all pairs: weights {best_weights}, hub correction {best_hub}")
    print(f"  MRR on all pairs: {mrr(candidate_scores[best], pairs):.2f}")
    print(f"  MRR on held-out titles: {np.mean(1 / np.array(held_out_ranks)):.2f}")


if __name__ == "__main__":
    main()
