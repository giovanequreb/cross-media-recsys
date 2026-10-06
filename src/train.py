"""Train the recommender network on the curated picks and the votes and save it to data/towers.npz.

data/curated.csv lists, for every title, about ten tracks hand-picked as good
matches. A small "two-tower" network learns from them: one linear layer turns a
title's description vectors into 64 numbers, another does the same for a track,
and training pulls each title towards its picked tracks and away from the rest.

If data/votes.csv exists (src/pull_votes.py downloads it), the votes join in: tracks
voted "fits" count as extra picks, tracks voted "doesn't fit" are pushed away.

The network is first tested on titles it has not seen (5-fold cross-validation
by title), compared with the hand-weighted baseline, and then trained on all.
"""

import argparse

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from model import (
    TOWERS_PATH, load_curated, load_facet_scores, load_model, load_vectors, load_votes,
    score_matrix, standardise, tower_scores,
)

FOLDS = 5
DIMENSIONS = 64  # size of the space both towers map into
DROPOUT = 0.5  # share of inputs hidden at each step, so the network cannot just memorise
WEIGHT_DECAY = 0.05  # keeps the weights small, for the same reason
LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 32
NEGATIVE_WEIGHT = 0.5  # how hard "doesn't fit" votes push compared with picks pulling


class Towers(nn.Module):
    """Two linear layers, one per medium, that map into a shared space."""

    def __init__(self, inputs: int) -> None:
        super().__init__()
        self.title = nn.Linear(inputs, DIMENSIONS)
        self.track = nn.Linear(inputs, DIMENSIONS)
        self.dropout = nn.Dropout(DROPOUT)
        self.sharpness = nn.Parameter(torch.tensor(2.5))  # learned scale of the similarities

    def forward(self, titles: torch.Tensor, tracks: torch.Tensor) -> torch.Tensor:
        """Return one row per title with a score for every track."""
        title_points = F.normalize(self.title(self.dropout(titles)), dim=1)
        track_points = F.normalize(self.track(self.dropout(tracks)), dim=1)
        return title_points @ track_points.T * self.sharpness.exp()


def train(
    title_vectors: np.ndarray, track_vectors: np.ndarray, positives: np.ndarray, negatives: np.ndarray, rows: np.ndarray
) -> dict:
    """Train the towers on the given title rows and return their weights as arrays.

    positives and negatives are (titles, tracks) weights: how much a track should be
    pulled towards a title, or pushed away from it. Titles with no positive are skipped.
    """
    torch.manual_seed(0)
    rows = rows[positives[rows].sum(axis=1) > 0]
    titles = torch.tensor(title_vectors.reshape(len(title_vectors), -1))
    tracks = torch.tensor(track_vectors.reshape(len(track_vectors), -1))
    targets = torch.tensor(positives, dtype=torch.float32)
    avoid = torch.tensor(negatives, dtype=torch.float32)

    network = Towers(titles.shape[1])
    optimiser = torch.optim.AdamW(network.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    network.train()
    for _ in range(EPOCHS):
        shuffled = torch.tensor(rows)[torch.randperm(len(rows))]
        for start in range(0, len(shuffled), BATCH_SIZE):
            batch = shuffled[start : start + BATCH_SIZE]
            scores = network(titles[batch], tracks)
            # For each title, the picked tracks should get the highest probabilities
            # among all tracks (cross-entropy averaged over that title's picks).
            log_probabilities = F.log_softmax(scores, dim=1)
            loss = -(log_probabilities * targets[batch]).sum(dim=1) / targets[batch].sum(dim=1)
            if avoid[batch].any():
                # "Doesn't fit" votes: penalise the probability given to those tracks, -log(1 - p).
                probabilities = log_probabilities.exp().clamp(max=0.999)
                unlikely = -(torch.log1p(-probabilities) * avoid[batch]).sum(dim=1) / avoid[batch].sum(dim=1).clamp(min=1e-9)
                loss = loss + NEGATIVE_WEIGHT * unlikely
            optimiser.zero_grad()
            loss.mean().backward()
            optimiser.step()

    return {
        "title_weight": network.title.weight.detach().numpy(),
        "title_bias": network.title.bias.detach().numpy(),
        "track_weight": network.track.weight.detach().numpy(),
        "track_bias": network.track.bias.detach().numpy(),
    }


def report(scores: np.ndarray, picked: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Return [precision@10, recall@50, share of titles with no picked track in the top 10]."""
    ranking = np.argsort(-scores[rows], axis=1)
    hits = np.take_along_axis(picked[rows], ranking, axis=1)
    return np.array([
        hits[:, :10].mean(),
        (hits[:, :50].sum(axis=1) / picked[rows].sum(axis=1)).mean(),
        (hits[:, :10].sum(axis=1) == 0).mean(),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--min-precision", type=float, default=0.0,
        help="do not save the network if precision@10 on unseen titles is below this (a guard for automatic retraining)",
    )
    arguments = parser.parse_args()

    all_facets, title_ids, track_ids, similarities = load_facet_scores()
    weights, hub_correction = load_model()
    _, title_vectors, track_vectors = load_vectors()
    picked = load_curated(title_ids, track_ids)
    vote_positive, vote_negative = load_votes(title_ids, track_ids)
    positives = picked.astype(np.float32) + vote_positive  # what the network is trained on
    baseline = score_matrix(all_facets, similarities, weights, hub_correction)
    print(f"Votes used: {int((vote_positive > 0).sum())} positive and {int((vote_negative > 0).sum())} negative (title, track) pairs")

    # The test only uses titles that have curated picks, and only the curated picks, so the
    # numbers stay comparable from run to run whatever the votes say.
    labelled = np.flatnonzero(picked.any(axis=1))
    order = np.random.default_rng(0).permutation(labelled)
    results: dict[str, list[np.ndarray]] = {"baseline (hand-set weights)": [], "network alone": [], "network + baseline": []}
    for fold in range(FOLDS):
        test = order[fold::FOLDS]
        towers = train(title_vectors, track_vectors, positives, vote_negative, rows=np.setdiff1d(np.arange(len(title_ids)), test))
        learned = tower_scores(title_vectors, track_vectors, towers)
        results["baseline (hand-set weights)"].append(report(baseline, picked, test))
        results["network alone"].append(report(learned, picked, test))
        results["network + baseline"].append(report(standardise(learned) + standardise(baseline), picked, test))

    print(f"Tested on unseen titles ({FOLDS}-fold cross-validation, {picked.sum()} curated picks)")
    print(f"{'':<30}{'precision@10':>13}{'recall@50':>11}{'no hit in top 10':>18}")
    for name, folds in results.items():
        precision, recall, no_hit = np.mean(folds, axis=0)
        print(f"{name:<30}{precision:>13.3f}{recall:>11.3f}{no_hit:>17.1%}")

    unseen_precision = np.mean(results["network + baseline"], axis=0)[0]
    if unseen_precision < arguments.min_precision:
        raise SystemExit(f"precision@10 on unseen titles is {unseen_precision:.3f}, below {arguments.min_precision}: network not saved")

    # Final network: trained on every title.
    everything = np.arange(len(title_ids))
    towers = train(title_vectors, track_vectors, positives, vote_negative, rows=everything)
    np.savez_compressed(TOWERS_PATH, **{name: values.astype(np.float16) for name, values in towers.items()})

    final = standardise(tower_scores(title_vectors, track_vectors, towers)) + standardise(baseline)
    precision, recall, no_hit = report(final, picked, labelled)
    print(f"{'on the titles it trained on':<30}{precision:>13.3f}{recall:>11.3f}{no_hit:>17.1%}")
    print(f"\nSaved the network to {TOWERS_PATH.name}")


if __name__ == "__main__":
    main()
