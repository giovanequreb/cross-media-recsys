"""Train the recommender network on the curated picks and save it to data/towers.npz.

data/curated.csv lists, for every title, about ten tracks hand-picked as good
matches. A small "two-tower" network learns from them: one linear layer turns a
title's description vectors into 64 numbers, another does the same for a track,
and training pulls each title towards its picked tracks and away from the rest.

The network is first tested on titles it has not seen (5-fold cross-validation
by title), compared with the hand-weighted baseline, and then trained on all.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from model import (
    TOWERS_PATH, load_curated, load_facet_scores, load_model, load_vectors,
    score_matrix, standardise, tower_scores,
)

FOLDS = 5
DIMENSIONS = 64  # size of the space both towers map into
DROPOUT = 0.5  # share of inputs hidden at each step, so the network cannot just memorise
WEIGHT_DECAY = 0.05  # keeps the weights small, for the same reason
LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 32


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


def train(title_vectors: np.ndarray, track_vectors: np.ndarray, picked: np.ndarray, rows: np.ndarray) -> dict:
    """Train the towers on the given title rows and return their weights as arrays."""
    torch.manual_seed(0)
    titles = torch.tensor(title_vectors.reshape(len(title_vectors), -1))
    tracks = torch.tensor(track_vectors.reshape(len(track_vectors), -1))
    targets = torch.tensor(picked, dtype=torch.float32)

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
    all_facets, title_ids, track_ids, similarities = load_facet_scores()
    weights, hub_correction = load_model()
    _, title_vectors, track_vectors = load_vectors()
    picked = load_curated(title_ids, track_ids)
    baseline = score_matrix(all_facets, similarities, weights, hub_correction)

    order = np.random.default_rng(0).permutation(len(title_ids))
    results: dict[str, list[np.ndarray]] = {"baseline (hand-set weights)": [], "network alone": [], "network + baseline": []}
    for fold in range(FOLDS):
        test = order[fold::FOLDS]
        towers = train(title_vectors, track_vectors, picked, rows=np.setdiff1d(order, test))
        learned = tower_scores(title_vectors, track_vectors, towers)
        results["baseline (hand-set weights)"].append(report(baseline, picked, test))
        results["network alone"].append(report(learned, picked, test))
        results["network + baseline"].append(report(standardise(learned) + standardise(baseline), picked, test))

    print(f"Tested on unseen titles ({FOLDS}-fold cross-validation, {picked.sum()} curated picks)")
    print(f"{'':<30}{'precision@10':>13}{'recall@50':>11}{'no hit in top 10':>18}")
    for name, folds in results.items():
        precision, recall, no_hit = np.mean(folds, axis=0)
        print(f"{name:<30}{precision:>13.3f}{recall:>11.3f}{no_hit:>17.1%}")

    # Final network: trained on every title.
    everything = np.arange(len(title_ids))
    towers = train(title_vectors, track_vectors, picked, rows=everything)
    np.savez_compressed(TOWERS_PATH, **{name: values.astype(np.float16) for name, values in towers.items()})

    final = standardise(tower_scores(title_vectors, track_vectors, towers)) + standardise(baseline)
    precision, recall, no_hit = report(final, picked, everything)
    print(f"{'on the titles it trained on':<30}{precision:>13.3f}{recall:>11.3f}{no_hit:>17.1%}")
    print(f"\nSaved the network to {TOWERS_PATH.name}")


if __name__ == "__main__":
    main()
