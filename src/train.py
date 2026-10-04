"""Train the recommender on the curated picks and save it to data/model.json.

data/curated.csv lists, for every title, about ten tracks hand-picked as good
matches. A logistic regression learns to tell picked (title, track) pairs from
all the others, using the inputs built in src/model.py.

The model is first tested on titles it has not seen (5-fold cross-validation by
title), compared with the hand-weighted baseline, and then fitted on everything.
"""

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from model import MODEL_PATH, TITLES_PATH, build_features, feature_names, load_vectors

FOLDS = 5
NEIGHBOUR_SHARPNESS = 20.0  # higher = only very similar titles lend their picks
REGULARISATION = 0.3  # the C of LogisticRegression: smaller = simpler model
CURATED_BOOST = 3.0  # added to the score of tracks hand-picked for the title itself
BASELINE_COLUMN = -2  # position of the "baseline" input, see model.feature_names


def fit(features: np.ndarray, picked: np.ndarray, rows: np.ndarray) -> tuple[StandardScaler, LogisticRegression]:
    """Fit the model on the (title, track) pairs of the given title rows."""
    inputs = features[rows].reshape(-1, features.shape[-1])
    targets = picked[rows].reshape(-1)
    scaler = StandardScaler().fit(inputs)
    # Picked pairs are about 2% of all pairs: "balanced" stops the model from just saying no.
    model = LogisticRegression(C=REGULARISATION, class_weight="balanced", max_iter=500)
    model.fit(scaler.transform(inputs), targets)
    return scaler, model


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
    title_count = len(pd.read_csv(TITLES_PATH))
    order = np.random.default_rng(0).permutation(title_count)

    baseline_results, trained_results = [], []
    for fold in range(FOLDS):
        test = order[fold::FOLDS]
        train = np.setdiff1d(order, test)
        # Only the training titles lend their picks to the neighbours input.
        _, _, features, picked = build_features(known=train, sharpness=NEIGHBOUR_SHARPNESS)
        scaler, model = fit(features, picked, train)
        scores = model.decision_function(scaler.transform(features.reshape(-1, features.shape[-1])))
        trained_results.append(report(scores.reshape(picked.shape), picked, test))
        baseline_results.append(report(features[..., BASELINE_COLUMN], picked, test))

    print(f"Tested on unseen titles ({FOLDS}-fold cross-validation, {picked.sum()} curated picks)")
    print(f"{'':<28}{'precision@10':>13}{'recall@50':>11}{'no hit in top 10':>18}")
    for name, results in (("hand-set weights", baseline_results), ("trained model", trained_results)):
        precision, recall, no_hit = np.mean(results, axis=0)
        print(f"{name:<28}{precision:>13.3f}{recall:>11.3f}{no_hit:>17.1%}")

    # Final model: fitted on every title.
    everything = np.arange(title_count)
    _, _, features, picked = build_features(known=everything, sharpness=NEIGHBOUR_SHARPNESS)
    scaler, model = fit(features, picked, everything)

    facets, _, _ = load_vectors()
    names = feature_names(facets)
    saved = json.loads(MODEL_PATH.read_text())
    saved["trained"] = {
        "features": names,
        "mean": scaler.mean_.round(6).tolist(),
        "scale": scaler.scale_.round(6).tolist(),
        "coefficients": model.coef_[0].round(6).tolist(),
        "intercept": round(float(model.intercept_[0]), 6),
        "neighbour_sharpness": NEIGHBOUR_SHARPNESS,
        "curated_boost": CURATED_BOOST,
    }
    MODEL_PATH.write_text(json.dumps(saved, indent=2) + "\n")

    strongest = sorted(zip(names, model.coef_[0]), key=lambda pair: -abs(pair[1]))[:8]
    print("\nStrongest inputs: " + ", ".join(f"{name} {weight:+.2f}" for name, weight in strongest))
    print(f"Saved the trained model to {MODEL_PATH.name}")


if __name__ == "__main__":
    main()
