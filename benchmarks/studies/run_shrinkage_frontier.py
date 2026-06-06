"""Milestone-1 result table: does marginal shrinkage move the CRPS/RMSE frontier?

Fits an honest ``mmd_rff`` distributional random forest on a real multi-target
regression dataset, then compares the raw forest weights against the same
weights after closed-form marginal shrinkage, on held-out test points.

Run with::

    pixi run python benchmarks/studies/run_shrinkage_frontier.py

External datasets must first be materialised once::

    pixi run python benchmarks/data/fetch_mtr.py
"""

import argparse

import numpy as np

from drforest.criteria.mmd_rff import MmdRffCriterion
from drforest.datasets import load_dataset
from drforest.features.rff import median_heuristic, sample_rff
from drforest.forest import DistributionalRandomForest
from drforest.metrics import componentwise_crps, mean_energy_score, rmse
from drforest.shrinkage import shrink
from drforest.targets import weighted_mean
from drforest.tree import TreeParams


def _split(n: int, *, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = max(1, int(round(test_fraction * n)))
    return perm[n_test:], perm[:n_test]


def _scores(W, Y_train: np.ndarray, Y_test: np.ndarray) -> dict[str, float]:
    pred_mean = weighted_mean(W, Y_train)
    crps = componentwise_crps(W, Y_train, Y_test).mean(axis=0)
    return {
        "RMSE": rmse(Y_test, pred_mean),
        "energy score": mean_energy_score(W, Y_train, Y_test),
        "CRPS mean": float(np.mean(crps)),
    }


def _one_run(data, *, seed: int, n_trees: int, n_features: int, shrink_features: int):
    train, test = _split(data.X.shape[0], test_fraction=0.25, seed=seed)
    X_train, Y_train = data.X[train], data.Y[train]
    X_test, Y_test = data.X[test], data.Y[test]

    forest = DistributionalRandomForest(
        criterion_factory=lambda Y: MmdRffCriterion.from_data(
            Y, n_features=n_features, bandwidth_rule=median_heuristic
        ),
        seed=seed,
        n_trees=n_trees,
        subsample=0.5,
        tree_params=TreeParams(min_samples_leaf=5, alpha=0.05, honesty_fraction=0.5, colsample=0.7),
    ).fit(X_train, Y_train)

    W = forest.weights(X_test)
    rff = sample_rff(Y_train.shape[1], shrink_features, median_heuristic(Y_train), np.random.default_rng(seed + 1))
    result = shrink(W, Y_train, rff=rff)
    return _scores(W, Y_train, Y_test), _scores(result.weights, Y_train, Y_test), result.alpha


def run(*, dataset: str, seed: int, repeats: int, n_trees: int, n_features: int, shrink_features: int) -> None:
    data = load_dataset(dataset)
    metrics = ("RMSE", "energy score", "CRPS mean")
    raw_runs = {m: [] for m in metrics}
    shr_runs = {m: [] for m in metrics}
    alphas = []

    for r in range(repeats):
        raw, shrunk, alpha = _one_run(
            data, seed=seed + r, n_trees=n_trees, n_features=n_features, shrink_features=shrink_features
        )
        for m in metrics:
            raw_runs[m].append(raw[m])
            shr_runs[m].append(shrunk[m])
        alphas.append(alpha)

    alpha_all = np.concatenate(alphas)
    print(
        f"dataset={dataset}  n={data.X.shape[0]}  d={data.Y.shape[1]}  repeats={repeats} (seeds {seed}..{seed + repeats - 1})"
    )
    print(f"forest: mmd_rff (median heuristic), n_trees={n_trees}, honest split-sample")
    print(
        f"shrinkage: marginal, alpha mean={alpha_all.mean():.4f}  median={np.median(alpha_all):.4f}  max={alpha_all.max():.4f}"
    )
    print()

    header = f"{'metric':<16}{'raw':>18}{'+marginal':>18}{'delta':>12}"
    print(header)
    print("-" * len(header))
    for m in metrics:
        raw_v = np.array(raw_runs[m])
        shr_v = np.array(shr_runs[m])
        delta = (shr_v - raw_v).mean()
        print(
            f"{m:<16}{raw_v.mean():>10.4f} ± {raw_v.std():<5.4f}"
            f"{shr_v.mean():>10.4f} ± {shr_v.std():<5.4f}{delta:>+12.4f}"
        )
    print()
    print("(delta = mean(shrunk - raw) across repeats; negative = shrinkage improves)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="enb")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5, help="number of train/test splits to average over")
    parser.add_argument("--n-trees", type=int, default=200)
    parser.add_argument("--n-features", type=int, default=200, help="RFF count for the split criterion")
    parser.add_argument("--shrink-features", type=int, default=1000, help="RFF count for shrinkage intensity")
    args = parser.parse_args()
    run(
        dataset=args.dataset,
        seed=args.seed,
        repeats=args.repeats,
        n_trees=args.n_trees,
        n_features=args.n_features,
        shrink_features=args.shrink_features,
    )


if __name__ == "__main__":
    main()
