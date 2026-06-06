"""Combined ablation: {splitting criterion} × {shrinkage variant} × {dataset}.

Extends the marginal-shrinkage frontier into a small grid so two questions get
answered at once:

1. Does distributional (``mmd_rff``) splitting buy anything over mean-only
   ``cart`` splitting, on RMSE *and* CRPS / energy score?
2. Does marginal shrinkage ever move the frontier — and if so, does the
   bias-corrected ``stein`` intensity beat the raw ``kmse`` one?

``enb`` is the strong-signal real dataset (shrinkage expected ≈ no-op); the
synthetic toys add weak-signal / noisy-leaf regimes where ``α̂`` should fire.

External datasets must first be materialised once with
``pixi run python benchmarks/data/fetch_mtr.py``. Then run with::

    pixi run python benchmarks/studies/run_ablation.py
"""

import argparse

import numpy as np

from drforest.criteria.cart import CartCriterion
from drforest.criteria.mmd_rff import MmdRffCriterion
from drforest.datasets import load_dataset
from drforest.features.rff import median_heuristic, sample_rff
from drforest.forest import DistributionalRandomForest
from drforest.metrics import componentwise_crps, mean_energy_score, rmse
from drforest.shrinkage import shrink
from drforest.targets import weighted_mean
from drforest.tree import TreeParams

CRITERIA = ("cart", "mmd_rff")
VARIANTS = ("raw", "kmse", "stein")  # raw forest, then the two shrinkage parameterizations
DEFAULT_DATASETS = ("enb", "shrinkage_toy", "paper_quantile_2")
METRICS = ("RMSE", "energy", "CRPS")


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
        "energy": mean_energy_score(W, Y_train, Y_test),
        "CRPS": float(np.mean(crps)),
    }


def _criterion_factory(criterion: str, n_features: int):
    if criterion == "cart":
        return lambda Y: CartCriterion()
    if criterion == "mmd_rff":
        return lambda Y: MmdRffCriterion.from_data(Y, n_features=n_features, bandwidth_rule=median_heuristic)
    raise ValueError(f"unknown criterion {criterion!r}")


def _one_run(data, *, criterion: str, seed: int, n_trees: int, n_features: int, shrink_features: int):
    train, test = _split(data.X.shape[0], test_fraction=0.25, seed=seed)
    X_train, Y_train = data.X[train], data.Y[train]
    X_test, Y_test = data.X[test], data.Y[test]

    forest = DistributionalRandomForest(
        criterion_factory=_criterion_factory(criterion, n_features),
        seed=seed,
        n_trees=n_trees,
        subsample=0.5,
        tree_params=TreeParams(min_samples_leaf=5, alpha=0.05, honesty_fraction=0.5, colsample=0.7),
    ).fit(X_train, Y_train)

    W = forest.weights(X_test)
    rff = sample_rff(Y_train.shape[1], shrink_features, median_heuristic(Y_train), np.random.default_rng(seed + 1))

    out = {"raw": (_scores(W, Y_train, Y_test), 0.0)}
    for param in ("kmse", "stein"):
        result = shrink(W, Y_train, rff=rff, parameterization=param)
        out[param] = (_scores(result.weights, Y_train, Y_test), float(result.alpha.mean()))
    return out


def run(*, datasets, seed: int, repeats: int, n_trees: int, n_features: int, shrink_features: int) -> None:
    for dataset in datasets:
        data = load_dataset(dataset)
        # acc[(criterion, variant)][metric] -> list over repeats; plus mean alpha
        acc = {(c, v): {m: [] for m in METRICS} for c in CRITERIA for v in VARIANTS}
        alpha = {(c, v): [] for c in CRITERIA for v in VARIANTS}

        for r in range(repeats):
            for criterion in CRITERIA:
                runs = _one_run(
                    data,
                    criterion=criterion,
                    seed=seed + r,
                    n_trees=n_trees,
                    n_features=n_features,
                    shrink_features=shrink_features,
                )
                for variant, (scores, a) in runs.items():
                    for m in METRICS:
                        acc[(criterion, variant)][m].append(scores[m])
                    alpha[(criterion, variant)].append(a)

        print(
            f"\n=== {dataset}  (n={data.X.shape[0]}, d={data.Y.shape[1]}, repeats={repeats}, seeds {seed}..{seed + repeats - 1}) ==="
        )
        header = f"{'criterion':<9}{'variant':<7}{'RMSE':>9}{'energy':>9}{'CRPS':>9}{'ᾱ':>9}"
        print(header)
        print("-" * len(header))
        for criterion in CRITERIA:
            for variant in VARIANTS:
                cell = acc[(criterion, variant)]
                vals = {m: np.mean(cell[m]) for m in METRICS}
                a = np.mean(alpha[(criterion, variant)])
                print(
                    f"{criterion:<9}{variant:<7}{vals['RMSE']:>9.4f}{vals['energy']:>9.4f}{vals['CRPS']:>9.4f}{a:>9.4f}"
                )
    print("\n(lower is better for all three metrics; ᾱ is mean shrinkage intensity)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--n-trees", type=int, default=200)
    parser.add_argument("--n-features", type=int, default=200)
    parser.add_argument("--shrink-features", type=int, default=1000)
    args = parser.parse_args()
    run(
        datasets=args.datasets,
        seed=args.seed,
        repeats=args.repeats,
        n_trees=args.n_trees,
        n_features=args.n_features,
        shrink_features=args.shrink_features,
    )


if __name__ == "__main__":
    main()
