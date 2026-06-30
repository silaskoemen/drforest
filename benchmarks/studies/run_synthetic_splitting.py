"""Raw splitting-criterion suite on the paper synthetic DGPs.

This study isolates the splitting geometry question by fitting raw forests only:
no shrinkage targets, no alpha formulas. It is the first gate for deciding
whether distributional splitting has a characterizable win regime.

Run with::

    pixi run python benchmarks/studies/run_synthetic_splitting.py
"""

import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np
from loguru import logger

from benchmarks.results_io import write_json_result
from benchmarks.studies.run_ablation import (
    _criterion_factory,
    _rho_bar,
    _scores,
    _split,
)
from drforest.datasets import load_dataset
from drforest.forest import DistributionalRandomForest
from drforest.targets import weighted_mean
from drforest.tree import TreeParams
from drforest.weights import assemble_weights

STUDY_NAME = "run_synthetic_splitting"
DEFAULT_DATASETS = ("paper_quantile_1", "paper_quantile_2", "paper_quantile_3", "paper_copula")
DEFAULT_CRITERIA = ("cart", "mmd_rff", "sliced_wasserstein")
METRICS = ("RMSE", "energy", "CRPS")
DEFAULT_MAX_CUTPOINTS = 32
DEFAULT_ADAPTIVE_SELECTED_FEATURES = 32


def _split_feature_counts(forest: DistributionalRandomForest) -> list[int]:
    counts = np.zeros(forest.trees[0].n_features_in, dtype=np.int64)
    for tree in forest.trees:
        internal = tree.feature[tree.feature >= 0]
        if internal.shape[0] == 0:
            continue
        feature_ids, feature_counts = np.unique(internal, return_counts=True)
        counts[feature_ids] += feature_counts
    return [int(value) for value in counts]


def _washout(
    forest: DistributionalRandomForest, X_test: np.ndarray, Y_train: np.ndarray, Y_test: np.ndarray
) -> dict[str, Any]:
    tree_scores = []
    tree_predictions = []
    for tree in forest.trees:
        W_tree = assemble_weights([tree], X_test, Y_train.shape[0])
        tree_scores.append(_scores(W_tree, Y_train, Y_test))
        tree_predictions.append(weighted_mean(W_tree, Y_train).reshape(-1))
    pred = np.vstack(tree_predictions)
    return {
        "rho_bar": _rho_bar(pred),
        "single_tree_mean": {m: float(np.mean([score[m] for score in tree_scores])) for m in METRICS},
        "single_tree_std": {m: float(np.std([score[m] for score in tree_scores])) for m in METRICS},
    }


def _one_run(
    data,
    *,
    criterion: str,
    seed: int,
    n_trees: int,
    n_features: int,
    max_cutpoints: int | None,
    honesty_fraction: float,
    sliced_projections: int | None = None,
    adaptive_pool_features: int | None = None,
    adaptive_selected_features: int = DEFAULT_ADAPTIVE_SELECTED_FEATURES,
) -> dict[str, Any]:
    train, test = _split(data.X.shape[0], test_fraction=0.25, seed=seed)
    X_train, Y_train = data.X[train], data.Y[train]
    X_test, Y_test = data.X[test], data.Y[test]

    forest = DistributionalRandomForest(
        criterion_factory=_criterion_factory(
            criterion,
            n_features,
            sliced_projections,
            adaptive_pool_features,
            adaptive_selected_features,
        ),
        seed=seed,
        n_trees=n_trees,
        subsample=0.5,
        tree_params=TreeParams(
            min_samples_leaf=5,
            alpha=0.05,
            honesty_fraction=honesty_fraction,
            colsample=0.7,
            max_cutpoints=max_cutpoints,
        ),
    ).fit(X_train, Y_train)

    W = forest.weights(X_test)
    return {
        "seed": seed,
        "criterion": criterion,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "scores": _scores(W, Y_train, Y_test),
        "washout": _washout(forest, X_test, Y_train, Y_test),
        "split_feature_counts": _split_feature_counts(forest),
    }


def run(
    *,
    datasets,
    criteria,
    seed: int,
    repeats: int,
    n_trees: int,
    n_features: int,
    max_cutpoints: int | None,
    honesty_fractions,
    sliced_projections: int | None = None,
    adaptive_pool_features: int | None = None,
    adaptive_selected_features: int = DEFAULT_ADAPTIVE_SELECTED_FEATURES,
    results_dir: Path | None = None,
    write_json: bool = True,
) -> dict[str, Any]:
    resolved_adaptive_pool_features = n_features if adaptive_pool_features is None else adaptive_pool_features
    payload = {
        "study": STUDY_NAME,
        "params": {
            "datasets": list(datasets),
            "criteria": list(criteria),
            "seed": seed,
            "repeats": repeats,
            "n_trees": n_trees,
            "n_features": n_features,
            "max_cutpoints": max_cutpoints,
            "honesty_fractions": list(honesty_fractions),
            "sliced_projections": n_features if sliced_projections is None else sliced_projections,
            "adaptive_pool_features": resolved_adaptive_pool_features,
            "adaptive_selected_features": adaptive_selected_features,
        },
        "datasets": [],
    }

    for dataset in datasets:
        logger.info(f"📊 Running synthetic splitting suite on dataset {dataset!r}")
        data = load_dataset(dataset)
        cells = [
            (criterion, float(honesty_fraction)) for honesty_fraction in honesty_fractions for criterion in criteria
        ]
        acc = {cell: {m: [] for m in METRICS} for cell in cells}
        washout: dict[tuple[str, float], list[dict[str, Any]]] = {cell: [] for cell in cells}
        split_counts = {cell: np.zeros(data.X.shape[1], dtype=np.int64) for cell in cells}
        run_records: list[dict[str, Any]] = []

        for r in range(repeats):
            for honesty_fraction in honesty_fractions:
                for criterion in criteria:
                    logger.info(f"Repeat: {r + 1}/{repeats} | criterion: {criterion} | honesty: {honesty_fraction:g}")
                    run_result = _one_run(
                        data,
                        criterion=criterion,
                        seed=seed + r,
                        n_trees=n_trees,
                        n_features=n_features,
                        max_cutpoints=max_cutpoints,
                        honesty_fraction=float(honesty_fraction),
                        sliced_projections=sliced_projections,
                        adaptive_pool_features=adaptive_pool_features,
                        adaptive_selected_features=adaptive_selected_features,
                    )
                    run_result["honesty_fraction"] = float(honesty_fraction)
                    run_records.append(run_result)
                    cell_key = (criterion, float(honesty_fraction))
                    scores = cast(dict[str, float], run_result["scores"])
                    for metric in METRICS:
                        acc[cell_key][metric].append(scores[metric])
                    washout[cell_key].append(cast(dict[str, Any], run_result["washout"]))
                    split_counts[cell_key] += np.asarray(run_result["split_feature_counts"], dtype=np.int64)

        logger.success(f"📊 Finished synthetic splitting suite on dataset {dataset!r}")
        logger.info(f"=== {dataset}  (n={data.X.shape[0]}, d={data.Y.shape[1]}, repeats={repeats}) ===")
        header = f"{'criterion':<20}{'honesty':>9}{'RMSE':>9}{'energy':>9}{'CRPS':>9}{'rho_bar':>9}"
        logger.info(header)
        logger.info("-" * len(header))
        summary = []
        for criterion, honesty_fraction in cells:
            cell_key = (criterion, honesty_fraction)
            cell = acc[cell_key]
            values = {m: float(np.mean(cell[m])) for m in METRICS}
            rho = float(np.nanmean([w["rho_bar"] for w in washout[cell_key]]))
            logger.info(
                f"{criterion:<20}{honesty_fraction:>9.2f}{values['RMSE']:>9.4f}{values['energy']:>9.4f}"
                f"{values['CRPS']:>9.4f}{rho:>9.4f}"
            )
            summary.append(
                {
                    "criterion": criterion,
                    "honesty_fraction": honesty_fraction,
                    "metrics": {
                        metric: {
                            "mean": float(np.mean(cell[metric])),
                            "std": float(np.std(cell[metric])),
                        }
                        for metric in METRICS
                    },
                    "washout": {
                        "rho_bar_mean": rho,
                        "single_tree_mean": {
                            m: float(np.mean([w["single_tree_mean"][m] for w in washout[cell_key]])) for m in METRICS
                        },
                        "single_tree_std_mean": {
                            m: float(np.mean([w["single_tree_std"][m] for w in washout[cell_key]])) for m in METRICS
                        },
                    },
                    "split_feature_counts": [int(value) for value in split_counts[cell_key]],
                }
            )
        payload["datasets"].append(
            {
                "name": data.name,
                "n": int(data.X.shape[0]),
                "p": int(data.X.shape[1]),
                "d": int(data.Y.shape[1]),
                "runs": run_records,
                "summary": summary,
            }
        )

    logger.info("\n(lower is better for RMSE, energy, and CRPS)")
    if write_json:
        path = write_json_result(STUDY_NAME, payload, results_dir)
        logger.info(f"wrote JSON: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--criteria", nargs="+", default=list(DEFAULT_CRITERIA))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--n-trees", type=int, default=200)
    parser.add_argument("--n-features", type=int, default=200)
    parser.add_argument("--honesty-fractions", nargs="+", type=float, default=[0.5])
    parser.add_argument("--sliced-projections", type=int, default=None)
    parser.add_argument("--adaptive-pool-features", type=int, default=None)
    parser.add_argument("--adaptive-selected-features", type=int, default=DEFAULT_ADAPTIVE_SELECTED_FEATURES)
    parser.add_argument("--max-cutpoints", type=int, default=DEFAULT_MAX_CUTPOINTS)
    parser.add_argument("--all-cutpoints", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--no-write-json", action="store_true")
    args = parser.parse_args()
    logger.info("🚀 Starting synthetic splitting suite")
    logger.info(f"Args: {args}")
    run(
        datasets=args.datasets,
        criteria=args.criteria,
        seed=args.seed,
        repeats=args.repeats,
        n_trees=args.n_trees,
        n_features=args.n_features,
        max_cutpoints=None if args.all_cutpoints else args.max_cutpoints,
        honesty_fractions=args.honesty_fractions,
        sliced_projections=args.sliced_projections,
        adaptive_pool_features=args.adaptive_pool_features,
        adaptive_selected_features=args.adaptive_selected_features,
        results_dir=args.results_dir,
        write_json=not args.no_write_json,
    )


if __name__ == "__main__":
    main()
