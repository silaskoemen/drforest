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
) -> dict[str, Any]:
    train, test = _split(data.X.shape[0], test_fraction=0.25, seed=seed)
    X_train, Y_train = data.X[train], data.Y[train]
    X_test, Y_test = data.X[test], data.Y[test]

    forest = DistributionalRandomForest(
        criterion_factory=_criterion_factory(criterion, n_features),
        seed=seed,
        n_trees=n_trees,
        subsample=0.5,
        tree_params=TreeParams(
            min_samples_leaf=5,
            alpha=0.05,
            honesty_fraction=0.5,
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
    results_dir: Path | None = None,
    write_json: bool = True,
) -> dict[str, Any]:
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
        },
        "datasets": [],
    }

    for dataset in datasets:
        logger.info(f"📊 Running synthetic splitting suite on dataset {dataset!r}")
        data = load_dataset(dataset)
        acc = {criterion: {m: [] for m in METRICS} for criterion in criteria}
        washout: dict[str, list[dict[str, Any]]] = {criterion: [] for criterion in criteria}
        run_records: list[dict[str, Any]] = []

        for r in range(repeats):
            for criterion in criteria:
                run_result = _one_run(
                    data,
                    criterion=criterion,
                    seed=seed + r,
                    n_trees=n_trees,
                    n_features=n_features,
                    max_cutpoints=max_cutpoints,
                )
                run_records.append(run_result)
                scores = cast(dict[str, float], run_result["scores"])
                for metric in METRICS:
                    acc[criterion][metric].append(scores[metric])
                washout[criterion].append(cast(dict[str, Any], run_result["washout"]))

        logger.success(f"📊 Finished synthetic splitting suite on dataset {dataset!r}")
        logger.info(f"=== {dataset}  (n={data.X.shape[0]}, d={data.Y.shape[1]}, repeats={repeats}) ===")
        header = f"{'criterion':<20}{'RMSE':>9}{'energy':>9}{'CRPS':>9}{'rho_bar':>9}"
        logger.info(header)
        logger.info("-" * len(header))
        summary = []
        for criterion in criteria:
            cell = acc[criterion]
            values = {m: float(np.mean(cell[m])) for m in METRICS}
            rho = float(np.nanmean([w["rho_bar"] for w in washout[criterion]]))
            logger.info(
                f"{criterion:<20}{values['RMSE']:>9.4f}{values['energy']:>9.4f}" f"{values['CRPS']:>9.4f}{rho:>9.4f}"
            )
            summary.append(
                {
                    "criterion": criterion,
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
                            m: float(np.mean([w["single_tree_mean"][m] for w in washout[criterion]])) for m in METRICS
                        },
                        "single_tree_std_mean": {
                            m: float(np.mean([w["single_tree_std"][m] for w in washout[criterion]])) for m in METRICS
                        },
                    },
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
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--n-trees", type=int, default=200)
    parser.add_argument("--n-features", type=int, default=200)
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
        results_dir=args.results_dir,
        write_json=not args.no_write_json,
    )


if __name__ == "__main__":
    main()
