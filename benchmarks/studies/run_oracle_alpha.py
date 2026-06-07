"""Oracle fixed-alpha shrinkage sweep.

This is a diagnostic, not a proposed estimator: it asks whether a shrinkage
target can improve full-forest metrics for any fixed alpha. If no alpha > 0
helps, the target/formula family is not worth polishing at the current forest
size.

Run with::

    pixi run python benchmarks/studies/run_oracle_alpha.py
"""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
from loguru import logger
from scipy.sparse import csr_matrix

from benchmarks.results_io import write_json_result
from benchmarks.studies.run_ablation import _criterion_factory, _split
from drforest.datasets import load_dataset
from drforest.forest import DistributionalRandomForest
from drforest.metrics import componentwise_crps, mean_energy_score, rmse
from drforest.mixture import MixtureWeights
from drforest.shrinkage import marginal_target, parent_target
from drforest.targets import weighted_mean
from drforest.tree import TreeParams

STUDY_NAME = "run_oracle_alpha"
DEFAULT_DATASETS = ("enb", "shrinkage_toy", "paper_quantile_2")
DEFAULT_CRITERIA = ("mmd_rff",)
DEFAULT_TARGETS = ("marginal", "parent")
DEFAULT_ALPHAS = (0.0, 0.01, 0.03, 0.1, 0.25, 0.5, 0.75, 1.0)
METRICS = ("RMSE", "energy", "CRPS")
DEFAULT_MAX_CUTPOINTS = 32


def _scores(W: object, Y_train: np.ndarray, Y_test: np.ndarray) -> dict[str, float]:
    pred_mean = weighted_mean(W, Y_train)
    crps = componentwise_crps(W, Y_train, Y_test).mean(axis=0)
    crps_mean = float(np.mean(crps))
    energy = crps_mean if Y_train.shape[1] == 1 else mean_energy_score(W, Y_train, Y_test)
    return {
        "RMSE": rmse(Y_test, pred_mean),
        "energy": energy,
        "CRPS": crps_mean,
    }


def _fixed_alpha_weights(W: csr_matrix, target: csr_matrix, alpha: float) -> MixtureWeights:
    return MixtureWeights(base=W, alpha=np.full(W.shape[0], alpha, dtype=np.float64), target=target)


def _one_run(
    data,
    *,
    criterion: str,
    seed: int,
    n_trees: int,
    n_features: int,
    max_cutpoints: int | None,
    targets: Sequence[str],
    alphas: Sequence[float],
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
    target_weights = {
        "marginal": marginal_target(W.shape[1]),
        "parent": parent_target(forest.trees, X_test, W.shape[1]),
    }
    variants: dict[str, dict[str, Any]] = {"raw": {"scores": _scores(W, Y_train, Y_test), "alpha": 0.0}}
    for target in targets:
        if target not in target_weights:
            raise ValueError(f"unknown target {target!r}; expected one of {sorted(target_weights)}")
        for alpha in alphas:
            weights = _fixed_alpha_weights(W, target_weights[target], float(alpha))
            variants[f"{target}_alpha_{alpha:g}"] = {
                "scores": _scores(weights, Y_train, Y_test),
                "alpha": float(alpha),
                "target": target,
            }
    return {
        "seed": seed,
        "criterion": criterion,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "variants": variants,
    }


def run(
    *,
    datasets: Sequence[str],
    criteria: Sequence[str],
    targets: Sequence[str],
    alphas: Sequence[float],
    seed: int,
    repeats: int,
    n_trees: int,
    n_features: int,
    max_cutpoints: int | None,
    results_dir: Path | None = None,
    write_json: bool = True,
) -> dict[str, Any]:
    if any(alpha < 0.0 or alpha > 1.0 for alpha in alphas):
        raise ValueError(f"all alphas must lie in [0, 1]; got {list(alphas)}")
    payload: dict[str, Any] = {
        "study": STUDY_NAME,
        "params": {
            "datasets": list(datasets),
            "criteria": list(criteria),
            "targets": list(targets),
            "alphas": [float(alpha) for alpha in alphas],
            "seed": seed,
            "repeats": repeats,
            "n_trees": n_trees,
            "n_features": n_features,
            "max_cutpoints": max_cutpoints,
        },
        "datasets": [],
    }

    for dataset in datasets:
        logger.info(f"📊 Running oracle-alpha sweep on dataset {dataset!r}")
        data = load_dataset(dataset)
        variant_names = ["raw"] + [f"{target}_alpha_{alpha:g}" for target in targets for alpha in alphas]
        acc = {(criterion, variant): {m: [] for m in METRICS} for criterion in criteria for variant in variant_names}
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
                    targets=targets,
                    alphas=alphas,
                )
                run_records.append(run_result)
                for variant, result in cast(dict[str, dict[str, Any]], run_result["variants"]).items():
                    scores = cast(dict[str, float], result["scores"])
                    for metric in METRICS:
                        acc[(criterion, variant)][metric].append(scores[metric])

        summary = []
        logger.success(f"📊 Finished oracle-alpha sweep on dataset {dataset!r}")
        logger.info(f"=== {dataset}  (n={data.X.shape[0]}, d={data.Y.shape[1]}, repeats={repeats}) ===")
        header = f"{'criterion':<20}{'variant':<24}{'RMSE':>9}{'energy':>9}{'CRPS':>9}"
        logger.info(header)
        logger.info("-" * len(header))
        for criterion in criteria:
            for variant in variant_names:
                cell = acc[(criterion, variant)]
                means = {m: float(np.mean(cell[m])) for m in METRICS}
                logger.info(
                    f"{criterion:<20}{variant:<24}{means['RMSE']:>9.4f}" f"{means['energy']:>9.4f}{means['CRPS']:>9.4f}"
                )
                summary.append(
                    {
                        "criterion": criterion,
                        "variant": variant,
                        "metrics": {
                            metric: {
                                "mean": float(np.mean(cell[metric])),
                                "std": float(np.std(cell[metric])),
                            }
                            for metric in METRICS
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

    logger.info("\n(lower is better for all metrics; fixed-alpha variants are oracle diagnostics)")
    if write_json:
        path = write_json_result(STUDY_NAME, payload, results_dir)
        logger.info(f"wrote JSON: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--criteria", nargs="+", default=list(DEFAULT_CRITERIA))
    parser.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGETS))
    parser.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--n-trees", type=int, default=200)
    parser.add_argument("--n-features", type=int, default=200)
    parser.add_argument("--max-cutpoints", type=int, default=DEFAULT_MAX_CUTPOINTS)
    parser.add_argument("--all-cutpoints", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--no-write-json", action="store_true")
    args = parser.parse_args()
    logger.info("🚀 Starting oracle-alpha sweep")
    logger.info(f"Args: {args}")
    run(
        datasets=args.datasets,
        criteria=args.criteria,
        targets=args.targets,
        alphas=args.alphas,
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
