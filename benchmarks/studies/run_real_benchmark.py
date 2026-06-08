"""Real-data DRF benchmark harness.

This is deliberately small: it validates real-data loaders, times fitting and
weight assembly, and compares the practical DRF criteria before the Rust builder
work broadens the grid.

Run with::

    pixi run python benchmarks/studies/run_real_benchmark.py \
      --datasets diabetes \
      --criteria cart mmd_rff \
      --honesty-fractions 0.5 0.0
"""

import argparse
import cProfile
import pstats
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
from loguru import logger

from benchmarks.real_datasets import make_real_dataset
from benchmarks.results_io import write_json_result
from benchmarks.studies.run_ablation import _criterion_factory, _scores
from drforest.forest import DistributionalRandomForest
from drforest.tree import TreeParams

STUDY_NAME = "run_real_benchmark"
DEFAULT_DATASETS = ("diabetes",)
DEFAULT_CRITERIA = ("cart", "mmd_rff")
DEFAULT_HONESTY_FRACTIONS = (0.5, 0.0)
DEFAULT_MAX_CUTPOINTS = 32
DEFAULT_ADAPTIVE_SELECTED_FEATURES = 32


def _fit_and_score(
    *,
    dataset_name: str,
    n_train: int,
    n_test: int,
    data_seed: int,
    model_seed: int,
    criterion: str,
    n_trees: int,
    n_features: int,
    max_cutpoints: int | None,
    honesty_fraction: float,
    adaptive_pool_features: int | None,
    adaptive_selected_features: int,
    profile_output: Path | None,
) -> dict[str, Any]:
    data = make_real_dataset(dataset_name, n_train=n_train, n_test=n_test, seed=data_seed)
    forest = DistributionalRandomForest(
        criterion_factory=_criterion_factory(
            criterion,
            n_features,
            None,
            adaptive_pool_features,
            adaptive_selected_features,
        ),
        seed=model_seed,
        n_trees=n_trees,
        subsample=0.5,
        tree_params=TreeParams(
            min_samples_leaf=5,
            alpha=0.05,
            honesty_fraction=honesty_fraction,
            colsample=0.7,
            max_cutpoints=max_cutpoints,
        ),
    )

    profiler = cProfile.Profile() if profile_output is not None else None
    start = time.perf_counter()
    if profiler is None:
        forest.fit(data.X_train, data.Y_train)
    else:
        profiler.enable()
        forest.fit(data.X_train, data.Y_train)
        profiler.disable()
    fit_time = time.perf_counter() - start

    if profiler is not None and profile_output is not None:
        profile_output.parent.mkdir(parents=True, exist_ok=True)
        with profile_output.open("w") as file:
            stats = pstats.Stats(profiler, stream=file).sort_stats("cumulative")
            stats.print_stats(80)

    start = time.perf_counter()
    W = forest.weights(data.X_test)
    weight_time = time.perf_counter() - start

    scores = _scores(W, data.Y_train, data.Y_test)
    return {
        "dataset": data.name,
        "criterion": criterion,
        "honesty_fraction": honesty_fraction,
        "data_seed": data_seed,
        "model_seed": model_seed,
        "n_train": int(data.X_train.shape[0]),
        "n_test": int(data.X_test.shape[0]),
        "p": int(data.X_train.shape[1]),
        "d": int(data.Y_train.shape[1]),
        "n_trees": n_trees,
        "n_features": n_features,
        "max_cutpoints": max_cutpoints,
        "fit_time": fit_time,
        "weight_time": weight_time,
        "n_leaves_mean": float(np.mean([tree.n_leaves for tree in forest.trees])),
        "n_nodes_mean": float(np.mean([tree.n_nodes for tree in forest.trees])),
        "scores": scores,
    }


def run(
    *,
    datasets,
    criteria,
    honesty_fractions,
    seed: int,
    repeats: int,
    n_train: int,
    n_test: int,
    n_trees: int,
    n_features: int,
    max_cutpoints: int | None,
    adaptive_pool_features: int | None = None,
    adaptive_selected_features: int = DEFAULT_ADAPTIVE_SELECTED_FEATURES,
    results_dir: Path | None = None,
    write_json: bool = True,
    profile_dir: Path | None = None,
) -> dict[str, Any]:
    resolved_adaptive_pool_features = n_features if adaptive_pool_features is None else adaptive_pool_features
    payload: dict[str, Any] = {
        "study": STUDY_NAME,
        "params": {
            "datasets": list(datasets),
            "criteria": list(criteria),
            "honesty_fractions": list(honesty_fractions),
            "seed": seed,
            "repeats": repeats,
            "n_train": n_train,
            "n_test": n_test,
            "n_trees": n_trees,
            "n_features": n_features,
            "max_cutpoints": max_cutpoints,
            "adaptive_pool_features": resolved_adaptive_pool_features,
            "adaptive_selected_features": adaptive_selected_features,
        },
        "runs": [],
        "summary": [],
    }

    acc: dict[tuple[str, str, float], dict[str, list[float]]] = {}
    for dataset in datasets:
        for criterion in criteria:
            for honesty_fraction in honesty_fractions:
                acc[(dataset, criterion, float(honesty_fraction))] = {
                    "RMSE": [],
                    "energy": [],
                    "CRPS": [],
                    "fit_time": [],
                    "weight_time": [],
                }

    for r in range(repeats):
        data_seed = seed + r
        model_seed = seed + 10_000 + r
        for dataset in datasets:
            for honesty_fraction in honesty_fractions:
                for criterion in criteria:
                    logger.info(
                        f"dataset={dataset} repeat={r + 1}/{repeats} criterion={criterion} "
                        f"honesty={honesty_fraction:g}"
                    )
                    profile_output = None
                    if profile_dir is not None:
                        profile_output = (
                            profile_dir
                            / f"{dataset}_{criterion}_honesty{float(honesty_fraction):g}_seed{data_seed}.prof.txt"
                        )
                    row = _fit_and_score(
                        dataset_name=dataset,
                        n_train=n_train,
                        n_test=n_test,
                        data_seed=data_seed,
                        model_seed=model_seed,
                        criterion=criterion,
                        n_trees=n_trees,
                        n_features=n_features,
                        max_cutpoints=max_cutpoints,
                        honesty_fraction=float(honesty_fraction),
                        adaptive_pool_features=adaptive_pool_features,
                        adaptive_selected_features=adaptive_selected_features,
                        profile_output=profile_output,
                    )
                    cast(list[dict[str, Any]], payload["runs"]).append(row)
                    cell = acc[(dataset, criterion, float(honesty_fraction))]
                    scores = cast(dict[str, float], row["scores"])
                    for metric in ("RMSE", "energy", "CRPS"):
                        cell[metric].append(scores[metric])
                    cell["fit_time"].append(cast(float, row["fit_time"]))
                    cell["weight_time"].append(cast(float, row["weight_time"]))

    for (dataset, criterion, honesty_fraction), cell in acc.items():
        summary = {
            "dataset": dataset,
            "criterion": criterion,
            "honesty_fraction": honesty_fraction,
            "metrics": {
                metric: {
                    "mean": float(np.mean(cell[metric])),
                    "std": float(np.std(cell[metric])),
                }
                for metric in ("RMSE", "energy", "CRPS")
            },
            "fit_time_mean": float(np.mean(cell["fit_time"])),
            "weight_time_mean": float(np.mean(cell["weight_time"])),
        }
        cast(list[dict[str, Any]], payload["summary"]).append(summary)
        logger.info(
            f"{dataset:<22}{criterion:<16} honesty={honesty_fraction:g} "
            f"RMSE={summary['metrics']['RMSE']['mean']:.4f} "
            f"CRPS={summary['metrics']['CRPS']['mean']:.4f} "
            f"fit={summary['fit_time_mean']:.3f}s weight={summary['weight_time_mean']:.3f}s"
        )

    if write_json:
        path = write_json_result(STUDY_NAME, payload, results_dir)
        logger.info(f"wrote JSON: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--criteria", nargs="+", default=list(DEFAULT_CRITERIA))
    parser.add_argument("--honesty-fractions", nargs="+", type=float, default=list(DEFAULT_HONESTY_FRACTIONS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--n-train", type=int, default=300)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--n-trees", type=int, default=100)
    parser.add_argument("--n-features", type=int, default=200)
    parser.add_argument("--max-cutpoints", type=int, default=DEFAULT_MAX_CUTPOINTS)
    parser.add_argument("--all-cutpoints", action="store_true")
    parser.add_argument("--adaptive-pool-features", type=int, default=None)
    parser.add_argument("--adaptive-selected-features", type=int, default=DEFAULT_ADAPTIVE_SELECTED_FEATURES)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--profile-dir", type=Path, default=None)
    parser.add_argument("--no-write-json", action="store_true")
    args = parser.parse_args()

    run(
        datasets=args.datasets,
        criteria=args.criteria,
        honesty_fractions=args.honesty_fractions,
        seed=args.seed,
        repeats=args.repeats,
        n_train=args.n_train,
        n_test=args.n_test,
        n_trees=args.n_trees,
        n_features=args.n_features,
        max_cutpoints=None if args.all_cutpoints else args.max_cutpoints,
        adaptive_pool_features=args.adaptive_pool_features,
        adaptive_selected_features=args.adaptive_selected_features,
        results_dir=args.results_dir,
        write_json=not args.no_write_json,
        profile_dir=args.profile_dir,
    )


if __name__ == "__main__":
    main()
