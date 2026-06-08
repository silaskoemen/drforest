"""Combined ablation: {splitting criterion} × {shrinkage variant} × {dataset}.

Extends the marginal-shrinkage frontier into a small grid so two questions get
answered at once:

1. Does distributional (``mmd_rff`` / ``sliced_wasserstein``) splitting buy
   anything over mean-only ``cart`` splitting, on RMSE *and* CRPS / energy score?
2. Does marginal shrinkage ever move the frontier — and if so, does the
   bias-corrected ``stein`` intensity beat the raw ``kmse`` one?

``enb`` is the strong-signal real dataset (shrinkage expected ≈ no-op); the
synthetic toys add weak-signal / noisy-leaf regimes where ``α̂`` should fire.

External datasets must first be materialised once with
``pixi run python benchmarks/data/fetch_mtr.py``. Then run with::

    pixi run python benchmarks/studies/run_ablation.py
"""

import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np
from loguru import logger

from benchmarks.results_io import write_json_result
from drforest.criteria.adaptive_mmd import AdaptiveMmdCriterion
from drforest.criteria.anisotropic_mmd import AnisotropicMmdCriterion
from drforest.criteria.cart import CartCriterion
from drforest.criteria.mmd_rff import MmdRffCriterion
from drforest.criteria.sliced_wasserstein import SlicedWassersteinCriterion
from drforest.datasets import load_dataset
from drforest.features.rff import (
    coordinatewise_median_heuristic,
    median_heuristic,
    sample_rff,
)
from drforest.forest import DistributionalRandomForest
from drforest.metrics import componentwise_crps, mean_energy_score, rmse
from drforest.shrinkage import parent_target, shrink, shrink_to_target
from drforest.targets import weighted_mean
from drforest.tree import TreeParams
from drforest.weights import assemble_weights

CRITERIA = ("cart", "mmd_rff", "sliced_wasserstein")
VARIANTS = ("raw", "marginal_kmse", "marginal_stein", "parent_kmse", "parent_stein")
DEFAULT_DATASETS = ("enb", "shrinkage_toy", "paper_quantile_2")
METRICS = ("RMSE", "energy", "CRPS")
STUDY_NAME = "run_ablation"
DEFAULT_MAX_CUTPOINTS = 32
DEFAULT_ADAPTIVE_SELECTED_FEATURES = 32


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


def _rho_bar(predictions: np.ndarray) -> float:
    """Average pairwise correlation between per-tree flattened predictions."""
    if predictions.shape[0] < 2:
        return float("nan")
    centered = predictions - predictions.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    valid = norms > 0.0
    if int(valid.sum()) < 2:
        return float("nan")
    normalized = centered[valid] / norms[valid, None]
    corr = normalized @ normalized.T
    upper = corr[np.triu_indices(corr.shape[0], k=1)]
    return float(np.mean(upper))


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


def _criterion_factory(
    criterion: str,
    n_features: int,
    sliced_projections: int | None = None,
    adaptive_pool_features: int | None = None,
    adaptive_selected_features: int = DEFAULT_ADAPTIVE_SELECTED_FEATURES,
):
    if criterion == "cart":
        return lambda Y: CartCriterion()
    if criterion == "mmd_rff":
        return lambda Y: MmdRffCriterion.from_data(Y, n_features=n_features, bandwidth_rule=median_heuristic)
    if criterion == "anisotropic_mmd":
        return lambda Y: AnisotropicMmdCriterion.from_data(
            Y, n_features=n_features, bandwidth_rule=coordinatewise_median_heuristic
        )
    if criterion == "sliced_wasserstein":
        n_projections = n_features if sliced_projections is None else sliced_projections
        return lambda Y: SlicedWassersteinCriterion.from_data(Y, n_projections=n_projections)
    if criterion == "adaptive_mmd":
        pool_features = n_features if adaptive_pool_features is None else adaptive_pool_features
        return lambda Y: AdaptiveMmdCriterion.from_data(
            Y,
            pool_features=pool_features,
            selected_features=adaptive_selected_features,
            bandwidth_rule=median_heuristic,
        )
    raise ValueError(f"unknown criterion {criterion!r}")


def _one_run(
    data,
    *,
    criterion: str,
    seed: int,
    n_trees: int,
    n_features: int,
    shrink_features: int,
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
    rff = sample_rff(Y_train.shape[1], shrink_features, median_heuristic(Y_train), np.random.default_rng(seed + 1))
    parent_weights = parent_target(forest.trees, X_test, W.shape[1])

    variants = {"raw": {"scores": _scores(W, Y_train, Y_test), "alpha_mean": 0.0}}
    for target in ("marginal", "parent"):
        for param in ("kmse", "stein"):
            if target == "parent":
                result = shrink_to_target(W, Y_train, rff=rff, target_weights=parent_weights, parameterization=param)
            else:
                result = shrink(W, Y_train, rff=rff, target=target, parameterization=param)
            variants[f"{target}_{param}"] = {
                "scores": _scores(result.weights, Y_train, Y_test),
                "alpha_mean": float(result.alpha.mean()),
            }
    return {
        "seed": seed,
        "criterion": criterion,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "variants": variants,
        "washout": _washout(forest, X_test, Y_train, Y_test),
    }


def run(
    *,
    datasets,
    seed: int,
    repeats: int,
    n_trees: int,
    n_features: int,
    shrink_features: int,
    max_cutpoints: int | None,
    honesty_fraction: float = 0.5,
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
            "seed": seed,
            "repeats": repeats,
            "n_trees": n_trees,
            "n_features": n_features,
            "shrink_features": shrink_features,
            "max_cutpoints": max_cutpoints,
            "honesty_fraction": honesty_fraction,
            "sliced_projections": n_features if sliced_projections is None else sliced_projections,
            "adaptive_pool_features": resolved_adaptive_pool_features,
            "adaptive_selected_features": adaptive_selected_features,
        },
        "datasets": [],
    }
    for dataset in datasets:
        logger.info(f"📊 Running ablation on dataset {dataset!r}")
        data = load_dataset(dataset)
        # acc[(criterion, variant)][metric] -> list over repeats; plus mean alpha
        acc = {(c, v): {m: [] for m in METRICS} for c in CRITERIA for v in VARIANTS}
        alpha = {(c, v): [] for c in CRITERIA for v in VARIANTS}
        washout: dict[str, list[dict[str, Any]]] = {c: [] for c in CRITERIA}
        run_records: list[dict[str, Any]] = []

        for r in range(repeats):
            for criterion in CRITERIA:
                logger.info(f"Run {r+1}/{repeats} | criterion {criterion}")
                run_result = _one_run(
                    data,
                    criterion=criterion,
                    seed=seed + r,
                    n_trees=n_trees,
                    n_features=n_features,
                    shrink_features=shrink_features,
                    max_cutpoints=max_cutpoints,
                    honesty_fraction=honesty_fraction,
                    sliced_projections=sliced_projections,
                    adaptive_pool_features=adaptive_pool_features,
                    adaptive_selected_features=adaptive_selected_features,
                )
                run_records.append(run_result)
                washout[criterion].append(cast(dict[str, Any], run_result["washout"]))
                for variant, result in cast(dict[str, dict[str, Any]], run_result["variants"]).items():
                    scores = cast(dict[str, float], result["scores"])
                    for m in METRICS:
                        acc[(criterion, variant)][m].append(scores[m])
                    alpha[(criterion, variant)].append(result["alpha_mean"])

        logger.success(f"📊 Finished running ablation on dataset {dataset!r}")
        logger.info(
            f"=== {dataset}  (n={data.X.shape[0]}, d={data.Y.shape[1]}, repeats={repeats}, seeds {seed}..{seed + repeats - 1}) ==="
        )
        header = f"{'criterion':<20}{'variant':<16}{'RMSE':>9}{'energy':>9}{'CRPS':>9}{'alpha':>9}"
        logger.info(header)
        logger.info("-" * len(header))
        summary = []
        for criterion in CRITERIA:
            for variant in VARIANTS:
                cell = acc[(criterion, variant)]
                vals = {m: np.mean(cell[m]) for m in METRICS}
                a = np.mean(alpha[(criterion, variant)])
                logger.info(
                    f"{criterion:<20}{variant:<16}{vals['RMSE']:>9.4f}{vals['energy']:>9.4f}"
                    f"{vals['CRPS']:>9.4f}{a:>9.4f}"
                )
                summary.append(
                    {
                        "criterion": criterion,
                        "variant": variant,
                        "metrics": {
                            m: {"mean": float(np.mean(cell[m])), "std": float(np.std(cell[m]))} for m in METRICS
                        },
                        "alpha_mean": float(a),
                    }
                )
        washout_summary = {
            criterion: {
                "rho_bar_mean": float(np.nanmean([w["rho_bar"] for w in rows])),
                "single_tree_mean": {m: float(np.mean([w["single_tree_mean"][m] for w in rows])) for m in METRICS},
                "single_tree_std_mean": {m: float(np.mean([w["single_tree_std"][m] for w in rows])) for m in METRICS},
            }
            for criterion, rows in washout.items()
        }
        logger.info("washout:")
        for criterion, values in washout_summary.items():
            logger.info(f"{criterion:<20} rho_bar={values['rho_bar_mean']:.4f}")
        payload["datasets"].append(
            {
                "name": data.name,
                "n": int(data.X.shape[0]),
                "p": int(data.X.shape[1]),
                "d": int(data.Y.shape[1]),
                "runs": run_records,
                "summary": summary,
                "washout_summary": washout_summary,
            }
        )
    logger.info("\n(lower is better for all three metrics; alpha is mean shrinkage intensity)")
    if write_json:
        path = write_json_result(STUDY_NAME, payload, results_dir)
        logger.info(f"wrote JSON: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--n-trees", type=int, default=200)
    parser.add_argument("--n-features", type=int, default=200)
    parser.add_argument("--honesty-fraction", type=float, default=0.5)
    parser.add_argument("--sliced-projections", type=int, default=None)
    parser.add_argument("--adaptive-pool-features", type=int, default=None)
    parser.add_argument("--adaptive-selected-features", type=int, default=DEFAULT_ADAPTIVE_SELECTED_FEATURES)
    parser.add_argument("--shrink-features", type=int, default=1000)
    parser.add_argument("--max-cutpoints", type=int, default=DEFAULT_MAX_CUTPOINTS)
    parser.add_argument("--all-cutpoints", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--no-write-json", action="store_true")
    args = parser.parse_args()
    logger.info("🚀 Starting ablation study")
    logger.info(f"Args: {args}")
    run(
        datasets=args.datasets,
        seed=args.seed,
        repeats=args.repeats,
        n_trees=args.n_trees,
        n_features=args.n_features,
        shrink_features=args.shrink_features,
        max_cutpoints=None if args.all_cutpoints else args.max_cutpoints,
        honesty_fraction=args.honesty_fraction,
        sliced_projections=args.sliced_projections,
        adaptive_pool_features=args.adaptive_pool_features,
        adaptive_selected_features=args.adaptive_selected_features,
        results_dir=args.results_dir,
        write_json=not args.no_write_json,
    )


if __name__ == "__main__":
    main()
