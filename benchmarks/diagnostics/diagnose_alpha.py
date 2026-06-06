"""Decompose the closed-form shrinkage intensity ``α̂`` per test point.

The marginal-shrinkage frontier on ``enb`` was a no-op (α̂ ≈ 0). This script
opens up the closed form to confirm *why*, and to compare the current
parameterization against the bias-corrected James–Stein variant.

Oracle shrinkage of an embedding ``μ̂`` toward target ``μ₀`` minimizes
``E‖(1-α)μ̂ + αμ₀ - μ*‖²``; with estimator variance ``V = E‖μ̂-μ*‖²`` and
squared bias ``D² = ‖μ*-μ₀‖²`` the optimum is ``α* = V / (V + D²)``.

Plug-ins (k(y,y)=1 ⇒ tr Σ_φ = 1-‖μ̂‖²):
    V        = (1 - ‖μ̂‖²) / n_eff                      (variance of the weighted CME)
    MMD²     = ‖μ̂ - μ₀‖²                                (biased: E[MMD²] = D² + V)
    D̂²(unb)  = MMD² - V                                 (bias-corrected squared bias)

    α̂_current = (1-‖μ̂‖²) / ((1-‖μ̂‖²) + n_eff·MMD²) = V / (V + MMD²)
    α̂_js      = V / (V + D̂²) = V / MMD²                 (bias-corrected denominator)

Materialise external datasets once with
``pixi run python benchmarks/data/fetch_mtr.py``, then run with::

    pixi run python benchmarks/diagnostics/diagnose_alpha.py
"""

import argparse

import numpy as np

from drforest.criteria.mmd_rff import MmdRffCriterion
from drforest.datasets import load_dataset
from drforest.features.rff import median_heuristic, sample_rff
from drforest.forest import DistributionalRandomForest
from drforest.shrinkage import marginal_target
from drforest.tree import TreeParams
from drforest.weights import embedding_norm_sq, mmd_to_target, n_eff


def _q(name: str, v: np.ndarray) -> None:
    qs = np.quantile(v, [0.0, 0.25, 0.5, 0.75, 1.0])
    print(
        f"{name:<14} min={qs[0]:.4g}  q25={qs[1]:.4g}  med={qs[2]:.4g}  q75={qs[3]:.4g}  max={qs[4]:.4g}  mean={v.mean():.4g}"
    )


def run(*, dataset: str, seed: int, n_trees: int, n_features: int, shrink_features: int) -> None:
    data = load_dataset(dataset)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(data.X.shape[0])
    n_test = max(1, int(round(0.25 * data.X.shape[0])))
    test, train = perm[:n_test], perm[n_test:]
    X_train, Y_train = data.X[train], data.Y[train]
    X_test = data.X[test]

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

    norm_sq = embedding_norm_sq(W, Y_train, rff)  # ‖μ̂‖²
    var_scale = np.maximum(1.0 - norm_sq, 0.0)  # tr Σ_φ estimate
    ne = n_eff(W)  # 1 / Σ w_i²
    mmd = mmd_to_target(W, marginal_target(W.shape[1]), Y_train, rff)  # ‖μ̂-μ₀‖² (biased)

    V = var_scale / ne  # variance of the CME
    D2_unb = np.maximum(mmd - V, 0.0)  # bias-corrected squared bias

    alpha_current = np.divide(V, V + mmd, out=np.zeros_like(V), where=(V + mmd) > 0)
    alpha_js = np.clip(np.divide(V, mmd, out=np.ones_like(V), where=mmd > 0), 0.0, 1.0)

    print(f"dataset={dataset}  n_train={X_train.shape[0]}  n_test={n_test}  d={Y_train.shape[1]}")
    print(f"forest n_trees={n_trees}  split RFF={n_features}  shrink RFF={shrink_features}\n")
    print("per-test-point distribution of the closed-form ingredients:")
    _q("‖μ̂‖²", norm_sq)
    _q("1-‖μ̂‖²", var_scale)
    _q("n_eff", ne)
    _q("V", V)
    _q("MMD²", mmd)
    _q("D̂²=MMD²-V", D2_unb)
    print()
    print("ratio that controls shrinkage:  V / MMD²  (= α̂_js)")
    _q("V/MMD²", alpha_js)
    print()
    print("how much of MMD² is just estimator variance V (bias-corrected fraction):")
    _q("V/MMD² frac", np.divide(V, mmd, out=np.ones_like(V), where=mmd > 0))
    print()
    _q("α̂_current", alpha_current)
    _q("α̂_js (unb)", alpha_js)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="enb")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-trees", type=int, default=200)
    parser.add_argument("--n-features", type=int, default=200)
    parser.add_argument("--shrink-features", type=int, default=1000)
    args = parser.parse_args()
    run(
        dataset=args.dataset,
        seed=args.seed,
        n_trees=args.n_trees,
        n_features=args.n_features,
        shrink_features=args.shrink_features,
    )


if __name__ == "__main__":
    main()
