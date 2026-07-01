<p align="center">
  <img src="https://raw.githubusercontent.com/silaskoemen/drforest/main/assets/drforest_logo.png" width="50%" alt="drforest logo">
</p>

# drforest

[![PyPI](https://img.shields.io/pypi/v/drforest?style=flat-square)](https://pypi.org/project/drforest/)
[![CI](https://img.shields.io/github/actions/workflow/status/silaskoemen/drforest/ci.yml?style=flat-square&branch=main)](https://github.com/silaskoemen/drforest/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](https://github.com/silaskoemen/drforest/blob/main/LICENSE)

`drforest` is a Python and Rust implementation of distributional random
forests. A fitted forest estimates a conditional distribution as sparse weights
over the training responses, then derives means, quantiles, CDFs, and
distributional scores from the same estimate.

The package supports scalar and multivariate responses. Split search is
Rust-backed and offers CART, Gaussian maximum mean discrepancy (MMD), adaptive
and anisotropic MMD variants, and sliced Wasserstein separation.

## Installation

Install the published package from PyPI:

```bash
python -m pip install drforest
```

Python 3.11 through 3.14 are supported. Binary wheels are published for Linux,
macOS, and Windows on supported architectures; pip falls back to the source
distribution when no compatible wheel is available.

## Quick start

The estimator accepts ordinary one-dimensional regression targets. By default,
it uses distribution-sensitive MMD splitting.

```python
import numpy as np

from drforest import DistributionalRandomForest

rng = np.random.default_rng(0)
X = rng.normal(size=(500, 4))
y = X[:, 0] + (0.5 + np.abs(X[:, 1])) * rng.normal(size=500)

X_train, X_test = X[:400], X[400:]
y_train, y_test = y[:400], y[400:]

forest = DistributionalRandomForest(
    criterion="mmd",
    n_estimators=20,
    subsample=0.5,
    min_samples_leaf=5,
    honesty_fraction=0.5,
    colsample=0.7,
    max_cutpoints=32,
    random_state=0,
).fit(X_train, y_train)

mean = forest.predict(X_test)
quantiles = forest.predict_quantiles(X_test, [0.1, 0.5, 0.9])
cdf = forest.predict_cdf(X_test, [-1.0, 0.0, 1.0])

assert mean.shape == (100,)
assert quantiles.shape == (100, 3)
assert cdf.shape == (100, 3)
```

For a two-dimensional response array with shape `(n_samples, n_outputs)`,
`predict` returns `(n_test, n_outputs)`. Quantile and CDF predictions return
`(n_test, n_outputs, n_values)`.

## Prediction interface

- `predict(X)` returns the conditional mean.
- `predict_quantiles(X, quantiles)` returns marginal conditional quantiles.
- `predict_cdf(X, thresholds)` evaluates marginal conditional CDFs.
- `predict_weights(X)` returns the sparse conditional weight matrix.

The prediction methods retain the training responses during `fit`; callers do
not need to pass them again.

## Split criteria

Pass a built-in name through `criterion`:

| Name | Split geometry | Built-in configuration |
| --- | --- | --- |
| `"mmd"` or `"mmd_rff"` | Gaussian MMD | 128 random Fourier features, median bandwidth |
| `"cart"` | Multivariate mean separation | No additional configuration |
| `"anisotropic_mmd"` | Coordinatewise-bandwidth MMD | 128 random Fourier features |
| `"adaptive_mmd"` | Adaptive-frequency MMD | 128 pooled, 32 selected features |
| `"sliced_wasserstein"` | Sliced Wasserstein distance | 128 projections |

If neither `criterion` nor `criterion_factory` is supplied, `"mmd"` is used.
An explicit criterion name takes precedence when both are supplied.

## Research and custom targets

The sparse weight matrix remains the core representation for custom targets,
metrics, shrinkage, and criterion experiments:

```python
from drforest.metrics import mean_crps
from drforest.targets import weighted_mean, weighted_quantile

weights = forest.predict_weights(X_test)
mean = weighted_mean(weights, forest.y_train_)
quantiles = weighted_quantile(weights, forest.y_train_, [0.1, 0.5, 0.9])
score = mean_crps(weights, forest.y_train_, y_test[:, None])
```

For a custom or specially configured split criterion, provide a factory that
receives the validated two-dimensional training responses once:

```python
from drforest import DistributionalRandomForest
from drforest.criteria import MmdRffCriterion
from drforest.features.rff import fixed_bandwidth

forest = DistributionalRandomForest(
    criterion_factory=lambda y: MmdRffCriterion.from_data(
        y,
        n_features=512,
        bandwidth_rule=fixed_bandwidth(1.0),
    ),
    n_estimators=5,
    max_cutpoints=32,
    random_state=0,
).fit(X_train, y_train)
```

## Statistical scope

Version `0.2.0` is a research release. The core implementation is tested, but
the public API may still change. The current fixed-fraction subsampling
interface is intended for prediction and benchmarking; it does not claim the
inference-valid regime required by the asymptotic forest theory. Missing-value
handling is not implemented.

The accompanying characterization note is available as a
[repository PDF](https://github.com/silaskoemen/drforest/blob/main/paper/main.pdf).
It studies when distributional splitting helps, compares the implemented
criteria, and gives a finite-node analysis of mean versus kernel-mean split
signal. An arXiv link and formal citation will be added after the note is posted.

## Design

```text
forest structure -> sparse conditional weights -> targets and scores
```

This keeps split geometry separate from downstream estimation. A criterion can
change without changing mean, quantile, CDF, or scoring implementations, and
the same fitted forest can serve multiple targets.

Trees support honest structure/leaf sample splitting, explicit per-tree and
per-node random-number streams, row subsampling, feature subsampling, and a
shared candidate-cutpoint cap. MMD bandwidths are configured once from the full
training response; random Fourier frequencies are resampled per node.

## Benchmarks and paper

Study entry points live in
[`benchmarks/studies`](https://github.com/silaskoemen/drforest/tree/main/benchmarks/studies):

```bash
pixi run python benchmarks/studies/run_synthetic_splitting.py
pixi run python benchmarks/studies/run_real_benchmark.py \
  --datasets diabetes \
  --criteria cart mmd_rff \
  --honesty-fractions 0.5 0.0
```

Generated datasets and result files are intentionally not tracked. Tables,
figures, LaTeX sources, and the compiled note are committed under
[`paper`](https://github.com/silaskoemen/drforest/tree/main/paper).

## Development

Install the development environment and pre-commit hooks with:

```bash
git clone https://github.com/silaskoemen/drforest.git
cd drforest
pixi install
pixi run build-rust
pixi run setup
```

Before submitting a change, run:

```bash
pixi run lint
pixi run pytest
pixi run build-wheel
pixi run check-wheel
```

Security issues should be reported through
[`SECURITY.md`](https://github.com/silaskoemen/drforest/blob/main/SECURITY.md).

## Citation

A citation for the characterization note will be added after its arXiv release.
Until then, cite the repository and the specific release used so the code and
results remain reproducible.

## License

`drforest` is released under the [MIT License](https://github.com/silaskoemen/drforest/blob/main/LICENSE).
