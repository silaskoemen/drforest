<p align="center">
  <img src="assets/drforest_logo.png" width="50%" alt="drforest logo">
</p>

# drforest

[![CI](https://img.shields.io/github/actions/workflow/status/silaskoemen/drforest/ci.yml?style=flat-square&branch=main)](https://github.com/silaskoemen/drforest/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)

`drforest` is a Python and Rust implementation of distributional random forests.
Instead of returning a single prediction, a fitted forest produces a sparse,
row-stochastic weight matrix over the training responses. Conditional means,
quantiles, CDFs, and distributional scores are then plug-ins on the same weights.

The package supports scalar and multivariate responses and provides several split
criteria behind one forest implementation:

- CART mean separation;
- Gaussian maximum mean discrepancy (MMD) using random Fourier features;
- anisotropic and adaptive-frequency MMD variants; and
- sliced Wasserstein separation.

The split search is Rust-backed, while the statistical interfaces and benchmark
harnesses remain in Python.

## Project status

Version `0.1.0` is a research release. The core implementation is tested, but the
public API may still change. The current fixed-fraction subsampling interface is
intended for prediction and benchmarking; it does not claim the inference-valid
regime required by the asymptotic forest theory. Missing-value handling is not
implemented.

The accompanying characterization note is available as a
[repository PDF](paper/main.pdf). It studies when distributional splitting helps,
compares the implemented criteria, and gives a finite-node analysis of mean versus
kernel-mean split signal. An arXiv link and formal citation will be added after the
note is posted.

## Installation

The initial release is installed from source with
[pixi](https://pixi.sh):

```bash
git clone https://github.com/silaskoemen/drforest.git
cd drforest
pixi install
pixi run build-rust
```

PyPI and conda-forge installation instructions will be added when packages are
published.

## Quick start

Responses are represented as a two-dimensional array with shape
`(n_samples, n_outputs)`, including scalar-response problems.

```python
import numpy as np

from drforest.criteria import MmdRffCriterion
from drforest.features.rff import median_heuristic
from drforest.forest import DistributionalRandomForest
from drforest.targets import weighted_mean, weighted_quantile
from drforest.tree import TreeParams

rng = np.random.default_rng(0)
X = rng.normal(size=(500, 4))
Y = (X[:, 0] + (0.5 + np.abs(X[:, 1])) * rng.normal(size=500))[:, None]

X_train, X_test = X[:400], X[400:]
Y_train = Y[:400]

forest = DistributionalRandomForest(
    criterion_factory=lambda y: MmdRffCriterion.from_data(
        y,
        n_features=128,
        bandwidth_rule=median_heuristic,
    ),
    seed=0,
    n_trees=100,
    subsample=0.5,
    tree_params=TreeParams(
        min_samples_leaf=5,
        honesty_fraction=0.5,
        colsample=0.7,
        max_cutpoints=32,
    ),
).fit(X_train, Y_train)

weights = forest.weights(X_test)
mean = weighted_mean(weights, Y_train)
quantiles = weighted_quantile(weights, Y_train, np.array([0.1, 0.5, 0.9]))

assert mean.shape == (100, 1)
assert quantiles.shape == (100, 1, 3)
```

To use ordinary CART splitting while retaining distributional forest weights,
replace the criterion factory with:

```python
from drforest.criteria import CartCriterion

criterion_factory = lambda y: CartCriterion()
```

The resulting weight matrix also works with `weighted_cdf` and with the CRPS,
energy-score, and RMSE functions in `drforest.metrics`.

## Design

The weight matrix is the central object:

```text
forest structure -> sparse conditional weights -> targets and scores
```

This keeps split geometry separate from downstream estimation. A criterion can be
changed without changing the mean, quantile, CDF, or scoring implementations, and
the same fitted forest can serve multiple targets.

Trees support honest structure/leaf sample splitting, explicit per-tree and
per-node random-number streams, row subsampling, feature subsampling, and a shared
candidate-cutpoint cap. The MMD bandwidth is configured once from the training
responses; random Fourier frequencies are resampled per node.

## Benchmarks and paper

Study entry points live in [`benchmarks/studies`](benchmarks/studies):

```bash
pixi run python benchmarks/studies/run_synthetic_splitting.py
pixi run python benchmarks/studies/run_real_benchmark.py \
  --datasets diabetes \
  --criteria cart mmd_rff \
  --honesty-fractions 0.5 0.0
```

Generated datasets and result files are intentionally not tracked. The tables and
figures used by the note are committed under [`paper`](paper), together with the
LaTeX source and compiled PDF.

## Development

Install the development environment and pre-commit hooks with:

```bash
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

Security issues should be reported through the process described in
[`SECURITY.md`](SECURITY.md).

## Citation

A citation for the characterization note will be added after its arXiv release.
Until then, cite the repository and the specific release used so that the code and
results remain reproducible.

## License

`drforest` is released under the [MIT License](LICENSE).
