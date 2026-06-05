"""Dataset loaders and registry for experiments."""

from drforest.datasets.synthetic import (
    Dataset,
    DatasetInfo,
    list_datasets,
    load_dataset,
    make_gaussian_copula,
    make_heterogeneous_regression,
    make_quantile_scenario,
    make_shrinkage_toy,
)

__all__ = [
    "Dataset",
    "DatasetInfo",
    "list_datasets",
    "load_dataset",
    "make_gaussian_copula",
    "make_heterogeneous_regression",
    "make_quantile_scenario",
    "make_shrinkage_toy",
]
