"""Run-once acquisition of external multi-target regression benchmarks.

This script is the **sole owner** of everything network- and checksum-related:
the pinned source registry (URL + SHA-256 + target-column count), the download,
the checksum verification, and the ARFF parsing. It writes the raw ARFF to
``data/raw`` and the parsed arrays to ``data/processed`` through the offline
persistence helpers in :mod:`drforest.datasets.external`. Experiments never call
this — they read the processed payload via ``load_dataset`` and stay offline.

Run with::

    pixi run python benchmarks/data/fetch_mtr.py            # all wired datasets
    pixi run python benchmarks/data/fetch_mtr.py enb        # a specific one

Set ``DRFOREST_DATA_DIR`` to relocate ``data/`` (defaults to the repo root).
"""

import argparse
import hashlib
import io
from dataclasses import dataclass
from urllib.request import urlopen

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from scipy.io import arff

from drforest.datasets.external import (
    EXTERNAL_MTR_NAMES,
    processed_path,
    raw_path,
    write_processed,
)


@dataclass(frozen=True)
class MtrSource:
    """A pinned, checksummed Mulan ARFF source with a fixed target-column count."""

    url: str
    sha256: str
    n_targets: int  # number of trailing attributes that are responses


MTR_SOURCES = {
    "enb": MtrSource(
        url=(
            "https://raw.githubusercontent.com/tsoumakas/mulan/"
            "49efcccb0666c59f57d942c13c6184af028c7650/data/multi-target/enb.arff"
        ),
        sha256="8277d9094ccda6da2cd1c2b79c90ca201fa3695e75aca173ffc43138dcab0071",
        n_targets=2,
    ),
}

# The offline read side advertises EXTERNAL_MTR_NAMES as fetchable; this script
# must supply a recipe for exactly those. Fail loudly if the two drift apart.
if set(MTR_SOURCES) != set(EXTERNAL_MTR_NAMES):
    raise RuntimeError(
        f"fetch recipes {sorted(MTR_SOURCES)} disagree with "
        f"external.EXTERNAL_MTR_NAMES {sorted(EXTERNAL_MTR_NAMES)}"
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def download_source(name: str) -> bytes:
    """Download one wired source and verify its pinned checksum; raise on mismatch."""
    source = MTR_SOURCES[name]
    with urlopen(source.url, timeout=30) as response:  # noqa: S310 - https URL pinned in MTR_SOURCES
        payload = response.read()
    digest = _sha256(payload)
    if digest != source.sha256:
        raise RuntimeError(f"checksum mismatch for {name!r}: expected {source.sha256}, got {digest} from {source.url}")
    return payload


def parse_mtr_arff(payload: bytes, n_targets: int) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]:
    """Parse an all-numeric Mulan ARFF into ``(X, Y, feature_names, response_names)``.

    The last ``n_targets`` attributes are the responses, per the Mulan layout.
    """
    if n_targets < 1:
        raise ValueError(f"n_targets must be >= 1; got {n_targets}")
    data, meta = arff.loadarff(io.StringIO(payload.decode("ascii")))
    names = tuple(meta.names())
    if n_targets >= len(names):
        raise ValueError(f"n_targets={n_targets} leaves no feature columns among {len(names)} attributes")

    matrix = structured_to_unstructured(data, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("ARFF payload contains non-finite or missing values")

    X = np.ascontiguousarray(matrix[:, :-n_targets])
    Y = np.ascontiguousarray(matrix[:, -n_targets:])
    return X, Y, names[:-n_targets], names[-n_targets:]


def fetch(name: str) -> None:
    """Materialise one wired dataset into ``data/raw`` and ``data/processed``."""
    if name not in MTR_SOURCES:
        raise SystemExit(f"unknown dataset {name!r}; wired datasets: {', '.join(sorted(MTR_SOURCES))}")
    source = MTR_SOURCES[name]

    payload = download_source(name)  # verifies the pinned checksum
    raw = raw_path(name)
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(payload)

    X, Y, feature_names, response_names = parse_mtr_arff(payload, source.n_targets)
    write_processed(name, X, Y, feature_names, response_names)
    print(f"{name}: X{X.shape} Y{Y.shape} -> {raw}  +  {processed_path(name)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="datasets to fetch (default: all wired)")
    args = parser.parse_args()
    for name in args.names or sorted(MTR_SOURCES):
        fetch(name)


if __name__ == "__main__":
    main()
