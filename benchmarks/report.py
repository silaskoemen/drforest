"""Aggregate benchmark result JSON into paper-ready tables and figures.

The study harnesses write per-run records to ``benchmarks/results/{study}/``.
This module turns those records into the artefacts the note needs:

* paired-difference tables (``criterion - reference``) with paired standard
  error and seed-level win rate, emitted as LaTeX;
* absolute mean (+/- std) tables per criterion;
* the California subsample curve (paired CRPS/energy difference vs ``n_train``).

Pairing is done at the run level: for each ``(dataset, honesty, seed)`` cell the
reference criterion is subtracted from every other criterion, so the split seed,
model seed, and tree hyperparameters are held fixed within a pair. Lower is
better for every metric reported here (RMSE, CRPS, energy), so a negative
difference means the row improves over the reference.

Examples::

    pixi run python -m benchmarks.report paired \
      --study run_real_benchmark --reference cart \
      --out paper/tables/real_vs_cart.tex

    pixi run python -m benchmarks.report curve \
      --reference cart --metric CRPS \
      --out paper/figures/california_subsample.png
"""

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.results_io import default_results_dir

METRICS = ("RMSE", "CRPS", "energy")


@dataclass(frozen=True)
class RunRecord:
    """One fitted forest: the unit that paired differences are computed over."""

    study: str
    dataset: str
    criterion: str
    honesty_fraction: float
    seed: int
    n_train: int
    scores: dict[str, float]
    fit_time: float | None


def _coerce_seed(run: dict[str, Any]) -> int:
    """Real benchmarks key pairs by ``data_seed``; synthetic ones by ``seed``."""
    if "data_seed" in run:
        return int(run["data_seed"])
    return int(run["seed"])


def iter_records(payload: dict[str, Any]) -> Iterator[RunRecord]:
    """Yield flat run records from either study schema.

    ``run_real_benchmark`` stores runs at the top level; the synthetic and
    ablation studies nest them under ``datasets[].runs``.
    """
    study = payload["study"]
    params = payload["params"]
    default_honesty = None
    honesties = params.get("honesty_fractions")
    if isinstance(honesties, list) and len(honesties) == 1:
        default_honesty = float(honesties[0])

    def _emit(run: dict[str, Any], dataset: str) -> RunRecord:
        honesty = run.get("honesty_fraction", default_honesty)
        if honesty is None:
            raise ValueError(f"run for {dataset!r} is missing honesty_fraction and params are ambiguous")
        return RunRecord(
            study=study,
            dataset=dataset,
            criterion=run["criterion"],
            honesty_fraction=float(honesty),
            seed=_coerce_seed(run),
            n_train=int(run["n_train"]),
            scores={m: float(run["scores"][m]) for m in METRICS},
            fit_time=float(run["fit_time"]) if "fit_time" in run else None,
        )

    if "runs" in payload:
        for run in payload["runs"]:
            yield _emit(run, run["dataset"])
    if "datasets" in payload:
        for dataset_block in payload["datasets"]:
            name = dataset_block["name"]
            for run in dataset_block["runs"]:
                yield _emit(run, name)


def load_payloads(study: str, paths: Iterable[Path] | None, results_dir: Path | None) -> list[dict[str, Any]]:
    """Load result JSON for ``study``: explicit ``paths`` or all files on disk."""
    if paths is not None:
        files = list(paths)
    else:
        directory = default_results_dir(study) if results_dir is None else results_dir / study
        files = sorted(directory.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no result JSON found for study {study!r}")
    return [json.loads(path.read_text()) for path in files]


def load_records(study: str, paths: Iterable[Path] | None = None, results_dir: Path | None = None) -> list[RunRecord]:
    records: list[RunRecord] = []
    for payload in load_payloads(study, paths, results_dir):
        records.extend(iter_records(payload))
    return records


def drop_degenerate_anisotropic(
    records: list[RunRecord],
    *,
    criterion: str = "anisotropic_mmd",
    reference: str = "mmd_rff",
) -> list[RunRecord]:
    """Remove ``criterion`` records that are bit-identical to ``reference``.

    Diagonal-bandwidth (anisotropic) MMD reduces *exactly* to isotropic MMD on a
    one-dimensional response: the coordinatewise median bandwidth equals the
    Euclidean median, and the length-1 frequency draw is the same realization as
    the scalar-scale draw, so the two criteria produce identical scores within a
    matched seed. Such records are not a distinct estimator, so listing them as a
    separate criterion in a scalar comparison is misleading. A record is dropped
    only when a ``reference`` record exists at the same
    ``(dataset, n_train, honesty, seed)`` and its scores match exactly, so genuine
    multivariate cells (where the criteria diverge) are untouched.
    """
    ref_scores: dict[tuple[str, int, float, int], dict[str, float]] = {
        (r.dataset, r.n_train, r.honesty_fraction, r.seed): r.scores for r in records if r.criterion == reference
    }
    kept: list[RunRecord] = []
    for r in records:
        if r.criterion == criterion:
            key = (r.dataset, r.n_train, r.honesty_fraction, r.seed)
            if ref_scores.get(key) == r.scores:
                continue
        kept.append(r)
    return kept


@dataclass(frozen=True)
class PairedCell:
    dataset: str
    n_train: int
    honesty_fraction: float
    criterion: str
    n_pairs: int
    diff_mean: dict[str, float]
    diff_se: dict[str, float]
    win_rate: dict[str, float]


def paired_differences(
    records: list[RunRecord],
    reference: str,
    *,
    metrics: tuple[str, ...] = METRICS,
) -> list[PairedCell]:
    """Difference every criterion against ``reference`` within matched cells.

    A pair is a ``(dataset, n_train, honesty, seed)`` for which both the row
    criterion and the reference ran. ``n_train`` is part of the cell identity so
    runs at different subsample sizes are never silently pooled. ``win_rate`` is
    the fraction of pairs where the row beats the reference (lower metric).
    """
    Cell = tuple[str, int, float]  # (dataset, n_train, honesty)
    # (cell, criterion, seed) -> scores
    by_key: dict[tuple[Cell, str, int], dict[str, float]] = {}
    cells_seen: dict[Cell, None] = {}
    criteria: list[str] = []
    for rec in records:
        cell = (rec.dataset, rec.n_train, rec.honesty_fraction)
        by_key[(cell, rec.criterion, rec.seed)] = rec.scores
        cells_seen.setdefault(cell, None)
        if rec.criterion not in criteria:
            criteria.append(rec.criterion)

    seeds_by_cell: dict[tuple[Cell, str], set[int]] = defaultdict(set)
    for cell, criterion, seed in by_key:
        seeds_by_cell[(cell, criterion)].add(seed)

    result: list[PairedCell] = []
    for cell in cells_seen:
        dataset, n_train, honesty = cell
        ref_seeds = seeds_by_cell.get((cell, reference))
        if not ref_seeds:
            continue
        for criterion in criteria:
            if criterion == reference:
                continue
            shared = sorted(ref_seeds & seeds_by_cell.get((cell, criterion), set()))
            if not shared:
                continue
            diffs = {
                m: np.array([by_key[(cell, criterion, s)][m] - by_key[(cell, reference, s)][m] for s in shared])
                for m in metrics
            }
            n = len(shared)
            result.append(
                PairedCell(
                    dataset=dataset,
                    n_train=n_train,
                    honesty_fraction=honesty,
                    criterion=criterion,
                    n_pairs=n,
                    diff_mean={m: float(diffs[m].mean()) for m in metrics},
                    diff_se={m: float(diffs[m].std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0 for m in metrics},
                    win_rate={m: float(np.mean(diffs[m] < 0)) for m in metrics},
                )
            )
    result.sort(key=lambda c: (c.dataset, c.n_train, c.honesty_fraction, c.criterion))
    return result


def _latex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def _caption_block(caption: str, label: str | None) -> list[str]:
    block = ["  \\centering", f"  \\caption{{{caption}}}"]
    if label is not None:
        block.append(f"  \\label{{{label}}}")
    return block


def _colspec(n_columns: int) -> str:
    """Left-align the first (label) column; center the rest.

    The metric cells are compound strings (mean $\\pm$ SE with win rate), so
    right-aligning them gives no decimal alignment; centering reads cleaner.
    """
    if n_columns < 1:
        raise ValueError(f"n_columns must be positive; got {n_columns}")
    return "l" + "c" * (n_columns - 1)


def _fit_open(fit: bool) -> list[str]:
    """Open an ``adjustbox`` that shrinks an oversized table to the text width.

    ``max width`` only scales down, so narrow tables are left untouched. Requires
    ``\\usepackage{adjustbox}`` in the document preamble.
    """
    return ["  \\begin{adjustbox}{max width=\\textwidth}"] if fit else []


def _fit_close(fit: bool) -> list[str]:
    return ["  \\end{adjustbox}"] if fit else []


def paired_to_latex(
    cells: list[PairedCell],
    reference: str,
    *,
    metrics: tuple[str, ...] = METRICS,
    caption: str | None = None,
    label: str | None = None,
    fit: bool = True,
) -> str:
    """Render paired cells as a booktabs LaTeX table."""
    if caption is None:
        caption = (
            f"Paired differences (criterion $-$ \\texttt{{{_latex_escape(reference)}}}). "
            "Negative means the row improves over the reference; "
            "values are mean $\\pm$ paired SE with seed-level win rate."
        )
    header = " & ".join(
        ["dataset", "$n_{\\mathrm{tr}}$", "honesty", "criterion", "pairs"] + [f"$\\Delta${m}" for m in metrics]
    )
    lines = [
        "\\begin{table}[h]",
        *_caption_block(caption, label),
        *_fit_open(fit),
        f"  \\begin{{tabular}}{{{_colspec(5 + len(metrics))}}}",
        "    \\toprule",
        f"    {header} \\\\",
        "    \\midrule",
    ]
    for cell in cells:
        entries = [
            _latex_escape(cell.dataset),
            str(cell.n_train),
            f"{cell.honesty_fraction:g}",
            _latex_escape(cell.criterion),
            str(cell.n_pairs),
        ]
        for m in metrics:
            entries.append(f"{cell.diff_mean[m]:+.4f} $\\pm$ {cell.diff_se[m]:.4f} ({cell.win_rate[m] * 100:.0f}\\%)")
        lines.append("    " + " & ".join(entries) + " \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}", *_fit_close(fit), "\\end{table}", ""]
    return "\n".join(lines)


@dataclass(frozen=True)
class AbsoluteCell:
    dataset: str
    n_train: int
    honesty_fraction: float
    criterion: str
    n: int
    mean: dict[str, float]
    std: dict[str, float]
    fit_time_mean: float | None


def absolute_means(records: list[RunRecord], *, metrics: tuple[str, ...] = METRICS) -> list[AbsoluteCell]:
    """Mean and std of each metric per ``(dataset, n_train, honesty, criterion)`` cell."""
    grouped: dict[tuple[str, int, float, str], list[RunRecord]] = defaultdict(list)
    for rec in records:
        grouped[(rec.dataset, rec.n_train, rec.honesty_fraction, rec.criterion)].append(rec)
    cells = []
    for (dataset, n_train, honesty, criterion), recs in grouped.items():
        fit_times = [r.fit_time for r in recs if r.fit_time is not None]
        cells.append(
            AbsoluteCell(
                dataset=dataset,
                n_train=n_train,
                honesty_fraction=honesty,
                criterion=criterion,
                n=len(recs),
                mean={m: float(np.mean([r.scores[m] for r in recs])) for m in metrics},
                std={m: float(np.std([r.scores[m] for r in recs])) for m in metrics},
                fit_time_mean=float(np.mean(fit_times)) if fit_times else None,
            )
        )
    cells.sort(key=lambda c: (c.dataset, c.n_train, c.honesty_fraction, c.criterion))
    return cells


def absolute_to_latex(
    cells: list[AbsoluteCell],
    *,
    metrics: tuple[str, ...] = METRICS,
    with_fit_time: bool = True,
    caption: str = "Absolute metric means ($\\pm$ std over seeds); lower is better.",
    label: str | None = None,
    fit: bool = True,
) -> str:
    header_cols = ["dataset", "$n_{\\mathrm{tr}}$", "honesty", "criterion"] + list(metrics)
    if with_fit_time:
        header_cols.append("fit (s)")
    header = " & ".join(header_cols)
    lines = [
        "\\begin{table}[h]",
        *_caption_block(caption, label),
        *_fit_open(fit),
        f"  \\begin{{tabular}}{{{_colspec(len(header_cols))}}}",
        "    \\toprule",
        f"    {header} \\\\",
        "    \\midrule",
    ]
    for cell in cells:
        entries = [
            _latex_escape(cell.dataset),
            str(cell.n_train),
            f"{cell.honesty_fraction:g}",
            _latex_escape(cell.criterion),
        ]
        for m in metrics:
            entries.append(f"{cell.mean[m]:.4f} $\\pm$ {cell.std[m]:.4f}")
        if with_fit_time:
            entries.append(f"{cell.fit_time_mean:.2f}" if cell.fit_time_mean is not None else "--")
        lines.append("    " + " & ".join(entries) + " \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}", *_fit_close(fit), "\\end{table}", ""]
    return "\n".join(lines)


@dataclass(frozen=True)
class ShrinkageRecord:
    """One ``(criterion, shrinkage variant)`` outcome from an ablation run."""

    study: str
    dataset: str
    criterion: str
    variant: str
    honesty_fraction: float
    seed: int
    n_train: int
    scores: dict[str, float]
    alpha_mean: float | None


def iter_shrinkage_records(payload: dict[str, Any]) -> Iterator[ShrinkageRecord]:
    """Yield one record per shrinkage variant from an ablation payload.

    Ablation runs nest post-hoc shrinkage outcomes under ``runs[].variants`` as
    ``{variant: {"alpha_mean": ..., "scores": {...}}}``. Honesty is fixed per
    ablation file, so it is read from ``params`` rather than the run.
    """
    study = payload["study"]
    params = payload["params"]
    honesty = params.get("honesty_fraction")
    if honesty is None:
        honesties = params.get("honesty_fractions")
        if isinstance(honesties, list) and len(honesties) == 1:
            honesty = honesties[0]
    if honesty is None:
        raise ValueError("ablation payload is missing a single honesty_fraction")
    honesty = float(honesty)

    for dataset_block in payload["datasets"]:
        dataset = dataset_block["name"]
        for run in dataset_block["runs"]:
            for variant, outcome in run["variants"].items():
                yield ShrinkageRecord(
                    study=study,
                    dataset=dataset,
                    criterion=run["criterion"],
                    variant=variant,
                    honesty_fraction=honesty,
                    seed=int(run["seed"]),
                    n_train=int(run["n_train"]),
                    scores={m: float(outcome["scores"][m]) for m in METRICS},
                    alpha_mean=(float(outcome["alpha_mean"]) if "alpha_mean" in outcome else None),
                )


def load_shrinkage_records(
    study: str, paths: Iterable[Path] | None = None, results_dir: Path | None = None
) -> list[ShrinkageRecord]:
    records: list[ShrinkageRecord] = []
    for payload in load_payloads(study, paths, results_dir):
        records.extend(iter_shrinkage_records(payload))
    return records


@dataclass(frozen=True)
class ShrinkageCell:
    dataset: str
    criterion: str
    honesty_fraction: float
    variant: str
    n_pairs: int
    alpha_mean: float
    diff_mean: dict[str, float]
    diff_se: dict[str, float]
    win_rate: dict[str, float]


def shrinkage_differences(
    records: list[ShrinkageRecord],
    reference: str = "raw",
    *,
    metrics: tuple[str, ...] = METRICS,
) -> list[ShrinkageCell]:
    """Pair every shrinkage variant against ``reference`` (default raw weights).

    A pair is one ``(dataset, criterion, n_train, honesty, seed)`` for which both
    the row variant and the reference variant ran. The split criterion and the
    fitted forest are identical within a pair, so the difference isolates the
    post-hoc shrinkage effect. Negative means shrinkage improves over raw
    weights. ``alpha_mean`` is the variant's mean shrinkage intensity.
    """
    Cell = tuple[str, str, int, float]  # (dataset, criterion, n_train, honesty)
    by_key: dict[tuple[Cell, str, int], ShrinkageRecord] = {}
    cells_seen: dict[Cell, None] = {}
    variants: list[str] = []
    for rec in records:
        cell = (rec.dataset, rec.criterion, rec.n_train, rec.honesty_fraction)
        by_key[(cell, rec.variant, rec.seed)] = rec
        cells_seen.setdefault(cell, None)
        if rec.variant not in variants:
            variants.append(rec.variant)

    seeds_by: dict[tuple[Cell, str], set[int]] = defaultdict(set)
    for cell, variant, seed in by_key:
        seeds_by[(cell, variant)].add(seed)

    result: list[ShrinkageCell] = []
    for cell in cells_seen:
        dataset, criterion, n_train, honesty = cell
        ref_seeds = seeds_by.get((cell, reference))
        if not ref_seeds:
            continue
        for variant in variants:
            if variant == reference:
                continue
            shared = sorted(ref_seeds & seeds_by.get((cell, variant), set()))
            if not shared:
                continue
            diffs = {
                m: np.array(
                    [by_key[(cell, variant, s)].scores[m] - by_key[(cell, reference, s)].scores[m] for s in shared]
                )
                for m in metrics
            }
            alphas = [a for s in shared if (a := by_key[(cell, variant, s)].alpha_mean) is not None]
            n = len(shared)
            result.append(
                ShrinkageCell(
                    dataset=dataset,
                    criterion=criterion,
                    honesty_fraction=honesty,
                    variant=variant,
                    n_pairs=n,
                    alpha_mean=float(np.mean(alphas)) if alphas else float("nan"),
                    diff_mean={m: float(diffs[m].mean()) for m in metrics},
                    diff_se={m: float(diffs[m].std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0 for m in metrics},
                    win_rate={m: float(np.mean(diffs[m] < 0)) for m in metrics},
                )
            )
    result.sort(key=lambda c: (c.dataset, c.criterion, c.honesty_fraction, c.variant))
    return result


def shrinkage_to_latex(
    cells: list[ShrinkageCell],
    reference: str = "raw",
    *,
    metrics: tuple[str, ...] = METRICS,
    caption: str | None = None,
    label: str | None = None,
    fit: bool = True,
) -> str:
    """Render shrinkage paired cells (variant $-$ raw) as a booktabs table."""
    if caption is None:
        caption = (
            "Shrinkage frontier: paired differences (variant $-$ "
            f"\\texttt{{{_latex_escape(reference)}}}) within a fixed criterion and "
            "forest. Negative means shrinkage improves over raw weights; "
            "$\\bar\\alpha$ is the mean shrinkage intensity."
        )
    header = " & ".join(
        ["dataset", "criterion", "honesty", "variant", "pairs", "$\\bar\\alpha$"] + [f"$\\Delta${m}" for m in metrics]
    )
    lines = [
        "\\begin{table}[h]",
        *_caption_block(caption, label),
        *_fit_open(fit),
        f"  \\begin{{tabular}}{{{_colspec(6 + len(metrics))}}}",
        "    \\toprule",
        f"    {header} \\\\",
        "    \\midrule",
    ]
    for cell in cells:
        entries = [
            _latex_escape(cell.dataset),
            _latex_escape(cell.criterion),
            f"{cell.honesty_fraction:g}",
            _latex_escape(cell.variant),
            str(cell.n_pairs),
            f"{cell.alpha_mean:.3f}",
        ]
        for m in metrics:
            entries.append(f"{cell.diff_mean[m]:+.4f} $\\pm$ {cell.diff_se[m]:.4f} ({cell.win_rate[m] * 100:.0f}\\%)")
        lines.append("    " + " & ".join(entries) + " \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}", *_fit_close(fit), "\\end{table}", ""]
    return "\n".join(lines)


def subsample_curve(
    records: list[RunRecord],
    reference: str,
    *,
    metric: str,
    out_path: Path,
    dataset: str | None = None,
) -> None:
    """Plot paired ``criterion - reference`` differences vs ``n_train``.

    Records are bucketed by ``n_train``; within each bucket paired differences
    are computed exactly as in :func:`paired_differences`, then the mean and
    paired SE are plotted as one line per criterion.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if dataset is not None:
        records = [r for r in records if r.dataset == dataset]
    if not records:
        raise ValueError("no records to plot")

    by_n: dict[int, list[RunRecord]] = defaultdict(list)
    for rec in records:
        by_n[rec.n_train].append(rec)

    # criterion -> list of (n_train, mean, se)
    series: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    for n_train in sorted(by_n):
        for cell in paired_differences(by_n[n_train], reference, metrics=(metric,)):
            series[cell.criterion].append((n_train, cell.diff_mean[metric], cell.diff_se[metric]))

    if not series:
        raise ValueError(f"no paired differences against reference {reference!r}")

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for criterion in sorted(series):
        points = sorted(series[criterion])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        es = [p[2] for p in points]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=criterion)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("$n_{\\mathrm{train}}$")
    ax.set_ylabel(f"$\\Delta${metric} (criterion $-$ {reference})")
    ax.set_title(f"Subsample curve: lower means the criterion beats {reference}")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _write(text: str, out: Path | None) -> None:
    if out is None:
        print(text)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text if text.endswith("\n") else text + "\n")
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--study", required=True)
    common.add_argument(
        "--paths",
        nargs="+",
        type=Path,
        default=None,
        help="explicit result files (default: all on disk)",
    )
    common.add_argument("--results-dir", type=Path, default=None)
    common.add_argument("--metrics", nargs="+", default=list(METRICS))
    common.add_argument("--out", type=Path, default=None)
    common.add_argument("--label", default=None, help="LaTeX \\label to attach to the table")
    common.add_argument(
        "--no-fit",
        dest="fit",
        action="store_false",
        help="do not wrap the table in an adjustbox that shrinks it to \\textwidth",
    )
    common.add_argument(
        "--keep-degenerate-anisotropic",
        action="store_true",
        help="keep anisotropic_mmd rows even where they are identical to mmd_rff "
        "(the 1-D collapse); by default such redundant rows are dropped",
    )

    p_paired = sub.add_parser("paired", parents=[common], help="paired-difference LaTeX table")
    p_paired.add_argument("--reference", required=True)

    sub.add_parser("absolute", parents=[common], help="absolute mean/std LaTeX table")

    p_shrink = sub.add_parser("shrinkage", parents=[common], help="shrinkage variant paired LaTeX table")
    p_shrink.add_argument("--reference", default="raw")

    p_curve = argparse.ArgumentParser(add_help=False)
    p_curve.add_argument("--study", default="run_real_benchmark")
    p_curve.add_argument("--paths", nargs="+", type=Path, default=None)
    p_curve.add_argument("--results-dir", type=Path, default=None)
    p_curve.add_argument("--reference", required=True)
    p_curve.add_argument("--metric", default="CRPS")
    p_curve.add_argument("--dataset", default="california_housing")
    p_curve.add_argument("--out", type=Path, required=True)
    p_curve.add_argument("--keep-degenerate-anisotropic", action="store_true")
    sub.add_parser("curve", parents=[p_curve], help="subsample curve figure")

    args = parser.parse_args()

    if args.command == "curve":
        records = load_records(args.study, args.paths, args.results_dir)
        if not args.keep_degenerate_anisotropic:
            records = drop_degenerate_anisotropic(records)
        subsample_curve(
            records,
            args.reference,
            metric=args.metric,
            out_path=args.out,
            dataset=args.dataset,
        )
        print(f"wrote {args.out}")
        return

    metrics = tuple(args.metrics)
    if args.command == "shrinkage":
        shrink_records = load_shrinkage_records(args.study, args.paths, args.results_dir)
        cells = shrinkage_differences(shrink_records, args.reference, metrics=metrics)
        _write(
            shrinkage_to_latex(cells, args.reference, metrics=metrics, label=args.label, fit=args.fit),
            args.out,
        )
        return

    records = load_records(args.study, args.paths, args.results_dir)
    if not args.keep_degenerate_anisotropic:
        records = drop_degenerate_anisotropic(records)
    if args.command == "paired":
        cells = paired_differences(records, args.reference, metrics=metrics)
        _write(
            paired_to_latex(cells, args.reference, metrics=metrics, label=args.label, fit=args.fit),
            args.out,
        )
    elif args.command == "absolute":
        cells = absolute_means(records, metrics=metrics)
        _write(
            absolute_to_latex(cells, metrics=metrics, label=args.label, fit=args.fit),
            args.out,
        )


if __name__ == "__main__":
    main()
