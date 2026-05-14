#!/usr/bin/env python3
"""Visualize dLLM observability experiment results."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPERIMENT_RE = re.compile(r"^(?P<config>sglang(?:-baseline)?)-(?P<threshold>\d+(?:\.\d+)?)$")
SUBEXPERIMENT_RE = re.compile(r"^(?P<config>sglang(?:-baseline)?)_(?P<batch_size>\d+)$")
FORWARD_COUNTS_KEY = "dllm_forward_counts_per_block"
BLOCK_LATENCIES_KEY = "dllm_block_completion_latencies"
TOTAL_TOKENS_PER_SECOND_KEY = "total_tokens_per_second"
COMPLETION_TOKENS_KEY = "completion_tokens"


@dataclass(frozen=True)
class RequestMetrics:
    forward_counts: list[float]
    block_latencies: list[float]
    throughput: float
    tokens_per_forward: float


@dataclass(frozen=True)
class Subexperiment:
    path: Path
    batch_size: int
    forward_counts: list[float]
    block_latencies: list[float]
    throughputs: list[float]
    tokens_per_forward: list[float]
    request_count: int


@dataclass(frozen=True)
class Experiment:
    path: Path
    name: str
    config: str
    threshold: str
    subexperiments: list[Subexperiment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create histogram figures for dLLM experiment result folders."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing experiment result folders. Default: results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures"),
        help="Directory where PNG figures are written. Default: results/figures",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved PNG files. Default: 300",
    )
    return parser.parse_args()


def require_object(parent: dict[str, Any], key: str, request_path: Path) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{request_path}: missing object field '{key}'")
    return value


def require_number(
    parent: dict[str, Any], field_path: str, key: str, request_path: Path
) -> float:
    value = parent.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"{request_path}: missing numeric field '{field_path}'")
    return float(value)


def require_metric_list(
    final_meta_info: dict[str, Any], request_path: Path, metric_key: str
) -> list[float]:
    values = final_meta_info.get(metric_key)
    if not isinstance(values, list):
        raise ValueError(
            f"{request_path}: missing list field 'final_meta_info.{metric_key}'"
        )

    numeric_values: list[float] = []
    for index, value in enumerate(values):
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"{request_path}: non-numeric value at "
                f"'final_meta_info.{metric_key}[{index}]': {value!r}"
            )
        numeric_values.append(float(value))
    return numeric_values


def load_request_metrics(request_path: Path) -> RequestMetrics:
    try:
        with request_path.open("r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{request_path}: invalid JSON: {exc}") from exc

    final_meta_info = data.get("final_meta_info")
    if not isinstance(final_meta_info, dict):
        raise ValueError(f"{request_path}: missing object field 'final_meta_info'")

    token_metrics = require_object(data, "token_metrics", request_path)
    forward_counts = require_metric_list(final_meta_info, request_path, FORWARD_COUNTS_KEY)
    block_latencies = require_metric_list(final_meta_info, request_path, BLOCK_LATENCIES_KEY)
    throughput = require_number(
        token_metrics,
        f"token_metrics.{TOTAL_TOKENS_PER_SECOND_KEY}",
        TOTAL_TOKENS_PER_SECOND_KEY,
        request_path,
    )
    completion_tokens = require_number(
        token_metrics,
        f"token_metrics.{COMPLETION_TOKENS_KEY}",
        COMPLETION_TOKENS_KEY,
        request_path,
    )

    total_forward_count = sum(forward_counts)
    if total_forward_count == 0:
        raise ValueError(
            f"{request_path}: sum of 'final_meta_info.{FORWARD_COUNTS_KEY}' is zero"
        )

    return RequestMetrics(
        forward_counts=forward_counts,
        block_latencies=block_latencies,
        throughput=throughput,
        tokens_per_forward=completion_tokens / total_forward_count,
    )


def load_subexperiment(path: Path, batch_size: int) -> Subexperiment:
    request_paths = sorted(path.glob("request_*.json"))
    forward_counts: list[float] = []
    block_latencies: list[float] = []
    throughputs: list[float] = []
    tokens_per_forward: list[float] = []

    for request_path in request_paths:
        request_metrics = load_request_metrics(request_path)
        forward_counts.extend(request_metrics.forward_counts)
        block_latencies.extend(request_metrics.block_latencies)
        throughputs.append(request_metrics.throughput)
        tokens_per_forward.append(request_metrics.tokens_per_forward)

    return Subexperiment(
        path=path,
        batch_size=batch_size,
        forward_counts=forward_counts,
        block_latencies=block_latencies,
        throughputs=throughputs,
        tokens_per_forward=tokens_per_forward,
        request_count=len(request_paths),
    )


def discover_experiments(results_dir: Path) -> list[Experiment]:
    if not results_dir.is_dir():
        raise ValueError(f"{results_dir}: results directory does not exist")

    experiments: list[Experiment] = []
    for experiment_path in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        experiment_match = EXPERIMENT_RE.match(experiment_path.name)
        if experiment_match is None:
            continue

        config = experiment_match.group("config")
        subexperiments: list[Subexperiment] = []
        for subexperiment_path in sorted(
            p for p in experiment_path.iterdir() if p.is_dir()
        ):
            subexperiment_match = SUBEXPERIMENT_RE.match(subexperiment_path.name)
            if subexperiment_match is None:
                continue
            if subexperiment_match.group("config") != config:
                continue

            batch_size = int(subexperiment_match.group("batch_size"))
            subexperiments.append(load_subexperiment(subexperiment_path, batch_size))

        subexperiments.sort(key=lambda subexperiment: subexperiment.batch_size)
        if not subexperiments:
            continue

        experiments.append(
            Experiment(
                path=experiment_path,
                name=experiment_path.name,
                config=config,
                threshold=experiment_match.group("threshold"),
                subexperiments=subexperiments,
            )
        )

    return experiments


def integer_bins(values: list[float]) -> list[float] | int:
    if not values:
        return 1
    min_value = int(min(values))
    max_value = int(max(values))
    return [value - 0.5 for value in range(min_value, max_value + 2)]


def continuous_bins(values: list[float], bin_count: int = 30) -> list[float]:
    if not values:
        return [0.0, 1.0]

    min_value = min(values)
    max_value = max(values)
    if min_value == max_value:
        value = min_value
        width = max(abs(value) * 0.05, 0.01)
        return [value - width, value + width]

    step = (max_value - min_value) / bin_count
    return [min_value + step * index for index in range(bin_count + 1)]


def add_summary_text(ax: plt.Axes, values: list[float], request_count: int) -> None:
    if values:
        summary = f"avg={mean(values):.3f}\nn={len(values)}\nrequests={request_count}"
    else:
        summary = f"avg=N/A\nn=0\nrequests={request_count}"
    ax.text(
        0.97,
        0.95,
        summary,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )


def plot_metric(
    experiment: Experiment,
    output_dir: Path,
    metric_name: str,
    output_suffix: str,
    title: str,
    x_label: str,
    value_getter,
    bins,
    dpi: int,
) -> Path:
    subplot_count = len(experiment.subexperiments)
    fig, axes = plt.subplots(
        1,
        subplot_count,
        figsize=(5.0 * subplot_count, 4.0),
        squeeze=False,
        sharey=True,
    )
    axes_list = list(axes[0])

    for ax, subexperiment in zip(axes_list, experiment.subexperiments):
        values = value_getter(subexperiment)
        ax.hist(values, bins=bins, edgecolor="black", linewidth=0.6)
        ax.set_title(f"bs={subexperiment.batch_size}")
        ax.set_xlabel(x_label)
        ax.grid(axis="y", alpha=0.3)
        add_summary_text(ax, values, subexperiment.request_count)

    axes_list[0].set_ylabel("Count")
    fig.suptitle(
        f"{experiment.name}: {title} "
        f"(config={experiment.config}, threshold={experiment.threshold})"
    )
    fig.tight_layout()

    output_path = output_dir / f"{experiment.name}_{output_suffix}.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path} ({metric_name})")
    return output_path


def plot_throughput_vs_tokens_per_forward(
    experiment: Experiment, output_dir: Path, dpi: int
) -> Path:
    x_values = [
        mean(subexperiment.tokens_per_forward)
        for subexperiment in experiment.subexperiments
    ]
    y_values = [
        mean(subexperiment.throughputs) for subexperiment in experiment.subexperiments
    ]
    labels = [
        f"bs={subexperiment.batch_size}" for subexperiment in experiment.subexperiments
    ]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(x_values, y_values, linewidth=1.2, alpha=0.8)
    ax.scatter(x_values, y_values, s=70, zorder=3)

    for x_value, y_value, label in zip(x_values, y_values, labels):
        ax.annotate(
            label,
            (x_value, y_value),
            textcoords="offset points",
            xytext=(6, 6),
            ha="left",
            va="bottom",
        )

    ax.set_xlabel("Average tokens per forward")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.grid(alpha=0.3)
    ax.set_title(
        f"{experiment.name}: throughput vs tokens per forward "
        f"(config={experiment.config}, threshold={experiment.threshold})"
    )
    fig.tight_layout()

    output_path = output_dir / f"{experiment.name}_throughput_vs_tokens_per_forward.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path} (throughput vs tokens per forward)")
    return output_path


def plot_experiment(experiment: Experiment, output_dir: Path, dpi: int) -> list[Path]:
    all_forward_counts = [
        value
        for subexperiment in experiment.subexperiments
        for value in subexperiment.forward_counts
    ]
    all_block_latencies = [
        value
        for subexperiment in experiment.subexperiments
        for value in subexperiment.block_latencies
    ]

    written_paths = [
        plot_metric(
            experiment=experiment,
            output_dir=output_dir,
            metric_name=FORWARD_COUNTS_KEY,
            output_suffix="forward_counts_hist",
            title="forward counts per block",
            x_label="Forward count per block",
            value_getter=lambda subexperiment: subexperiment.forward_counts,
            bins=integer_bins(all_forward_counts),
            dpi=dpi,
        ),
        plot_metric(
            experiment=experiment,
            output_dir=output_dir,
            metric_name=BLOCK_LATENCIES_KEY,
            output_suffix="block_latencies_hist",
            title="block completion latencies",
            x_label="Block completion latency (s)",
            value_getter=lambda subexperiment: subexperiment.block_latencies,
            bins=continuous_bins(all_block_latencies),
            dpi=dpi,
        ),
        plot_throughput_vs_tokens_per_forward(
            experiment=experiment,
            output_dir=output_dir,
            dpi=dpi,
        ),
    ]
    return written_paths


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = discover_experiments(args.results_dir)
    if not experiments:
        raise ValueError(f"{args.results_dir}: no matching experiment folders found")

    written_paths: list[Path] = []
    for experiment in experiments:
        written_paths.extend(plot_experiment(experiment, output_dir, args.dpi))

    print(f"Wrote {len(written_paths)} figure(s) to {output_dir}")


if __name__ == "__main__":
    main()
