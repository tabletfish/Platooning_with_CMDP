#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        print(f"Skipping missing CSV: {path}")
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _series(rows: list[dict[str, str]], key: str) -> list[float]:
    values = []
    for row in rows:
        try:
            values.append(float(row.get(key, "nan")))
        except ValueError:
            values.append(float("nan"))
    return values


def _plot_progress(progress_files: list[Path], labels: list[str], output_dir: Path) -> None:
    metrics = [
        ("Metrics/EpRet", "Episode Return", "training_return.png"),
        ("Metrics/EpCost", "Episode Cost", "training_cost.png"),
        ("Env/MinTHW", "Minimum THW", "training_min_thw.png"),
        ("Env/CurrentTHW", "Current THW", "training_current_thw.png"),
        ("Metrics/LagrangeMultiplier/Mean", "Lagrange Multiplier", "training_lagrange.png"),
    ]
    for key, title, filename in metrics:
        plt.figure(figsize=(8, 4.5))
        has_data = False
        for path, label in zip(progress_files, labels, strict=False):
            rows = _read_csv(path)
            if not rows or key not in rows[0]:
                continue
            x = _series(rows, "TotalEnvSteps") if "TotalEnvSteps" in rows[0] else list(range(len(rows)))
            y = _series(rows, key)
            plt.plot(x, y, label=label)
            has_data = True
        if not has_data:
            plt.close()
            continue
        plt.title(title)
        plt.xlabel("Environment steps")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=200)
        plt.close()


def _plot_trace(trace_files: list[Path], labels: list[str], output_dir: Path) -> None:
    metrics = [
        ("spacing_error", "Spacing Error", "trace_spacing_error.png"),
        ("actual_distance", "Actual Distance", "trace_distance.png"),
        ("thw", "THW", "trace_thw.png"),
        ("cost", "THW Cost", "trace_cost.png"),
        ("action", "Control Action", "trace_action.png"),
        ("ego_vel", "Ego Velocity", "trace_ego_velocity.png"),
    ]
    for key, title, filename in metrics:
        plt.figure(figsize=(8, 4.5))
        has_data = False
        for path, label in zip(trace_files, labels, strict=False):
            rows = _read_csv(path)
            if not rows or key not in rows[0]:
                continue
            x = _series(rows, "step")
            y = _series(rows, key)
            plt.plot(x, y, label=label)
            has_data = True
        if not has_data:
            plt.close()
            continue
        plt.title(title)
        plt.xlabel("Step")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=200)
        plt.close()


def _plot_bar(trace_files: list[Path], labels: list[str], output_dir: Path) -> None:
    if not trace_files:
        return
    metrics = [
        ("cost", "Mean Step Cost", "summary_mean_cost.png"),
        ("thw", "Minimum THW", "summary_min_thw.png"),
        ("spacing_error", "Mean Absolute Spacing Error", "summary_spacing_error.png"),
        ("action", "Mean Absolute Action", "summary_action.png"),
    ]
    rows_by_file = [_read_csv(path) for path in trace_files]
    for key, title, filename in metrics:
        values = []
        for rows in rows_by_file:
            series = [value for value in _series(rows, key) if value == value]
            if not series:
                values.append(float("nan"))
            elif key == "thw":
                values.append(min(series))
            elif key in {"spacing_error", "action"}:
                values.append(sum(abs(value) for value in series) / len(series))
            else:
                values.append(sum(series) / len(series))
        plt.figure(figsize=(7, 4.5))
        plt.bar(labels, values)
        plt.title(title)
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=200)
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training and evaluation figures for platooning experiments.")
    parser.add_argument("--progress", nargs="*", default=[], help="progress.csv files for PPO/PPOLag training curves")
    parser.add_argument("--progress-labels", nargs="*", default=[], help="labels for progress files")
    parser.add_argument("--trace", nargs="*", default=[], help="evaluation trace CSV files from PLATOON_EVAL_TRACE_CSV")
    parser.add_argument("--trace-labels", nargs="*", default=[], help="labels for trace files")
    parser.add_argument("--output-dir", default="figures/latest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    progress_files = [Path(path) for path in args.progress]
    progress_labels = args.progress_labels or [path.parent.name for path in progress_files]
    trace_files = [Path(path) for path in args.trace]
    trace_labels = args.trace_labels or [path.stem for path in trace_files]

    if progress_files:
        _plot_progress(progress_files, progress_labels, output_dir)
    if trace_files:
        _plot_trace(trace_files, trace_labels, output_dir)
        _plot_bar(trace_files, trace_labels, output_dir)
    print(output_dir)


if __name__ == "__main__":
    main()
