#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path("/home/jungjinwoo/Platooning_with_CMDP")
LOG_DIR = ROOT / "logs"
CSV_OUT = LOG_DIR / "experiment_summary.csv"
MD_OUT = LOG_DIR / "experiment_summary.md"


METRIC_KEYS = [
    "episode_return",
    "episode_cost",
    "control_efficiency",
    "traffic_disturbance",
    "jerk_cost",
    "thw_cost",
    "danger_duration",
]


def parse_summary(path: Path) -> dict[str, str]:
    row: dict[str, str] = {"summary_file": str(path)}
    section = None
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Run ID:"):
            row["run_id"] = line.split(":", 1)[1].strip()
        elif line.startswith("Preset:"):
            row["preset"] = line.split(":", 1)[1].strip()
        elif line == "--- PPO summary ---":
            section = "ppo"
        elif line == "--- PID summary ---":
            section = "pid"
        elif section and ":" in line:
            key, value = [part.strip() for part in line.split(":", 1)]
            row[f"{section}_{key}"] = value
    return row


rows = []
for summary_path in sorted(LOG_DIR.glob("exp_*/summary.txt")):
    rows.append(parse_summary(summary_path))

fieldnames = ["run_id", "preset", "summary_file"]
for prefix in ["ppo", "pid"]:
    for key in METRIC_KEYS:
        fieldnames.append(f"{prefix}_{key}")

with CSV_OUT.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

with MD_OUT.open("w") as f:
    header = ["run_id", "preset", "ppo_episode_return", "pid_episode_return", "ppo_traffic_disturbance", "pid_traffic_disturbance", "ppo_jerk_cost", "pid_jerk_cost"]
    f.write("| " + " | ".join(header) + " |\n")
    f.write("|" + "|".join([" --- " for _ in header]) + "|\n")
    for row in rows:
        values = [row.get(col, "") for col in header]
        f.write("| " + " | ".join(values) + " |\n")

print(CSV_OUT)
print(MD_OUT)
print(f"rows={len(rows)}")
