"""Compare frozen study 003 checkpoints with retraining on diverse synthetic sounds."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--baseline-runs", default="runs/scale003")
    args = parser.parse_args()
    out, baseline = Path(args.out), Path(args.baseline_runs)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a fresh output directory")
    for seed in (7, 19, 42):
        if not (baseline / f"seed{seed}/best.pt").is_file():
            raise ValueError(
                "Reproduce scale003 first; all three baseline checkpoints are required"
            )
    out.mkdir(parents=True, exist_ok=True)
    config = "configs/diversity004.json"

    def command(arguments, name):
        with (out / f"{name}.log").open("w") as log:
            subprocess.run(
                [sys.executable, *arguments], stdout=log, stderr=subprocess.STDOUT, check=True
            )
        print(name + " complete", flush=True)

    command(
        [
            "-m",
            "felt.evaluate",
            "--checkpoint",
            str(baseline / "seed7/best.pt"),
            "--dataset-config",
            config,
            "--split",
            "val",
            "--out",
            str(out / "baseline-val-seed7"),
        ],
        "baseline-validation",
    )
    command(
        [
            "scripts/run_scale_study.py",
            "--config",
            config,
            "--prior-config",
            "configs/scale003.json",
            "--out",
            str(out / "trained"),
        ],
        "diverse-training",
    )
    comparison = {"complete": False, "runs": {}}
    for seed in (7, 19, 42):
        original = out / f"baseline-test-seed{seed}"
        command(
            [
                "-m",
                "felt.evaluate",
                "--checkpoint",
                str(baseline / f"seed{seed}/best.pt"),
                "--dataset-config",
                config,
                "--split",
                "test",
                "--out",
                str(original),
            ],
            f"baseline-test-seed{seed}",
        )
        a = json.loads((original / "metrics.json").read_text())
        b = json.loads((out / f"trained/seed{seed}/test/metrics.json").read_text())
        if a["dataset_id"] != b["dataset_id"]:
            raise ValueError("Baseline and retrained model used different evaluation data")
        comparison["runs"][str(seed)] = {"original": a, "retrained": b}
    comparison["complete"] = True
    (out / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")


if __name__ == "__main__":
    main()
