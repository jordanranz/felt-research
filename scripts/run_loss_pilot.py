"""Run the three preregistered loss comparisons sequentially on validation data."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from felt.data import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="configs/pilot001.json")
    parser.add_argument("--out", default="runs/pilot001")
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    if plan["evaluation_split"] != "val":
        raise ValueError("This tuning pilot only evaluates validation data")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose a fresh pilot output directory")
    out.mkdir(parents=True, exist_ok=True)
    base = load_config(plan["base_config"])
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    results = {}
    for name, loss_name in plan["variants"].items():
        config = dict(base, loss=loss_name)
        config_path = out / f"{name}.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        run = out / name
        started = time.perf_counter()
        with (out / f"{name}.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "felt.train",
                    "--config",
                    str(config_path),
                    "--mode",
                    plan["mode"],
                    "--out",
                    str(run),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        train_seconds = time.perf_counter() - started
        started = time.perf_counter()
        with (out / f"{name}-evaluation.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "felt.evaluate",
                    "--checkpoint",
                    str(run / "best.pt"),
                    "--split",
                    "val",
                    "--out",
                    str(run / "evaluation"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        evaluation_seconds = time.perf_counter() - started
        report = json.loads((run / "evaluation/metrics.json").read_text())
        results[name] = {
            "train_wall_seconds": round(train_seconds, 3),
            "evaluation_wall_seconds": round(evaluation_seconds, 3),
            "metrics": report["metrics"]["model"],
            "dataset_id": report["dataset_id"],
        }
        print(json.dumps({"variant": name, **results[name]}), flush=True)
    if len({result["dataset_id"] for result in results.values()}) != 1:
        raise ValueError("Pilot variants did not use identical data")
    reference = results["baseline"]["metrics"]
    gates = plan["gates_relative_to_baseline"]
    for name, result in results.items():
        if name == "baseline":
            continue
        m = result["metrics"]
        result["gates"] = {
            "quiet": m["silence_mean_intensity"]
            <= reference["silence_mean_intensity"] * gates["silence_mean_intensity_max_ratio"],
            "startup": m["startup_rmse"]
            <= reference["startup_rmse"] * gates["startup_rmse_max_ratio"],
            "active": m["active_rmse"] <= reference["active_rmse"] * gates["active_rmse_max_ratio"],
            "events": m["event_f1"] >= reference["event_f1"] - gates["event_f1_max_drop"],
        }
        result["passes_all_gates"] = all(result["gates"].values())
    summary = {"plan": plan, "results": results, "test_set_evaluated": False}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Pilot complete: {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
