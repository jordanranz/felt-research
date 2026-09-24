"""Fixed 10x synthetic study. No tuning after held-out test evaluation."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from felt.data import digest, families, load_config, make_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default="configs/scale003.json")
    parser.add_argument("--prior-config", action="append", default=["configs/experiment001.json"])
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a fresh study directory")
    out.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    if config["variants_per_family"] > 100:
        raise ValueError("synthetic-v1 seed spacing requires at most 100 variants")
    prior_groups = set()
    for path in args.prior_config:
        prior_groups.update(map(digest, families(load_config(path))))
    overlap = prior_groups & set(map(digest, families(config)))
    if overlap:
        raise ValueError("New study overlaps earlier development families")
    arrays, manifest = make_dataset(config)
    groups = {
        split: {r["family"] for r in manifest["records"] if r["split"] == split} for split in arrays
    }
    if any(
        groups[a] & groups[b] for a, b in [("train", "val"), ("train", "test"), ("val", "test")]
    ):
        raise ValueError("Split leakage")
    summary = {
        "config": config,
        "initialization_seeds": [7, 19, 42],
        "dataset_id": manifest["dataset_id"],
        "prior_family_overlap": len(overlap),
        "prior_configs": args.prior_config,
        "feature_target_bytes": sum(a.nbytes for pair in arrays.values() for a in pair),
        "clips": {key: len(pair[0]) for key, pair in arrays.items()},
        "runs": {},
        "complete": False,
    }
    del arrays, manifest

    def save():
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    def command(arguments, logfile):
        start = time.perf_counter()
        with logfile.open("w") as log:
            subprocess.run(
                [sys.executable, *arguments], stdout=log, stderr=subprocess.STDOUT, check=True
            )
        return time.perf_counter() - start

    save()
    for seed in summary["initialization_seeds"]:
        run = out / f"seed{seed}"
        cfg = out / f"seed{seed}.json"
        cfg.write_text(json.dumps(dict(config, initialization_seed=seed), indent=2) + "\n")
        base = ["-m", "felt.train", "--config", str(cfg), "--out", str(run)]
        seconds = command(base + ["--stop-after", "3"], out / f"seed{seed}-pilot.log")
        history = json.loads((run / "history.json").read_text())
        if not all(0 <= r["val_loss"] < 10 for r in history):
            raise ValueError("Pilot loss invalid")
        print(json.dumps({"seed": seed, "three_epoch_wall_seconds": seconds}), flush=True)
        if seconds > 180:
            raise RuntimeError("Timing budget exceeded; checkpoint retained")
        seconds += command(base + ["--resume", str(run / "last.pt")], out / f"seed{seed}-train.log")
        summary["runs"][str(seed)] = {"train_wall_seconds": seconds}
        save()
    # Every seed is fixed before any test evaluation. No best-seed selection.
    for seed in summary["initialization_seeds"]:
        run = out / f"seed{seed}"
        for split in ("val", "test"):
            command(
                [
                    "-m",
                    "felt.evaluate",
                    "--checkpoint",
                    str(run / "best.pt"),
                    "--split",
                    split,
                    "--out",
                    str(run / split),
                ],
                out / f"seed{seed}-{split}.log",
            )
            report = json.loads((run / split / "metrics.json").read_text())
            if report["dataset_id"] != summary["dataset_id"]:
                raise ValueError("Dataset changed")
            summary["runs"][str(seed)][split] = report["metrics"]
        save()
    summary["complete"] = True
    save()
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
