"""Bounded validation-only architecture checks; no scaled run is launched."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from felt.data import features, load_config, make_dataset
from felt.inference import EnvelopeStream
from felt.models import EnvelopeModel


def probes(checkpoint):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = ck["config"]
    torch.set_num_threads(config["threads"])
    model = EnvelopeModel(ck["mode"], config["hidden_channels"], config["model_version"]).eval()
    model.load_state_dict(ck["state_dict"])
    sr = config["sample_rate"]
    t = np.arange(sr * 3) / sr
    zero = np.zeros(len(t), dtype=np.float32)
    impulse = zero.copy()
    impulse[sr : sr + 160] = 0.5
    sources = {
        "silence": (zero, []),
        "beat_only": (zero, [0.5, 1.0, 1.5, 2.0]),
        "impulse": (impulse, []),
        "tone_55hz": ((0.25 * np.sin(2 * np.pi * 55 * t)).astype(np.float32), []),
        "noise": (np.random.default_rng(2026).uniform(-0.3, 0.3, len(t)).astype(np.float32), []),
    }
    result = {}
    with torch.no_grad():
        for name, (wav, beats) in sources.items():
            output = model(torch.from_numpy(features(wav, beats, config)[None]))
            result[name] = {
                "min": float(output.min()),
                "max": float(output.max()),
                "mean": float(output.mean()),
                "finite_bounded": bool(
                    torch.isfinite(output).all() and output.min() >= 0 and output.max() <= 1
                ),
            }
        x = torch.from_numpy(make_dataset(config)[0]["val"][0][:1])
        whole = model(x)
        stream = EnvelopeStream(model)
        pieces, durations = [], []
        for index in range(x.shape[-1]):
            started = time.perf_counter()
            pieces.append(stream.predict(x[:, :, index : index + 1]))
            elapsed = (time.perf_counter() - started) * 1000
            if index >= 28:
                durations.append(elapsed)
        chunked = torch.cat(pieces, dim=-1)
    result["streaming"] = {
        "matches_whole_clip": bool(torch.allclose(whole, chunked, atol=1e-6, rtol=1e-5)),
        "max_abs_error": float((whole - chunked).abs().max()),
        "single_frame_cpu_p95_ms": float(np.percentile(durations, 95)),
        "timing_scope": "Model and feature-history handling only; excludes audio features and I/O",
    }
    return result


def gates(candidate, baseline):
    c, b = candidate["metrics"], baseline["metrics"]
    p = candidate["probes"]
    return {
        "quiet_halved": c["silence_mean_intensity"] <= b["silence_mean_intensity"] * 0.5,
        "startup_halved": c["startup_rmse"] <= b["startup_rmse"] * 0.5,
        "active_preserved": c["active_rmse"] <= b["active_rmse"] * 1.2,
        "events_preserved": c["event_f1"] >= b["event_f1"] - 0.02,
        "active_absolute": c["active_rmse"] <= 0.08,
        "events_absolute": c["event_f1"] >= 0.95,
        "silence_peak": p["silence"]["max"] <= 0.02,
        "probe_bounds": all(v["finite_bounded"] for k, v in p.items() if k != "streaming"),
        "streaming_equivalence": p["streaming"]["matches_whole_clip"],
        "frame_budget": p["streaming"]["single_frame_cpu_p95_ms"] < 10,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["screen", "confirm"], required=True)
    parser.add_argument(
        "--candidate", choices=["warmup-sigmoid-v2", "warmup-linear-v2"], default="warmup-linear-v2"
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a fresh preflight directory")
    out.mkdir(parents=True, exist_ok=True)
    base = load_config("configs/experiment001.json")
    seeds = [42] if args.stage == "screen" else [7, 19, 42, 73, 101]
    versions = (
        ["causal-tcn-v1", "warmup-sigmoid-v2", "warmup-linear-v2"]
        if args.stage == "screen"
        else ["causal-tcn-v1", args.candidate]
    )
    summary = {
        "stage": args.stage,
        "initialization_seeds": seeds,
        "dataset_seed": base["seed"],
        "test_set_evaluated": False,
        "runs": {},
    }
    (out / "plan.json").write_text(json.dumps({**summary, "versions": versions}, indent=2) + "\n")
    for seed in seeds:
        for version in versions:
            name = f"{version}-seed{seed}"
            config = dict(base, initialization_seed=seed, model_version=version)
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
                        "--out",
                        str(run),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            train_seconds = time.perf_counter() - started
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
            report = json.loads((run / "evaluation/metrics.json").read_text())
            result = {
                "metrics": report["metrics"]["model"],
                "probes": probes(run / "best.pt"),
                "train_wall_seconds": round(train_seconds, 3),
                "total_wall_seconds": round(time.perf_counter() - started, 3),
                "dataset_id": report["dataset_id"],
            }
            if version != "causal-tcn-v1":
                result["gates"] = gates(result, summary["runs"][f"causal-tcn-v1-seed{seed}"])
                result["passed"] = all(result["gates"].values())
            summary["runs"][name] = result
            (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "run": name,
                        "metrics": result["metrics"],
                        "seconds": result["total_wall_seconds"],
                        "gates": result.get("gates"),
                        "silence_peak": result["probes"]["silence"]["max"],
                    }
                ),
                flush=True,
            )
    if len({r["dataset_id"] for r in summary["runs"].values()}) != 1:
        raise ValueError("Dataset identity changed across comparisons")
    summary["completed"] = True
    summary["ready_for_larger_synthetic_study"] = args.stage == "confirm" and all(
        result["passed"] for result in summary["runs"].values() if "passed" in result
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.stage == "confirm" and not summary["ready_for_larger_synthetic_study"]:
        raise SystemExit("Preflight failed one or more gates; inspect summary.json before scaling")


if __name__ == "__main__":
    main()
