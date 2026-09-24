"""Evaluate the validation-selected checkpoint once on held-out synthetic families."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from felt.data import features, load_config, make_dataset, synthesize, teacher
from felt.models import EnvelopeModel
from felt.patterns import pattern, save
from felt.preview import render, write_wav
from felt.train import environment


def metrics(predicted, target, hz=100):
    active = target > 0.1
    silence = target < 0.02
    errors = (predicted - target) ** 2
    hits = missing = extra = 0
    offsets = []
    # Pulse events are upward crossings of a fixed 0.2 threshold, matched within 30 ms.
    for p, y in zip(predicted, target, strict=True):
        onsets_p = np.flatnonzero(np.diff(np.r_[False, p >= 0.2].astype(int)) == 1)
        onsets_y = np.flatnonzero(np.diff(np.r_[False, y >= 0.2].astype(int)) == 1)
        unmatched = set(onsets_p.tolist())
        for onset in onsets_y:
            candidate = min(unmatched, key=lambda x: (abs(x - onset), x)) if unmatched else None
            if candidate is not None and abs(candidate - onset) / hz <= 0.030001:
                hits += 1
                offsets.append(abs(candidate - onset) * 1000 / hz)
                unmatched.remove(candidate)
            else:
                missing += 1
        extra += len(unmatched)
    precision = hits / (hits + extra) if hits + extra else 0.0
    recall = hits / (hits + missing) if hits + missing else 0.0
    return {
        "rmse": float(np.sqrt(errors.mean())),
        "startup_rmse": float(np.sqrt(errors[:, :29].mean())),
        "first_frame_mean_intensity": float(predicted[:, 0].mean()),
        "active_rmse": float(np.sqrt(errors[active].mean())) if active.any() else None,
        "silence_mean_intensity": float(predicted[silence].mean()) if silence.any() else None,
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "matched_timing_mae_ms": float(np.mean(offsets)) if offsets else None,
        "matched_events": hits,
        "missed_events": missing,
        "extra_events": extra,
    }


def evaluate(checkpoint_path, out, split="test", dataset_config=None):
    if split not in ("val", "test"):
        raise ValueError("Evaluation split must be val or test")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    training_config = checkpoint["config"]
    config = load_config(dataset_config) if dataset_config else training_config
    for key in (
        "sample_rate",
        "frame_rate",
        "preprocessing_version",
        "target_version",
        "lookahead_frames",
    ):
        if config[key] != training_config[key]:
            raise ValueError(f"Incompatible evaluation contract: {key}")
    torch.set_num_threads(config["threads"])
    splits, manifest = make_dataset(config)
    if dataset_config is None and checkpoint["dataset_id"] != manifest["dataset_id"]:
        raise ValueError("Evaluation dataset does not match training manifest")
    model = EnvelopeModel(
        checkpoint["mode"], training_config["hidden_channels"], training_config["model_version"]
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    x, y = splits[split]
    with torch.no_grad():
        prediction = model(torch.from_numpy(x)).numpy()
        sample = torch.from_numpy(x[:1])
        for _ in range(5):
            model(sample)
        durations = []
        for _ in range(30):
            start = time.perf_counter()
            model(sample)
            durations.append((time.perf_counter() - start) * 1000)
    outputs = {
        "model": prediction,
        "beat_pulses": 0.3 * x[:, 4],
        "constant_train_mean": np.full_like(y, splits["train"][1].mean()),
        "silence": np.zeros_like(y),
        "procedural_oracle": np.stack([teacher(row) for row in x]),
    }
    report = {
        "mode": checkpoint["mode"],
        "dataset_id": manifest["dataset_id"],
        "training_dataset_id": checkpoint["dataset_id"],
        "external_dataset": dataset_config is not None,
        "training_config": training_config,
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
        "config": config,
        "training_environment": checkpoint["environment"],
        "evaluation_environment": environment(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "evaluation_split": split,
        "evaluated_clips": len(y),
        "metrics": {
            name: metrics(value, y, config["frame_rate"]) for name, value in outputs.items()
        },
        "cpu_whole_clip_latency_ms": {
            "median": float(np.median(durations)),
            "p95": float(np.percentile(durations, 95)),
            "warmup": 5,
            "repetitions": 30,
        },
        "latency_scope": f"Model only, full {config['seconds']}-second clip; excludes preprocessing and playback. "
        "This is not a streaming latency measurement.",
    }
    records = [r for r in manifest["records"] if r["split"] == split]
    profiles = sorted({r["sound_profile"] for r in records if "sound_profile" in r})
    report["sound_profiles"] = {
        name: {
            "clips": sum(r.get("sound_profile") == name for r in records),
            "metrics": {
                label: metrics(
                    value[[r.get("sound_profile") == name for r in records]],
                    y[[r.get("sound_profile") == name for r in records]],
                    config["frame_rate"],
                )
                for label, value in outputs.items()
            },
        }
        for name in profiles
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    record = next(r for r in manifest["records"] if r["split"] == split)
    wav, beats, _ = synthesize(record["pattern"], record["seed"], config)
    assert np.array_equal(features(wav, beats, config), x[0])
    previews = {}
    for name in ("model", "beat_pulses", "procedural_oracle"):
        value = pattern(outputs[name][0], config["frame_rate"])
        save(value, out / f"{name}.json")
        previews[name] = render(value, config["sample_rate"])
        write_wav(out / f"{name}.wav", previews[name], config["sample_rate"])
    write_wav(out / "source.wav", wav, config["sample_rate"])
    stereo = np.stack(
        [np.pad(wav, (0, len(previews["model"]) - len(wav))) * 0.5, previews["model"]], axis=1
    )
    write_wav(out / "comparison.wav", stereo, config["sample_rate"])
    times = (np.arange(y.shape[1]) + 1) / config["frame_rate"]
    figure, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True, layout="constrained")
    axes[0].plot(np.arange(len(wav)) / config["sample_rate"], wav, color="#718096", lw=0.5)
    for beat in beats:
        axes[0].axvline(beat, color="#805ad5", alpha=0.3, lw=0.7)
    axes[0].set_ylabel("Audio / beats")
    axes[0].set_title(f"Felt Research · first {split} synthetic clip")
    for name, color in (
        ("procedural_oracle", "#1a202c"),
        ("model", "#008080"),
        ("beat_pulses", "#dd6b20"),
    ):
        axes[1].plot(times, outputs[name][0], label=name, color=color, lw=1.2)
    axes[1].set(xlabel="Time (seconds)", ylabel="Requested intensity", ylim=(-0.02, 1.02))
    axes[1].legend(loc="upper right")
    figure.savefig(out / "comparison.png", dpi=160)
    plt.close(figure)
    print(json.dumps(report["metrics"], indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument(
        "--dataset-config",
        help="Explicit external dataset; records training and evaluation identities",
    )
    args = parser.parse_args()
    evaluate(args.checkpoint, args.out, args.split, args.dataset_config)


if __name__ == "__main__":
    main()
