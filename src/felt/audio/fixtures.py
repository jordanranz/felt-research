"""Export reproducible synthetic fixtures without duplicating the v1 generator."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from felt.data import features, make_dataset, synthesize
from felt.patterns import pattern, save
from felt.preview import render, write_wav


def export_fixtures(config, out, split="test", count=12):
    if split not in ("train", "val", "test"):
        raise ValueError("Split must be train, val, or test")
    available = config["families"][split] * config["variants_per_family"]
    if not isinstance(count, int) or not 1 <= count <= available:
        raise ValueError(f"Count must be between 1 and {available}")
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory is not empty; choose a new directory")
    started = time.perf_counter()
    splits, manifest = make_dataset(config)
    records = [r for r in manifest["records"] if r["split"] == split][:count]
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for index, record in enumerate(records):
        directory = out / f"{split}-{index:04d}"
        directory.mkdir()
        wav, beats, _ = synthesize(record["pattern"], record["seed"], config)
        x = features(wav, beats, config)
        np.testing.assert_array_equal(x, splits[split][0][index])
        value = pattern(splits[split][1][index], config["frame_rate"])
        write_wav(directory / "source.wav", wav, config["sample_rate"])
        (directory / "beats.json").write_text(json.dumps(beats.tolist(), indent=2) + "\n")
        save(value, directory / "target.json")
        write_wav(
            directory / "target-preview.wav",
            render(value, config["sample_rate"]),
            config["sample_rate"],
        )
        np.save(directory / "features.npy", x, allow_pickle=False)
        artifacts = {
            path.name: {
                "path": path.relative_to(out).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in sorted(directory.iterdir())
        }
        entries.append({"id": directory.name, "source": record, "artifacts": artifacts})
    bundle = {
        "schema_version": 1,
        "kind": "felt.synthetic_fixtures",
        "dataset_id": manifest["dataset_id"],
        "split": split,
        "config": config,
        "features": {
            "version": config["preprocessing_version"],
            "layout": "channels,time",
            "channels": ["rms", "low_energy", "high_energy", "beat_impulse", "beat_decay"],
            "frame_rate_hz": config["frame_rate"],
            "first_available_seconds": 1 / config["frame_rate"],
        },
        "audio_note": "source.wav is PCM16 for inspection. Training uses the original float32 "
        "waveform regenerated from the source seed; PCM16 quantization changes samples.",
        "clips": entries,
    }
    (out / "index.json").write_text(json.dumps(bundle, indent=2) + "\n")
    return {
        "clips": count,
        "audio_seconds": count * config["seconds"],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "bytes": sum(p.stat().st_size for p in out.rglob("*") if p.is_file()),
        "index": str(out / "index.json"),
        "dataset_id": manifest["dataset_id"],
    }
