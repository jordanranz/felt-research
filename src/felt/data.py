"""Synthetic rhythms, causal features, and a deliberately simple procedural teacher."""

import hashlib
import json
from pathlib import Path

import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_config(path):
    config = json.loads(Path(path).read_text())
    if config["sample_rate"] % config["frame_rate"]:
        raise ValueError("Sample rate must be divisible by envelope frame rate")
    if config["seconds"] * config["frame_rate"] % 1:
        raise ValueError("Duration must contain a whole number of frames")
    if config["lookahead_frames"] != 0 or config["precision"] != "float32":
        raise ValueError("Experiment 001 supports causal float32 only")
    expected = {
        "generator_version": "synthetic-v1",
        "preprocessing_version": "frame-features-v1",
        "target_version": "causal-rule-v1",
        "model_version": "causal-tcn-v1",
        "loss": "active-weighted-mse-v1",
    }
    for key, value in expected.items():
        if config[key] != value:
            raise ValueError(f"Unsupported {key}: {config[key]}")
    if any(
        config[key] <= 0
        for key in (
            "sample_rate",
            "frame_rate",
            "seconds",
            "variants_per_family",
            "batch_size",
            "epochs",
            "hidden_channels",
            "threads",
            "learning_rate",
        )
    ) or any(config["families"][split] <= 0 for split in ("train", "val", "test")):
        raise ValueError("Counts, rates and learning rate must be positive")
    return config


def families(config):
    """Canonicalize cyclic shifts so rotated versions cannot cross split boundaries."""
    rng = np.random.default_rng(config["seed"])
    seen, result = set(), []
    total = sum(config["families"].values())
    if total > 1000:
        raise ValueError("This small synthetic generator supports at most 1000 families")
    while len(result) < total:
        pattern = tuple(int(x) for x in rng.integers(0, 4, 16))
        canonical = min(pattern[i:] + pattern[:i] for i in range(16))
        if canonical in seen or sum(x != 0 for x in canonical) < 4:
            continue
        seen.add(canonical)
        result.append(canonical)
    return result


def synthesize(pattern, seed, config):
    rng = np.random.default_rng(seed)
    sr = config["sample_rate"]
    seconds = config["seconds"]
    bpm = float(rng.uniform(80, 170))
    offset = float(rng.uniform(0.05, 0.3))
    beats = np.arange(offset, seconds, 60 / bpm)
    waveform = np.zeros(round(seconds * sr), dtype=np.float32)
    for index, onset in enumerate(np.arange(offset, seconds, 30 / bpm)):
        instrument = pattern[index % len(pattern)]
        if not instrument:
            continue
        t = np.arange(round(0.18 * sr)) / sr
        strength = float(rng.uniform(0.3, 0.9))
        if instrument == 1:
            voice = np.sin(2 * np.pi * (65 * t + 2 * (1 - np.exp(-30 * t))))
            voice *= np.exp(-25 * t)
        elif instrument == 2:
            voice = rng.normal(0, 0.35, len(t)) * np.exp(-40 * t)
        else:
            noise = rng.normal(0, 0.2, len(t) + 1)
            voice = np.diff(noise) * np.exp(-90 * t)
        start = round(onset * sr)
        length = min(len(voice), len(waveform) - start)
        waveform[start : start + length] += (strength * voice[:length]).astype(np.float32)
    return np.clip(waveform, -1, 1), beats, bpm


def features(waveform, beats, config):
    """Each frame uses its own audio bin, available at that bin's END timestamp."""
    sr, hz = config["sample_rate"], config["frame_rate"]
    width = sr // hz
    frames = np.asarray(waveform, dtype=np.float32).reshape(-1, width)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    spectrum = np.abs(np.fft.rfft(frames, axis=1)) / width
    frequencies = np.fft.rfftfreq(width, 1 / sr)
    low = np.sqrt(np.sum(spectrum[:, frequencies < 400] ** 2, axis=1))
    high = np.sqrt(np.sum(spectrum[:, frequencies >= 1000] ** 2, axis=1))
    audio = np.clip(np.stack([rms, low, high]) * 4, 0, 1)
    impulses = np.zeros(len(frames), dtype=np.float32)
    bins = np.floor(np.asarray(beats) * hz).astype(int)
    impulses[bins[(bins >= 0) & (bins < len(frames))]] = 1
    decay = np.zeros_like(impulses)
    for index, value in enumerate(impulses):
        decay[index] = max(value, decay[index - 1] * 0.8 if index else 0)
    return np.concatenate([audio, impulses[None], decay[None]]).astype(np.float32)


def teacher(x):
    """Known rule; matching this is pipeline validation, not learned human preference."""
    drive = np.clip(0.7 * x[0] + 0.3 * x[3], 0, 1)
    result = np.zeros_like(drive)
    for index, value in enumerate(drive):
        result[index] = max(value, result[index - 1] * 0.78 if index else 0)
    return result


def make_dataset(config):
    patterns = families(config)
    splits, records, cursor = {}, [], 0
    content_hash = hashlib.sha256()
    for split in ("train", "val", "test"):
        xs, ys = [], []
        for family_index in range(cursor, cursor + config["families"][split]):
            pattern = patterns[family_index]
            for variant in range(config["variants_per_family"]):
                seed = config["seed"] + 10000 + family_index * 100 + variant
                wav, beats, bpm = synthesize(pattern, seed, config)
                x = features(wav, beats, config)
                y = teacher(x)
                xs.append(x)
                ys.append(y)
                records.append(
                    {
                        "split": split,
                        "family": digest(pattern),
                        "pattern": list(pattern),
                        "seed": seed,
                        "bpm": bpm,
                        "beats": beats.tolist(),
                        "audio_sha256": hashlib.sha256(wav.tobytes()).hexdigest(),
                    }
                )
                content_hash.update(x.tobytes())
                content_hash.update(y.tobytes())
        splits[split] = np.stack(xs), np.stack(ys)
        cursor += config["families"][split]
    manifest = {
        "schema_version": 1,
        "generator": config["generator_version"],
        "records": records,
        "features_targets_sha256": content_hash.hexdigest(),
    }
    manifest["dataset_id"] = digest(manifest)
    return splits, manifest
