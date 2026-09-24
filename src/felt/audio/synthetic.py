"""Versioned procedural sound diversity; no external recordings or samples."""

import numpy as np

PROFILES = ("percussion", "tonal_bed", "noise_bed", "sparse_dynamics")


def profile(seed):
    return PROFILES[seed % len(PROFILES)]


def synthesize_diverse(pattern, seed, config):
    rng = np.random.default_rng(seed)
    sr, seconds = config["sample_rate"], config["seconds"]
    bpm = float(rng.uniform(60, 200))
    offset = float(rng.uniform(0.05, 0.3))
    beats = np.arange(offset, seconds, 60 / bpm)
    waveform = np.zeros(round(seconds * sr), dtype=np.float32)
    kind = profile(seed)
    swing = rng.uniform(0, 0.22)
    kick_hz, kick_decay = rng.uniform(35, 110), rng.uniform(10, 45)
    snare_hz, snare_decay = rng.uniform(140, 320), rng.uniform(12, 65)
    hat_decay = rng.uniform(25, 130)
    for index, onset in enumerate(np.arange(offset, seconds, 30 / bpm)):
        instrument = pattern[index % len(pattern)]
        if not instrument or (kind == "sparse_dynamics" and rng.random() < 0.5):
            continue
        onset += swing * 30 / bpm if index % 2 else 0
        start = round(onset * sr)
        if start >= len(waveform):
            continue
        t = np.arange(round(0.4 * sr)) / sr
        strength = float(rng.uniform(0.08, 0.95))
        if instrument == 1:
            sweep = rng.uniform(0.5, 3)
            voice = np.sin(2 * np.pi * (kick_hz * t + sweep * (1 - np.exp(-30 * t))))
            voice *= np.exp(-kick_decay * t)
        elif instrument == 2:
            voice = rng.normal(0, 0.25, len(t)) + 0.2 * np.sin(2 * np.pi * snare_hz * t)
            voice *= np.exp(-snare_decay * t)
        else:
            voice = np.diff(rng.normal(0, 0.15, len(t) + 1)) * np.exp(-hat_decay * t)
        length = min(len(voice), len(waveform) - start)
        waveform[start : start + length] += (strength * voice[:length]).astype(np.float32)
    t = np.arange(len(waveform)) / sr
    fade = np.minimum(t / 0.03, 1) * np.minimum((seconds - t) / 0.03, 1)
    if kind == "tonal_bed":
        fundamental = rng.uniform(45, 440)
        bed = sum(
            np.sin(2 * np.pi * fundamental * harmonic * t + rng.uniform(0, 6.28)) / harmonic
            for harmonic in (1, 2, 3)
        )
        waveform += (rng.uniform(0.01, 0.12) * bed * fade).astype(np.float32)
    elif kind == "noise_bed":
        waveform += (rng.normal(0, rng.uniform(0.005, 0.06), len(t)) * fade).astype(np.float32)
    waveform *= float(rng.uniform(0.2, 1.2))
    return np.clip(waveform, -1, 1), beats, bpm
