"""Fixed-gain 180 Hz sonification. This is not an actuator waveform or perceptual proxy."""

import argparse
import json
import wave
from pathlib import Path

import numpy as np

from felt.patterns import validate


def render(value, sample_rate=16000):
    values = validate(value)
    hz, start = value["frame_rate_hz"], value["start_time_seconds"]
    length = round((start + len(values) / hz) * sample_rate)
    times = np.arange(length) / sample_rate
    bins = np.floor((times - start) * hz + 1e-8).astype(int)
    amplitude = np.zeros(length)
    valid = (bins >= 0) & (bins < len(values))
    amplitude[valid] = values[bins[valid]]
    # Trailing 3 ms ramp smooths amplitude steps without shifting them earlier.
    width = max(1, round(0.003 * sample_rate))
    amplitude = np.convolve(amplitude, np.ones(width) / width, mode="full")[:length]
    output = 0.25 * amplitude * np.sin(2 * np.pi * 180 * times)
    fade = min(round(0.005 * sample_rate), length)
    output[-fade:] *= np.linspace(1, 0, fade)
    return output.astype(np.float32)


def write_wav(path, audio, sample_rate=16000):
    audio = np.asarray(audio)
    with wave.open(str(path), "wb") as file:
        file.setnchannels(1 if audio.ndim == 1 else audio.shape[1])
        file.setsampwidth(2)
        file.setframerate(sample_rate)
        file.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    write_wav(args.out, render(json.loads(Path(args.pattern).read_text())))


if __name__ == "__main__":
    main()
