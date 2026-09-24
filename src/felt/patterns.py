import json
from pathlib import Path

import numpy as np


def pattern(envelope, frame_rate=100):
    result = {
        "schema_version": 1,
        "kind": "normalized_intensity_envelope",
        "frame_rate_hz": frame_rate,
        "start_time_seconds": 1 / frame_rate,
        "intensity": np.asarray(envelope).tolist(),
    }
    validate(result)
    return result


def validate(value):
    if value["schema_version"] != 1 or value["kind"] != "normalized_intensity_envelope":
        raise ValueError("Unsupported pattern schema")
    hz, start = value["frame_rate_hz"], value["start_time_seconds"]
    if not np.isfinite(hz) or hz <= 0 or not np.isfinite(start) or start < 0:
        raise ValueError("Invalid envelope timebase")
    values = np.asarray(value["intensity"], dtype=np.float32)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Intensity must be a finite, nonempty vector")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("Intensity must be between zero and one")
    return values


def save(value, path):
    validate(value)
    Path(path).write_text(json.dumps(value, indent=2) + "\n")
