import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from felt.audio.fixtures import export_fixtures
from felt.data import load_config, make_dataset


def test_fixture_export_is_reproducible_and_matches_training(tmp_path):
    config = load_config(Path(__file__).parents[1] / "configs/experiment001.json")
    config.update(families={"train": 2, "val": 1, "test": 1}, variants_per_family=2, seconds=1)
    a, b = tmp_path / "a", tmp_path / "b"
    export_fixtures(config, a, count=2)
    export_fixtures(config, b, count=2)
    bundle = json.loads((a / "index.json").read_text())
    assert (a / "index.json").read_bytes() == (b / "index.json").read_bytes()
    splits, manifest = make_dataset(config)
    assert bundle["dataset_id"] == manifest["dataset_id"]
    for index, clip in enumerate(bundle["clips"]):
        for artifact in clip["artifacts"].values():
            content = (a / artifact["path"]).read_bytes()
            assert hashlib.sha256(content).hexdigest() == artifact["sha256"]
            assert content == (b / artifact["path"]).read_bytes()
        actual = np.load(a / clip["artifacts"]["features.npy"]["path"], allow_pickle=False)
        np.testing.assert_array_equal(actual, splits["test"][0][index])
    with pytest.raises(ValueError, match="not empty"):
        export_fixtures(config, a, count=1)
    with pytest.raises(ValueError, match="Count"):
        export_fixtures(config, tmp_path / "invalid", count=999)
