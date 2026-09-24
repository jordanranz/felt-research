import json
from pathlib import Path

import numpy as np
import pytest

from felt.audio.synthetic import PROFILES
from felt.data import load_config, make_dataset
from felt.evaluate import evaluate
from felt.train import train


def test_original_generator_matches_frozen_legacy_outputs():
    config = load_config(Path(__file__).parents[1] / "configs/experiment001.json")
    arrays, manifest = make_dataset(config)
    fixture = Path(__file__).parent / "fixtures/synthetic-v1"
    expected = json.loads(fixture.with_suffix(".json").read_text())
    records = [
        {k: r[k] for k in ("split", "family", "pattern", "seed", "bpm", "beats")}
        for r in manifest["records"]
    ]
    assert len(records) == len(expected["records"])
    for actual, frozen_record in zip(records, expected["records"], strict=True):
        for key in ("split", "family", "pattern", "seed"):
            assert actual[key] == frozen_record[key]
        np.testing.assert_allclose(actual["bpm"], frozen_record["bpm"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(actual["beats"], frozen_record["beats"], rtol=0, atol=1e-12)
    # Floating-point audio/FFT bytes may differ across operating systems.
    # Same-environment exact regeneration is separately tested in test_contract.
    with np.load(fixture.with_suffix(".npz"), allow_pickle=False) as frozen:
        for split, pair in arrays.items():
            for kind, actual in zip(("features", "targets"), pair, strict=True):
                np.testing.assert_allclose(actual, frozen[f"{split}_{kind}"], rtol=1e-5, atol=1e-7)


def test_diverse_reproducibility_profiles_and_group_isolation(config):
    config = dict(config, generator_version="synthetic-diverse-v2", variants_per_family=4)
    arrays, manifest = make_dataset(config)
    again, repeated = make_dataset(config)
    assert manifest == repeated
    groups = {}
    for split, (x, y) in arrays.items():
        np.testing.assert_array_equal(x, again[split][0])
        np.testing.assert_array_equal(y, again[split][1])
        assert np.isfinite(x).all() and np.isfinite(y).all()
        assert x.min() >= 0 and x.max() <= 1
        records = [r for r in manifest["records"] if r["split"] == split]
        assert {r["sound_profile"] for r in records} == set(PROFILES)
        groups[split] = {r["family"] for r in records}
    assert not groups["train"] & groups["val"]
    assert not groups["train"] & groups["test"]
    assert not groups["val"] & groups["test"]
    with pytest.raises(ValueError, match="seed spacing"):
        make_dataset(dict(config, variants_per_family=101))


def test_external_evaluation_records_both_identities_and_checks_contract(config, tmp_path):
    train(config, tmp_path / "train")
    external = dict(
        config, generator_version="synthetic-diverse-v2", seed=99, variants_per_family=4
    )
    path = tmp_path / "external.json"
    path.write_text(json.dumps(external))
    report = evaluate(tmp_path / "train/best.pt", tmp_path / "eval", "val", path)
    assert report["external_dataset"] is True
    assert report["training_dataset_id"] != report["dataset_id"]
    assert len(report["checkpoint_sha256"]) == 64
    assert set(report["sound_profiles"]) == set(PROFILES)
    assert sum(p["clips"] for p in report["sound_profiles"].values()) == report["evaluated_clips"]
    assert report["metrics"]["procedural_oracle"]["rmse"] == 0
    path.write_text(json.dumps(dict(external, sample_rate=8000)))
    with pytest.raises(ValueError, match="Incompatible evaluation contract: sample_rate"):
        evaluate(tmp_path / "train/best.pt", tmp_path / "invalid", "val", path)
