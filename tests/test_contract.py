import json
from pathlib import Path

import numpy as np
import pytest
import torch

from felt.data import features, load_config, make_dataset, synthesize, teacher
from felt.models import EnvelopeModel
from felt.patterns import pattern, validate
from felt.preview import render
from felt.train import train


@pytest.fixture
def config():
    value = load_config(Path(__file__).parents[1] / "configs/experiment001.json")
    value.update(
        families={"train": 3, "val": 2, "test": 2},
        variants_per_family=2,
        seconds=1,
        epochs=3,
        hidden_channels=4,
        batch_size=2,
        threads=1,
    )
    return value


def test_reproducible_data_and_group_isolation(config):
    splits, manifest = make_dataset(config)
    again, other = make_dataset(config)
    assert manifest == other
    groups = {
        key: {r["family"] for r in manifest["records"] if r["split"] == key} for key in splits
    }
    assert not groups["train"] & groups["val"]
    assert not groups["train"] & groups["test"]
    assert not groups["val"] & groups["test"]
    for key in splits:
        np.testing.assert_array_equal(splits[key][0], again[key][0])
        assert np.isfinite(splits[key][1]).all()
        assert np.all((splits[key][1] >= 0) & (splits[key][1] <= 1))


def test_preprocessing_teacher_and_model_causality(config):
    wav, beats, _ = synthesize([1, 2, 0, 3] * 4, 42, config)
    original = features(wav, beats, config)
    cutoff = 40
    changed = wav.copy()
    changed[cutoff * 160 :] = 0.9
    modified = features(changed, np.r_[beats[beats < 0.4], 0.5, 0.8], config)
    np.testing.assert_array_equal(original[:, :cutoff], modified[:, :cutoff])
    np.testing.assert_array_equal(teacher(original)[:cutoff], teacher(modified)[:cutoff])
    model = EnvelopeModel(hidden=4).eval()
    with torch.no_grad():
        a = model(torch.from_numpy(original[None]))
        b = model(torch.from_numpy(modified[None]))
    torch.testing.assert_close(a[:, :cutoff], b[:, :cutoff], rtol=0, atol=0)


def test_frame_end_timestamp_and_audio_gain(config):
    wav = np.zeros(config["sample_rate"], dtype=np.float32)
    x = features(wav, [0.005, 0.02], config)
    assert np.flatnonzero(x[3]).tolist() == [0, 2]
    value = pattern(np.r_[1.0, np.zeros(9)])
    assert value["start_time_seconds"] == 0.01
    audio = render(value)
    assert np.all(audio[:160] == 0)
    half = render(pattern(np.r_[0.5, np.zeros(9)]))
    np.testing.assert_allclose(half, audio * 0.5)
    with pytest.raises(ValueError):
        validate(pattern([float("nan")]))


def test_epoch_boundary_resume_is_exact(config, tmp_path):
    train(config, tmp_path / "continuous")
    train(config, tmp_path / "interrupted", stop_after=1)
    train(config, tmp_path / "interrupted", resume=tmp_path / "interrupted/last.pt")
    a = torch.load(tmp_path / "continuous/last.pt", weights_only=False)
    b = torch.load(tmp_path / "interrupted/last.pt", weights_only=False)
    assert a["history"] == b["history"]
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    wrong = dict(config, seed=999)
    with pytest.raises(ValueError, match="same resolved config"):
        train(wrong, tmp_path / "invalid", resume=tmp_path / "interrupted/last.pt")


def test_evaluation_and_export(config, tmp_path):
    from felt.evaluate import evaluate, metrics

    train(config, tmp_path / "run")
    result = evaluate(tmp_path / "run/best.pt", tmp_path / "report")
    assert result["metrics"]["procedural_oracle"]["rmse"] == 0
    assert (tmp_path / "report/comparison.wav").stat().st_size > 44
    validate(json.loads((tmp_path / "report/model.json").read_text()))
    truth = np.zeros((1, 50))
    truth[0, 10:15] = 0.5
    shifted = np.zeros_like(truth)
    shifted[0, 12:17] = 0.5
    score = metrics(shifted, truth)
    assert score["matched_timing_mae_ms"] == 20
    assert score["event_f1"] == 1
    assert metrics(np.zeros_like(truth), truth)["missed_events"] == 1
