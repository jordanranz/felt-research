import numpy as np
import pytest
import torch

from felt.data import make_dataset
from felt.inference import EnvelopeStream
from felt.models import MODEL_VERSIONS, EnvelopeModel
from felt.train import train


@pytest.mark.parametrize("version", MODEL_VERSIONS)
def test_chunk_inference_matches_whole_clip_and_has_no_future_access(version):
    torch.manual_seed(9)
    model = EnvelopeModel(hidden=4, version=version).eval()
    inputs = torch.rand(2, 5, 105)
    stream = EnvelopeStream(model)
    with torch.no_grad():
        whole = model(inputs)
        chunks = torch.cat([stream.predict(c) for c in inputs.split(7, dim=-1)], dim=-1)
        changed = inputs.clone()
        changed[:, :, 40:] = 9
        future_changed = model(changed)
    torch.testing.assert_close(chunks, whole, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(whole[:, :40], future_changed[:, :40], rtol=0, atol=0)
    assert torch.isfinite(whole).all() and whole.min() >= 0 and whole.max() <= 1
    stream.reset()
    torch.testing.assert_close(stream.predict(inputs), whole)


def test_warmup_removes_silence_boundary_transient():
    torch.manual_seed(42)
    original = EnvelopeModel(hidden=4)
    warm = EnvelopeModel(hidden=4, version="warmup-sigmoid-v2")
    warm.load_state_dict(original.state_dict())
    with torch.no_grad():
        expected = original(torch.zeros(1, 5, 100))[:, 28:]
        actual = warm(torch.zeros(1, 5, 72))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual, actual[:, :1].expand_as(actual), rtol=0, atol=0)


def test_initialization_seed_does_not_change_dataset(config):
    a, ma = make_dataset(dict(config, initialization_seed=10))
    b, mb = make_dataset(dict(config, initialization_seed=20))
    assert ma == mb
    for split in a:
        np.testing.assert_array_equal(a[split][0], b[split][0])


def test_new_model_resume_is_exact(config, tmp_path):
    config = dict(config, initialization_seed=7, model_version="warmup-linear-v2")
    train(config, tmp_path / "full")
    train(config, tmp_path / "split", stop_after=1)
    train(config, tmp_path / "split", resume=tmp_path / "split/last.pt")
    a = torch.load(tmp_path / "full/last.pt", weights_only=False)
    b = torch.load(tmp_path / "split/last.pt", weights_only=False)
    assert a["history"] == b["history"]
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
