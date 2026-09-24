import numpy as np
import pytest
import torch

from felt.evaluate import metrics
from felt.models import loss


@pytest.mark.parametrize(
    "kind", ["active-weighted-mse-v1", "region-balanced-mse-v1", "region-startup-mse-v1"]
)
@pytest.mark.parametrize("level", [0.0, 0.05, 0.5])
def test_losses_handle_missing_regions_and_backpropagate(kind, level):
    target = torch.full((2, 40), level)
    prediction = torch.full_like(target, 0.25, requires_grad=True)
    value = loss(prediction, target, kind)
    value.backward()
    assert torch.isfinite(value)
    assert torch.isfinite(prediction.grad).all()
    assert loss(target, target, kind).item() == 0


def test_startup_loss_penalizes_equal_errors_more_at_clip_start():
    target = torch.zeros(1, 100)
    early, late = target.clone(), target.clone()
    early[:, 0] = 0.5
    late[:, 90] = 0.5
    assert loss(early, target, "region-balanced-mse-v1") == loss(
        late, target, "region-balanced-mse-v1"
    )
    assert loss(early, target, "region-startup-mse-v1") > loss(
        late, target, "region-startup-mse-v1"
    )
    m = metrics(early.numpy(), target.numpy())
    assert m["first_frame_mean_intensity"] == 0.5
    assert m["startup_rmse"] == pytest.approx(np.sqrt(0.25 / 29))
