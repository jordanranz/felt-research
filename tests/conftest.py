from pathlib import Path

import pytest

from felt.data import load_config


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
