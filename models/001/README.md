# Experiment 001 reference weights

These original Felt Research weights are released under the repository's
[MIT License](../../LICENSE), Copyright (c) 2026 Jordan Ranz.

| File | Inputs | Parameters |
|---|---|---:|
| `combined.pt` | Audio features and supplied beat timestamps | 11,169 |
| `audio.pt` | Audio features | 10,849 |
| `beat.pt` | Supplied beat timestamps | 10,689 |

All three checkpoints were trained from random initialization using this repository's
synthetic generator and procedural targets. They contain no third-party pretrained
weights, music recordings, or participant data. The models approximate a known rule;
they are not trained on human haptic preferences and have not been validated on hardware.

Each checkpoint contains a state dict, input mode, training config, dataset ID, and
training environment/provenance. These are the unchanged validation-selected `best.pt`
files from the original runs. Optimizer and random-number states are not included.
Use `torch.load(path, map_location="cpu", weights_only=True)` to load them.

[manifest.json](manifest.json) records the MIT license, SHA-256 checksum, dataset ID,
and original training source commit for each checkpoint. These files are licensed
explicitly; this release makes no commitment about the license or publication of future weights.

## Evaluation

From the repository root:

```bash
uv sync --locked
uv run python -m felt.evaluate --checkpoint models/001/combined.pt --out runs/reference-evaluation
```

Repeat with `audio.pt` or `beat.pt` for the ablations. The evaluator regenerates the
synthetic data and verifies the dataset ID. See the [experiment report](../../reports/001/README.md)
for all metrics, data splits, and limitations. The combined model has residual quiet-region
intensity and a false startup pulse. Audio previews are inspection tools, not actuator signals.
