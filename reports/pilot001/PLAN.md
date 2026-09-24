# Short pilot: silence and startup behavior

Plan recorded before executing the pilot. This tunes on validation data only.
The already-published Experiment 001 test results motivated the question; they are not
used to select among these candidates or to claim a fresh independent test outcome.

Use the unchanged generator, train/validation splits, combined inputs, seed 42,
30-epoch budget, model, batch size, and optimizer recipe. Run sequentially on CPU.
Measure whole-process training and evaluation wall time, including data generation,
startup, and artifact writes. This is not GPU benchmarking.

Compare:

1. Existing active-weighted MSE.
2. Region-balanced MSE: 50% mean active error, 35% mean quiet error, 15% mean transition
   error. Active targets exceed 0.1; quiet targets are below 0.02. Empty regions contribute zero.
3. The same balanced loss plus mean squared error over the first 29 frames, matching
   the model's receptive-field length. This includes real early pulses, not just silence.

Each run selects its checkpoint by its own validation loss. Those loss values have
different scales and must not be ranked against each other. Compare shared metrics instead.

A candidate advances to multi-seed validation only if all gates hold relative to baseline:

- Quiet-region mean predicted intensity is at most 50% of baseline.
- Startup RMSE over the first 29 frames is at most 50% of baseline.
- Active-region RMSE increases by at most 20%.
- Event F1 drops by at most 0.02 absolute.

These are engineering screening thresholds, not established perceptual thresholds.
Do not extend runs, change coefficients, or evaluate test data in response to pilot outcomes.
If no candidate passes, document the tradeoff and propose the next experiment.
One seed cannot establish robustness. Before a multi-seed study, separate model seeds
from dataset seeds to keep data fixed across initializations.

```bash
uv run python scripts/run_loss_pilot.py --out runs/pilot001
```
