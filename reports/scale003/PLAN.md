# Scale study 003

Freeze the warmup-linear-v2 candidate and current teacher, features, optimizer and
30-epoch schedule. Train three initialization seeds, 7, 19 and 42, on a new dataset
seed 314159 with 480/120/120 rhythm families and four variants each. This gives
1,920 training, 480 validation and 480 test clips, ten times the previous counts.
Verify zero canonical-family overlap with all earlier development splits before
training. This expands rhythm coverage but does not expand instrument timbres.

Use CPU with four threads. Start each run with three epochs, preserving the full
schedule, then resume the exact checkpoint. Stop if that pilot exceeds 180 seconds
or produces invalid validation loss. Feature/target arrays total about 55 MB;
framework activations and temporary allocations add memory. This bounded study does
not require a streaming dataset loader or saving all source WAVs.

Finish all three training runs before test evaluation. Checkpoint selection uses
validation loss only. Evaluate all three seeds once on the new test split and
report all results, including procedural and beat-pulse baselines. Do not select a
winning seed or tune using test results. Descriptive acceptance targets are active
RMSE <= 0.08, event F1 >= 0.95, quiet mean <= 0.02 and startup RMSE <= 0.02 for
every seed. Failure is a reportable outcome, not permission for test-set tuning.

This is a synthetic scaling and generalization check within the same generator,
not evidence of real-music quality, listener preference, or actuator performance.
Run with `uv run python scripts/run_scale_study.py --out runs/scale003`.
