# Preflight 002: model boundaries and readiness for a larger synthetic study

The failed loss pilot motivates changes to the model's startup behavior and output
parameterization. Keep the original active-weighted loss, data seed 42, architecture
width, and 30-epoch optimizer budget. Test data is not used for model selection.

## Diagnostic and development screening

Evaluate original weights on all-zero features and with 28 zero input frames prefixed.
This tests whether hidden-layer zero padding contributes to the startup transient.

Train three model versions at initialization seed 42 on the same data:

- `causal-tcn-v1`: unchanged baseline.
- `warmup-sigmoid-v2`: add 28 silent input-history frames, then discard their outputs.
- `warmup-linear-v2`: same history plus a bounded linear output, clamped to [0,1],
  with output bias initialized to 0.1. This permits exact zero; check for dead outputs.

All processing stays causal. Silent history is an explicit reset assumption; it is not
valid to reset at arbitrary chunk boundaries. A stateful reference predictor retains
28 frames between chunks. Existing v1 weights retain their old interpretation.

Screen on validation using the previous loss pilot's gates: halve quiet mean and startup
RMSE, active RMSE no worse than 1.2 times baseline, event F1 no worse than baseline minus .02.
If one qualifies, run paired baseline/candidate comparisons at initialization seeds
7, 19, 42, 73, 101. Seed 42 is a development seed, not independent confirmation.
Keep dataset seed 42 fixed; initialization seeds must not affect data or splits.

## Readiness gates

- Candidate passes all four relative gates for every paired seed.
- Candidate validation event F1 >= .95 and active RMSE <= .08 at every seed.
- No collapsed runs, NaNs, or out-of-range outputs.
- Complete silence with no beats: maximum predicted intensity <= .02, at every seed.
- Whole-clip and chunked predictions match within 1e-6 absolute / 1e-5 relative tolerance.
- Causality, old-checkpoint loading, and exact CPU checkpoint resumption pass.
- Silence, beat-only, impulse, tone, and noise probes produce finite bounded outputs.
- Per-frame CPU model-inference p95 below 10 ms after warmup; explicitly exclude
  feature extraction, audio I/O, transport, actuator delays, and OS tail-latency guarantees.

These are engineering gates for expanding a synthetic experiment, not perceptual claims.
Do not launch a scaled or overnight run. Publish failed candidates and measured runtime.
If a development candidate fails, record it before designing a follow-up; do not silently
weaken gates. Fresh sound generators and a newly reserved final holdout are needed for
claims beyond this repeatedly used synthetic generator and validation set.
