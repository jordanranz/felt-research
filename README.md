# Felt Research

Reproducible model training and evaluation for haptics and sensory interaction.

**First completed run:** [Experiment 001 results and audio previews](reports/001/README.md).

**Follow-up:** [Short loss pilot](reports/pilot001/README.md). Two alternatives failed the
validation gates; the original baseline remains the reference model.

**Preflight:** [Startup fixes and five-seed checks](reports/preflight002/README.md).
The opt-in `warmup-linear-v2` candidate passed all gates for a larger synthetic study.
Use `configs/preflight002-candidate.json`; the default Experiment 001 config and released
weights remain unchanged for reproduction.

**Scaling:** [Tenfold dataset study](reports/scale003/README.md). Three fixed seeds passed
on new synthetic rhythm families.

**Sound diversity:** [Transfer and retraining results](reports/diversity004/README.md).
Broader procedural audio improved mean held-out event F1 from 0.879 to 0.982.
Sustained tones remain the weakest profile. [Physical playback evaluation](docs/physical-evaluation.md)
is the next proposed check; synthetic accuracy does not establish useful sensations.

The first task maps synthetic audio and supplied beat timestamps to a causal,
100 Hz intensity envelope. The target is a published procedural rule. Learning it
validates this pipeline; it does not establish perceptual quality or novelty.

Product applications, custom ring firmware, CAD, and PCB engineering live outside
this repository. No Spotify, Deezer, Splice, participant, or third-party audio is included.

## Run locally

Repository boundaries:

- `src/felt/`: reusable Python research code and audio utilities.
- `apps/website/` and `apps/visualizer/`: documented locations for future research web apps.
- `scripts/`: thin utility commands, starting with reproducible test-audio export.
- `configs/`, `reports/`, and `models/`: experiment settings, findings, and released weights.

See [utility commands](scripts/README.md) for fixture generation and the training workflow.
The web applications are not implemented yet and add no frontend dependency to training.

Install Python 3.12 and [uv](https://docs.astral.sh/uv/), then:

```bash
uv sync --locked
uv run pytest
uv run python -m felt.train --config configs/experiment001.json --out runs/combined
uv run python -m felt.evaluate --checkpoint runs/combined/best.pt --out runs/combined/evaluation
uv run python -m felt.preview --pattern runs/combined/evaluation/model.json --out runs/preview.wav
```

CPU is the reference backend. Use `--device mps` on supported Macs or `--device cuda`
on a compatible NVIDIA installation. GPU runs need their own verification; cross-device
bitwise equality is not promised. The environment lock supports platform-specific dependencies.

Compare input ablations using separate run directories:

```bash
uv run python -m felt.train --mode audio --out runs/audio
uv run python -m felt.train --mode beat --out runs/beat
```

Evaluate each `best.pt` using the same evaluation command. Checkpoint selection uses
validation loss only. Test results must not become a hyperparameter search target.

## Resume

```bash
uv run python -m felt.train --out runs/resume-demo --stop-after 5
uv run python -m felt.train --out runs/resume-demo --resume runs/resume-demo/last.pt
```

Checkpoints are atomic and resumable at **epoch boundaries**, not arbitrary batches.
The dataset is regenerated and checked by hash. The config, code hash, key dependency
versions, and platform must match. Checkpoints include the optimizer, scheduler,
best model, history, and random-number states. Floating-point synthesis and FFT
results can differ slightly across platforms, changing the dataset byte hash.
Exact resume is therefore environment-specific; cross-platform transfer evaluation
requires an explicit `--dataset-config` and records both dataset identities. Load full training checkpoints only
from trusted sources because they contain pickled Python state.

## Data and target

The generator produces repeating 16-step patterns from silence, kick, snare, and hat.
It groups cyclic rotations into one family before splitting. Each family has four
variations in tempo, onset offset, and sound strength. Defaults produce 192 training,
48 validation, and 48 test clips of eight seconds each at 16 kHz.

Audio features are frame RMS and low/high spectral energy with fixed scaling, plus
a beat impulse and decaying beat trace. There is no clip-level normalization.
The teacher combines `0.7 * audio_RMS_feature + 0.3 * beat_impulse`, clipped to `[0,1]`,
with an exponential release of `0.78` per frame. The original model is a small causal TCN with
a 29-frame receptive field and sigmoid output. The opt-in `warmup-linear-v2` model
adds silent prehistory and a bounded linear output head. Weighted MSE gives target frames above
0.1 five times the weight. AdamW uses cosine learning-rate decay and gradient clipping.

The deterministic teacher is an oracle with zero target error. The model is not expected
to beat it. Other baselines are beat pulses, the training-target mean, and silence.

## Time and output contract

Frame `i` reads audio samples `[160*i, 160*(i+1))` and beat timestamps in that same
10 ms bin. Its output becomes available at `(i+1)*0.01` seconds and lasts 10 ms.
This is zero lookahead beyond the current completed frame, with up to 10 ms of
framing delay plus computation. Future beats are not encoded. Supplied beat timestamps
remain an idealized input; live beat tracking is outside this experiment.

`pattern.json` has a schema version, frame rate, start time, and intensity array.
Intensity is a normalized request, not acceleration, motor frequency, or perceptual
equivalence between devices. No hardware drive signal is exported.

Audio previews use a fixed 180 Hz tone at fixed gain, a causal 3 ms amplitude smoother,
and a short final fade. They do not normalize each prediction. `comparison.wav` places
the source audio on the left and the model's sonification on the right. These are
debugging tools, not evidence of comfortable physical haptics.

## Evaluation and artifacts

Each run records its resolved config, complete dataset manifest, environment, JSON
metric history, and checkpoints. Evaluation exports RMSE, active-region RMSE, silence
intensity, threshold-crossing event precision/recall, and matched-event timing error.
Events use a fixed 0.2 threshold and a 30 ms matching tolerance. Timing error excludes
misses, so always read it beside event recall. These thresholds are experiment conventions.

Latency is measured on CPU for one whole clip after warmup and excludes preprocessing
and playback. It is not a streaming latency benchmark. Multi-seed synthetic studies and feature-streaming checks are documented in the
reports above. Real recordings and human tests are still needed before broader claims.

`runs/`, datasets, and weights are ignored by Git except for the explicitly released
reference weights under `models/001/`. Selected reports and synthetic previews
are published under `reports/`. JSON tracking keeps the first pipeline self-contained;
a hosted or local experiment-tracking service can be added when run comparison needs it.

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv build
```

Tests cover family isolation, deterministic generation, preprocessing/model causality,
timebase and preview gain, exact CPU epoch-boundary resumption, and evaluation exports.

## Project status and provenance

Early research infrastructure, built with AI-assisted implementation and review.
This repository implements its own synthetic generator and small TCN; it contains no
copied Sound2Hap or HapticGen code, weights, or datasets. Those projects informed the
broader research discussion. Human preference learning and hardware calibration remain future work.

## License

The original code, documentation, synthetic example artifacts, and the original
Experiment 001 weights explicitly listed in [models/001](models/001/README.md)
are released under the [MIT License](LICENSE).

This license applies to the released materials, including those named weights.
It does not license unpublished product code, ring engineering, or future model releases.
Third-party dependencies retain their own licenses. Any future third-party datasets or
derived models must document their applicable terms separately.

The reference checkpoints are available without training:

```bash
uv run python -m felt.evaluate --checkpoint models/001/combined.pt --out runs/reference-evaluation
```
