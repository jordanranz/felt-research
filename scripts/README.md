# Research utility commands

Run these from the repository root with `uv run`. Scripts coordinate workflows;
reusable generation logic belongs in `src/felt/` and is tested there.

## Generate test audio

```bash
uv run python scripts/generate_test_audio.py --split test --count 12 --out data/fixtures/demo
```

Each clip gets source WAV audio, beat timestamps, target envelope JSON, an audible target
preview, and feature arrays. `index.json` records configuration, dataset identity, source
seeds, family IDs, relative paths, and file checksums. Exporting again with the same config
to a new directory produces the same fixture contents. Nonempty directories are refused.

The exporter reuses Experiment 001's generator. It does not change the training dataset,
train a model, or produce learned predictions. WAV files are PCM16 inspection copies;
training uses deterministically regenerated float32 audio. The default test split has 48
clips; `--count` selects the first N in manifest order, not new random examples.

## Training and evaluation

The startup/model preflight is separate from the earlier loss pilot:

```bash
uv run python scripts/run_preflight.py --stage screen --out runs/new-screen
uv run python scripts/run_preflight.py --stage confirm --candidate warmup-linear-v2 --out runs/new-confirm
```

It runs sequentially, keeps data fixed across initialization seeds, evaluates validation
data only, and records gates and runtime probes. See `reports/preflight002/PLAN.md`.

For the bounded, validation-only loss comparison:

```bash
uv run python scripts/run_loss_pilot.py --out runs/pilot001
```

The plan lives in `configs/pilot001.json`; its fixed gates and interpretation are in
`reports/pilot001/`. Use a fresh output directory when reproducing it. The command exits
with an error on a failed subprocess and never starts an overnight job automatically.
For other tuning evaluations, explicitly pass `--split val` to `felt.evaluate`.

Training currently generates its full dataset in memory. Exporting WAV fixtures is
optional, not an input step for training:

```bash
uv run python -m felt.train --out runs/new-combined
uv run python -m felt.evaluate --checkpoint runs/new-combined/best.pt --out runs/new-combined/evaluation
```

The first experiment contains 288 eight-second clips, or 38.4 minutes of synthetic audio.
Generation computes samples directly and does not play them in real time.

Overnight work becomes useful for larger datasets, multiple seeds, or longer training.
First run a short validation job, inspect previews and loss, estimate runtime and storage,
then run a bounded sweep with separate run directories. Save resumable checkpoints and
keep held-out test data out of tuning. More epochs or more versions of the same simple
generator do not establish generalization to real music. No overnight job is scheduled here.

## Bounded 10x synthetic study

```bash
uv run python scripts/run_scale_study.py --out runs/scale003
```

The fixed plan is in `reports/scale003/PLAN.md`. It checks disjoint rhythm families
against prior development data, trains three predetermined initialization seeds,
and evaluates all three on a fresh held-out test split after training finishes.
Each run begins with a three-epoch timing check and resumes its saved checkpoint.
It retains the same synthetic instruments and teacher; this is not a real-music test.

## Sound diversity and transfer

```bash
uv run python scripts/run_diversity_study.py --baseline-runs runs/scale003 --out runs/diversity004
```

Reproduce study 003 first if its three local checkpoints are unavailable. This command
checks the first fixed baseline on diverse validation clips, trains three models on
`synthetic-diverse-v2`, then compares every original/retrained pair on the same new
test split. The generator uses only procedural sounds. The plan is in
`reports/diversity004/PLAN.md`.

For explicit evaluation on a different dataset with the same feature/target contract:

```bash
uv run python -m felt.evaluate --checkpoint runs/scale003/seed7/best.pt --dataset-config configs/diversity004.json --split val --out runs/transfer-check
```

Both training and evaluation dataset IDs, the checkpoint checksum, and profile-level
metrics are recorded. Model architecture always comes from the checkpoint. Default
evaluation still requires the original dataset identity. The constant-mean comparison
uses the evaluation dataset's training split, including for transfer evaluations.
